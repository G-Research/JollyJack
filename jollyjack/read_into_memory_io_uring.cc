#include "arrow/status.h"
#include "arrow/io/memory.h"
#include "arrow/util/parallel.h"
#include "parquet/column_reader.h"
#include "parquet/types.h"

#include "jollyjack.h"

#include <iostream>
#include <stdlib.h>
#include <liburing.h>
#include <fcntl.h>

#if defined(__x86_64__)
  #include <immintrin.h>
#endif

using arrow::Status;

void ReadIntoMemoryIOUring (const std::string& path
    , std::shared_ptr<parquet::FileMetaData> file_metadata
    , void* buffer
    , size_t buffer_size
    , size_t stride0_size
    , size_t stride1_size
    , std::vector<int> column_indices
    , const std::vector<int> &row_groups
    , const std::vector<int64_t> &target_row_ranges
    , const std::vector<std::string> &column_names
    , const std::vector<int> &target_column_indices
    , bool pre_buffer
    , bool use_threads
    , int64_t expected_rows)
{
  if (target_row_ranges.size() % 2 != 0)
  {
    throw std::logic_error("target_row_ranges must contain pairs of [start, end) indices");
  }

  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    auto msg = std::string("Failed to open file: ") + path + " - " + strerror(errno);
  }

  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  auto arrowReaderProperties = parquet::default_arrow_reader_properties();
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::OpenFile(path, false, reader_properties, file_metadata);
  file_metadata = parquet_reader->metadata();

  if (column_names.size() > 0)
  {
      column_indices.reserve(column_names.size());
      auto schema = file_metadata->schema();
      for (auto column_name : column_names)
      {
        auto column_index = schema->ColumnIndex(column_name);
         
        if (column_index < 0)
        {
          auto msg = std::string("Column '") + column_name + "' was not found!";
          throw std::logic_error(msg);
        }

        column_indices.push_back(column_index);
      }
  }

  struct Request
  {
    int row_group;
    int column_counter;
    int column_index;
    int64_t offset;
    int64_t length;
    int64_t target_row = 0;
    std::shared_ptr<arrow::Buffer> buffer;
  };
    
  std::vector<Request> requests(column_indices.size());
  struct io_uring ring = {};
  unsigned int queue_depth = requests.size();

   // Initialize io_uring
  int ret = io_uring_queue_init(queue_depth, &ring, 0);
  if (ret < 0) {
    auto msg = std::string("Failed to initialize io_uring: ") + strerror(-ret);
    throw std::logic_error(msg);
  }

  std::vector<int> row_groups_for_read_ranges(1);
  std::vector<int> column_indices_for_read_ranges(1);
  int64_t target_row = 0;
  size_t target_row_ranges_idx = 0;
  for (size_t r_idx = 0; r_idx < row_groups.size(); r_idx++)
  {
    size_t requestIndex = 0;
    auto row_group = row_groups[r_idx];
    auto row_group_metadata = file_metadata->RowGroup(row_group);
    for (size_t c_idx = 0; c_idx < column_indices.size(); c_idx++)
    {
      Request& request = requests[requestIndex++];
      request.row_group = row_group;
      request.column_counter = c_idx;
      request.column_index = column_indices[c_idx];      
      request.target_row = target_row;
      row_groups_for_read_ranges[0] = request.row_group;
      column_indices_for_read_ranges[0] = request.column_index;
      
      auto const& read_ranges = parquet_reader->GetReadRanges(row_groups_for_read_ranges, column_indices_for_read_ranges, 0, 1).ValueOrDie();
      request.offset = read_ranges[0].offset;
      request.length = read_ranges[0].length;
    }

    for (size_t requestIndex = 0; requestIndex < requests.size(); requestIndex++)
    {
      Request& request = requests[requestIndex];
      struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
      if (!sqe) {
        break; // Queue full, try again later
      }

      // Allocate buffer
      auto buffer_result = arrow::AllocateBuffer(request.length);
      if (!buffer_result.ok()) {
      throw std::logic_error(std::string("Unable to AllocateResizableBuffer: ") + buffer_result.status().message());
      }

      request.buffer = std::move(buffer_result).ValueOrDie();

      // Prepare read operation
      io_uring_prep_read(sqe, fd, request.buffer->mutable_data(), 
                        request.length, request.offset);
      io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(requestIndex));
    }

    io_uring_submit(&ring);

    for (size_t request_counter = 0; request_counter < requests.size(); )
    {
      struct io_uring_cqe* cqe;
      int ret = io_uring_wait_cqe_timeout(&ring, &cqe, NULL); 
      if (ret == -ETIME || ret == -EINTR) {
        continue;
      }

      request_counter++;

      if (ret < 0) 
      {
        auto msg = std::string("Failed to wait io_uring: ") + strerror(-ret);
        throw std::logic_error(msg);
      }

      // Process completion
      uint64_t requestIndex = reinterpret_cast<uint64_t>(io_uring_cqe_get_data(cqe));
      Request& request = requests[requestIndex];

      if (cqe->res < 0) {
        // Error occurred
        auto msg = std::string("Read failed: ") + strerror(-cqe->res);
        throw std::logic_error(msg);
      } 
      else if (cqe->res != request.length) {
        // Success? - resize buffer to actual bytes read and complete future
        auto msg = std::string("Read failed? cqe->res != request.length: ") + std::to_string(cqe->res) + " != " + std::to_string(request.length);
        throw std::logic_error(msg);
      }

      io_uring_cqe_seen(&ring, cqe);

      const auto& col_metadata = row_group_metadata->ColumnChunk(request.column_index);
      std::shared_ptr<parquet::ArrowInputStream> data = std::make_shared<::arrow::io::BufferReader>(request.buffer);
      auto page_reader = parquet::PageReader::Open(data, col_metadata->num_values(), col_metadata->compression(), reader_properties, false, nullptr);
      auto descr = file_metadata->schema()->Column(request.column_index);
      auto column_reader = parquet::ColumnReader::Make (descr, std::move(page_reader));

      auto status = ReadColumn (request.column_counter
      , request.target_row
      , column_reader
      , row_group_metadata.get()
      , buffer
      , buffer_size
      , stride0_size
      , stride1_size
      , column_indices
      , target_column_indices
      , target_row_ranges
      , target_row_ranges_idx
      );

      request.buffer = nullptr;

      if (status != arrow::Status::OK())
      {
        throw std::logic_error(status.message());
      }
    }

    target_row += row_group_metadata->num_rows();
    if (target_row_ranges.size() > 0)
    {
      auto rows = row_group_metadata->num_rows();
      while (true)
      {
        auto range_rows = target_row_ranges[target_row_ranges_idx + 1] - target_row_ranges[target_row_ranges_idx];
        target_row_ranges_idx += 2;
        if (rows == range_rows)
          break;

        rows -= range_rows;
      }
    }
  }

  // Clean up io_uring
  if (ring.ring_fd != 0)
    io_uring_queue_exit(&ring);

  if (target_row_ranges.size() > 0)
  {
    if (target_row_ranges_idx != target_row_ranges.size())
    {
      auto msg = std::string("Expected to read ") + std::to_string(target_row_ranges.size() / 2) + " row ranges, but read only " + std::to_string(target_row_ranges_idx / 2) + "!";
      throw std::logic_error(msg);
    }
  }
  else
  {
    if (target_row != expected_rows)
    {
      auto msg = std::string("Expected to read ") + std::to_string(expected_rows) + " rows, but read only " + std::to_string(target_row) + "!";
      throw std::logic_error(msg);
    }
  };
}

