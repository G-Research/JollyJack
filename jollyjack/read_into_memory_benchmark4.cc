#include "arrow/status.h"
#include "arrow/io/memory.h"
#include "arrow/util/parallel.h"
#include "parquet/column_reader.h"
#include "parquet/types.h"

#include "jollyjack.h"

#include <liburing.h>
#include <iostream>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

#if defined(__x86_64__)
  #include <immintrin.h>
#endif

using arrow::Status;


struct Request {
  int64_t offset;
  int64_t length;
  std::shared_ptr<arrow::Buffer> buffer;
};

// io_uring, no coalescing
void ReadIntoMemory_benchmark4(
  const std::string& path,
  std::shared_ptr<parquet::FileMetaData> file_metadata,
  void* buffer,
  size_t buffer_size,
  size_t stride0_size,
  size_t stride1_size,
  std::vector<int> column_indices,
  const std::vector<int>& row_groups,
  const std::vector<int64_t>& target_row_ranges,
  const std::vector<std::string>& column_names,
  const std::vector<int>& target_column_indices,
  bool pre_buffer,
  bool use_threads,
  int64_t expected_rows, 
  arrow::io::CacheOptions cache_options)
{
  std::atomic<size_t> read_bytes(0);
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::logic_error("Failed to open file: " + path + " - " + strerror(errno));
  }

  auto reader_properties = parquet::default_reader_properties();
  auto parquet_reader = parquet::ParquetFileReader::OpenFile(path, false, reader_properties, file_metadata);

  if (pre_buffer)
  {
    parquet_reader->PreBuffer(row_groups, column_indices, arrow::io::default_io_context(), cache_options);
  }

  // Initialize io_uring
  struct io_uring ring = {};
  int ret = io_uring_queue_init(column_indices.size(), &ring, 0);
  if (ret < 0) {
    throw std::logic_error(
      "Failed to initialize io_uring: " + std::string(strerror(-ret))
    );
  }

  std::vector<int> single_row_group(1);
  std::vector<int> single_column(1);
  std::vector<Request> requests;
  requests.resize(column_indices.size()); // reserve enough memory to avoid reallocations

  // Process each row group separately to maintain target_row tracking
  for (int row_group : row_groups) {
    size_t read_range_idx = 0;
    auto row_group_reader = parquet_reader->RowGroup(row_group);
    std::shared_ptr<parquet::RowGroupMetaData> row_group_metadata = file_metadata->RowGroup(row_group);
    single_row_group[0] = row_group;

    for (size_t c_idx = 0; c_idx < column_indices.size(); c_idx++) {
      single_column[0] = column_indices[c_idx];
      
      auto &request = requests[read_range_idx++];
      auto ranges = parquet_reader->GetReadRanges(
        single_row_group, single_column, 0, 1
      ).ValueOrDie();

      request.length = ranges[0].length;
      request.offset = ranges[0].offset;
    }

    for (size_t i = 0; i < requests.size(); i++) {
      auto& request = requests[i];
      
      struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
      if (!sqe) {
        throw std::logic_error("Failed to get SQE from io_uring");
      }

      // Allocate buffer for this coalesced request
      auto buffer_result = arrow::AllocateBuffer(request.length);
      if (!buffer_result.ok()) {
        throw std::logic_error(
          "Unable to AllocateBuffer: " + 
          buffer_result.status().message()
        );
      }
      request.buffer = std::move(buffer_result.ValueOrDie());

      // Prepare and queue read operation
      io_uring_prep_read(
        sqe, fd, request.buffer->mutable_data(),
        request.length, request.offset
      );
      io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(i));
    }

    auto submitted = io_uring_submit(&ring);

    size_t completed = 0;
    while (completed < requests.size()) {
      struct io_uring_cqe* cqe;
      int ret = io_uring_wait_cqe_timeout(&ring, &cqe, NULL);
    
      if (ret == -ETIME || ret == -EINTR) {
        continue;
      }
    
      if (ret < 0) {
        throw std::logic_error(
          "Failed to wait io_uring: " + std::string(strerror(-ret))
        );
      }

      completed++;

      // Extract request index and validate completion
      size_t request_idx = reinterpret_cast<size_t>(io_uring_cqe_get_data(cqe));
      auto& request = requests[request_idx];
      request.buffer.reset();
      read_bytes.fetch_add(cqe->res);
      io_uring_cqe_seen(&ring, cqe);
    }
  }

  close(fd);  
  io_uring_queue_exit(&ring);
  *(float*)buffer = (float)read_bytes.fetch_add(0);
}
