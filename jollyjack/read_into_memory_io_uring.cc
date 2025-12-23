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

/**
 * Custom RandomAccessFile implementation that supports buffered reads
 * and integrates with Arrow's I/O system for efficient Parquet reading.
 */
class FantomReader : public arrow::io::RandomAccessFile {
 public:
  explicit FantomReader(int fd, int block_size);
  ~FantomReader() override;

  arrow::Result<int64_t> ReadAt(
    int64_t position, int64_t nbytes, void* out
  ) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(
    int64_t position, int64_t nbytes
  ) override;
  arrow::Result<int64_t> GetSize() override;
  
  bool closed() const override;
  arrow::Status Seek(int64_t position) override;  
  arrow::Status Close() override;
  arrow::Result<int64_t> Tell() const override;
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override;
  
  // Set a pre-loaded buffer for a specific file offset range
  void SetBuffer(int64_t buffer_offset, std::shared_ptr<arrow::Buffer> buffer);
  int64_t GetBlockSize();
 private:
  int fd_;
  int64_t pos_= 0;
  int64_t file_size_ = 0;
  int64_t block_size_ = 0;
  std::shared_ptr<arrow::Buffer> cached_buffer_;
  int64_t cached_buffer_offset_ = 0;
};

FantomReader::FantomReader(int fd, int block_size)
    : fd_(fd), pos_(0), file_size_(0) {
  struct stat file_stats;
  if (fstat(fd_, &file_stats) < 0) {
    throw std::runtime_error("Failed to get file statistics with fstat");
  }

  file_size_ = file_stats.st_size;
  block_size_ = std::max(block_size, 1);
}

FantomReader::~FantomReader() {
  (void)Close();
}

arrow::Status FantomReader::Close() {
  return arrow::Status::OK();
}

bool FantomReader::closed() const {
  return false;
}

arrow::Result<int64_t> FantomReader::GetSize() {
  return file_size_;
}

arrow::Result<int64_t> FantomReader::ReadAt(int64_t position, int64_t nbytes, void* out) {
  return pread(fd_, out, nbytes, position);
}

arrow::Result<std::shared_ptr<arrow::Buffer>> FantomReader::ReadAt(int64_t position, int64_t nbytes) {
  // Check if the request can be served from the cached buffer
  if (cached_buffer_ != nullptr && 
      position >= cached_buffer_offset_ && 
      cached_buffer_offset_ + cached_buffer_->size() >= position + nbytes) {
    return arrow::SliceBuffer(cached_buffer_, position - cached_buffer_offset_, nbytes);
  }

  // If we have a cached buffer but can't serve the request, throw an error
  if (cached_buffer_ != nullptr) {
    auto error_message = std::string("ReadAt failed - request cannot be served from cached buffer. ") +
          "Buffer offset=" + std::to_string(cached_buffer_offset_) + 
          ", buffer size=" + std::to_string(cached_buffer_->size()) + 
          ", requested position=" + std::to_string(position) + 
          ", requested bytes=" + std::to_string(nbytes);
    return arrow::Status::UnknownError(error_message);
  }

  // Calculate aligned offset and length using bit hacks
  int64_t align_down_mask = ~(block_size_ - 1);
  int64_t aligned_start_offset = position & align_down_mask;
  int64_t original_end_offset = position + nbytes;
  int64_t aligned_end_offset = (original_end_offset + block_size_ - 1) & align_down_mask;
  int64_t aligned_read_length = aligned_end_offset - aligned_start_offset;
  int64_t updated_nbytes = nbytes + (position - aligned_start_offset);

  // Allocate new buffer and read from file
  ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(aligned_read_length, block_size_));
  ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(aligned_start_offset, aligned_read_length, buffer->mutable_data()));

  if (bytes_read < updated_nbytes) {
    throw std::logic_error(
      "Incomplete read: expected " + std::to_string(nbytes) + 
      " bytes, but read " + std::to_string(bytes_read) + " bytes"
    );
  }

  return arrow::SliceBuffer(std::move(buffer), (position - aligned_start_offset), nbytes);
}

arrow::Status FantomReader::Seek(int64_t position) {
  pos_= position;
  return arrow::Status::OK();
}

arrow::Result<int64_t> FantomReader::Tell() const {
  return pos_;
}

arrow::Result<int64_t> FantomReader::Read(int64_t nbytes, void* out) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
  memcpy(out, buffer->data(), buffer->size());
  pos_+= buffer->size();
  return buffer->size();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> FantomReader::Read(int64_t nbytes) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
  pos_+= buffer->size();
  return buffer;
}

void FantomReader::SetBuffer(int64_t buffer_offset, std::shared_ptr<arrow::Buffer> buffer) {
  cached_buffer_ = buffer;
  cached_buffer_offset_ = buffer_offset;
}

int64_t FantomReader::GetBlockSize() {
  return block_size_;
}

// Represents a single column that needs to be read
struct ColumnReadOperation {
  int column_array_index;     // Index in the output column array
  int parquet_column_index;   // Index in the Parquet file schema
  std::shared_ptr<parquet::ColumnReader> column_reader;
};

// Represents a coalesced I/O request that reads multiple columns in one operation
struct CoalescedIORequest {
  int64_t file_offset;
  int64_t read_length;
  std::shared_ptr<arrow::ResizableBuffer> read_buffer;
  std::vector<ColumnReadOperation> column_operations;
};

// Information about a column's file range for coalescing optimization
struct ColumnFileRange {
  int64_t file_offset;
  int64_t data_length;
  size_t column_array_index;
  
  int64_t end_offset() const { return file_offset + data_length; }
};

// Validate that row ranges are properly paired
void ValidateRowRangePairs(const std::vector<int64_t>& row_ranges) {
  if (row_ranges.size() % 2 != 0) {
    throw std::logic_error(
      "Row ranges must contain pairs of [start, end) indices, but got odd number of elements"
    );
  }
}

// Open Parquet file and create necessary readers
std::tuple<int, std::shared_ptr<FantomReader>, std::unique_ptr<parquet::ParquetFileReader>>
OpenParquetFileForReading(const std::string& file_path, std::shared_ptr<parquet::FileMetaData> metadata, int flags, int block_size) {
  int fd = open(file_path.c_str(), flags);
  if (fd < 0) {
    throw std::logic_error("Failed to open file: " + file_path + " - " + strerror(errno));
  }

  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  auto fantom_reader = std::make_shared<FantomReader>(fd, block_size);
  auto parquet_reader = parquet::ParquetFileReader::Open(fantom_reader, reader_properties, metadata);

  return {fd, fantom_reader, std::move(parquet_reader)};
}

// Convert column names to column indices using schema lookup
void ResolveColumnNameToIndices(
  std::vector<int>& resolved_column_indices,
  const std::vector<std::string>& column_names,
  const std::shared_ptr<parquet::FileMetaData>& file_metadata
) {
  if (column_names.empty()) {
    return;
  }

  resolved_column_indices.reserve(column_names.size());
  auto schema = file_metadata->schema();
  
  for (const auto& column_name : column_names) {
    int column_index = schema->ColumnIndex(column_name);

    if (column_index < 0) {
      throw std::logic_error("Column '" + column_name + "' was not found!");
    }

    resolved_column_indices.push_back(column_index);
  }
}

// Get column file ranges sorted by offset for efficient coalescing
std::vector<ColumnFileRange> GetSortedColumnRanges(
  parquet::ParquetFileReader* parquet_reader,
  int row_group_index,
  const std::vector<int>& column_indices
) {
  std::vector<int> single_row_group = {row_group_index};
  std::vector<int> single_column(1);

  std::vector<ColumnFileRange> column_ranges;
  column_ranges.reserve(column_indices.size());

  // Get individual read ranges for each column
  for (size_t column_array_index = 0; column_array_index < column_indices.size(); column_array_index++) {
    single_column[0] = column_indices[column_array_index];

    auto ranges = parquet_reader->GetReadRanges(single_row_group, single_column, 0, 1).ValueOrDie();
    column_ranges.emplace_back();
    ColumnFileRange& range_info = column_ranges.back();
    range_info.file_offset = ranges[0].offset;
    range_info.data_length = ranges[0].length;
    range_info.column_array_index = column_array_index;
  }

  // Sort by file offset for efficient matching with coalesced ranges
  std::sort(column_ranges.begin(), column_ranges.end(),
    [](const ColumnFileRange& first, const ColumnFileRange& second) {
      return first.file_offset < second.file_offset;
    });

  return column_ranges;
}

// Match individual column ranges to coalesced read ranges
std::vector<CoalescedIORequest> MatchColumnsToCoalescedRanges(
  const std::vector<ColumnFileRange>& sorted_column_ranges,
  const std::vector<arrow::io::ReadRange>& coalesced_ranges,
  const std::vector<int>& column_indices
) {
  std::vector<CoalescedIORequest> coalesced_requests;
  coalesced_requests.reserve(coalesced_ranges.size());

  auto it = sorted_column_ranges.begin();
  for (const auto& coalesced_range : coalesced_ranges) {
    // Construct directly in the vector
    coalesced_requests.emplace_back();
    CoalescedIORequest& request = coalesced_requests.back();
    request.file_offset = coalesced_range.offset;
    request.read_length = coalesced_range.length;
    int64_t range_end = coalesced_range.offset + coalesced_range.length;

    // Iterate only through potentially overlapping columns
    for (; it != sorted_column_ranges.end(); ++it) {
      if (it->file_offset >= range_end) {
        break;
      }
      
      request.column_operations.emplace_back();
      ColumnReadOperation &column_op = request.column_operations.back();
      column_op.column_array_index = it->column_array_index;
      column_op.parquet_column_index = column_indices[it->column_array_index];
    }
  }

  return coalesced_requests;
}

// Submit all coalesced I/O requests to io_uring for parallel execution
void SubmitIORequests(
  struct io_uring& ring,
  std::vector<CoalescedIORequest>& io_requests,
  int fd,
  int64_t block_size
) {
  for (size_t request_index = 0; request_index < io_requests.size(); request_index++) {
    auto& request = io_requests[request_index];
    
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
    if (!sqe) {
      throw std::logic_error("Failed to get submission queue entry from io_uring");
    }

    // Calculate aligned offset and length using bit hacks
    int64_t align_down_mask = ~(block_size - 1);
    int64_t aligned_start_offset = request.file_offset & align_down_mask;
    int64_t original_end_offset = request.file_offset + request.read_length;
    int64_t aligned_end_offset = (original_end_offset + block_size - 1) & align_down_mask;
    int64_t aligned_read_length = aligned_end_offset - aligned_start_offset;
    request.read_length = request.read_length + request.file_offset - aligned_start_offset;
    request.file_offset = aligned_start_offset;

    auto buffer_result = arrow::AllocateResizableBuffer(aligned_read_length, block_size);
    if (!buffer_result.ok()) {
      throw std::logic_error("Failed to allocate buffer: " + buffer_result.status().message());
    }
    request.read_buffer = std::move(buffer_result.ValueOrDie());

    // Prepare read operation for io_uring
    io_uring_prep_read(
      sqe, fd, request.read_buffer->mutable_data(),
      aligned_read_length, aligned_start_offset
    );
    io_uring_sqe_set_flags(sqe, IOSQE_ASYNC);
    io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(request_index));
  }

  int submitted_count = io_uring_submit(&ring);
  if (submitted_count < 0) {
    throw std::logic_error("Failed to submit io_uring operations: " + std::string(strerror(-submitted_count)));
  }
}

// Process a single completed I/O request and read all its columns
void ProcessSingleIOCompletion(
  int64_t current_target_row,
  CoalescedIORequest& completed_request,
  const std::shared_ptr<FantomReader>& fantom_reader,
  parquet::RowGroupMetaData* row_group_metadata,
  const std::vector<int>& column_indices,
  const std::vector<int64_t>& target_row_ranges,
  const std::vector<int>& target_column_indices,
  void* out,
  size_t buffer_size,
  size_t stride0_size,
  size_t stride1_size,
  size_t target_row_ranges_index
) {    
  // Process each column covered by this coalesced read
  for (const auto& column_operation : completed_request.column_operations) {
    auto read_status = ReadColumn(
      column_operation.column_array_index,
      current_target_row,
      column_operation.column_reader.get(),
      row_group_metadata, 
      out,
      buffer_size,
      stride0_size,
      stride1_size,
      column_indices,
      target_column_indices,
      target_row_ranges,
      target_row_ranges_index
    );

    if (!read_status.ok()) {
      throw std::logic_error("Column read failed: " + read_status.message());
    }
  }
}

// Wait for io_uring completions and setup column readers
arrow::Status WaitForIOCompletionsAndSetupReaders(
  struct io_uring& ring,
  std::vector<CoalescedIORequest>& io_requests,
  const std::shared_ptr<FantomReader>& fantom_reader,
  parquet::RowGroupReader* row_group_reader,
  std::vector<size_t>* completed_requests,
  size_t* const requests_to_complete,
  bool use_threads
) {
  size_t min_completed_requests = use_threads ? 128 : 1;
  completed_requests->clear();

  // Wait for all I/O operations to complete
  while (*requests_to_complete > 0) {
    struct io_uring_cqe* completion_entry;

    if (completed_requests->size() < min_completed_requests)
    {
      int wait_result = io_uring_wait_cqe(&ring, &completion_entry);
      if (wait_result < 0) {
        throw std::logic_error("Failed to wait for io_uring completion: " + std::string(strerror(-wait_result)));
      }
    }
    else
    {
      int wait_result = io_uring_peek_cqe(&ring, &completion_entry);
      if (wait_result != 0)
        return arrow::Status::OK();
    }

    (*requests_to_complete)--;

    // Validate the completion
    size_t request_index = reinterpret_cast<size_t>(io_uring_cqe_get_data(completion_entry));
    CoalescedIORequest& completed_request = io_requests[request_index];
    completed_requests->push_back(request_index);

    if (completion_entry->res < 0) {
      auto msg = std::string("I/O operation failed: ") + std::string(strerror(-completion_entry->res));
      return arrow::Status::UnknownError(msg);
    }

    if (completion_entry->res < completed_request.read_length) {
      auto msg = std::string("Incomplete read: expected ") + std::to_string(completed_request.read_length) + 
        " bytes, got " + std::to_string(completion_entry->res);
      return arrow::Status::UnknownError(msg);
    }

    bool shrink_to_fit = false;
    completed_request.read_buffer->Resize(completed_request.read_length, shrink_to_fit);
    fantom_reader->SetBuffer(completed_request.file_offset, completed_request.read_buffer);

    // Create column readers for each column in this request
    for (auto& column_operation : completed_request.column_operations) {
      column_operation.column_reader = row_group_reader->Column(column_operation.parquet_column_index);
    }

    io_uring_cqe_seen(&ring, completion_entry);
  }

  return arrow::Status::OK();
}

// Process all completed I/O requests, optionally in parallel
void ProcessAllCompletedRequests(
  struct io_uring& ring,
  std::vector<CoalescedIORequest>& io_requests,
  const std::shared_ptr<FantomReader>& fantom_reader,
  int64_t current_target_row,
  parquet::RowGroupReader* row_group_reader,
  parquet::RowGroupMetaData* row_group_metadata,
  const std::vector<int>& column_indices,
  const std::vector<int64_t>& target_row_ranges,
  const std::vector<int>& target_column_indices,
  void* out,
  size_t buffer_size,
  size_t stride0_size,
  size_t stride1_size,
  bool use_threads,
  size_t target_row_ranges_index
) {
  std::vector<size_t> completed_requests;
  size_t requests_to_complete = io_requests.size();
  completed_requests.reserve(io_requests.size());

  arrow::Status final_status = arrow::Status::OK();
  while (requests_to_complete > 0)
  {
    // Wait for all I/O operations and setup readers
    auto waiting_status = WaitForIOCompletionsAndSetupReaders(ring, io_requests, fantom_reader, row_group_reader, &completed_requests, &requests_to_complete, use_threads);
    if (!waiting_status.ok())
      final_status = waiting_status;

    // Process all requests, potentially in parallel
    auto processing_status = ::arrow::internal::OptionalParallelFor(
      use_threads,
      static_cast<int>(completed_requests.size()),
      [&](int i) -> Status {
        try {
          size_t request_index = completed_requests[i];
          ProcessSingleIOCompletion(
            current_target_row, io_requests[request_index], fantom_reader, row_group_metadata,
            column_indices, target_row_ranges, target_column_indices,
            out, buffer_size, stride0_size, stride1_size, target_row_ranges_index
          );
          return Status::OK();
        } catch (const std::exception& error) {
          return Status::UnknownError("Request processing failed: " + std::string(error.what()));
        }
      }
    );

    if (!processing_status.ok())
      final_status = processing_status;
  }

  if (!final_status.ok()) {
    throw std::logic_error("Parallel processing failed: " + final_status.message());
  }
}

// Main function to read Parquet data into memory using io_uring for optimal I/O performance
void ReadIntoMemoryIOUring(
  const std::string& parquet_file_path,
  std::shared_ptr<parquet::FileMetaData> file_metadata,
  void* out,
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
  bool use_o_direct,
  int64_t expected_total_rows, 
  arrow::io::CacheOptions cache_options)
{
  ValidateRowRangePairs(target_row_ranges);

  int flags = O_RDONLY;
  int block_size = 0;
  if (use_o_direct)
  {
    flags |= O_DIRECT;
    block_size = 4096;
  }

  auto [fd, fantom_reader, parquet_reader] = OpenParquetFileForReading(parquet_file_path, file_metadata, flags, block_size);
  file_metadata = parquet_reader->metadata();

  ResolveColumnNameToIndices(column_indices, column_names, file_metadata);

  #define IORING_SETUP_COOP_TASKRUN 256
  #define IORING_SETUP_SINGLE_ISSUER 4096
  #define IORING_SETUP_DEFER_TASKRUN 8192

  // Initialize io_uring with enough capacity for all columns
  struct io_uring ring = {};
  int ret = io_uring_queue_init(column_indices.size(), &ring, IORING_SETUP_COOP_TASKRUN | IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN);
  if (ret < 0) {
    throw std::logic_error(
      "Failed to initialize io_uring: " + std::string(strerror(-ret))
    );
  }

  try {
    int64_t current_target_row = 0;
    size_t target_row_ranges_index = 0;

    // Process each row group
    for (int row_group_index : row_groups) {
      const auto row_group_reader = parquet_reader->RowGroup(row_group_index);
      const auto row_group_metadata = file_metadata->RowGroup(row_group_index);
      const auto rows_in_group = row_group_metadata->num_rows();

      // Get individual column ranges sorted by file offset
      auto sorted_column_ranges = GetSortedColumnRanges(parquet_reader.get(), row_group_index, column_indices);

      std::vector<::arrow::io::ReadRange> maybe_coalesced_ranges;

      if (pre_buffer)
      {
        // In arrow, read ranges are coalesced only if the pre_buffer is True
        std::vector<int> single_row_group = {row_group_index};
        maybe_coalesced_ranges = parquet_reader->GetReadRanges(
            single_row_group, column_indices, 
            cache_options.hole_size_limit, cache_options.range_size_limit
          ).ValueOrDie();

        if (!use_o_direct)
        {
          fantom_reader->WillNeed(maybe_coalesced_ranges);
        }
      }
      else
      {
        for (const auto& column_range: sorted_column_ranges)
        {
          ::arrow::io::ReadRange read_range = 
          {
            .offset = column_range.file_offset,
            .length = column_range.data_length,
          };

          maybe_coalesced_ranges.push_back(read_range);
        }
      }

      // Create coalesced I/O requests to minimize file operations
      auto coalesced_requests = MatchColumnsToCoalescedRanges(sorted_column_ranges, maybe_coalesced_ranges, column_indices);

      // Submit all I/O requests asynchronously
      SubmitIORequests(ring, coalesced_requests, fd, fantom_reader->GetBlockSize());

      // Wait for completions and process all columns
      ProcessAllCompletedRequests(
        ring, coalesced_requests, fantom_reader, current_target_row, 
        row_group_reader.get(), row_group_metadata.get(), column_indices, target_row_ranges,
        target_column_indices, out, buffer_size,
        stride0_size, stride1_size, use_threads, target_row_ranges_index
      );

      current_target_row += rows_in_group;

      // Update row ranges index if using targeted row ranges
      if (!target_row_ranges.empty()) {
        auto remaining_rows = rows_in_group;
        while (remaining_rows > 0) {
          auto range_rows = target_row_ranges[target_row_ranges_index + 1] - target_row_ranges[target_row_ranges_index];
          target_row_ranges_index += 2;
          remaining_rows -= range_rows;
          
          if (remaining_rows == 0) break;
        }
      }
    }
  
    // Validate that we processed the expected amount of data
    if (target_row_ranges.size() > 0)
    {
      if (target_row_ranges_index != target_row_ranges.size())
      {
        auto msg = std::string("Expected to read ") + std::to_string(target_row_ranges.size() / 2) + " row ranges, but read only " + std::to_string(target_row_ranges_index / 2) + "!";
        throw std::logic_error(msg);
      }
    }
    else
    {
      if (current_target_row != expected_total_rows)
      {
        auto msg = std::string("Expected to read ") + std::to_string(expected_total_rows) + " rows, but read only " + std::to_string(current_target_row) + "!";
        throw std::logic_error(msg);
      }
    }
  } catch (...) {
    io_uring_queue_exit(&ring);
    close(fd);
    throw;
  }

  io_uring_queue_exit(&ring);
  close(fd);
}