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
  explicit FantomReader(int fd);
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

FantomReader::FantomReader(int fd)
    : fd_(fd), pos_(0), file_size_(0) {
  struct stat file_stats;
  if (fstat(fd_, &file_stats) < 0) {
    throw std::runtime_error("Failed to get file statistics with fstat");
  }

  file_size_ = file_stats.st_size;
  block_size_ = file_stats.st_blksize;
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

  // Allocate buffer for this coalesced request
  // Calculate aligned offset and length using bit hacks
  int64_t align_down_mask = ~(block_size_ - 1);
  // Mask for checking alignment, or effectively for aligning up with offset
  // This is not directly used as a mask for 'align up' in the same way as 'align down'
  int64_t aligned_start_offset = position & align_down_mask;
  int64_t original_end_offset = position + nbytes;
  // Align up for the end offset: (val + align - 1) & ~(align - 1)
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
OpenParquetFileForReading(const std::string& file_path, std::shared_ptr<parquet::FileMetaData> metadata, int flags) {
  int fd = open(file_path.c_str(), flags);
  if (fd < 0) {
    throw std::logic_error("Failed to open file: " + file_path + " - " + strerror(errno));
  }

  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  auto fantom_reader = std::make_shared<FantomReader>(fd);
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

    ColumnFileRange range_info;
    range_info.file_offset = ranges[0].offset;
    range_info.data_length = ranges[0].length;
    range_info.column_array_index = column_array_index;
    
    column_ranges.push_back(range_info);
  }

  // Sort by file offset for efficient matching with coalesced ranges
  std::sort(column_ranges.begin(), column_ranges.end(),
    [](const ColumnFileRange& first, const ColumnFileRange& second) {
      return first.file_offset < second.file_offset;
    });

  return column_ranges;
}

// Match individual column ranges to coalesced read ranges
void MatchColumnsToCoalescedRanges(
  std::vector<CoalescedIORequest>& coalesced_requests,
  const std::vector<ColumnFileRange>& sorted_column_ranges,
  const std::vector<arrow::io::ReadRange>& coalesced_ranges,
  const std::vector<int>& column_indices
) {
  coalesced_requests.reserve(coalesced_ranges.size());

  for (const auto& coalesced_range : coalesced_ranges) {
    CoalescedIORequest request;
    request.file_offset = coalesced_range.offset;
    request.read_length = coalesced_range.length;
    
    int64_t range_end = coalesced_range.offset + coalesced_range.length;
    
    // Find all column ranges that overlap with this coalesced range
    for (const auto& column_range : sorted_column_ranges) {
      // Early termination: if column starts after coalesced range ends
      if (column_range.file_offset >= range_end) {
        break;
      }

      // Check for overlap between column range and coalesced range
      bool ranges_overlap = (column_range.file_offset < range_end && 
                           column_range.end_offset() > coalesced_range.offset);

      if (ranges_overlap) {
        ColumnReadOperation column_op;
        column_op.column_array_index = column_range.column_array_index;
        column_op.parquet_column_index = column_indices[column_range.column_array_index];
        
        request.column_operations.push_back(column_op);
      }
    }

    coalesced_requests.push_back(request);
  }
}

// Create optimized coalesced read requests to minimize I/O operations
std::vector<CoalescedIORequest> CreateCoalescedIORequests(
  parquet::ParquetFileReader* parquet_reader,
  const std::shared_ptr<parquet::FileMetaData>& file_metadata,
  int row_group_index,
  const std::vector<int>& column_indices, 
  const arrow::io::CacheOptions& cache_options
) {
  // Get individual column ranges sorted by file offset
  auto sorted_column_ranges = GetSortedColumnRanges(parquet_reader, row_group_index, column_indices);

  // Get coalesced ranges from Parquet reader
  std::vector<int> single_row_group = {row_group_index};
  auto coalesced_ranges = parquet_reader->GetReadRanges(
    single_row_group, column_indices, 
    cache_options.hole_size_limit, cache_options.range_size_limit
  ).ValueOrDie();

  // Match columns to coalesced ranges
  std::vector<CoalescedIORequest> coalesced_requests;
  MatchColumnsToCoalescedRanges(coalesced_requests, sorted_column_ranges, coalesced_ranges, column_indices);
  
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

    // Allocate buffer for this coalesced request
    // Calculate aligned offset and length using bit hacks
    int64_t align_down_mask = ~(block_size - 1);
    // Mask for checking alignment, or effectively for aligning up with offset
    // This is not directly used as a mask for 'align up' in the same way as 'align down'
    int64_t aligned_start_offset = request.file_offset & align_down_mask;
    int64_t original_end_offset = request.file_offset + request.read_length;
    // Align up for the end offset: (val + align - 1) & ~(align - 1)
    int64_t aligned_end_offset = (original_end_offset + block_size - 1) & align_down_mask;
    int64_t aligned_read_length = aligned_end_offset - aligned_start_offset;

    // allign file offset + bump the read_lengtrh
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
      column_operation.column_reader,
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
void WaitForIOCompletionsAndSetupReaders(
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
        return;
    }

    (*requests_to_complete)--;

    // Validate the completion
    size_t request_index = reinterpret_cast<size_t>(io_uring_cqe_get_data(completion_entry));
    CoalescedIORequest& completed_request = io_requests[request_index];
    completed_requests->push_back(request_index);

    if (completion_entry->res < 0) {
      throw std::logic_error("I/O operation failed: " + std::string(strerror(-completion_entry->res)));
    }

    if (completion_entry->res < completed_request.read_length) {
      throw std::logic_error(
        "Incomplete read: expected " + std::to_string(completed_request.read_length) + 
        " bytes, got " + std::to_string(completion_entry->res)
      );
    }

    completed_request.read_buffer->Resize(completed_request.read_length);
    fantom_reader->SetBuffer(completed_request.file_offset, completed_request.read_buffer);

    // Create column readers for each column in this request
    for (auto& column_operation : completed_request.column_operations) {
      column_operation.column_reader = row_group_reader->Column(column_operation.parquet_column_index);
    }

    io_uring_cqe_seen(&ring, completion_entry);
  }
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

  while (requests_to_complete > 0)
  {
    // Wait for all I/O operations and setup readers
    WaitForIOCompletionsAndSetupReaders(ring, io_requests, fantom_reader, row_group_reader, &completed_requests, &requests_to_complete, use_threads);

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

    if (!processing_status.ok()) {
      throw std::logic_error("Parallel processing failed: " + processing_status.message());
    }
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
  const std::vector<int>& target_row_groups,
  const std::vector<int64_t>& target_row_ranges,
  const std::vector<std::string>& column_names,
  const std::vector<int>& target_column_indices,
  bool pre_buffer,  // Currently unused but kept for API compatibility
  bool use_threads,
  int64_t expected_total_rows, 
  arrow::io::CacheOptions cache_options)
{
  ValidateRowRangePairs(target_row_ranges);

  int flags = O_RDONLY;
  char *env_value = getenv("JJ_experimental_O_DIRECT");
  if (env_value)
    flags |= O_DIRECT;

  auto [fd, fantom_reader, parquet_reader] = OpenParquetFileForReading(parquet_file_path, file_metadata, flags);
  file_metadata = parquet_reader->metadata();

  ResolveColumnNameToIndices(column_indices, column_names, file_metadata);

  // Initialize io_uring with enough capacity for all columns
  struct io_uring ring = {};
  int ret = io_uring_queue_init(column_indices.size(), &ring, 0);
  if (ret < 0) {
    throw std::logic_error(
      "Failed to initialize io_uring: " + std::string(strerror(-ret))
    );
  }

  try {
    int64_t current_target_row = 0;
    size_t target_row_ranges_index = 0;

    // Process each row group
    for (int row_group_index : target_row_groups) {
      const auto row_group_reader = parquet_reader->RowGroup(row_group_index);
      const auto row_group_metadata = file_metadata->RowGroup(row_group_index);
      const auto rows_in_group = row_group_metadata->num_rows();

      // Create coalesced I/O requests to minimize file operations
      auto coalesced_requests = CreateCoalescedIORequests(
        parquet_reader.get(), file_metadata, row_group_index, column_indices, cache_options
      );

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