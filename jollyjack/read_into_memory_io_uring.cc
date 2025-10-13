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
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(
    int64_t nbytes
  ) override;
  void SetBuffer(long offset, std::shared_ptr<arrow::Buffer> buffer);
  
 private:
  int fd_;
  int64_t pos_ = 0;
  int64_t size_ = 0;
  std::shared_ptr<arrow::Buffer> buffer_;
  int64_t buffer_offset_;
};

FantomReader::FantomReader(int fd)
    : fd_(fd), pos_(0), size_(0) {
  struct stat st;
  if (fstat(fd_, &st) < 0) {
    throw std::runtime_error("fstat failed");
  }

  size_ = st.st_size;
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
  return size_;
}

arrow::Result<int64_t> FantomReader::ReadAt(int64_t position, int64_t nbytes, void* out) {
  return pread(fd_, out, nbytes, position);
}

arrow::Result<std::shared_ptr<arrow::Buffer>> FantomReader::ReadAt(int64_t position, int64_t nbytes) {
  if (buffer_ != nullptr && position >= buffer_offset_ && buffer_offset_ + buffer_->size() >= position + nbytes) {
    return arrow::SliceBuffer(buffer_, position - buffer_offset_, nbytes);
  }

  if (buffer_ != nullptr)
  {
    auto msg = std::string("ReadAt failed, because read cannot be serveeed from a buffer, buffer_offset=") + std::to_string(buffer_offset_) 
          + ", buffer_size=" + std::to_string(buffer_->size()) + ", position_to_read=" + std::to_string(position) + ", nbytes_to_read=" + std::to_string(nbytes) + "!";
    return arrow::Status::UnknownError(msg);
  }

  ARROW_ASSIGN_OR_RAISE(
    auto buffer, arrow::AllocateResizableBuffer(nbytes)
  );

  ARROW_ASSIGN_OR_RAISE(
    int64_t bytes_read, 
    ReadAt(position, nbytes, buffer->mutable_data())
  );

  if (bytes_read < nbytes) {
    RETURN_NOT_OK(buffer->Resize(bytes_read));
    buffer->ZeroPadding();
  }

  return std::shared_ptr<arrow::Buffer>(std::move(buffer));
}

arrow::Status FantomReader::Seek(int64_t position) {
  pos_ = position;
  return arrow::Status::OK();
}

arrow::Result<int64_t> FantomReader::Tell() const {
  return pos_;
}

arrow::Result<int64_t> FantomReader::Read(int64_t nbytes, void* out) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
  memcpy(out, buffer->data(), buffer->size());
  pos_ += buffer->size();
  return buffer->size();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> FantomReader::Read(int64_t nbytes) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
  pos_ += buffer->size();
  return buffer;
}

void FantomReader::SetBuffer(long offset, std::shared_ptr<arrow::Buffer> buffer) {
  buffer_ = buffer;
  buffer_offset_ = offset;
}

// Represents a single column read operation
struct ColumnRead {
  int column_counter;
  int column_index;
};

// Represents a coalesced I/O request that may serve multiple columns
struct CoalescedRequest {
  int row_group;
  int64_t offset;
  int64_t length;
  int64_t target_row;
  std::shared_ptr<parquet::RowGroupMetaData> row_group_metadata;
  std::shared_ptr<parquet::RowGroupReader> row_group_reader;
  std::shared_ptr<arrow::Buffer> buffer;
  std::vector<ColumnRead> column_reads;
};

// Validate that target_row_ranges contains pairs
void ValidateTargetRowRanges(const std::vector<int64_t>& target_row_ranges) {
  if (target_row_ranges.size() % 2 != 0) {
    throw std::logic_error(
      "target_row_ranges must contain pairs of [start, end) indices"
    );
  }
}

// Open file and create parquet reader
std::tuple<int, std::shared_ptr<FantomReader>, std::unique_ptr<parquet::ParquetFileReader>>
OpenParquetFile(const std::string& path, std::shared_ptr<parquet::FileMetaData> file_metadata) 
{
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::logic_error("Failed to open file: " + path + " - " + strerror(errno));
  }

  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  auto fantom_reader = std::make_shared<FantomReader>(fd);
  auto parquet_reader = parquet::ParquetFileReader::Open(fantom_reader, reader_properties, file_metadata);

  return {fd, fantom_reader, std::move(parquet_reader)};
}

// Resolve column names to column indices
void ResolveColumnIndices(
  std::vector<int>& column_indices,
  const std::vector<std::string>& column_names,
  const std::shared_ptr<parquet::FileMetaData>& file_metadata
) {
  if (column_names.empty()) {
    return;
  }

  column_indices.reserve(column_names.size());
  auto schema = file_metadata->schema();
  
  for (const auto& column_name : column_names)
  {
    int column_index = schema->ColumnIndex(column_name);
    
    if (column_index < 0) {
      throw std::logic_error("Column '" + column_name + "' was not found!");
    }

    column_indices.push_back(column_index);
  }
}

// Check if a column chunk falls within a read range
bool ColumnInRange(
  const parquet::ColumnChunkMetaData* column_metadata,
  int64_t range_offset,
  int64_t range_length
) {
  int64_t col_offset = column_metadata->data_page_offset();
  if (column_metadata->has_dictionary_page()) {
    col_offset = std::min(
      col_offset, column_metadata->dictionary_page_offset()
    );
  }
  
  int64_t col_length = column_metadata->total_compressed_size();
  int64_t col_end = col_offset + col_length;
  int64_t range_end = range_offset + range_length;
  
  // Check if column overlaps with range
  return col_offset < range_end && col_end > range_offset;
}

// Represents a read range with associated metadata
struct ColumnRangeInfo {
  int64_t offset;
  int64_t length;
  size_t column_idx;
  
  int64_t end() const { return offset + length; }
};

// Create coalesced read requests using GetReadRanges efficiently
std::vector<CoalescedRequest> CreateCoalescedRequests(
  parquet::ParquetFileReader* parquet_reader,
  const std::shared_ptr<parquet::FileMetaData>& file_metadata,
  const std::vector<int>& row_groups,
  const std::vector<int>& column_indices, 
  const arrow::io::CacheOptions& cache_options
) {
  std::vector<CoalescedRequest> coalesced_requests;
  coalesced_requests.reserve(column_indices.size() * row_groups.size()); // reserve enough memory to avoid reallocations

  int64_t current_target_row = 0;
  
  std::vector<int> single_row_group(1);
  std::vector<int> single_column(1);
  
  // Process each row group separately to maintain target_row tracking
  for (int row_group : row_groups) {
    auto row_group_reader = parquet_reader->RowGroup(row_group);
    std::shared_ptr<parquet::RowGroupMetaData> row_group_metadata = file_metadata->RowGroup(row_group);

    single_row_group[0] = row_group;

    // Get individual read ranges for each column - O(columns)
    std::vector<ColumnRangeInfo> column_ranges;
    column_ranges.resize(column_indices.size());

    for (size_t c_idx = 0; c_idx < column_indices.size(); c_idx++) {
      single_column[0] = column_indices[c_idx];

      auto ranges = parquet_reader->GetReadRanges(
        single_row_group, single_column, 0, 1
      ).ValueOrDie();

      ColumnRangeInfo &info = column_ranges[c_idx];
      info.offset = ranges[0].offset;
      info.length = ranges[0].length;
      info.column_idx = c_idx;
    }

    // Sort column ranges by offset for efficient matching - O(columns * log(columns))
    std::sort(column_ranges.begin(), column_ranges.end(),
      [](const ColumnRangeInfo& a, const ColumnRangeInfo& b) {
        return a.offset < b.offset;
      });

    auto coalesced_ranges = parquet_reader->GetReadRanges(single_row_group, column_indices, cache_options.hole_size_limit, cache_options.range_size_limit).ValueOrDie();
    
    // Match column ranges to coalesced ranges using two pointers - O(coalesced + columns)
    size_t col_idx = 0;
    
    for (const auto& coalesced_range : coalesced_ranges) {
      coalesced_requests.emplace_back();
      CoalescedRequest &request = coalesced_requests[coalesced_requests.size() - 1];

      request.offset = coalesced_range.offset;
      request.length = coalesced_range.length;
      request.row_group = row_group;
      request.target_row = current_target_row;
      request.row_group_reader = row_group_reader;
      request.row_group_metadata = row_group_metadata;   

      int64_t coalesced_end = coalesced_range.offset + coalesced_range.length;
      
      // Find all column ranges that overlap with this coalesced range
      // Start from where we left off (columns are sorted)
      // Scan forward to find all overlapping columns
      for (size_t i = col_idx; i < column_ranges.size(); i++) {
        const auto& col_range = column_ranges[i];
        
        // If this column starts after the coalesced range ends, we're done
        if (col_range.offset >= coalesced_end) {
          break;
        }
        
        // Check for overlap
        if (col_range.offset < coalesced_end && 
            col_range.end() > coalesced_range.offset) {
          
          request.column_reads.emplace_back();
          ColumnRead &column_read = request.column_reads[request.column_reads.size() - 1];
          column_read.column_counter = col_range.column_idx;
          column_read.column_index = column_indices[col_range.column_idx];
        }
      }
    }

    current_target_row += row_group_metadata->num_rows();
  }
  
  return coalesced_requests;
}

// Allocate buffers and submit all coalesced requests to io_uring
void SubmitCoalescedRequests(
  struct io_uring& ring,
  std::vector<CoalescedRequest>& requests,
  int fd
) {
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

  /*
  auto start = std::chrono::system_clock::now();
  auto submitted = io_uring_submit(&ring);

  auto end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cerr << " io_uring_submit:" << std::to_string(elapsed.count()) << "ms, submitted=" << std::to_string(submitted) << std::endl;
  */
}

// Calculate target_row_ranges_idx for a specific row group
size_t CalculateTargetRowRangesIndex(
  int target_row_group,
  const std::vector<int>& row_groups,
  const std::shared_ptr<parquet::FileMetaData>& file_metadata,
  const std::vector<int64_t>& target_row_ranges
) 
{
  if (target_row_ranges.empty()) {
    return 0;
  }

  size_t idx = 0;
  for (int row_group : row_groups) {
    if (row_group == target_row_group) {
      break;
    }
    
    auto metadata = file_metadata->RowGroup(row_group);
    int64_t rows = metadata->num_rows();
    
    while (rows > 0) {
      int64_t range_rows = target_row_ranges[idx + 1] - target_row_ranges[idx];
      idx += 2;
      rows -= range_rows;
    }
  }
  
  return idx;
}

// Process all columns covered by a coalesced request
void ProcessCompletion(
  CoalescedRequest& request,
  const std::shared_ptr<FantomReader>& fantom_reader,
  const std::shared_ptr<parquet::FileMetaData>& file_metadata,
  const std::vector<int>& row_groups,
  const std::vector<int>& column_indices,
  const std::vector<int64_t>& target_row_ranges,
  const std::vector<int>& target_column_indices,
  void* buffer,
  size_t buffer_size,
  size_t stride0_size,
  size_t stride1_size
) {    
  size_t target_row_ranges_idx = CalculateTargetRowRangesIndex(request.row_group, row_groups, file_metadata, target_row_ranges);

  // Set the buffer once for all columns in this request
  fantom_reader->SetBuffer(request.offset, request.buffer);
  
  // Process each column covered by this coalesced read
  for (const auto& column_read : request.column_reads) 
  {
    auto column_reader = request.row_group_reader->Column(column_read.column_index);

    auto status = ReadColumn(
      column_read.column_counter,
      request.target_row,
      column_reader,
      request.row_group_metadata.get(), 
      buffer,
      buffer_size,
      stride0_size,
      stride1_size,
      column_indices,
      target_column_indices,
      target_row_ranges,
      target_row_ranges_idx
    );

    if (!status.ok()) {
      throw std::logic_error(status.message());
    }
  }
  
  request.buffer.reset();
}

// Wait for and process all io_uring completions
void ProcessCompletions(
  struct io_uring& ring,
  std::vector<CoalescedRequest>& requests,
  const std::shared_ptr<FantomReader>& fantom_reader,
  const std::shared_ptr<parquet::FileMetaData>& file_metadata,
  const std::vector<int>& row_groups,
  const std::vector<int>& column_indices,
  const std::vector<int64_t>& target_row_ranges,
  const std::vector<int>& target_column_indices,
  void* buffer,
  size_t buffer_size,
  size_t stride0_size,
  size_t stride1_size
) {
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
    CoalescedRequest& request = requests[request_idx];

    if (cqe->res < 0) {
      throw std::logic_error(
        "Read failed: " + std::string(strerror(-cqe->res))
      );
    }
    
    if (cqe->res != request.length) {
        throw std::logic_error("Read failed - incomplete: " + std::to_string(cqe->res) + " != " + std::to_string(request.length)
      );
    }

    io_uring_cqe_seen(&ring, cqe);

    ProcessCompletion(
      request, fantom_reader, file_metadata, row_groups,
      column_indices, target_row_ranges, target_column_indices,
      buffer, buffer_size, stride0_size, stride1_size
    );
  }
}

// Validate final row counts
void ValidateResults(
  const std::vector<int>& row_groups,
  const std::shared_ptr<parquet::FileMetaData>& file_metadata,
  const std::vector<int64_t>& target_row_ranges,
  int64_t expected_rows
) {
  int64_t total_rows = 0;
  size_t range_idx = 0;
  
  for (int row_group : row_groups) {
    auto metadata = file_metadata->RowGroup(row_group);
    total_rows += metadata->num_rows();
    
    if (!target_row_ranges.empty()) {
      int64_t rows = metadata->num_rows();
      while (rows > 0) {
        int64_t range_rows = 
          target_row_ranges[range_idx + 1] - target_row_ranges[range_idx];
        range_idx += 2;
        rows -= range_rows;
      }
    }
  }

  if (!target_row_ranges.empty()) {
    if (range_idx != target_row_ranges.size()) {
      throw std::logic_error("Expected to read " + 
        std::to_string(target_row_ranges.size() / 2) +
        " row ranges, but read only " + 
        std::to_string(range_idx / 2) + "!"
      );
    }
  } else {
    if (total_rows != expected_rows) {
      throw std::logic_error(
        "Expected to read " + std::to_string(expected_rows) +
        " rows, but read only " + std::to_string(total_rows) + "!"
      );
    }
  }
}

void ReadIntoMemoryIOUring(
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
  ValidateTargetRowRanges(target_row_ranges);

  auto [fd, fantom_reader, parquet_reader] = OpenParquetFile(path, file_metadata);
  file_metadata = parquet_reader->metadata();

  ResolveColumnIndices(column_indices, column_names, file_metadata);

  // Create coalesced requests (fewer I/O operations than columns)
  auto coalesced_requests = CreateCoalescedRequests(parquet_reader.get(), file_metadata, row_groups, column_indices, cache_options);

  // Initialize io_uring
  struct io_uring ring = {};
  int ret = io_uring_queue_init(coalesced_requests.size(), &ring, 0);
  if (ret < 0) {
    throw std::logic_error(
      "Failed to initialize io_uring: " + std::string(strerror(-ret))
    );
  }

  try {
    SubmitCoalescedRequests(ring, coalesced_requests, fd);

    ProcessCompletions(
      ring, coalesced_requests, fantom_reader, file_metadata,
      row_groups, column_indices, target_row_ranges,
      target_column_indices, buffer, buffer_size,
      stride0_size, stride1_size
    );

    ValidateResults(row_groups, file_metadata, target_row_ranges, expected_rows);
  } catch (...) {
    io_uring_queue_exit(&ring);
    close(fd);
    throw;
  }

  io_uring_queue_exit(&ring);
  close(fd);
}