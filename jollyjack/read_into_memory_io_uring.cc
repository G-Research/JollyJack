/*
Both suites pass on the final build. The process still showing is one of my own wait loops, which matched its own command line, so I'll stop the leftover waiters.

  Ran 2 shell commands · 1 failed

io_uring_odirect is now about 30% faster with use_threads=False and about 2.3× faster with use_threads=True. Buffered io_uring also improved. The change is in jollyjack/read_into_memory_io_uring.cc and is not committed.

Results from your command, comparing master with the new build on the same machine:

┌──────────────────┬─────────────┬──────────────┬──────────────┐
│     Backend      │ use_threads │    master    │     new      │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ io_uring_odirect │ False       │ 138–140 Gb/s │ 180–200 Gb/s │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ io_uring_odirect │ True        │ 72–82 Gb/s   │ 181–184 Gb/s │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ io_uring         │ False       │ 123–125 Gb/s │ 144–171 Gb/s │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ io_uring         │ True        │ 71–76 Gb/s   │ 144–153 Gb/s │
└──────────────────┴─────────────┴──────────────┴──────────────┘

The drives top out at about 212 Gb/s in the benchmark's units (26.5 GiB/s with fio), so O_DIRECT now runs at roughly 85–95% of the hardware limit.

What was wrong:
- Too many reads at once. Each reader submitted one read per column, 6000 per file, all at the same time. With 16 workers that put about 30,000 reads in the queue, each waiting about 89 ms. A separate fio test showed these drives are fastest with only about 16 MB in flight across the whole process; deeper queues are slower.
- Your cache hypothesis was right. Each reader held about 384 MB of freshly allocated buffers, much more than L2 (2 MB per core) or L3 (320 MB). With the change, LLC load misses dropped about 4.7× (440M to 93M) and total CPU cycles dropped by about 40%.
- Duplicate reads. Each 64 KB column was rounded out to 4 KB blocks on its own, so neighbouring columns read their shared block twice. That added 6.4% extra disk traffic.
- IOSQE_ASYNC hurt O_DIRECT. It sent every read through kernel worker threads: 153 Gb/s with it, 190 without. For buffered reads it does help, so it stays on there.

What I changed:
- Limited window: each reader keeps 2–16 reads in flight, under a shared budget of 16 MB for the whole process, so it adapts to any worker count. Each completed read is decoded and its buffer freed right away, so the allocator reuses memory that is still in cache.
- Merged reads: adjacent columns are now combined into reads of up to 256 KB when pre_buffer=False. With O_DIRECT, the shared blocks are read only once.
- IOSQE_ASYNC is now off for O_DIRECT and on for buffered reads.
- Hang fix: if a submission slot can't be obtained, the request is no longer counted as in flight.

Tests: the full suite passes on the final build for both backends: 4404 passed with io_uring and 4400 passed with io_uring_odirect. An earlier run without a backend set hit one test_readme failure because I ran the tests from a copied folder without the README.

Single reader: this is where a small window could have hurt. All my cases came out faster: small 4 KB columns and 128 KB columns, on both the NVMe array and the root EBS volume. For example, small columns with O_DIRECT on EBS went from 58 ms to 10 ms.

Decision for you: the tuning settings are environment variables: JJ_IOURING_INFLIGHT_BYTES, JJ_IOURING_INFLIGHT, JJ_IOURING_INFLIGHT_MIN, JJ_IOURING_MAX_READ and JJ_IOURING_ASYNC. The defaults are tuned for this 8-drive RAID0 setup and may not suit other storage. Should they stay as documented settings, or become fixed constants?
*/

#include "arrow/status.h"
#include "arrow/io/memory.h"
#include "arrow/util/parallel.h"
#include "parquet/column_reader.h"
#include "parquet/types.h"

#include "jollyjack.h"

#include <liburing.h>
#include <algorithm>
#include <atomic>
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

#define IORING_SETUP_COOP_TASKRUN 256
#define IORING_SETUP_SINGLE_ISSUER 4096
#define IORING_SETUP_DEFER_TASKRUN 8192

class ThreadRing {
 public:
  struct io_uring& Get(size_t entries) {
    if (entries_ == 0 || entries > entries_ || pid_ != getpid()) {
      Reset();
      int ret = io_uring_queue_init(entries, &ring_, IORING_SETUP_COOP_TASKRUN | IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN);
      if (ret < 0) {
        throw std::logic_error(
          "Failed to initialize io_uring: " + std::string(strerror(-ret))
        );
      }
      entries_ = entries;
      pid_ = getpid();
    }
    return ring_;
  }

  void Reset() {
    if (entries_ > 0)
      io_uring_queue_exit(&ring_);
    ring_ = {};
    entries_ = 0;
  }

  ~ThreadRing() { Reset(); }

 private:
  struct io_uring ring_ = {};
  size_t entries_ = 0;
  pid_t pid_ = 0;
};

static thread_local ThreadRing thread_ring;


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
    close(fd_);
    throw std::runtime_error("Failed to get file statistics with fstat");
  }

  file_size_ = file_stats.st_size;
  block_size_ = std::max(block_size, 1);
}

FantomReader::~FantomReader() {
  (void)Close();
}

arrow::Status FantomReader::Close() {
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
  return arrow::Status::OK();
}

bool FantomReader::closed() const {
  return fd_ < 0;
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
  std::shared_ptr<parquet::ColumnChunkMetaData> column_chunk_metadata;
};

// Represents a coalesced I/O request that reads multiple columns in one operation
struct CoalescedIORequest {
  int64_t file_offset;
  int64_t read_length;
  std::shared_ptr<arrow::ResizableBuffer> read_buffer;
  std::vector<ColumnReadOperation> column_operations;
  int64_t in_flight_bytes = 0; // taken from the process-wide in-flight budget
};

// Information about a column's file range for coalescing optimization
struct ColumnFileRange {
  int64_t file_offset;
  int64_t data_length;
  size_t column_array_index;
  std::shared_ptr<parquet::ColumnChunkMetaData> metadata;

  int64_t end_offset() const { return file_offset + data_length; }
};

// Must match parquet's ComputeColumnChunkRange
arrow::io::ReadRange ColumnChunkRange(const parquet::ColumnChunkMetaData &column, int64_t source_size, bool pad_dictionary_header)
{
  constexpr int64_t kMaxDictHeaderSize = 100;
  int64_t col_start = column.data_page_offset();
  if (column.has_dictionary_page() && column.dictionary_page_offset() > 0 && col_start > column.dictionary_page_offset())
    col_start = column.dictionary_page_offset();
  int64_t col_length = column.total_compressed_size();
  if (col_start < 0 || col_length < 0 || col_length > source_size - col_start)
    throw parquet::ParquetException("Invalid column metadata (corrupt file?)");
  if (pad_dictionary_header)
    col_length += std::min<int64_t>(kMaxDictHeaderSize, source_size - (col_start + col_length));
  return {col_start, col_length};
}

std::vector<ColumnFileRange> GetSortedColumnRanges(
  parquet::FileMetaData* file_metadata,
  int64_t source_size,
  int row_group_index,
  const std::vector<int>& column_indices
) {
  const auto row_group = file_metadata->RowGroup(row_group_index);
  const bool pad = file_metadata->writer_version().VersionLt(parquet::ApplicationVersion::PARQUET_816_FIXED_VERSION());
  std::vector<ColumnFileRange> column_ranges(column_indices.size());
  for (size_t i = 0; i < column_indices.size(); i++) {
    std::shared_ptr<parquet::ColumnChunkMetaData> metadata = row_group->ColumnChunk(column_indices[i]);
    const auto range = ColumnChunkRange(*metadata, source_size, pad);
    column_ranges[i] = {range.offset, range.length, i, std::move(metadata)};
  }
  std::sort(column_ranges.begin(), column_ranges.end(),
    [](const ColumnFileRange& first, const ColumnFileRange& second) {
      return first.file_offset < second.file_offset;
    });
  return column_ranges;
}

// Must match arrow's CoalesceReadRanges; ranges are sorted by offset
std::vector<arrow::io::ReadRange> CoalesceSortedRanges(const std::vector<ColumnFileRange>& sorted, int64_t hole_size_limit, int64_t range_size_limit)
{
  std::vector<arrow::io::ReadRange> coalesced;
  int64_t start = 0, end = 0;
  bool open = false;
  for (const auto& r : sorted) {
    if (r.data_length == 0)
      continue;
    if (open && (r.end_offset() - start > range_size_limit || r.file_offset - end > hole_size_limit)) {
      coalesced.push_back({start, end - start});
      open = false;
    }
    if (!open) {
      start = r.file_offset;
      end = r.end_offset();
      open = true;
    } else {
      end = std::max(end, r.end_offset());
    }
  }
  if (open)
    coalesced.push_back({start, end - start});
  return coalesced;
}

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
      column_op.column_chunk_metadata = it->metadata;
    }
  }

  return coalesced_requests;
}

// Tunables, read once from the environment; the defaults were tuned on a RAID0 of 8 NVMe drives
static int64_t EnvInt(const char* name, int64_t default_value) {
  const char* value = getenv(name);
  return value ? atoll(value) : default_value;
}

struct IOUringTuning {
  // Bytes in flight across all readers of the process
  int64_t total_in_flight_bytes = EnvInt("JJ_IOURING_INFLIGHT_BYTES", 16 * 1024 * 1024);
  // Reads in flight of one reader, at least min and at most max
  int64_t min_in_flight = EnvInt("JJ_IOURING_INFLIGHT_MIN", 2);
  int64_t max_in_flight = EnvInt("JJ_IOURING_INFLIGHT", 16);
  int64_t max_read_size = EnvInt("JJ_IOURING_MAX_READ", 256 * 1024);
  // IOSQE_ASYNC: -1 = only for buffered reads (they block when issued inline), 0 = never, 1 = always
  int64_t force_async = EnvInt("JJ_IOURING_ASYNC", -1);
};

// Bytes in flight across all readers of the process
static std::atomic<int64_t> process_in_flight_bytes{0};

static const IOUringTuning& GetTuning() {
  static const IOUringTuning tuning;
  return tuning;
}

// Merges column ranges whose block-aligned reads touch or overlap, so the shared blocks are read only once.
// Ranges are sorted by offset.
std::vector<arrow::io::ReadRange> CoalesceAdjacentRanges(const std::vector<ColumnFileRange>& sorted, int64_t block_size, int64_t max_read_size)
{
  std::vector<arrow::io::ReadRange> coalesced;
  const int64_t align_down_mask = ~(block_size - 1);
  int64_t start = 0, end = 0;
  bool open = false;
  for (const auto& r : sorted) {
    if (r.data_length == 0)
      continue;
    if (open) {
      const int64_t aligned_end = (end + block_size - 1) & align_down_mask;
      const int64_t new_end = std::max(end, r.end_offset());
      const int64_t aligned_new_end = (new_end + block_size - 1) & align_down_mask;
      const bool touches = (r.file_offset & align_down_mask) <= aligned_end;
      const bool fits = aligned_new_end - (start & align_down_mask) <= max_read_size;
      if (touches && fits) {
        end = new_end;
        continue;
      }
      coalesced.push_back({start, end - start});
    }
    start = r.file_offset;
    end = r.end_offset();
    open = true;
  }
  if (open)
    coalesced.push_back({start, end - start});
  return coalesced;
}

// Prepares a read for one coalesced request, allocating its block-aligned buffer
void PrepareIORequest(
  struct io_uring_sqe* sqe,
  CoalescedIORequest& request,
  size_t request_index,
  int fd,
  int64_t block_size,
  bool force_async
) {
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
    io_uring_prep_nop(sqe);
    io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(request_index));
    throw std::logic_error("Failed to allocate buffer: " + buffer_result.status().message());
  }
  request.read_buffer = std::move(buffer_result.ValueOrDie());

  io_uring_prep_read(
    sqe, fd, request.read_buffer->mutable_data(),
    aligned_read_length, aligned_start_offset
  );
  if (force_async)
    io_uring_sqe_set_flags(sqe, IOSQE_ASYNC);
  io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(request_index));
}

// Process a single completed I/O request and read all its columns
void ProcessSingleIOCompletion(
  int64_t current_target_row,
  CoalescedIORequest& completed_request,
  parquet::RowGroupMetaData* row_group_metadata,
  const std::vector<int>& column_indices,
  const std::vector<int64_t>& target_row_ranges,
  const std::vector<int>& target_column_indices,
  void* out,
  size_t buffer_size,
  size_t stride0_size,
  size_t stride1_size,
  size_t target_row_ranges_index,
  bool nan_fast_path
) {
  // Process each column covered by this coalesced read
  for (auto& column_operation : completed_request.column_operations) {
    auto read_status = ReadColumn(
      column_operation.column_array_index,
      current_target_row,
      column_operation.column_reader.get(),
      row_group_metadata,
      column_operation.column_chunk_metadata.get(),
      out,
      buffer_size,
      stride0_size,
      stride1_size,
      column_indices,
      target_column_indices,
      target_row_ranges,
      target_row_ranges_index,
      nan_fast_path
    );

    // Release the reader, it holds a slice of the read buffer
    column_operation.column_reader.reset();

    if (!read_status.ok()) {
      throw std::logic_error("Column read failed: " + read_status.message());
    }
  }

  // Release the read buffer, so the allocator can hand the (cache-hot) memory to the next read
  completed_request.read_buffer.reset();
}

// Validates a completion and creates the column readers of its request
arrow::Status SetupCompletedRequest(
  CoalescedIORequest& completed_request,
  int result,
  const std::shared_ptr<FantomReader>& fantom_reader,
  parquet::RowGroupReader* row_group_reader
) {
  if (result < 0) {
    return arrow::Status::UnknownError(std::string("I/O operation failed: ") + std::string(strerror(-result)));
  }

  if (result < completed_request.read_length) {
    return arrow::Status::UnknownError(std::string("Incomplete read: expected ") + std::to_string(completed_request.read_length) +
      " bytes, got " + std::to_string(result));
  }

  bool shrink_to_fit = false;
  ARROW_RETURN_NOT_OK(completed_request.read_buffer->Resize(completed_request.read_length, shrink_to_fit));
  fantom_reader->SetBuffer(completed_request.file_offset, completed_request.read_buffer);

  // Create column readers for each column in this request
  for (auto& column_operation : completed_request.column_operations) {
    const auto *descr = row_group_reader->metadata()->schema()->Column(column_operation.parquet_column_index);

    if (descr->physical_type() == parquet::Type::FIXED_LEN_BYTE_ARRAY)
      column_operation.column_reader = MakeFLBAReader(row_group_reader, column_operation.parquet_column_index, descr, column_operation.column_chunk_metadata.get(), fantom_reader);
    else
      column_operation.column_reader = row_group_reader->Column(column_operation.parquet_column_index);
  }

  return arrow::Status::OK();
}

// Reads and processes all requests through a bounded window of in-flight reads.
// Keeping only a few reads in flight keeps the read buffers small and cache-hot,
// and the devices are not flooded with thousands of queued requests.
void ReadAndProcessRequests(
  struct io_uring& ring,
  size_t max_in_flight,
  std::vector<CoalescedIORequest>& io_requests,
  int fd,
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
  size_t target_row_ranges_index,
  bool nan_fast_path
) {
  const int64_t block_size = fantom_reader->GetBlockSize();
  const auto& tuning = GetTuning();
  // O_DIRECT reads are issued asynchronously by the block layer, punting them to io-wq only adds overhead
  const bool force_async = tuning.force_async < 0 ? block_size <= 1 : tuning.force_async != 0;
  const size_t min_in_flight = std::clamp<int64_t>(tuning.min_in_flight, 1, max_in_flight);
  const int64_t align_down_mask = ~(block_size - 1);
  size_t next_to_submit = 0;
  size_t in_flight = 0;
  int64_t in_flight_bytes = 0;
  int64_t prepared_bytes = 0;

  // Returns the bytes of this reader to the process-wide budget, also when unwinding
  struct InFlightGuard {
    int64_t& bytes;
    ~InFlightGuard() { process_in_flight_bytes.fetch_sub(bytes); }
  } in_flight_guard{in_flight_bytes};

  // A reader always keeps min_in_flight reads, more only while the process-wide budget allows
  auto acquire_slot = [&](size_t reader_in_flight, CoalescedIORequest& request) {
    if (reader_in_flight >= max_in_flight)
      return false;
    const int64_t bytes = ((request.file_offset + request.read_length + block_size - 1) & align_down_mask) - (request.file_offset & align_down_mask);
    const int64_t previous = process_in_flight_bytes.fetch_add(bytes);
    if (reader_in_flight < min_in_flight || previous + bytes <= tuning.total_in_flight_bytes) {
      prepared_bytes += bytes;
      request.in_flight_bytes = bytes;
      return true;
    }
    process_in_flight_bytes.fetch_sub(bytes);
    return false;
  };
  arrow::Status status = arrow::Status::OK();
  std::vector<size_t> completed_requests;
  completed_requests.reserve(max_in_flight);

  // Queues reads until the window is full, stops queuing once anything failed
  auto fill_window = [&]() {
    size_t prepared = 0;
    while (status.ok() && next_to_submit < io_requests.size() && acquire_slot(in_flight + prepared, io_requests[next_to_submit])) {
      struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
      if (!sqe) {
        // Not queued, return its bytes
        process_in_flight_bytes.fetch_sub(io_requests[next_to_submit].in_flight_bytes);
        prepared_bytes -= io_requests[next_to_submit].in_flight_bytes;
        status = arrow::Status::UnknownError("Failed to get submission queue entry from io_uring");
        break;
      }
      try {
        PrepareIORequest(sqe, io_requests[next_to_submit], next_to_submit, fd, block_size, force_async);
      } catch (const std::exception& error) {
        status = arrow::Status::UnknownError(error.what());
      }
      next_to_submit++;
      prepared++;
    }
    return prepared;
  };

  auto submit = [&](size_t prepared, unsigned wait_nr) {
    if (prepared == 0 && wait_nr == 0)
      return;
    int ret;
    do {
      ret = io_uring_submit_and_wait(&ring, wait_nr);
    } while (ret == -EINTR);
    if (ret < 0) {
      // Nothing was submitted, the prepared reads are not in flight
      process_in_flight_bytes.fetch_sub(prepared_bytes);
      prepared_bytes = 0;
      if (status.ok())
        status = arrow::Status::UnknownError("Failed to submit io_uring operations: " + std::string(strerror(-ret)));
      throw std::logic_error(status.message());
    }
    in_flight += prepared;
    in_flight_bytes += prepared_bytes;
    prepared_bytes = 0;
  };

  submit(fill_window(), 0);

  while (in_flight > 0) {
    // Wait for at least one completion, then harvest everything that is ready
    struct io_uring_cqe* cqe;
    int wait_result;
    do {
      wait_result = io_uring_wait_cqe(&ring, &cqe);
    } while (wait_result == -EINTR);
    if (wait_result < 0) {
      throw std::logic_error("Failed to wait for io_uring completion: " + std::string(strerror(-wait_result)));
    }

    completed_requests.clear();
    unsigned head;
    unsigned harvested = 0;
    int64_t harvested_bytes = 0;
    io_uring_for_each_cqe(&ring, head, cqe) {
      harvested++;
      size_t request_index = reinterpret_cast<size_t>(io_uring_cqe_get_data(cqe));
      int result = cqe->res;
      harvested_bytes += io_requests[request_index].in_flight_bytes;
      if (!status.ok())
        continue;
      try {
        status = SetupCompletedRequest(io_requests[request_index], result, fantom_reader, row_group_reader);
      } catch (const std::exception& error) {
        status = arrow::Status::UnknownError(error.what());
      }
      if (status.ok())
        completed_requests.push_back(request_index);
    }
    io_uring_cq_advance(&ring, harvested);
    in_flight -= harvested;
    in_flight_bytes -= harvested_bytes;
    process_in_flight_bytes.fetch_sub(harvested_bytes);

    // Refill the window before processing, so the devices stay busy meanwhile
    submit(fill_window(), 0);

    if (!status.ok())
      continue; // drain

    auto processing_status = ::arrow::internal::OptionalParallelFor(
      use_threads && completed_requests.size() > 1,
      static_cast<int>(completed_requests.size()),
      [&](int i) -> Status {
        try {
          ProcessSingleIOCompletion(
            current_target_row, io_requests[completed_requests[i]], row_group_metadata,
            column_indices, target_row_ranges, target_column_indices,
            out, buffer_size, stride0_size, stride1_size, target_row_ranges_index, nan_fast_path
          );
          return Status::OK();
        } catch (const std::exception& error) {
          return Status::UnknownError("Request processing failed: " + std::string(error.what()));
        }
      }
    );

    if (!processing_status.ok())
      status = processing_status;
  }

  if (!status.ok()) {
    throw std::logic_error("Parallel processing failed: " + status.message());
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
  bool nan_fast_path,
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
  nan_fast_path = nan_fast_path && file_metadata->writer_version().application_ == "parquet-cpp-arrow";

  ResolveColumnNameToIndices(column_indices, column_names, file_metadata);

  const size_t max_in_flight = std::max<int64_t>(1, GetTuning().max_in_flight);
  struct io_uring& ring = thread_ring.Get(max_in_flight);

  try {
    int64_t current_target_row = 0;
    size_t target_row_ranges_index = 0;

    // Process each row group
    for (int row_group_index : row_groups) {
      const auto row_group_reader = parquet_reader->RowGroup(row_group_index);
      const auto row_group_metadata = file_metadata->RowGroup(row_group_index);
      const auto rows_in_group = row_group_metadata->num_rows();

      // Get individual column ranges sorted by file offset
      PARQUET_ASSIGN_OR_THROW(auto source_size, fantom_reader->GetSize());
      auto sorted_column_ranges = GetSortedColumnRanges(file_metadata.get(), source_size, row_group_index, column_indices);

      if (nan_fast_path) {
        // All-NaN chunks are filled here, before any I/O is submitted, and left out of the reads below.
        auto all_nan_begin = std::stable_partition(sorted_column_ranges.begin(), sorted_column_ranges.end(),
          [&](const ColumnFileRange& range) {
            const auto* descr = file_metadata->schema()->Column(column_indices[range.column_array_index]);
            return !IsAllNaNChunk(descr, range.metadata.get(), rows_in_group);
          });

        for (auto it = all_nan_begin; it != sorted_column_ranges.end(); ++it) {
          auto read_status = ReadColumn(
            it->column_array_index, current_target_row, nullptr, row_group_metadata.get(), it->metadata.get(),
            out, buffer_size, stride0_size, stride1_size, column_indices, target_column_indices,
            target_row_ranges, target_row_ranges_index, nan_fast_path);
          if (!read_status.ok()) {
            throw std::logic_error("Column read failed: " + read_status.message());
          }
        }

        sorted_column_ranges.erase(all_nan_begin, sorted_column_ranges.end());
      }

      std::vector<::arrow::io::ReadRange> maybe_coalesced_ranges;

      if (pre_buffer)
      {
        // In arrow, read ranges are coalesced only if the pre_buffer is True
        maybe_coalesced_ranges = CoalesceSortedRanges(sorted_column_ranges, cache_options.hole_size_limit, cache_options.range_size_limit);

        if (!use_o_direct)
        {
          fantom_reader->WillNeed(maybe_coalesced_ranges);
        }
      }
      else
      {
        // Merge adjacent columns into fewer, larger reads; with O_DIRECT their shared boundary blocks are read once
        maybe_coalesced_ranges = CoalesceAdjacentRanges(sorted_column_ranges, fantom_reader->GetBlockSize(), GetTuning().max_read_size);
      }

      // Create coalesced I/O requests to minimize file operations
      auto coalesced_requests = MatchColumnsToCoalescedRanges(sorted_column_ranges, maybe_coalesced_ranges, column_indices);

      // Read and process all requests through a bounded window of in-flight reads
      ReadAndProcessRequests(
        ring, max_in_flight, coalesced_requests, fd, fantom_reader, current_target_row,
        row_group_reader.get(), row_group_metadata.get(), column_indices, target_row_ranges,
        target_column_indices, out, buffer_size,
        stride0_size, stride1_size, use_threads, target_row_ranges_index, nan_fast_path
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
    thread_ring.Reset();
    throw;
  }
}