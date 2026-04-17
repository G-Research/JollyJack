#pragma once

#include "parquet/arrow/reader.h"

// Alternative pre-buffer implementation that keeps the range-coalescing benefit
// of Arrow's PreBuffer for slow / remote storage without its NUMA-hostile
// user-space buffer staging.
//
// Default pre_buffer=true uses Arrow's ReadRangeCache, which dispatches every
// prefetch read to a shared IO thread pool. The buffer is allocated on
// whichever NUMA node the IO thread happens to live on, and the worker that
// later decodes from that buffer typically sits on a different node -- so
// almost every decode load misses LLC and hits remote DRAM (measured ~97% LLC
// miss rate, 3x slowdown vs pre_buffer=false on warm page cache).
//
// This path instead:
//   1. Computes the coalesced byte ranges via the public
//      ParquetFileReader::GetReadRanges API (same coalescing Arrow would do).
//   2. Issues a single posix_fadvise(WILLNEED) over those ranges via
//      RandomAccessFile::WillNeed. On NFS / cold page cache this kicks off
//      background kernel readahead of a few large ranges. On warm page cache
//      it is a no-op.
//   3. Falls through to the standard (pre_buffer=false) decode path, so each
//      column chunk is read by the same thread that decodes it, with the
//      resulting buffer allocated on that thread's NUMA node and staying hot
//      in L1/L2 across the read-then-decode cycle.
//
// When pre_buffer is false, this function is equivalent to the default
// ReadIntoMemory path.
void ReadIntoMemoryCoalescedSync(
    std::shared_ptr<arrow::io::RandomAccessFile> source,
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
    arrow::io::CacheOptions cache_options);
