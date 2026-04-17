#include "coalesced_sync_reader.h"

#include "jollyjack.h"

#include "arrow/io/interfaces.h"
#include "parquet/file_reader.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

void ReadIntoMemoryCoalescedSync(
    std::shared_ptr<arrow::io::RandomAccessFile> source,
    std::shared_ptr<parquet::FileMetaData> file_metadata,
    void* buffer, size_t buffer_size, size_t stride0_size, size_t stride1_size,
    std::vector<int> column_indices, const std::vector<int>& row_groups,
    const std::vector<int64_t>& target_row_ranges,
    const std::vector<std::string>& column_names,
    const std::vector<int>& target_column_indices, bool pre_buffer, bool use_threads,
    int64_t expected_rows, arrow::io::CacheOptions cache_options) {
  if (!pre_buffer) {
    ReadIntoMemory(std::move(source), std::move(file_metadata), buffer, buffer_size,
                   stride0_size, stride1_size, std::move(column_indices), row_groups,
                   target_row_ranges, column_names, target_column_indices,
                   /*pre_buffer=*/false, use_threads, expected_rows, cache_options);
    return;
  }

  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  auto tmp_reader =
      parquet::ParquetFileReader::Open(source, reader_properties, file_metadata);
  auto shared_metadata = tmp_reader->metadata();

  if (!column_names.empty()) {
    column_indices.reserve(column_names.size());
    auto schema = shared_metadata->schema();
    for (const auto& column_name : column_names) {
      int column_index = schema->ColumnIndex(column_name);
      if (column_index < 0) {
        auto msg = std::string("Column '") + column_name + "' was not found!";
        throw std::logic_error(msg);
      }
      column_indices.push_back(column_index);
    }
  }

  auto ranges_result =
      tmp_reader->GetReadRanges(row_groups, column_indices,
                                cache_options.hole_size_limit,
                                cache_options.range_size_limit);
  if (!ranges_result.ok()) {
    throw std::logic_error(ranges_result.status().message());
  }
  auto ranges = ranges_result.MoveValueUnsafe();

  std::sort(ranges.begin(), ranges.end(),
            [](const arrow::io::ReadRange& a, const arrow::io::ReadRange& b) {
              return a.offset < b.offset;
            });

  // posix_fadvise(WILLNEED) on the coalesced ranges. For NFS / cold page cache
  // this triggers background kernel readahead of a few large ranges instead of
  // thousands of tiny per-column fetches. For warm page cache it is a no-op.
  std::ignore = source->WillNeed(ranges);

  tmp_reader.reset();

  // Regular decode path: each column does its own synchronous per-chunk ReadAt
  // on the worker thread that decodes it. Allocations stay small, NUMA-local,
  // and hot in L1/L2 across the read-then-decode for a single column.
  ReadIntoMemory(std::move(source), std::move(shared_metadata), buffer, buffer_size,
                 stride0_size, stride1_size, std::move(column_indices), row_groups,
                 target_row_ranges, /*column_names=*/{}, target_column_indices,
                 /*pre_buffer=*/false, use_threads, expected_rows, cache_options);
}
