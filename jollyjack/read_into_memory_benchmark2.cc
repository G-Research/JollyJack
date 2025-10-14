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

// pread, with coalescing
void ReadIntoMemory_benchmark2(
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

  std::vector<int> single_row_group(1);

  // Process each row group separately to maintain target_row tracking
  for (int row_group : row_groups) {
    single_row_group[0] = row_group;
    auto read_ranges = parquet_reader->GetReadRanges(single_row_group, column_indices, cache_options.hole_size_limit, cache_options.range_size_limit).ValueOrDie();
    /*
    auto start = std::chrono::system_clock::now();
    auto read_ranges = parquet_reader->GetReadRanges(single_row_group, column_indices, cache_options.hole_size_limit, cache_options.range_size_limit).ValueOrDie();
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "ReadIntoMemory_benchmark2::GetReadRanges:" << std::to_string(elapsed.count()) << "ms, read_ranges.size()=" << std::to_string(read_ranges.size()) << std::endl;
  */
    auto result = ::arrow::internal::OptionalParallelFor(use_threads, read_ranges.size(),
            [&](int target_column) { 
              try
              {
                auto &read_range = read_ranges[target_column];
                ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(read_range.length));
                auto result = pread(fd, buffer->mutable_data(), read_range.length, read_range.offset);
                if (result != read_range.length )
                  return arrow::Status::IOError("");

                read_bytes.fetch_add(result);
                return arrow::Status::OK();
              }
              catch(const parquet::ParquetException& e)
              {
                return arrow::Status::UnknownError(e.what());
              }});

    if (!result.ok()) {
      throw std::logic_error(result.message());
    }
  }

  close(fd);
  *(float*)buffer = (float)read_bytes.fetch_add(0);
}