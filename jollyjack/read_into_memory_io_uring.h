#include "parquet/arrow/reader.h"

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
    , bool use_o_direct
    , int64_t expected_rows
    , arrow::io::CacheOptions cache_options);

