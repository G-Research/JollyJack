#include "parquet/arrow/reader.h"

void ReadIntoMemory (std::shared_ptr<arrow::io::RandomAccessFile> source
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
    , int64_t expected_rows);

void CopyToRowMajor (void* src_buffer,
    size_t src_stride0_size,
    size_t src_stride1_size,
    int src_rows,
    int src_cols,
    void* dst_buffer,
    size_t dst_stride0_size,
    size_t dst_stride1_size,
    std::vector<int> row_indices);