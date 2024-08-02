#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/arrow/schema.h"

void ReadData(const char *parquet_path, std::shared_ptr<parquet::FileMetaData> file_metadata
    , void* data, size_t buffer_size
    , size_t stride0_size, size_t stride1_size
    , const std::vector<int> &row_groups
    , const std::vector<int> &column_indices
    , bool pre_buffer);
