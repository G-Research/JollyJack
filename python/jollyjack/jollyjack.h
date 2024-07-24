#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/arrow/schema.h"

void ReadColumnChunk(const parquet::FileMetaData& file_metadata, const char *parquet_path, void* data, int row_group, int column);
void ReadColumnsF32(const char *parquet_path, std::shared_ptr<parquet::FileMetaData> file_metadata, void* data, size_t stride_size, int row_group, 
                                                    const std::vector<uint32_t> &column_indices);
