#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/arrow/schema.h"

void ReadColumnChunk(const parquet::FileMetaData& file_metadata, const char *parquet_path, void* data, int row_group, int column);