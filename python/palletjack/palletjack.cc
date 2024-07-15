#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/result.h"
#include "arrow/util/logging.h"
#include "arrow/util/type_fwd.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/arrow/schema.h"
#include "parquet/column_reader.h"

#include "palletjack.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>

using arrow::Status;

void ReadColumnChunk(const parquet::FileMetaData &file_metadata, const char *parquet_path, void *data, int row_group, int column)
{
  std::cerr
      << " ReadColumnChunk rows" << file_metadata.num_rows()
      << " row_groups:" << file_metadata.num_row_groups()
      << " columns:" << file_metadata.num_columns()
      << std::endl;

  std::cerr
      << " row_group:" << row_group
      << " column:" << column
      << std::endl;
 
  // ((double*)data)[0] = 7.12;
  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::OpenFile(parquet_path, false, reader_properties);
  auto row_group_reader = parquet_reader->RowGroup(row_group);
  auto column_reader = row_group_reader->Column(column);
  auto float_reader = static_cast<parquet::DoubleReader*>(column_reader.get());
  auto row_group_metadata = file_metadata.RowGroup(row_group);
  auto num_rows = row_group_metadata->num_rows();

  int64_t values_read = 0;
  auto read_levels = float_reader->ReadBatch(num_rows + 5, nullptr, nullptr, (double*)data, &values_read);
  values_read = values_read;

  num_rows = num_rows;
}