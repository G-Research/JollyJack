#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/result.h"
#include "arrow/util/logging.h"
#include "arrow/util/type_fwd.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/arrow/schema.h"
#include "parquet/column_reader.h"

#include "jollyjack.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>

using arrow::Status;

void ReadColumnsF32(const char *parquet_path, std::shared_ptr<parquet::FileMetaData> file_metadata, void *data, size_t stride_size, int row_group,
                    const std::vector<uint32_t> &column_indices)
{
  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::OpenFile(parquet_path, false, reader_properties);
  auto row_group_reader = parquet_reader->RowGroup(row_group);
  auto row_group_metadata = file_metadata->RowGroup(row_group);
  auto num_rows = row_group_metadata->num_rows();
  num_rows = num_rows;

#ifdef DEBUG
  std::cerr
      << " ReadColumnChunk rows:" << file_metadata->num_rows()
      << " metadata row_groups:" << file_metadata->num_row_groups()
      << " metadata columns:" << file_metadata->num_columns()
      << " columns.size:" << column_indices.size()
      << " columns.size:" << data
      << std::endl;

  std::cerr
      << " row_group:" << row_group
      << " num_rows:" << num_rows
      << " stride_size:" << stride_size
      << std::endl;
#endif

  for (int numpy_column = 0; numpy_column < column_indices.size(); numpy_column++)
  {
    auto parquet_column = column_indices[numpy_column];
    auto column_reader = row_group_reader->Column(parquet_column);

#ifdef DEBUG
    std::cerr
        << " numpy_column:" << numpy_column
        << " parquet_column:" << parquet_column
        << " logical_type:" << column_reader->descr()->logical_type()->ToString()
        << " physical_type:" << column_reader->descr()->physical_type()
        << std::endl;
#endif

    if (column_reader->descr()->physical_type() == parquet::Type::FLOAT)
    {
      auto float_reader = static_cast<parquet::FloatReader *>(column_reader.get());
      int64_t values_read = 0;
      char *byte_ptr = (char *)data;
      auto read_levels = float_reader->ReadBatch(num_rows, nullptr, nullptr, (float *)&byte_ptr[stride_size * numpy_column], &values_read);
      if (values_read != num_rows)
      {
        auto msg = std::string("Expected to read ") + std::to_string(num_rows) + " values, but read " + std::to_string(values_read) + "!";
        throw std::logic_error(msg);
      }
    }
  }
}