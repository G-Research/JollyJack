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

void ReadData(const char *parquet_path, std::shared_ptr<parquet::FileMetaData> file_metadata
    , void* data, size_t buffer_size
    , size_t stride0_size, size_t stride1_size
    , const std::vector<int> &row_groups, const std::vector<int> &column_indices
    , bool pre_buffer)
{
  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  auto arrowReaderProperties = parquet::default_arrow_reader_properties();

  std::unique_ptr<parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::OpenFile(parquet_path, false, reader_properties, file_metadata);
  
  if (pre_buffer)
  {
    parquet_reader->PreBuffer(row_groups, column_indices, arrowReaderProperties.io_context(), arrowReaderProperties.cache_options());
  }

  int64_t current_row = 0;
  for (int row_group : row_groups)
  {
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
        << " stride0_size:" << stride0_size
        << " stride1_size:" << stride1_size
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

      int64_t values_read = 0;
      char *base_ptr = (char *)data;
      size_t target_offset = stride0_size * current_row + stride1_size * numpy_column;

      if (buffer_size < target_offset + num_rows * stride0_size)
      {
          auto msg = std::string("Buffer overflow would happen, not executing the read!");
          throw std::logic_error(msg);
      }

      if (column_reader->descr()->physical_type() == parquet::Type::FLOAT)
      {
        auto float_reader = static_cast<parquet::FloatReader *>(column_reader.get());
        auto read_levels = float_reader->ReadBatch(num_rows, nullptr, nullptr, (float *)&base_ptr[stride1_size * numpy_column], &values_read);
        if (values_read != num_rows)
        {
          auto msg = std::string("Expected to read ") + std::to_string(num_rows) + " values, but read " + std::to_string(values_read) + "!";
          throw std::logic_error(msg);
        }
      }

      current_row += num_rows;
    }
  }
}