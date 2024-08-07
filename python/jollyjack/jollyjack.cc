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

void ReadIntoMemory (const char *parquet_path
    , std::shared_ptr<parquet::FileMetaData> file_metadata
    , void* buffer
    , size_t buffer_size
    , size_t stride0_size
    , size_t stride1_size
    , const std::vector<int> &row_groups
    , const std::vector<int> &column_indices
    , bool pre_buffer)
{
  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  auto arrowReaderProperties = parquet::default_arrow_reader_properties();

  std::unique_ptr<parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::OpenFile(parquet_path, false, reader_properties, file_metadata);
  
  if (pre_buffer)
  {
    parquet_reader->PreBuffer(row_groups, column_indices, arrowReaderProperties.io_context(), arrowReaderProperties.cache_options());
  }

  int64_t target_row = 0;
  for (int row_group : row_groups)
  {
    const auto row_group_reader = parquet_reader->RowGroup(row_group);
    const auto row_group_metadata = file_metadata->RowGroup(row_group);
    const auto num_rows = row_group_metadata->num_rows();

#ifdef DEBUG
    std::cerr
        << " ReadColumnChunk rows:" << file_metadata->num_rows()
        << " metadata row_groups:" << file_metadata->num_row_groups()
        << " metadata columns:" << file_metadata->num_columns()
        << " columns.size:" << column_indices.size()
        << " buffer_size:" << buffer_size
        << std::endl;

    std::cerr
        << " row_group:" << row_group
        << " num_rows:" << num_rows
        << " stride0_size:" << stride0_size
        << " stride1_size:" << stride1_size
        << std::endl;
#endif

    for (int target_column = 0; target_column < column_indices.size(); target_column++)
    {
      auto parquet_column = column_indices[target_column];
      auto column_reader = row_group_reader->Column(parquet_column);

#ifdef DEBUG
      std::cerr
          << " target_column:" << target_column
          << " parquet_column:" << parquet_column
          << " logical_type:" << column_reader->descr()->logical_type()->ToString()
          << " physical_type:" << column_reader->descr()->physical_type()
          << std::endl;
#endif

      int64_t values_read = 0;
      char *base_ptr = (char *)buffer;
      size_t target_offset = stride0_size * target_row + stride1_size * target_column;

      if (buffer_size < target_offset + num_rows * stride0_size)
      {
          auto msg = std::string("Buffer overrun protection, target_row:" + std::to_string(target_row) + " target_column:" + std::to_string(target_column) );
          throw std::logic_error(msg);
      }

      switch (column_reader->descr()->physical_type())
      {
        case parquet::Type::DOUBLE:
        {
          if (stride0_size != 8)
          {
            auto msg = std::string("Column " + std::to_string(parquet_column) + " has DOUBLE data type, but the target value size is " + std::to_string(stride0_size) + "!");
            throw std::logic_error(msg);
          }

          int64_t rows_to_read = num_rows;
          auto typed_reader = static_cast<parquet::DoubleReader *>(column_reader.get());
          while (rows_to_read > 0)
          {
            int64_t tmp_values_read = 0;
            auto read_levels = typed_reader->ReadBatch(rows_to_read, nullptr, nullptr, (double *)&base_ptr[target_offset + values_read * stride0_size], &tmp_values_read);
            values_read += tmp_values_read;
            rows_to_read -= tmp_values_read;
          }
          break;
        }

        case parquet::Type::FLOAT:
        {
          if (stride0_size != 4)
          {
            auto msg = std::string("Column " + std::to_string(parquet_column) + " has FLOAT data type, but the target value size is " + std::to_string(stride0_size) + "!");
            throw std::logic_error(msg);
          }

          int64_t rows_to_read = num_rows;
          auto typed_reader = static_cast<parquet::FloatReader *>(column_reader.get());
          while (rows_to_read > 0)
          {
            int64_t tmp_values_read = 0;
            auto read_levels = typed_reader->ReadBatch(rows_to_read, nullptr, nullptr, (float *)&base_ptr[target_offset + values_read * stride0_size], &tmp_values_read);
            values_read += tmp_values_read;
            rows_to_read -= tmp_values_read;
          }
          break;
        }

        case parquet::Type::FIXED_LEN_BYTE_ARRAY:
        {
          if (stride0_size != column_reader->descr()->type_length())
          {
            auto msg = std::string("Column " + std::to_string(parquet_column) + " has FIXED_LEN_BYTE_ARRAY data type with size " + std::to_string(column_reader->descr()->type_length()) + 
              ", but the target value size is " + std::to_string(stride0_size) + "!");
            throw std::logic_error(msg);
          }

          const size_t warp_size = 1024;
          parquet::FixedLenByteArray flba [warp_size];
          int64_t rows_to_read = num_rows;
          auto typed_reader = static_cast<parquet::FixedLenByteArrayReader *>(column_reader.get());

          while (rows_to_read > 0)
          {
              int64_t tmp_values_read = 0;
              auto read_levels = typed_reader->ReadBatch(warp_size, nullptr, nullptr, flba, &tmp_values_read);
              if (tmp_values_read > 0)
              {
                if (flba[tmp_values_read - 1].ptr - flba[0].ptr != (tmp_values_read - 1) * stride0_size)
                {
                  // TODO(marcink)  We could copy each FLB pointed value one by one instead of throwing an exception.
                  //                However, at the time of this implementation, non-contiguous memory is impossible, so that exception is not expected to occur anyway.
                  auto msg = std::string("Unexpected situation, FLBA memory is not contiguous for olumn:" + std::to_string(parquet_column) + " !");
                  throw std::logic_error(msg);
                }

                memcpy(&base_ptr[target_offset + values_read * stride0_size], flba[0].ptr, tmp_values_read * stride0_size);
                values_read += tmp_values_read;
                rows_to_read -= tmp_values_read;
              }
          }

          break;
        }

        default:
        {
          auto msg = std::string("Column " + std::to_string(parquet_column) + " has unsupported data type: " + std::to_string(column_reader->descr()->physical_type()) + "!");
          throw std::logic_error(msg);
        }
      }
      
      if (values_read != num_rows)
      {
        auto msg = std::string("Expected to read ") + std::to_string(num_rows) + " values, but read " + std::to_string(values_read) + "!";
        throw std::logic_error(msg);
      }
    }

    target_row += num_rows;
  }
}