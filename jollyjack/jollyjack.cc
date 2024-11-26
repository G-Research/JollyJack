
#include "arrow/status.h"
#include "arrow/util/parallel.h"
#include "parquet/column_reader.h"

#include "jollyjack.h"

#include <iostream>
#include <stdlib.h>

using arrow::Status;

arrow::Status ReadColumn (int column_index
    , int64_t target_row
    , std::shared_ptr<parquet::RowGroupReader> row_group_reader
    , parquet::RowGroupMetaData *row_group_metadata
    , void* buffer
    , size_t buffer_size
    , size_t stride0_size
    , size_t stride1_size
    , const std::vector<int> &column_indices
    , const std::vector<int> &target_column_indices
    ) noexcept
{
  std::string column_name;
  const auto num_rows = row_group_metadata->num_rows();
  const auto parquet_column = column_indices[column_index];
  
  try
  {
    const auto column_reader = row_group_reader->Column(parquet_column);
    column_name = column_reader->descr()->name();

    int target_column = column_index;
    if (target_column_indices.size() > 0)
      target_column = target_column_indices[column_index];

    #ifdef DEBUG
        std::cerr
            << " column_index:" << column_index
            << " target_column:" << target_column
            << " parquet_column:" << parquet_column
            << " logical_type:" << column_reader->descr()->logical_type()->ToString()
            << " physical_type:" << column_reader->descr()->physical_type()
            << std::endl;
    #endif

    int64_t values_read = 0;
    char *base_ptr = (char *)buffer;
    size_t target_offset = stride0_size * target_row + stride1_size * target_column;
    size_t required_size = target_offset + num_rows * stride0_size;

    if (target_offset >= buffer_size)
    {        
        auto msg = std::string("Buffer overrun error:")          
          + " Attempted to read " + std::to_string(num_rows) + " rows into location [" + std::to_string(target_row)
          + ", " + std::to_string(target_column) + "], but that is beyond target's boundaries.";

        return arrow::Status::UnknownError(msg);
    }

    if (required_size > buffer_size)
    {
        auto left_space = (buffer_size - target_offset) / stride0_size;
        auto msg = std::string("Buffer overrun error:")          
          + " Attempted to read " + std::to_string(num_rows) + " rows into location [" + std::to_string(target_row)
          + ", " + std::to_string(target_column) + "], but there was space available for only " + std::to_string(left_space) + " rows.";

        return arrow::Status::UnknownError(msg);
    }

    switch (column_reader->descr()->physical_type())
    {
      case parquet::Type::DOUBLE:
      {
        if (stride0_size != 8)
        {
          auto msg = std::string("Column[" + std::to_string(parquet_column) + "] ('"  + column_name + "') has DOUBLE data type, but the target value size is " + std::to_string(stride0_size) + "!");
          return arrow::Status::UnknownError(msg);
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
          auto msg = std::string("Column[" + std::to_string(parquet_column) + "] ('"  + column_name + "') has FLOAT data type, but the target value size is " + std::to_string(stride0_size) + "!");
          return arrow::Status::UnknownError(msg);
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
          auto msg = std::string("Column[" + std::to_string(parquet_column) + "] ('"  + column_name + "') has FIXED_LEN_BYTE_ARRAY data type with size " + std::to_string(column_reader->descr()->type_length()) + 
            ", but the target value size is " + std::to_string(stride0_size) + "!");
          return arrow::Status::UnknownError(msg);
        }

        const int64_t warp_size = 1024;
        parquet::FixedLenByteArray flba [warp_size];
        int64_t rows_to_read = num_rows;
        auto typed_reader = static_cast<parquet::FixedLenByteArrayReader *>(column_reader.get());

        while (rows_to_read > 0)
        {
            int64_t tmp_values_read = 0;
            auto read_levels = typed_reader->ReadBatch(std::min(warp_size, rows_to_read), nullptr, nullptr, flba, &tmp_values_read);
            if (tmp_values_read > 0)
            {
              if (flba[tmp_values_read - 1].ptr - flba[0].ptr != (tmp_values_read - 1) * stride0_size)
              {
                // TODO(marcink)  We could copy each FLB pointed value one by one instead of throwing an exception.
                //                However, at the time of this implementation, non-contiguous memory is impossible, so that exception is not expected to occur anyway.
                auto msg = std::string("Unexpected, FLBA memory is not contiguous when reading olumn:" + std::to_string(parquet_column) + " !");
                return arrow::Status::UnknownError(msg);
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
        auto msg = std::string("Column[" + std::to_string(parquet_column) + "] ('"  + column_name + "') has unsupported data type: " + std::to_string(column_reader->descr()->physical_type()) + "!");
        return arrow::Status::UnknownError(msg);
      }
    }

    if (values_read != num_rows)
    {
      auto msg = std::string("Column[" + std::to_string(parquet_column) + "] ('"  + column_name + "'): Expected to read ") + std::to_string(num_rows) + " values, but read only " + std::to_string(values_read) + "!";
      return arrow::Status::UnknownError(msg);
    }
  }
  catch(const parquet::ParquetException& e)
  {
    if (e.what() == std::string("Unexpected end of stream"))
    {
      auto msg = std::string(e.what() + std::string(". Column[" + std::to_string(parquet_column) + "] ('"  + column_name + "') contains null values?"));
      return arrow::Status::UnknownError(msg);
    }

    return arrow::Status::UnknownError(e.what());
  }

  return arrow::Status::OK();
}

void ReadIntoMemory (std::shared_ptr<arrow::io::RandomAccessFile> source
    , std::shared_ptr<parquet::FileMetaData> file_metadata
    , void* buffer
    , size_t buffer_size
    , size_t stride0_size
    , size_t stride1_size
    , std::vector<int> column_indices
    , const std::vector<int> &row_groups
    , const std::vector<std::string> &column_names
    , const std::vector<int> &target_column_indices
    , bool pre_buffer
    , bool use_threads
    , int64_t expected_rows)
{
  arrow::io::RandomAccessFile *random_access_file = nullptr;
  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  auto arrowReaderProperties = parquet::default_arrow_reader_properties();

  std::unique_ptr<parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::Open(source, reader_properties, file_metadata);
  file_metadata = parquet_reader->metadata();

  if (column_names.size() > 0)
  {
      column_indices.reserve(column_names.size());
      auto schema = file_metadata->schema();
      for (auto column_name : column_names)
      {
        auto column_index = schema->ColumnIndex(column_name);
         
        if (column_index < 0)
        {
          auto msg = std::string("Column '") + column_name + "' was not found!";
          throw std::logic_error(msg);
        }

        column_indices.push_back(column_index);
      }
  }

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
        << " column_indices.size:" << column_indices.size()
        << " buffer_size:" << buffer_size
        << std::endl;

    std::cerr
        << " row_group:" << row_group
        << " num_rows:" << num_rows
        << " stride0_size:" << stride0_size
        << " stride1_size:" << stride1_size
        << std::endl;
#endif

  auto result = ::arrow::internal::OptionalParallelFor(use_threads, column_indices.size(),
            [&](int target_column) { 
              return ReadColumn(target_column
                , target_row
                , row_group_reader
                , row_group_metadata.get()
                , buffer
                , buffer_size
                , stride0_size
                , stride1_size
                , column_indices
                , target_column_indices);
              });
    if (result != arrow::Status::OK())
    {
      throw std::logic_error(result.message());
    }

    target_row += num_rows;
  }

  if (target_row != expected_rows)
  {
    auto msg = std::string("Expected to read ") + std::to_string(expected_rows) + " rows, but read only " + std::to_string(target_row) + "!";
    throw std::logic_error(msg);
  }
}

void TransposeShuffle (void* src_buffer, size_t src_stride0_size, size_t src_stride1_size, int src_rows, int src_cols,
    void* dst_buffer, size_t dst_stride0_size, size_t dst_stride1_size,
    std::vector<int> row_indices)
{
  uint8_t *src_ptr = (uint8_t *)src_buffer;
  uint8_t *dst_ptr = (uint8_t *)dst_buffer;
  const int BLOCK_SIZE = 32;
  char *env_value = getenv("JJ_TRANSPOSE_SHUFFLE");
  int variant = 1;
  if (env_value != NULL)
  {
    variant = atoi(env_value);
  }

  for (auto row_index : row_indices)
  {
      if (row_index < 0 || row_index >= src_cols)
      {          
          auto msg = std::string("Row index = '" + std::to_string(row_index) + "' is not in the expected range [0, " + std::to_string(src_cols) + ")!");
          throw std::logic_error(msg);
      }
  }

  if (variant == 1)
  {
    for (int src_col = 0; src_col < src_cols; src_col++)
    {    
      int dst_row = row_indices[src_col];
      size_t src_offset = src_stride1_size * src_col;
      size_t dst_offset = dst_stride0_size * dst_row;

      for (int src_row = 0; src_row < src_rows; src_row++, src_offset += src_stride0_size, dst_offset += dst_stride1_size)
      {
        switch (src_stride0_size)
        {
          case 1:*(uint8_t*)&dst_ptr[dst_offset] = *(uint8_t*)&src_ptr[src_offset]; break;
          case 2:*(uint16_t*)&dst_ptr[dst_offset] = *(uint16_t*)&src_ptr[src_offset]; break;
          case 4:*(uint32_t*)&dst_ptr[dst_offset] = *(uint32_t*)&src_ptr[src_offset]; break;
          case 8:*(uint64_t*)&dst_ptr[dst_offset] = *(uint64_t*)&src_ptr[src_offset]; break;
        }
      }
    }
  }

  // It turns out to be slower than the "variant 1"

  if (variant == 2)
  {  
    for (int block_col = 0; block_col < src_cols; block_col += BLOCK_SIZE)
    {            
      int src_col_limit = std::min (src_cols, block_col + BLOCK_SIZE);
      for (int block_row = 0; block_row < src_rows; block_row += BLOCK_SIZE)
      {
        int src_row_limit = std::min (src_rows, block_row + BLOCK_SIZE);
        for (int src_col = block_col; src_col < src_col_limit; src_col++)
        {
          for (int src_row = block_row; src_row < src_row_limit; src_row++)
          {
            int dst_row = row_indices[src_col];
            int dst_col = src_row;
            size_t src_offset = src_stride0_size * src_row + src_stride1_size * src_col;
            size_t dst_offset = dst_stride0_size * dst_row + dst_stride1_size * dst_col;

            switch (src_stride0_size)
            {
              case 1:*(uint8_t*)&dst_ptr[dst_offset] = *(uint8_t*)&src_ptr[src_offset]; break;
              case 2:*(uint16_t*)&dst_ptr[dst_offset] = *(uint16_t*)&src_ptr[src_offset]; break;
              case 4:*(uint32_t*)&dst_ptr[dst_offset] = *(uint32_t*)&src_ptr[src_offset]; break;
              case 8:*(uint64_t*)&dst_ptr[dst_offset] = *(uint64_t*)&src_ptr[src_offset]; break;
            }
          }
        }
      }
    }
  }
  

}
