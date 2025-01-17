#include "arrow/status.h"
#include "arrow/util/parallel.h"
#include "parquet/column_reader.h"
#include "parquet/types.h"

#include "jollyjack.h"

#include <iostream>
#include <stdlib.h>

#if defined(__x86_64__)
  #include <immintrin.h>
#endif

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
    , const std::vector<int64_t> &target_row_ranges
    , size_t target_row_ranges_idx
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

    const auto column_chunk_metadata = row_group_metadata->ColumnChunk(parquet_column);
    for (const auto encoding : column_chunk_metadata->encodings())
    {
      const char *unsupported_encoding = nullptr;

      // Dictionary encoding is not supported for float16 values because FLBA pointers point to non-contiguous memory.
      // Additionally, dictionary encoding prevents proper null value detection across all data types, so we disable it entirely.
      if (encoding == parquet::Encoding::RLE_DICTIONARY)
      {
        unsupported_encoding = "RLE_DICTIONARY";
      }

      if (encoding == parquet::Encoding::PLAIN_DICTIONARY)
      {
        unsupported_encoding = "PLAIN_DICTIONARY";
      }

      // DELTA_BYTE_ARRAY encoding is not supported for float16 values because FLBA pointers reference non-contiguous memory.
      if (encoding == parquet::Encoding::DELTA_BYTE_ARRAY)
      {
        unsupported_encoding = "DELTA_BYTE_ARRAY";
      }

      if (unsupported_encoding)
      {
        auto msg = std::string("Cannot read column=") + std::to_string(parquet_column) + " due to unsupported_encoding=" + unsupported_encoding + "!";
        return arrow::Status::UnknownError(msg);
      }
    }

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
    
    int64_t rows_to_read = num_rows;
    while (true)
    {
      if (target_row_ranges.size() > 0)
      {
        if (target_row_ranges_idx + 1 >= target_row_ranges.size())
        {
          auto msg = std::string("Requested to read ") + std::to_string(rows_to_read + values_read) + " rows"
              + ", but the current row group has " + std::to_string(num_rows) + " rows.";

          return arrow::Status::UnknownError(msg);
        }

        target_row = target_row_ranges[target_row_ranges_idx];
        rows_to_read = target_row_ranges[target_row_ranges_idx + 1] - target_row;

        if (rows_to_read + values_read > num_rows)
        {
            auto msg = std::string("Requested to read ") + std::to_string(rows_to_read + values_read) + " rows"
              + ", but the current row group has only " + std::to_string(num_rows) + " rows.";

            return arrow::Status::UnknownError(msg);
        }
      }

      size_t target_offset = stride0_size * target_row + stride1_size * target_column;
      size_t required_size = target_offset + rows_to_read * stride0_size;

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

          auto typed_reader = static_cast<parquet::DoubleReader *>(column_reader.get());
          while (rows_to_read > 0)
          {
            int64_t tmp_values_read = 0;
            auto read_levels = typed_reader->ReadBatch(rows_to_read, nullptr, nullptr, (double *)&base_ptr[target_offset], &tmp_values_read);
            target_offset += tmp_values_read * stride0_size;
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

          auto typed_reader = static_cast<parquet::FloatReader *>(column_reader.get());
          while (rows_to_read > 0)
          {
            int64_t tmp_values_read = 0;
            auto read_levels = typed_reader->ReadBatch(rows_to_read, nullptr, nullptr, (float *)&base_ptr[target_offset], &tmp_values_read);
            target_offset += tmp_values_read * stride0_size;
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

                memcpy(&base_ptr[target_offset], flba[0].ptr, tmp_values_read * stride0_size);
                target_offset += tmp_values_read * stride0_size;
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

      if (values_read == num_rows)
        break;

      target_row_ranges_idx += 2;
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
    , const std::vector<int64_t> &target_row_ranges
    , const std::vector<std::string> &column_names
    , const std::vector<int> &target_column_indices
    , bool pre_buffer
    , bool use_threads
    , int64_t expected_rows)
{
  if (target_row_ranges.size() % 2 != 0)
  {
    throw std::logic_error("target_row_ranges must contain pairs of [start, end) indices");
  }

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
  size_t target_row_ranges_idx = 0;
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
                , target_column_indices
                , target_row_ranges
                , target_row_ranges_idx);
              });
    if (result != arrow::Status::OK())
    {
      throw std::logic_error(result.message());
    }

    target_row += num_rows;
    if (target_row_ranges.size() > 0)
    {
      auto rows = num_rows;
      while (true)
      {
        auto range_rows = target_row_ranges[target_row_ranges_idx + 1] - target_row_ranges[target_row_ranges_idx];
        target_row_ranges_idx += 2;
        if (rows == range_rows)
          break;

        rows -= range_rows;
      }
    }
  }

    if (target_row_ranges.size() > 0)
    {
      if (target_row_ranges_idx != target_row_ranges.size())
      {
        auto msg = std::string("Expected to read ") + std::to_string(target_row_ranges.size() / 2) + " row ranges, but read only " + std::to_string(target_row_ranges_idx / 2) + "!";
        throw std::logic_error(msg);
      }
    }
    else
    {
      if (target_row != expected_rows)
      {
        auto msg = std::string("Expected to read ") + std::to_string(expected_rows) + " rows, but read only " + std::to_string(target_row) + "!";
        throw std::logic_error(msg);
      }
    }
}

void CopyToRowMajor (void* src_buffer, size_t src_stride0_size, size_t src_stride1_size, int src_rows, int src_cols,
    void* dst_buffer, size_t dst_stride0_size, size_t dst_stride1_size,
    std::vector<int> row_indices)
{
  uint8_t *src_ptr = (uint8_t *)src_buffer;
  uint8_t *dst_ptr = (uint8_t *)dst_buffer;
  const int BLOCK_SIZE = 32;
  char *env_value = getenv("JJ_copy_to_row_major");

#if defined(__x86_64__)
  int variant = 2;
#else
  int variant = 1;
#endif

  if (env_value != NULL)
  {
    variant = atoi(env_value);
  }

  if (variant == 1)
  {
    size_t src_offset_0 = 0;
    size_t dst_offset_0 = 0;
    for (int block_col = 0; block_col < src_cols; block_col += BLOCK_SIZE, src_offset_0 += src_stride1_size * BLOCK_SIZE, dst_offset_0 += dst_stride1_size * BLOCK_SIZE)
    {
      int src_col_limit = std::min (src_cols, block_col + BLOCK_SIZE);
      size_t src_offset_1 = src_offset_0;
      for (int block_row = 0; block_row < src_rows; block_row += BLOCK_SIZE, src_offset_1 += src_stride0_size * BLOCK_SIZE)
      {
        int src_row_limit = std::min (src_rows, block_row + BLOCK_SIZE);
        size_t src_offset_2 = src_offset_1;
        for (int src_row = block_row; src_row < src_row_limit; src_row++, src_offset_2 += src_stride0_size)
        {
          int dst_row = row_indices[src_row];
          size_t src_offset = src_offset_2;
          size_t dst_offset = dst_stride0_size * dst_row + dst_offset_0;
          for (int src_col = block_col; src_col < src_col_limit; src_col++, dst_offset += dst_stride1_size, src_offset += src_stride1_size)
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
    }
  }

#if defined(__x86_64__)
  if (variant == 2)
  {
    // Special fast path for 4-byte elements using SSE
    if (src_stride0_size == 4)
    {
        const int SSE_VECTOR_SIZE = 4; // Number of 32-bit elements in SSE vector

        size_t src_offset_0 = 0;
        size_t dst_offset_0 = 0;
        for (int block_col = 0; block_col < src_cols; block_col += BLOCK_SIZE,
              src_offset_0 += src_stride1_size * BLOCK_SIZE,
              dst_offset_0 += dst_stride1_size * BLOCK_SIZE)
        {
            int src_col_limit = std::min(src_cols, block_col + BLOCK_SIZE);
            size_t src_offset_1 = src_offset_0;
            
            for (int block_row = 0; block_row < src_rows; block_row += BLOCK_SIZE,
                  src_offset_1 += src_stride0_size * BLOCK_SIZE)
            {
                int src_row_limit = std::min(src_rows, block_row + BLOCK_SIZE);
                size_t src_offset_2 = src_offset_1;
                
                for (int src_row = block_row; src_row < src_row_limit; src_row++,
                      src_offset_2 += src_stride0_size)
                {
                    int dst_row = row_indices[src_row];
                    size_t src_offset = src_offset_2;
                    size_t dst_offset = dst_stride0_size * dst_row + dst_offset_0;
                    
                    // Process 4 elements at a time using SSE
                    for (int src_col = block_col; src_col <= src_col_limit - SSE_VECTOR_SIZE;
                          src_col += SSE_VECTOR_SIZE,
                          dst_offset += dst_stride1_size * SSE_VECTOR_SIZE,
                          src_offset += src_stride1_size * SSE_VECTOR_SIZE)
                    {
                          // Load 4 scattered elements into a contiguous vector
                        __m128i v = _mm_set_epi32(
                          *(int*)&src_ptr[src_offset + 3 * src_stride1_size],
                          *(int*)&src_ptr[src_offset + 2 * src_stride1_size],
                          *(int*)&src_ptr[src_offset + 1 * src_stride1_size],
                          *(int*)&src_ptr[src_offset]
                        );

                        // Store the vector to destination (destination is contiguous in memory)
                        _mm_storeu_si128((__m128i*)&dst_ptr[dst_offset], v);
                    }

                    // Handle remaining elements
                    for (int src_col = src_col_limit - (src_col_limit - block_col) % SSE_VECTOR_SIZE;
                          src_col < src_col_limit;
                          src_col++,
                          dst_offset += dst_stride1_size,
                          src_offset += src_stride1_size)
                    {
                        *(uint32_t*)&dst_ptr[dst_offset] = *(uint32_t*)&src_ptr[src_offset];
                    }
                }
            }
        }
    }
    else if (src_stride0_size == 2)
    {
      const int SSE_VECTOR_SIZE = 8; // Number of 16-bit elements in SSE vector

      size_t src_offset_0 = 0;
      size_t dst_offset_0 = 0;
      for (int block_col = 0; block_col < src_cols; block_col += BLOCK_SIZE, 
            src_offset_0 += src_stride1_size * BLOCK_SIZE, 
            dst_offset_0 += dst_stride1_size * BLOCK_SIZE)
      {
        int src_col_limit = std::min(src_cols, block_col + BLOCK_SIZE);
        size_t src_offset_1 = src_offset_0;
        
        for (int block_row = 0; block_row < src_rows; block_row += BLOCK_SIZE,
              src_offset_1 += src_stride0_size * BLOCK_SIZE)
        {
            int src_row_limit = std::min(src_rows, block_row + BLOCK_SIZE);
            size_t src_offset_2 = src_offset_1;
            
            for (int src_row = block_row; src_row < src_row_limit; src_row++,
                  src_offset_2 += src_stride0_size)
            {
                int dst_row = row_indices[src_row];
                size_t src_offset = src_offset_2;
                size_t dst_offset = dst_stride0_size * dst_row + dst_offset_0;
                
                // Process 4 elements at a time using SSE
                for (int src_col = block_col; src_col <= src_col_limit - SSE_VECTOR_SIZE; 
                      src_col += SSE_VECTOR_SIZE,
                      dst_offset += dst_stride1_size * SSE_VECTOR_SIZE,
                      src_offset += src_stride1_size * SSE_VECTOR_SIZE)
                {
                      // Load 8 scattered elements into a contiguous vector
                    __m128i v = _mm_set_epi16(
                      *(short*)&src_ptr[src_offset + 7 * src_stride1_size],
                      *(short*)&src_ptr[src_offset + 6 * src_stride1_size],
                      *(short*)&src_ptr[src_offset + 5 * src_stride1_size],
                      *(short*)&src_ptr[src_offset + 4 * src_stride1_size],
                      *(short*)&src_ptr[src_offset + 3 * src_stride1_size],
                      *(short*)&src_ptr[src_offset + 2 * src_stride1_size],
                      *(short*)&src_ptr[src_offset + 1 * src_stride1_size],
                      *(short*)&src_ptr[src_offset + 0 * src_stride1_size]
                    );

                    // Store the vector to destination (destination is contiguous in memory)
                    _mm_storeu_si128((__m128i*)&dst_ptr[dst_offset], v);
                }
                
                // Handle remaining elements
                for (int src_col = src_col_limit - (src_col_limit - block_col) % SSE_VECTOR_SIZE;
                      src_col < src_col_limit;
                      src_col++,
                      dst_offset += dst_stride1_size,
                      src_offset += src_stride1_size)
                {
                    *(uint16_t*)&dst_ptr[dst_offset] = *(uint16_t*)&src_ptr[src_offset];
                }
            }
        }
      }
    }
    else 
    {
      // Fall back to original implementation for other sizes
      size_t src_offset_0 = 0;
      size_t dst_offset_0 = 0;
      for (int block_col = 0; block_col < src_cols; block_col += BLOCK_SIZE,
            src_offset_0 += src_stride1_size * BLOCK_SIZE,
            dst_offset_0 += dst_stride1_size * BLOCK_SIZE)
      {
        int src_col_limit = std::min(src_cols, block_col + BLOCK_SIZE);
        size_t src_offset_1 = src_offset_0;
        
        for (int block_row = 0; block_row < src_rows; block_row += BLOCK_SIZE,
              src_offset_1 += src_stride0_size * BLOCK_SIZE)
        {
            int src_row_limit = std::min(src_rows, block_row + BLOCK_SIZE);
            size_t src_offset_2 = src_offset_1;
            
            for (int src_row = block_row; src_row < src_row_limit; src_row++,
                  src_offset_2 += src_stride0_size)
            {
                int dst_row = row_indices[src_row];
                size_t src_offset = src_offset_2;
                size_t dst_offset = dst_stride0_size * dst_row + dst_offset_0;
                
                for (int src_col = block_col; src_col < src_col_limit; src_col++,
                      dst_offset += dst_stride1_size,
                      src_offset += src_stride1_size)
                {
                    switch (src_stride0_size)
                    {
                        case 1: *(uint8_t*)&dst_ptr[dst_offset] = *(uint8_t*)&src_ptr[src_offset]; break;
                        case 2: *(uint16_t*)&dst_ptr[dst_offset] = *(uint16_t*)&src_ptr[src_offset]; break;
                        case 4: *(uint32_t*)&dst_ptr[dst_offset] = *(uint32_t*)&src_ptr[src_offset]; break;
                        case 8: *(uint64_t*)&dst_ptr[dst_offset] = *(uint64_t*)&src_ptr[src_offset]; break;
                    }
                }
            }
        }
      }
    }
  }
#endif

}