#include "arrow/io/memory.h"
#include "arrow/status.h"
#include "arrow/util/parallel.h"
#include "parquet/column_reader.h"
#include "parquet/column_page.h"
#include "parquet/file_reader.h"
#include "parquet/types.h"

#include "jollyjack.h"

#include "arrow/util/compression.h"

#include <iostream>
#include <stdlib.h>

#if defined(__x86_64__)
  #include <immintrin.h>
#endif

using arrow::Status;

struct ColumnIndex
{
  int column;
  int index;
};

namespace {

class ZstdPageDecoder
{
public:
  ZstdPageDecoder()
  {
    PARQUET_ASSIGN_OR_THROW(codec_, arrow::util::Codec::Create(arrow::Compression::ZSTD));
    PARQUET_ASSIGN_OR_THROW(decompressor_, codec_->MakeDecompressor());
  }

  const uint8_t *Decompress(const uint8_t *src, int64_t src_size, int64_t uncompressed_size)
  {
    if (buffer_.size() < (size_t)uncompressed_size)
      buffer_.resize(uncompressed_size);

    PARQUET_THROW_NOT_OK(decompressor_->Reset());
    int64_t read = 0, written = 0;
    while (read < src_size || !decompressor_->IsFinished())
    {
      auto result = decompressor_->Decompress(src_size - read, src + read, uncompressed_size - written,
          buffer_.data() + written);
      PARQUET_THROW_NOT_OK(result.status());
      read += result->bytes_read;
      written += result->bytes_written;
      if (result->need_more_output || (result->bytes_read == 0 && result->bytes_written == 0))
        break;
    }

    if (written != uncompressed_size || !decompressor_->IsFinished())
      throw parquet::ParquetException(std::string("ZSTD decompression produced ") + std::to_string(written)
          + " bytes, but the page header says " + std::to_string(uncompressed_size));

    return buffer_.data();
  }

private:
  std::unique_ptr<arrow::util::Codec> codec_;
  std::shared_ptr<arrow::util::Decompressor> decompressor_;
  std::vector<uint8_t> buffer_;
};

thread_local ZstdPageDecoder zstd_page_decoder;

void CopyValues(uint8_t *dst, const uint8_t *src, size_t n)
{
#if defined(__x86_64__)
  if (n >= 4096)
  {
    const size_t head = (64 - ((uintptr_t)dst & 63)) & 63;
    memcpy(dst, src, head);
    dst += head, src += head, n -= head;
    for (; n >= 64; n -= 64, src += 64, dst += 64)
    {
      __m128i a = _mm_loadu_si128((const __m128i *)src), b = _mm_loadu_si128((const __m128i *)(src + 16));
      __m128i c = _mm_loadu_si128((const __m128i *)(src + 32)), d = _mm_loadu_si128((const __m128i *)(src + 48));
      _mm_stream_si128((__m128i *)dst, a);
      _mm_stream_si128((__m128i *)(dst + 16), b);
      _mm_stream_si128((__m128i *)(dst + 32), c);
      _mm_stream_si128((__m128i *)(dst + 48), d);
    }
    memcpy(dst, src, n);
    _mm_sfence();
    return;
  }
#endif
  memcpy(dst, src, n);
}

struct PageHeaderLite
{
  int32_t type = -1, uncompressed_size = -1, compressed_size = -1;
  int32_t num_values = -1, encoding = -1, def_level_encoding = -1, num_nulls = 0;
  int32_t def_levels_bytes = 0, rep_levels_bytes = 0;
  bool is_compressed = true;
};

class CompactReader
{
public:
  CompactReader(const uint8_t *p, const uint8_t *end) : p_(p), end_(end) {}
  const uint8_t *pos() const { return p_; }

  uint8_t Byte()
  {
    if (p_ >= end_)
      throw parquet::ParquetException("Truncated page header");
    return *p_++;
  }

  uint64_t Varint()
  {
    uint64_t r = 0;
    for (int shift = 0; shift < 64; shift += 7)
    {
      const uint8_t b = Byte();
      r |= uint64_t(b & 0x7f) << shift;
      if (!(b & 0x80))
        return r;
    }
    throw parquet::ParquetException("Corrupt varint in page header");
  }

  int32_t I32() { const uint64_t v = Varint(); return int32_t((v >> 1) ^ -(v & 1)); }

  void Skip(int type)
  {
    switch (type)
    {
      case 1: case 2: return;
      case 3: Byte(); return;
      case 4: case 5: case 6: Varint(); return;
      case 7: Advance(8); return;
      case 8: Advance(Varint()); return;
      case 9: case 10:
      {
        const uint8_t h = Byte();
        uint64_t n = h >> 4;
        if (n == 15)
          n = Varint();
        for (uint64_t i = 0; i < n; i++)
          SkipElement(h & 0x0f);
        return;
      }
      case 11:
      {
        const uint64_t n = Varint();
        if (n == 0)
          return;
        const uint8_t kv = Byte();
        for (uint64_t i = 0; i < n; i++)
        {
          SkipElement(kv >> 4);
          SkipElement(kv & 0x0f);
        }
        return;
      }
      case 12: SkipStruct(); return;
      default: throw parquet::ParquetException("Unexpected thrift type in page header");
    }
  }

  template <typename F>
  void Struct(F &&field)
  {
    int16_t id = 0;
    while (true)
    {
      const uint8_t h = Byte();
      if (h == 0)
        return;
      const int type = h & 0x0f;
      if (h >> 4)
        id += h >> 4;
      else
        id = int16_t(I32());
      field(id, type);
    }
  }

  void SkipStruct() { Struct([&](int16_t, int type) { Skip(type); }); }

private:
  void SkipElement(int type)
  {
    if (type == 1 || type == 2)
      Byte();  // bools in containers take a byte each
    else
      Skip(type);
  }

  void Advance(uint64_t n)
  {
    if (n > uint64_t(end_ - p_))
      throw parquet::ParquetException("Truncated page header");
    p_ += n;
  }

  const uint8_t *p_, *end_;
};

const uint8_t *ParsePageHeader(const uint8_t *p, const uint8_t *end, PageHeaderLite *h)
{
  CompactReader r(p, end);
  r.Struct([&](int16_t id, int type) {
    if (type == 5 && id >= 1 && id <= 3)
      (id == 1 ? h->type : id == 2 ? h->uncompressed_size : h->compressed_size) = r.I32();
    else if (type == 12 && id == 5)
      r.Struct([&](int16_t id5, int type5) {
        if (type5 == 5 && id5 == 1) h->num_values = r.I32();
        else if (type5 == 5 && id5 == 2) h->encoding = r.I32();
        else if (type5 == 5 && id5 == 3) h->def_level_encoding = r.I32();
        else r.Skip(type5);
      });
    else if (type == 12 && id == 8)
      r.Struct([&](int16_t id8, int type8) {
        if (type8 == 5 && id8 == 1) h->num_values = r.I32();
        else if (type8 == 5 && id8 == 2) h->num_nulls = r.I32();
        else if (type8 == 5 && id8 == 4) h->encoding = r.I32();
        else if (type8 == 5 && id8 == 5) h->def_levels_bytes = r.I32();
        else if (type8 == 5 && id8 == 6) h->rep_levels_bytes = r.I32();
        else if ((type8 == 1 || type8 == 2) && id8 == 7) h->is_compressed = type8 == 1;
        else r.Skip(type8);
      });
    else
      r.Skip(type);
  });
  if (h->uncompressed_size < 0 || h->compressed_size < 0 || h->compressed_size > end - r.pos())
    throw parquet::ParquetException("Corrupt page header");
  return r.pos();
}

}  // namespace

// Reader for FLBA columns. On the fast path (PLAIN-encoded, flat, null-free),
// copies values directly from decompressed page buffers into the caller's
// output. Otherwise delegates to a regular Arrow ColumnReader.
class ContiguousFLBAReader : public parquet::ColumnReader
{
public:
  ContiguousFLBAReader(parquet::RowGroupReader *row_group_reader,
                       int column_index,
                       const parquet::ColumnDescriptor *descr,
                       const parquet::ColumnChunkMetaData *column_chunk_metadata,
                       const std::shared_ptr<arrow::io::RandomAccessFile> &source)
      : descr_(descr)
      , type_length_(descr->type_length())
      , max_def_level_(descr->max_definition_level())
  {
    // The fast path only handles PLAIN-encoded, flat columns read as a whole chunk.
    bool fast_path = true;
    for (const auto encoding : column_chunk_metadata->encodings())
    {
      if (encoding != parquet::Encoding::PLAIN && encoding != parquet::Encoding::RLE)
        fast_path = false;
    }

    if (fast_path && descr->max_repetition_level() == 0)
    {
      if (source != nullptr && (column_chunk_metadata->compression() == arrow::Compression::ZSTD
          || column_chunk_metadata->compression() == arrow::Compression::UNCOMPRESSED))
        OpenChunk(source, column_chunk_metadata);
      else
        page_reader_ = row_group_reader->GetColumnPageReader(column_index);
    }
    else
      fallback_reader_ = row_group_reader->Column(column_index);  // delegate to Arrow
  }

  int type_length() const { return type_length_; }

  // Reads up to batch_size values into dst (a raw buffer of batch_size * type_length bytes).
  int64_t ReadBatch(int64_t batch_size, void *dst_raw, int64_t *values_read)
  {
    uint8_t *dst = static_cast<uint8_t *>(dst_raw);

    if (fallback_reader_)
    {
      constexpr int64_t kScratch = 1024;
      parquet::FixedLenByteArray flba[kScratch];
      int64_t n = 0;
      static_cast<parquet::FixedLenByteArrayReader*>(fallback_reader_.get())->ReadBatch(
          std::min<int64_t>(batch_size, kScratch), nullptr, nullptr, flba, &n);

      if (n > 0)
      {
        if (flba[0].ptr + (n - 1) * type_length_ != flba[n - 1].ptr)
          throw parquet::ParquetException("Unexpected, FLBA memory is not contiguous");
        memcpy(dst, flba[0].ptr, n * type_length_);
      }

      *values_read = n;
      return n;
    }

    if (page_pos_ >= page_values_)
    {
      if (eof_ || !NextDataPage()) { eof_ = true; *values_read = 0; return 0; }
    }

    const int64_t n = std::min<int64_t>(batch_size, page_values_ - page_pos_);
    CopyValues(dst, page_values_ptr_ + page_pos_ * type_length_, n * type_length_);
    page_pos_ += n;
    *values_read = n;
    return n;
  }

  bool HasNext() override
  {
    if (fallback_reader_)
      return fallback_reader_->HasNext();
    if (page_pos_ < page_values_)
      return true;
    if (eof_)
      return false;
    eof_ = !NextDataPage();
    return !eof_;
  }

  parquet::Type::type type() const override { return parquet::Type::FIXED_LEN_BYTE_ARRAY; }
  const parquet::ColumnDescriptor *descr() const override { return descr_; }
  parquet::ExposedEncoding GetExposedEncoding() override
  {
    throw std::logic_error("Not Implemented");
  }

protected:
  void SetExposedEncoding(parquet::ExposedEncoding /*encoding*/) override 
  {
    throw std::logic_error("Not Implemented");
  }

private:

  void OpenChunk(const std::shared_ptr<arrow::io::RandomAccessFile> &source,
      const parquet::ColumnChunkMetaData *column_chunk_metadata)
  {
    int64_t chunk_start = column_chunk_metadata->data_page_offset();
    if (column_chunk_metadata->has_dictionary_page() && column_chunk_metadata->dictionary_page_offset() > 0
        && chunk_start > column_chunk_metadata->dictionary_page_offset())
      chunk_start = column_chunk_metadata->dictionary_page_offset();

    PARQUET_ASSIGN_OR_THROW(chunk_, source->ReadAt(chunk_start, column_chunk_metadata->total_compressed_size()));
    chunk_pos_ = chunk_->data();
    chunk_end_ = chunk_pos_ + chunk_->size();
    remaining_values_ = column_chunk_metadata->num_values();
    compressed_ = column_chunk_metadata->compression() == arrow::Compression::ZSTD;
    fast_pages_ = true;
  }

  bool NextFastDataPage()
  {
    while (chunk_pos_ < chunk_end_ && remaining_values_ > 0)
    {
      PageHeaderLite h;
      const uint8_t *payload = ParsePageHeader(chunk_pos_, chunk_end_, &h);
      chunk_pos_ = payload + h.compressed_size;
      if (h.type != 0 && h.type != 3)
        continue;
      if (h.encoding != 0)
        throw parquet::ParquetException(std::string("Unexpected data page encoding ")
            + parquet::EncodingToString(static_cast<parquet::Encoding::type>(h.encoding))
            + " on the FIXED_LEN_BYTE_ARRAY fast path");
      if (h.num_values < 0)
        throw parquet::ParquetException("Corrupt page header");
      remaining_values_ -= h.num_values;

      if (h.type == 3)
      {
        if (h.num_nulls != 0)
          throw parquet::ParquetException("Column contains null values");
        const int64_t levels = int64_t(h.def_levels_bytes) + h.rep_levels_bytes;
        if (h.def_levels_bytes < 0 || h.rep_levels_bytes < 0 || levels > h.compressed_size || levels > h.uncompressed_size)
          throw parquet::ParquetException("Corrupt page header");
        const int64_t values_size = h.uncompressed_size - levels;
        const uint8_t *values_ptr = payload + levels;
        if (compressed_ && h.is_compressed)
          values_ptr = zstd_page_decoder.Decompress(values_ptr, h.compressed_size - levels, values_size);
        else if (h.compressed_size != h.uncompressed_size)
          throw parquet::ParquetException("Corrupt page header");
        if (int64_t(h.num_values) * type_length_ > values_size)
          throw parquet::ParquetException("Column contains null values");
        if (h.num_values == 0)
          continue;
        page_values_ptr_ = values_ptr;
        page_values_ = h.num_values;
        page_pos_ = 0;
        return true;
      }

      if (!compressed_ && h.compressed_size != h.uncompressed_size)
        throw parquet::ParquetException("Corrupt page header");
      const uint8_t *page_data = compressed_ ? zstd_page_decoder.Decompress(payload, h.compressed_size, h.uncompressed_size) : payload;
      const uint8_t *page_end = page_data + h.uncompressed_size;
      const uint8_t *values_ptr = page_data;
      if (max_def_level_ > 0)
      {
        if (h.def_level_encoding != 3)
          throw parquet::ParquetException(std::string("Unsupported definition-level encoding ")
              + parquet::EncodingToString(static_cast<parquet::Encoding::type>(h.def_level_encoding))
              + " on the FIXED_LEN_BYTE_ARRAY fast path");
        if (values_ptr + 4 > page_end)
          throw parquet::ParquetException("Corrupt definition-level section");
        int32_t def_len = 0;
        memcpy(&def_len, values_ptr, sizeof(int32_t));  // little-endian length prefix
        if (def_len < 0)
          throw parquet::ParquetException("Invalid definition-level length");
        values_ptr += 4 + def_len;
      }

      if (values_ptr < page_data || values_ptr + int64_t(h.num_values) * type_length_ > page_end)
        throw parquet::ParquetException("Column contains null values");

      if (h.num_values == 0)
        continue;

      page_values_ptr_ = values_ptr;
      page_values_ = h.num_values;
      page_pos_ = 0;
      return true;
    }
    return false;
  }

  // Advance to the next data page and locate its contiguous value region.
  // Returns false at end of column. Throws on unexpected encodings/nulls.
  bool NextDataPage()
  {
    if (fast_pages_)
      return NextFastDataPage();
    while (true)
    {
      current_page_ = page_reader_->NextPage();
      if (current_page_ == nullptr)
        return false;

      const auto page_type = current_page_->type();
      if (page_type != parquet::PageType::DATA_PAGE
          && page_type != parquet::PageType::DATA_PAGE_V2)
        continue;  // skip dictionary / other pages

      auto data_page = std::static_pointer_cast<parquet::DataPage>(current_page_);
      if (data_page->encoding() != parquet::Encoding::PLAIN)
        throw parquet::ParquetException(std::string("Unexpected data page encoding ")
            + parquet::EncodingToString(data_page->encoding())
            + " on the FIXED_LEN_BYTE_ARRAY fast path");

      const uint8_t *page_data = data_page->data();
      const int64_t page_size = data_page->size();
      const uint8_t *page_end = page_data + page_size;
      const uint8_t *values_ptr = page_data;
      const int64_t page_values = data_page->num_values();

      if (page_type == parquet::PageType::DATA_PAGE_V2)
      {
        // V2 stores levels uncompressed with explicit byte lengths and an
        // explicit null count in the header.
        auto data_page_v2 = std::static_pointer_cast<parquet::DataPageV2>(current_page_);
        if (data_page_v2->num_nulls() != 0)
          throw parquet::ParquetException("Column contains null values");

        values_ptr += data_page_v2->repetition_levels_byte_length()
                    + data_page_v2->definition_levels_byte_length();
      }
      else if (max_def_level_ > 0)
      {
        // V1: skip the definition-level section (no repetition levels, since the
        // column is flat). num_values includes the (zero) nulls.
        const auto def_encoding = std::static_pointer_cast<parquet::DataPageV1>(current_page_)->definition_level_encoding();
        if (def_encoding == parquet::Encoding::RLE)
        {
          if (values_ptr + 4 > page_end)
            throw parquet::ParquetException("Corrupt definition-level section");
          int32_t def_len = 0;
          memcpy(&def_len, values_ptr, sizeof(int32_t));  // little-endian length prefix
          if (def_len < 0)
            throw parquet::ParquetException("Invalid definition-level length");
          values_ptr += 4 + def_len;
        }
        else
        {
          throw parquet::ParquetException(std::string("Unsupported definition-level encoding ")
              + parquet::EncodingToString(def_encoding)
              + " on the FIXED_LEN_BYTE_ARRAY fast path");
        }
      }

      if (values_ptr < page_data || values_ptr + page_values * type_length_ > page_end)
        throw parquet::ParquetException("Column contains null values");

      if (page_values == 0)
        continue;  // skip empty pages so page_values_ > 0 always holds after return

      page_values_ptr_ = values_ptr;
      page_values_ = page_values;
      page_pos_ = 0;
      return true;
    }
  }

  const parquet::ColumnDescriptor *descr_;
  const int type_length_;
  const int16_t max_def_level_;

  // Set when the fast path applies; otherwise fallback_reader_ is used.
  std::unique_ptr<parquet::PageReader> page_reader_;
  std::shared_ptr<parquet::ColumnReader> fallback_reader_;

  bool eof_ = false;
  bool fast_pages_ = false;
  bool compressed_ = false;
  std::shared_ptr<arrow::Buffer> chunk_;
  const uint8_t *chunk_pos_ = nullptr;
  const uint8_t *chunk_end_ = nullptr;
  int64_t remaining_values_ = 0;
  std::shared_ptr<parquet::Page> current_page_;  // keeps the current page buffer alive
  const uint8_t *page_values_ptr_ = nullptr;
  int64_t page_values_ = 0;
  int64_t page_pos_ = 0;
};

std::shared_ptr<parquet::ColumnReader> MakeFLBAReader(parquet::RowGroupReader *row_group_reader,
    int column_index, const parquet::ColumnDescriptor *descr,
    const parquet::ColumnChunkMetaData *column_chunk_metadata,
    const std::shared_ptr<arrow::io::RandomAccessFile> &source)
{
  return std::make_shared<ContiguousFLBAReader>(row_group_reader, column_index, descr, column_chunk_metadata, source);
}

arrow::Status ReadColumn (int column_index
    , int64_t target_row
    , parquet::ColumnReader *column_reader
    , parquet::RowGroupMetaData *row_group_metadata
    , const parquet::ColumnChunkMetaData *column_chunk_metadata
    , void* buffer
    , size_t buffer_size
    , size_t stride0_size
    , size_t stride1_size
    , const std::vector<int> &column_indices
    , const std::vector<int> &target_column_indices
    , const std::vector<int64_t> &target_row_ranges
    , size_t target_row_ranges_idx
    )
{
  std::string column_name;
  const auto num_rows = row_group_metadata->num_rows();
  const auto parquet_column = column_indices[column_index];

  try
  {
    column_name = column_reader->descr()->name();

    int target_column = column_index;
    if (target_column_indices.size() > 0)
      target_column = target_column_indices[column_index];

    for (const auto encoding : column_chunk_metadata->encodings())
    {
      bool unsupported_encoding = false;

      // 1. Dictionary encoding is not supported for float16 values because FLBA pointers point to non-contiguous memory.
      // 2. Dictionary encoding prevents proper null value detection across all data types, so we disable it entirely.
      if (encoding == parquet::Encoding::RLE_DICTIONARY || encoding == parquet::Encoding::PLAIN_DICTIONARY)
      {
        unsupported_encoding = true;
      }

      // DELTA_BYTE_ARRAY encoding is not supported for float16 values because FLBA pointers reference non-contiguous memory.
      if (encoding == parquet::Encoding::DELTA_BYTE_ARRAY)
      {
        unsupported_encoding = true;
      }

      if (unsupported_encoding)
      {
        auto msg = std::string("Cannot read column=") + std::to_string(parquet_column) + " due to unsupported_encoding=" + parquet::EncodingToString(encoding) + "!";
        return arrow::Status::UnknownError(msg);
      }
    }

    #ifdef DEBUG
        std::cerr
            << " column_index:" << column_index
            << " target_column:" << target_column
            << " parquet_column:" << parquet_column
            << " physical_type:" << column_chunk_metadata->type()
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

      switch (column_chunk_metadata->type())
      {
        case parquet::Type::DOUBLE:
        {
          if (stride0_size != 8)
          {
            auto msg = std::string("Column[" + std::to_string(parquet_column) + "] ('"  + column_name + "') has DOUBLE data type, but the target value size is " + std::to_string(stride0_size) + "!");
            return arrow::Status::UnknownError(msg);
          }

          auto typed_reader = static_cast<parquet::DoubleReader *>(column_reader);
          while (rows_to_read > 0)
          {
            int64_t tmp_values_read = 0;
            std::ignore = typed_reader->ReadBatch(rows_to_read, nullptr, nullptr, (double *)&base_ptr[target_offset], &tmp_values_read);
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

          auto typed_reader = static_cast<parquet::FloatReader *>(column_reader);
          while (rows_to_read > 0)
          {
            int64_t tmp_values_read = 0;
            std::ignore = typed_reader->ReadBatch(rows_to_read, nullptr, nullptr, (float *)&base_ptr[target_offset], &tmp_values_read);
            target_offset += tmp_values_read * stride0_size;
            values_read += tmp_values_read;
            rows_to_read -= tmp_values_read;
          }
          break;
        }

        case parquet::Type::FIXED_LEN_BYTE_ARRAY:
        {
          auto flba_reader = static_cast<ContiguousFLBAReader *>(column_reader);
          if ((int32_t)stride0_size != flba_reader->type_length())
          {
            auto msg = std::string("Column[" + std::to_string(parquet_column) + "] ('"  + column_name + "') has FIXED_LEN_BYTE_ARRAY data type with size " + std::to_string(flba_reader->type_length()) +
              ", but the target value size is " + std::to_string(stride0_size) + "!");
            return arrow::Status::UnknownError(msg);
          }
          while (rows_to_read > 0)
          {
            int64_t tmp_values_read = 0;
            std::ignore = flba_reader->ReadBatch(rows_to_read, &base_ptr[target_offset], &tmp_values_read);
            if (tmp_values_read == 0)
              break;
            target_offset += tmp_values_read * stride0_size;
            values_read += tmp_values_read;
            rows_to_read -= tmp_values_read;
          }

          break;
        }

        case parquet::Type::INT32:
        {
          if (stride0_size != 4)
          {
            auto msg = std::string("Column[" + std::to_string(parquet_column) + "] ('" + column_name + "') has INT32 data type, but the target value size is " + std::to_string(stride0_size) + "!");
            return arrow::Status::UnknownError(msg);
          }

          auto typed_reader = static_cast<parquet::Int32Reader *>(column_reader);
          while (rows_to_read > 0)
          {
            int64_t tmp_values_read = 0;
            std::ignore = typed_reader->ReadBatch(rows_to_read, nullptr, nullptr, (int32_t *)&base_ptr[target_offset], &tmp_values_read);
            target_offset += tmp_values_read * stride0_size;
            values_read += tmp_values_read;
            rows_to_read -= tmp_values_read;
          }
          break;
        }

        case parquet::Type::INT64:
        {
          if (stride0_size != 8)
          {
            auto msg = std::string("Column[" + std::to_string(parquet_column) + "] ('" + column_name + "') has INT64 data type, but the target value size is " + std::to_string(stride0_size) + "!");
            return arrow::Status::UnknownError(msg);
          }

          auto typed_reader = static_cast<parquet::Int64Reader *>(column_reader);
          while (rows_to_read > 0)
          {
            int64_t tmp_values_read = 0;
            std::ignore = typed_reader->ReadBatch(rows_to_read, nullptr, nullptr, (int64_t *)&base_ptr[target_offset], &tmp_values_read);
            target_offset += tmp_values_read * stride0_size;
            values_read += tmp_values_read;
            rows_to_read -= tmp_values_read;
          }
          break;
        }

        default:
        {
          auto msg = std::string("Column[" + std::to_string(parquet_column) + "] ('"  + column_name + "') has unsupported data type: " + std::to_string(column_chunk_metadata->type()) + "!");
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
    auto what = std::string(e.what());
    if (what.find("Column contains null values") != std::string::npos
        || what.find("Unexpected end of stream") != std::string::npos)
    {
      auto msg = what + ". Column[" + std::to_string(parquet_column) + "] ('"  + column_name + "') contains null values?";
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
    , bool prefetch_page_cache
    , bool use_threads
    , int64_t expected_rows
    , arrow::io::CacheOptions cache_options)
{
  if (target_row_ranges.size() % 2 != 0)
  {
    throw std::logic_error("target_row_ranges must contain pairs of [start, end) indices");
  }

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

  if (prefetch_page_cache)
  {
    PARQUET_ASSIGN_OR_THROW(auto read_ranges, parquet_reader->GetReadRanges(row_groups,
                              column_indices,
                              cache_options.hole_size_limit,
                              cache_options.range_size_limit
                            ));
    auto status = source->WillNeed(read_ranges);
    if (!status.ok()) {
      throw std::runtime_error(status.message());
    }
  }

  if (pre_buffer)
  {
    parquet_reader->PreBuffer(row_groups, column_indices, arrowReaderProperties.io_context(), cache_options);
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

  // Warm up the lazy cache in a single-threaded pass to avoid thread contention
  // during parallel reads.
  if (pre_buffer && use_threads && cache_options.lazy)
  {
    for (auto c_idx: column_indices)
    {
      std::ignore = row_group_reader->Column(c_idx);
    }
  }

  auto result = ::arrow::internal::OptionalParallelFor(use_threads, column_indices.size(),
            [&](int i) {
              try
              {
                const int parquet_column = column_indices[i];
                const auto column_chunk_metadata = row_group_metadata->ColumnChunk(parquet_column);
                const auto *descr = row_group_metadata->schema()->Column(parquet_column);
                std::shared_ptr<parquet::ColumnReader> column_reader;
                if (descr->physical_type() == parquet::Type::FIXED_LEN_BYTE_ARRAY)
                  column_reader = MakeFLBAReader(row_group_reader.get(), parquet_column, descr, column_chunk_metadata.get(),
                      pre_buffer ? nullptr : source);
                else
                  column_reader = row_group_reader->Column(parquet_column);

                return ReadColumn(i
                  , target_row
                  , column_reader.get()
                  , row_group_metadata.get()
                  , column_chunk_metadata.get()
                  , buffer
                  , buffer_size
                  , stride0_size
                  , stride1_size
                  , column_indices
                  , target_column_indices
                  , target_row_ranges
                  , target_row_ranges_idx);
              }
              catch(const parquet::ParquetException& e)
              {
                return arrow::Status::UnknownError(e.what());
              }});

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

void PrefetchPageCache(
    std::shared_ptr<arrow::io::RandomAccessFile> source,
    std::shared_ptr<parquet::FileMetaData> file_metadata,
    std::vector<int> column_indices,
    const std::vector<int>& row_groups,
    const std::vector<std::string>& column_names,
    arrow::io::CacheOptions cache_options) {
  parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
  auto reader = parquet::ParquetFileReader::Open(source, reader_properties, file_metadata);
  auto metadata = reader->metadata();

  if (!column_names.empty()) {
    column_indices.reserve(column_names.size());
    auto schema = metadata->schema();
    for (const auto& name : column_names) {
      int idx = schema->ColumnIndex(name);
      if (idx < 0) {
        throw std::logic_error(std::string("Column '") + name + "' was not found!");
      }
      column_indices.push_back(idx);
    }
  }

  PARQUET_ASSIGN_OR_THROW(auto read_ranges, reader->GetReadRanges(row_groups, 
                            column_indices,
                            cache_options.hole_size_limit,
                            cache_options.range_size_limit
                          ));

  auto status = source->WillNeed(read_ranges);
  if (!status.ok()) {
    throw std::runtime_error(status.message());
  }
}

#ifdef WITH_IO_URING
#else
void ReadIntoMemoryIOUring (const std::string& path
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
    , bool use_o_direct
    , int64_t expected_rows
    , arrow::io::CacheOptions cache_options)
{
  throw std::runtime_error("io_uring is not available on this platform!"); 
}

#endif