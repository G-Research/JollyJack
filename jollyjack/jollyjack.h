#include "parquet/arrow/reader.h"

enum JollyJackReaderBackend
{
  kJollyJackDefaultReader = 0,
  kJollyJackIOUringReader = 1,
  kJollyJackIOUringODirectReader = 2,
};

// Prefix of the PalletJackCAPI table from palletjack_capi.h in G-Research/PalletJack, handed over as the PyCapsule
// palletjack.palletjack_capi. PalletJack only appends members to the table, so declaring the members up to the last
// one used here is enough. Members jollyjack does not call are void* placeholders that keep the layout.
#define PALLETJACK_CAPI_CAPSULE_NAME "palletjack.palletjack_capi"

using PalletJackReadMetadataFromBufferV1Func = std::shared_ptr<parquet::FileMetaData> (*)(const unsigned char *index_data,
    size_t index_data_length,
    const std::vector<uint32_t> &row_groups,
    const std::vector<uint32_t> &column_indices,
    const std::vector<std::string> &column_names,
    bool schema_only);

struct PalletJackCAPI
{
  int version;
  // Added in version 1
  void *GenerateMetadataIndexToBufferV1;
  void *GenerateMetadataIndexToFileV1;
  PalletJackReadMetadataFromBufferV1Func ReadMetadataFromBufferV1;
};

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
    , arrow::io::CacheOptions cache_options);

// Reads a whole batch of (source, row group) pairs into one array, on a native thread pool, without the GIL.
// Task i is the four int64 values at tasks[4 * i]: (source index, row group, first row in the array, rows).
void ReadIntoMemoryBatch (const std::vector<std::string> &paths
    , const std::vector<const unsigned char*> &index_data
    , const std::vector<size_t> &index_data_lengths
    , const std::vector<int> &source_column_set
    , const std::vector<std::vector<std::string>> &column_names
    , const std::vector<std::vector<int>> &target_column_indices
    , const int64_t *tasks
    , int64_t task_count
    , PalletJackReadMetadataFromBufferV1Func read_metadata
    , void* buffer
    , size_t stride0_size
    , size_t stride1_size
    , int64_t columns
    , int n_threads
    , bool pre_buffer
    , bool use_threads
    , JollyJackReaderBackend backend
    , arrow::io::CacheOptions cache_options);

void CopyToRowMajor (void* src_buffer,
    size_t src_stride0_size,
    size_t src_stride1_size,
    int src_rows,
    int src_cols,
    void* dst_buffer,
    size_t dst_stride0_size,
    size_t dst_stride1_size,
    std::vector<int> row_indices);

// Calls posix_fadvise(POSIX_FADV_WILLNEED) on the byte ranges for the requested columns/row groups.
void PrefetchPageCache(
    std::shared_ptr<arrow::io::RandomAccessFile> source,
    std::shared_ptr<parquet::FileMetaData> file_metadata,
    std::vector<int> column_indices,
    const std::vector<int>& row_groups,
    const std::vector<std::string>& column_names,
    arrow::io::CacheOptions cache_options);


// A source, if given, is read in full when the reader is created, so it should be cheap to read.
std::shared_ptr<parquet::ColumnReader> MakeFLBAReader(parquet::RowGroupReader *row_group_reader,
    int column_index, const parquet::ColumnDescriptor *descr,
    const parquet::ColumnChunkMetaData *column_chunk_metadata,
    const std::shared_ptr<arrow::io::RandomAccessFile> &source = nullptr);

arrow::Status ReadColumn (int column_index
    , int64_t target_row
    , parquet::ColumnReader* column_reader
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
    );