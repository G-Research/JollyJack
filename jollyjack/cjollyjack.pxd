from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from libc.stdint cimport int64_t, uint32_t
from pyarrow._parquet cimport *

cdef extern from "jollyjack.h":
    cdef enum JollyJackReaderBackend:
        kJollyJackDefaultReader
        kJollyJackIOUringReader
        kJollyJackIOUringODirectReader

    ctypedef shared_ptr[CFileMetaData] (*PalletJackReadMetadataFunc)(const unsigned char *index_data
        , size_t index_data_length
        , const vector[uint32_t] &row_groups
        , const vector[uint32_t] &column_indices
        , const vector[string] &column_names
        , bint schema_only)

    cdef const char* JOLLYJACK_READ_METADATA_CAPSULE_NAME

    cdef void ReadIntoMemoryBatch (const vector[string] &paths
        , const vector[const unsigned char*] &index_data
        , const vector[size_t] &index_data_lengths
        , const vector[int] &source_column_set
        , const vector[vector[string]] &column_names
        , const vector[vector[int]] &target_column_indices
        , const int64_t *tasks
        , int64_t task_count
        , PalletJackReadMetadataFunc read_metadata
        , void* buffer
        , size_t stride0_size
        , size_t stride1_size
        , int64_t columns
        , int n_threads
        , bool pre_buffer
        , bool use_threads
        , JollyJackReaderBackend backend
        , CCacheOptions cache_options
        ) except + nogil

    cdef void ReadIntoMemory (shared_ptr[CRandomAccessFile] source
        , shared_ptr[CFileMetaData] file_metadata
        , void* buffer
        , size_t buffer_size
        , size_t stride0_size
        , size_t stride1_size
        , vector[int] column_indices
        , const vector[int] &row_groups
        , const vector[int64_t] &target_row_ranges
        , const vector[string] &column_names
        , const vector[int] &target_column_indices
        , bool pre_buffer
        , bool prefetch_page_cache
        , bool use_threads
        , int64_t expected_rows
        , CCacheOptions cache_options
        ) except + nogil

    cdef void CopyToRowMajor (void* src_buffer,
        size_t src_stride0_size,
        size_t src_stride1_size,
        int src_rows,
        int src_cols,
        void* dst_buffer,
        size_t dst_stride0_size,
        size_t dst_stride1_size,
        vector[int] row_indices) except + nogil

    cdef void PrefetchPageCache (shared_ptr[CRandomAccessFile] source
        , shared_ptr[CFileMetaData] file_metadata
        , vector[int] column_indices
        , const vector[int] &row_groups
        , const vector[string] &column_names
        , CCacheOptions cache_options
        ) except + nogil

    cdef shared_ptr[CRandomAccessFile] GetIOUringReader1 (const string& path) except + nogil
    cdef shared_ptr[CRandomAccessFile] GetDirectReader (const string& path) except + nogil

cdef extern from "read_into_memory_io_uring.h":
    cdef void ReadIntoMemoryIOUring (const string& path
        , shared_ptr[CFileMetaData] file_metadata
        , void* buffer
        , size_t buffer_size
        , size_t stride0_size
        , size_t stride1_size
        , vector[int] column_indices
        , const vector[int] &row_groups
        , const vector[int64_t] &target_row_ranges
        , const vector[string] &column_names
        , const vector[int] &target_column_indices
        , bool pre_buffer
        , bool use_threads
        , bool use_o_direct
        , int64_t expected_rows        
        , CCacheOptions cache_options
        ) except + nogil

