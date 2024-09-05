from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from pyarrow._parquet cimport *

cdef extern from "jollyjack.h":
    cdef void ReadIntoMemory(shared_ptr[CRandomAccessFile] source
        , shared_ptr[CFileMetaData] file_metadata
        , void* buffer
        , size_t buffer_size
        , size_t stride0_size
        , size_t stride1_size
        , vector[int] column_indices
        , const vector[int] &row_groups
        , const vector[string] &column_names
        , const vector[int] &target_column_indices
        , bool pre_buffer
        , bool use_threads
        , int64_t expected_rows
        ) except + nogil
