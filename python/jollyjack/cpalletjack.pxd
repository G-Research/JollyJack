from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libc.stdint cimport uint32_t
from pyarrow._parquet cimport *

cdef extern from "jollyjack.h":
    cdef void ReadColumnChunk(const CFileMetaData& file_metadata, const char *parquet_path, void* data, int row_group, int column) except + nogil
    cdef void ReadColumnsF32(const char *parquet_path, shared_ptr[CFileMetaData] file_metadata, void* data, size_t stride_size, int row_group, const vector[uint32_t] column_indices) except + nogil
