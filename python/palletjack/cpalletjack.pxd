from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libc.stdint cimport uint32_t
from pyarrow._parquet cimport *

cdef extern from "palletjack.h":
    cdef void ReadColumnChunk(const CFileMetaData& file_metadata, const char *parquet_path, void* data, int row_group, int column) except + nogil
