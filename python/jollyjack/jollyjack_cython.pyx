# distutils: include_dirs = .

import cython
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
cimport numpy as cnp
from cython.operator cimport dereference as deref
from cython.cimports.jollyjack import cjollyjack
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
from pyarrow._parquet cimport *

cpdef void read_into_numpy_f32(parquet_path, FileMetaData metadata, cnp.ndarray[cnp.float32_t, ndim=2] np_array, row_group_idx, column_indices):
    cdef string encoded_path = parquet_path.encode('utf8') if parquet_path is not None else "".encode('utf8')
    cdef uint32_t crow_group_idx = row_group_idx
    cdef vector[uint32_t] ccolumn_indices = column_indices
    cdef uint32_t cstride_size = np_array.strides[1]
    cdef void* cdata = np_array.data

    # Ensure the input is a 2D array
    assert np_array.ndim == 2

    # Ensure the row and column indices are within the array bounds
    assert ccolumn_indices.size() == np_array.shape[1]
    assert np_array.strides[0] == 4 #f32 size

    cjollyjack.ReadColumnsF32(encoded_path.c_str(), metadata.sp_metadata, np_array.data, cstride_size, crow_group_idx, ccolumn_indices)

    return
