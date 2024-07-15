# distutils: include_dirs = .

import cython
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
cimport numpy as cnp
from cython.operator cimport dereference as deref
from cython.cimports.palletjack import cpalletjack
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
from pyarrow._parquet cimport *

cpdef void read_column_chunk(FileMetaData metadata, parquet_path, cnp.ndarray[cnp.float64_t, ndim=2] np_array, row_idx, column_idx):
    cdef string encoded_path = parquet_path.encode('utf8') if parquet_path is not None else "".encode('utf8')

    # Ensure the input is a 2D array
    assert np_array.ndim == 2

    # Ensure that the subarray is C-contiguous
    if not np_array.flags['F_CONTIGUOUS']:
        raise ValueError("np_array must be C-contiguous")

    # Ensure the row and column indices are within the array bounds
    assert 0 <= row_idx < np_array.shape[0]
    assert 0 <= column_idx < np_array.shape[1]
    
    # Get the pointer to the specified element
    cdef double* ptr = <double*> np_array.data
    
    cpalletjack.ReadColumnChunk(deref(metadata.sp_metadata), encoded_path.c_str(), &ptr[column_idx * np_array.shape[0] + row_idx], row_idx, column_idx)

    return
