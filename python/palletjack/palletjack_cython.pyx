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

cpdef void read_column_chunk_f32(FileMetaData metadata, parquet_path, cnp.ndarray[cnp.float32_t, ndim=2] np_array, row_idx, column_idx, row_group_idx):
    cdef string encoded_path = parquet_path.encode('utf8') if parquet_path is not None else "".encode('utf8')

    # Ensure the input is a 2D array
    assert np_array.ndim == 2

    # Ensure that the subarray is C-contiguous
    # if not np_array.flags['F_CONTIGUOUS']:
    #     raise ValueError("np_array must be F-contiguous")

    # Ensure the row and column indices are within the array bounds
    assert 0 <= row_idx < np_array.shape[0]
    assert 0 <= column_idx < np_array.shape[1]
    
    # Get the pointer to the specified element
    cdef uint8_t* ptr = <uint8_t*> np_array.data
    print (np_array.strides[0])
    print (np_array.strides[1])
    cdef uint8_t* ptr1 = &(ptr[column_idx * np_array.strides[0] + row_idx])
    cdef float* ptr2=<float*>ptr1
    ptr2[0] = 1.23

    # cpalletjack.ReadColumnChunk(deref(metadata.sp_metadata), encoded_path.c_str(), &ptr[column_idx * np_array.shape[0] + row_idx], row_group_idx, column_idx)
    
    return

cpdef void read_into_numpy_f32(parquet_path, FileMetaData metadata, cnp.ndarray[cnp.float32_t, ndim=2] np_array, row_group_idx, column_indices):
    cdef string encoded_path = parquet_path.encode('utf8') if parquet_path is not None else "".encode('utf8')
    cdef uint32_t crow_group_idx = row_group_idx
    cdef vector[uint32_t] ccolumn_indices = column_indices
    cdef uint32_t cstride_size = np_array.strides[1]

    # Ensure the input is a 2D array
    assert np_array.ndim == 2

    # Ensure that the subarray is C-contiguous
    # if not np_array.flags['F_CONTIGUOUS']:
    #     raise ValueError("np_array must be F-contiguous")

    # Ensure the row and column indices are within the array bounds
    assert ccolumn_indices.size() == np_array.shape[1]
    assert np_array.strides[0] == 4 #f32 size

    cpalletjack.ReadColumns(encoded_path.c_str(), metadata.sp_metadata, np_array.data, cstride_size, crow_group_idx, ccolumn_indices)

    return
