# distutils: include_dirs = .

import cython
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
cimport numpy as cnp
import torch

from cython.operator cimport dereference as deref
from cython.cimports.jollyjack import cjollyjack

from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp cimport bool
import ctypes

from libc.stdint cimport uint32_t
from pyarrow._parquet cimport *
from cpython cimport PyCapsule_GetPointer, PyCapsule_Import

cpdef void read_into_torch (parquet_path, FileMetaData metadata, tensor, row_group_indices, column_indices, pre_buffer=False):
    
    assert tensor.dim() == 2, f"Unexpected tensor.dim(), {tensor.dim()} != 2"

    storage = tensor.untyped_storage()
    np_array = np.ctypeslib.as_array((ctypes.c_float * len(storage)).from_address(storage.data_ptr()))
    cdef cnp.ndarray cnp_array = np_array
    cnp_array[0] = 1.01
    
    cdef string encoded_path = parquet_path.encode('utf8') if parquet_path is not None else "".encode('utf8')
    cdef vector[int] crow_group_indices = row_group_indices
    cdef vector[int] ccolumn_indices = column_indices
    cdef uint32_t cstride0_size = tensor.stride()[0]
    cdef uint32_t cstride1_size = tensor.stride()[1]
    cdef void* cdata = cnp_array.data
    cdef uint32_t element_size = tensor.element_size()
    cdef uint32_t cshape0_size = tensor.shape[0]
    cdef uint32_t cshape1_size = tensor.shape[1]
    cdef uint32_t cbuffer_size = element_size * (cstride0_size * cshape0_size + cstride1_size * cshape1_size)

    return

cpdef void read_into_numpy (parquet_path, FileMetaData metadata, cnp.ndarray np_array, row_group_indices, column_indices, pre_buffer=False):
    cdef string encoded_path = parquet_path.encode('utf8') if parquet_path is not None else "".encode('utf8')
    cdef vector[int] crow_group_indices = row_group_indices
    cdef vector[int] ccolumn_indices = column_indices
    cdef uint32_t cstride0_size = np_array.strides[0]
    cdef uint32_t cstride1_size = np_array.strides[1]
    cdef void* cdata = np_array.data
    cdef bool cpre_buffer = pre_buffer
    cdef uint32_t cbuffer_size = np_array.shape[0] * np_array.strides[0] + np_array.shape[1] * np_array.strides[1]

    # Ensure the input is a 2D array
    assert np_array.ndim == 2, f"Unexpected np_array.ndim, {np_array.ndim} != 2"

    # Ensure the row and column indices are within the array bounds
    assert ccolumn_indices.size() == np_array.shape[1], f"Requested to read {ccolumn_indices.size()} columns , but the number of columns in numpy array is {np_array.shape[1]}"
    assert np_array.strides[0] <= np_array.strides[1], f"Expected array in a Fortran order"

    with nogil:
        cjollyjack.ReadIntoMemory (encoded_path.c_str(), metadata.sp_metadata
            , np_array.data
            , cbuffer_size
            , cstride0_size
            , cstride1_size
            , crow_group_indices
            , ccolumn_indices
            , cpre_buffer)
        return

    return
