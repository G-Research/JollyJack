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
from libcpp cimport bool
from libc.stdint cimport uint32_t
from pyarrow._parquet cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_python cimport *
from pyarrow.lib cimport (get_reader)

cpdef void read_into_torch (object source, FileMetaData metadata, tensor, row_group_indices, column_indices = [], column_names = [], pre_buffer=False, use_threads=False, use_memory_map = False):

    import torch

    read_into_numpy (source = source
        , metadata = metadata
        , np_array = tensor.numpy()
        , row_group_indices = row_group_indices
        , column_indices = column_indices
        , column_names = column_names
        , pre_buffer = pre_buffer
        , use_threads = use_threads
        , use_memory_map = use_memory_map
    )

    return

cpdef void read_into_numpy (object source, FileMetaData metadata, cnp.ndarray np_array, row_group_indices, column_indices = [], column_names = [], pre_buffer=False, use_threads = False, use_memory_map = False):
    """
    Read parquet data directly into numpy array

    Parameters
    ----------
    source : str, pathlib.Path, pyarrow.NativeFile, or file-like object
    use_memory_map : bool, default False
    metadata : FileMetaData, optional
    pre_buffer : bool, default False
    """

    cdef vector[int] crow_group_indices = row_group_indices
    cdef vector[int] ccolumn_indices = column_indices
    cdef uint32_t cstride0_size = np_array.strides[0]
    cdef uint32_t cstride1_size = np_array.strides[1]
    cdef void* cdata = np_array.data
    cdef bool cpre_buffer = pre_buffer
    cdef bool cuse_threads = use_threads
    cdef vector[string] ccolumn_names = [c.encode('utf8') for c in column_names]
    cdef uint32_t cbuffer_size = (np_array.shape[0]) * cstride0_size + (np_array.shape[1] - 1) * cstride1_size

    # Ensure the input is a 2D array
    assert np_array.ndim == 2, f"Unexpected np_array.ndim, {np_array.ndim} != 2"

    # Ensure the row and column indices are within the array bounds

    assert max(ccolumn_indices.size(), ccolumn_names.size()) == np_array.shape[1], f"Requested to read {ccolumn_indices.size()} columns, but the number of columns in numpy array is {np_array.shape[1]}"
    assert np_array.strides[0] <= np_array.strides[1], f"Expected array in a Fortran-style (column-major) order"

    cdef shared_ptr[CRandomAccessFile] rd_handle
    get_reader(source, use_memory_map, &rd_handle)

    with nogil:
        cjollyjack.ReadIntoMemory (rd_handle, metadata.sp_metadata
            , np_array.data
            , cbuffer_size
            , cstride0_size
            , cstride1_size
            , crow_group_indices
            , ccolumn_indices
            , ccolumn_names
            , cpre_buffer
            , cuse_threads)
        return

    return
