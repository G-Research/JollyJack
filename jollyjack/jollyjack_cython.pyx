# distutils: include_dirs = .

import cython
import pyarrow.parquet as pq
cimport numpy as cnp

from cython.cimports.jollyjack import cjollyjack

from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport uint32_t
from pyarrow._parquet cimport *
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

    cdef vector[int] crow_group_indices = row_group_indices
    cdef vector[int] ccolumn_indices = column_indices.keys() if isinstance(column_indices, dict) else column_indices
    cdef uint32_t cstride0_size = np_array.strides[0]
    cdef uint32_t cstride1_size = np_array.strides[1]
    cdef void* cdata = np_array.data
    cdef bool cpre_buffer = pre_buffer
    cdef bool cuse_threads = use_threads
    cdef vector[string] ccolumn_names = [c.encode('utf8') for c in column_names]
    cdef uint32_t cbuffer_size = (np_array.shape[0]) * cstride0_size + (np_array.shape[1] - 1) * cstride1_size
    cdef shared_ptr[CFileMetaData] c_metadata
    if metadata is not None:
        c_metadata = metadata.sp_metadata

    target_column_indices = []
    if column_indices and isinstance(column_indices, dict):
        target_column_indices = column_indices.values()
    elif column_indices:
        assert len(column_indices) == np_array.shape[1], f"Requested to read {len(column_indices)} columns, but the number of columns in numpy array is {np_array.shape[1]}"
       
    if isinstance(column_names, dict):
        target_column_indices = column_names.values()
    elif column_names:
        assert len(column_names) == np_array.shape[1], f"Requested to read {len(column_names)} columns, but the number of columns in numpy array is {np_array.shape[1]}"

    cdef vector[int] ctarget_column_indices = target_column_indices

    # Ensure that only one input is set
    assert (column_indices or column_names) and (not column_indices or not column_names), f"Either column_indices or column_names needs to be set"

    # Ensure the input is a 2D array, Fortran-styl
    assert np_array.ndim == 2, f"Unexpected np_array.ndim, {np_array.ndim} != 2"
    assert np_array.strides[0] <= np_array.strides[1], f"Expected array in a Fortran-style (column-major) order"

    cdef int64_t cexpected_rows = np_array.shape[0]
    cdef shared_ptr[CRandomAccessFile] rd_handle
    get_reader(source, use_memory_map, &rd_handle)

    with nogil:
        cjollyjack.ReadIntoMemory (rd_handle
            , c_metadata
            , np_array.data
            , cbuffer_size
            , cstride0_size
            , cstride1_size
            , ccolumn_indices
            , crow_group_indices
            , ccolumn_names
            , ctarget_column_indices
            , cpre_buffer
            , cuse_threads
            , cexpected_rows)
        return

    return
