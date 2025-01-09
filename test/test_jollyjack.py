import unittest
import tempfile

import jollyjack as jj
import palletjack as pj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import platform
import os
import itertools
import torch
from pyarrow import fs
from functools import wraps

chunk_size = 3
n_row_groups = 2
n_columns = 5
n_rows = n_row_groups * chunk_size
current_dir = os.path.dirname(os.path.realpath(__file__))

os_name = platform.system()

numpy_to_torch_dtype_dict = {
        np.bool       : torch.bool,
        np.uint8      : torch.uint8,
        np.int8       : torch.int8,
        np.int16      : torch.int16,
        np.int32      : torch.int32,
        np.int64      : torch.int64,
        np.float16    : torch.float16,
        np.float32    : torch.float32,
        np.float64    : torch.float64,
        np.complex64  : torch.complex64,
        np.complex128 : torch.complex128
    }

def get_table(n_rows, n_columns, data_type = pa.float32()):
    # Generate a random 2D array of floats using NumPy
    # Each column in the array represents a column in the final table
    data = np.random.rand(n_rows, n_columns).astype(np.float32)

    # Convert the NumPy array to a list of PyArrow Arrays, one for each column
    pa_arrays = [pa.array(data[:, i]).cast(data_type, safe = False) for i in range(n_columns)]
    schema = pa.schema([(f'column_{i}', data_type) for i in range(n_columns)])
    # Create a PyArrow Table from the Arrays
    return pa.Table.from_arrays(pa_arrays, schema=schema)

def for_each_parameter():
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(self: unittest.TestCase):
            for pre_buffer, use_threads, use_memory_map in itertools.product ([False, True], [False, True], [False, True]):
                with self.subTest((pre_buffer, use_threads, use_memory_map)):
                    test_func(self, pre_buffer = pre_buffer, use_threads = use_threads, use_memory_map = use_memory_map)

        return wrapper
    return decorator

class TestJollyJack(unittest.TestCase):

    @for_each_parameter()
    def test_read_with_palletjack(self, pre_buffer, use_threads, use_memory_map):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows, n_columns)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            index_path = path + '.index'
            pj.generate_metadata_index(path, index_path)

            # Create an array of zeros
            np_array = np.zeros((n_rows, n_columns), dtype='f', order='F')

            row_begin = 0
            row_end = 0
            # Basic list of slices
            slices = [slice(0, 10, 2), slice(20, 30), slice(40, 50)]
            for rg in range(n_row_groups):
                column_indices=list(range(n_columns))
                metadata = pj.read_metadata(index_path, row_groups=[rg], column_indices=column_indices)

                row_begin = row_end
                row_end = row_begin + metadata.num_rows
                subset_view = np_array[row_begin:row_end, :] 
                jj.read_into_numpy (source = path
                                    , metadata = metadata
                                    , np_array = subset_view
                                    , row_group_indices = [0]
                                    , column_indices = column_indices
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map
                                    , row_ranges = slices)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all()
            self.assertTrue(np.array_equal(np_array, expected_data))
            pr.close()

if __name__ == '__main__':
    unittest.main()
