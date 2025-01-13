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
    def test_read_entire_table_with_slices(self, pre_buffer, use_threads, use_memory_map):

        for dtype in [pa.float16(), pa.float32(), pa.float64()]:
            with tempfile.TemporaryDirectory() as tmpdirname:
                path = os.path.join(tmpdirname, "my.parquet")
                table = get_table(n_rows, n_columns, data_type=dtype)
                pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)
                pr = pq.ParquetReader()
                pr.open(path)

                # Create an array of zeros
                expected_data = pr.read_all()
                expected_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
                expected_array[:chunk_size] = expected_data[chunk_size:]
                expected_array[chunk_size:] = expected_data[:chunk_size]
                row_ranges = [slice (chunk_size, 2 * chunk_size), slice(0, 1), slice (1, chunk_size), ]
                np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(pr.metadata.num_row_groups)
                                    , column_indices = range(pr.metadata.num_columns)
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map
                                    , row_ranges = row_ranges)

                self.assertTrue(np.array_equal(np_array, expected_array))
                pr.close()

    @for_each_parameter()
    def test_read_with_slices_to_large(self, pre_buffer, use_threads, use_memory_map):

        for dtype in [pa.float16(), pa.float32(), pa.float64()]:
            with tempfile.TemporaryDirectory() as tmpdirname:
                path = os.path.join(tmpdirname, "my.parquet")
                table = get_table(n_rows, n_columns, data_type=dtype)
                pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)
                pr = pq.ParquetReader()
                pr.open(path)

                row_ranges = [slice (0, 2 * chunk_size), ]
                np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
                
                with self.assertRaises(RuntimeError) as context:
                    jj.read_into_numpy (source = path
                                        , metadata = None
                                        , np_array = np_array
                                        , row_group_indices = range(pr.metadata.num_row_groups)
                                        , column_indices = range(pr.metadata.num_columns)
                                        , pre_buffer = pre_buffer
                                        , use_threads = use_threads
                                        , use_memory_map = use_memory_map
                                        , row_ranges = row_ranges)

                self.assertTrue(f"Requested to read {2 * chunk_size} rows, but the current row group has only {chunk_size} rows" in str(context.exception), context.exception)

if __name__ == '__main__':
    unittest.main()
