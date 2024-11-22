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

def get_table(n_rows, n_columns, data_type = pa.float32()):
    # Generate a random 2D array of floats using NumPy
    # Each column in the array represents a column in the final table
    data = np.random.rand(n_rows, n_columns).astype(np.float32)

    # Convert the NumPy array to a list of PyArrow Arrays, one for each column
    pa_arrays = [pa.array(data[:, i]).cast(data_type, safe = False) for i in range(n_columns)]
    schema = pa.schema([(f'column_{i}', data_type) for i in range(n_columns)])
    # Create a PyArrow Table from the Arrays
    return pa.Table.from_arrays(pa_arrays, schema=schema)

class TestJollyJack(unittest.TestCase):

    def test_transpose_shuffled(self):

        for (n_rows, n_columns) in [(5,6), (1, 1), (100, 200), ]:
            for dtype in [pa.float16(), pa.float32(), pa.float64()]:
                with self.subTest((n_rows, n_columns, dtype)):

                    src_array = get_table(n_rows, n_columns, data_type = dtype).to_pandas().to_numpy()
                    dst_array = np.zeros((n_columns, n_rows), dtype=dtype.to_pandas_dtype(), order='C')
                    jj.transpose_shuffled(src_array = src_array, dst_array = dst_array, row_indices = range(n_columns))
                    self.assertTrue(np.array_equal(src_array.T, dst_array), f"{src_array.T}\n!=\n{dst_array}")

                    # Reversed rows
                    jj.transpose_shuffled(src_array = src_array, dst_array = dst_array, row_indices = [n_columns - i - 1 for i in range(n_columns)])
                    expected_array = src_array.T[::-1, :]
                    self.assertTrue(np.array_equal(expected_array, dst_array), f"{src_array.T}\n!=\n{dst_array}")

                    # Subsets
                    src_view = src_array[1:(n_rows - 1), 1:(n_columns - 1)] 
                    dst_array = np.zeros((n_columns, n_rows), dtype=dtype.to_pandas_dtype(), order='C')
                    dst_view = dst_array[1:(n_columns - 1), 1:(n_rows - 1)] 
                    jj.transpose_shuffled(src_array = src_view, dst_array = dst_view, row_indices = range(n_columns - 2))
                    self.assertTrue(np.array_equal(src_view.T, dst_view), f"{src_view.T}\n!=\n{dst_view}")

    def test_transpose_shuffled_arg_validation(self):

        for (n_rows, n_columns) in [(5,6), ]:
            for dtype in [pa.float16(), pa.float32(), pa.float64()]:
                with self.subTest((n_rows, n_columns, dtype)):

                    src_array = get_table(n_rows, n_columns, data_type = dtype).to_pandas().to_numpy()
                    
                    with self.assertRaises(AssertionError) as context:
                        dst_array = np.zeros((n_columns + 1, n_rows), dtype=dtype.to_pandas_dtype(), order='C')
                        jj.transpose_shuffled(src_array = src_array, dst_array = dst_array, row_indices = range(n_columns))
                    self.assertTrue(f"src_array.shape[1] != dst_array.shape[0], {n_columns} != {n_columns + 1}" in str(context.exception), context.exception)
                    
                    with self.assertRaises(AssertionError) as context:
                        dst_array = np.zeros((n_columns, n_rows + 1), dtype=dtype.to_pandas_dtype(), order='C')
                        jj.transpose_shuffled(src_array = src_array, dst_array = dst_array, row_indices = range(n_columns))
                    self.assertTrue(f"src_array.shape[0] != dst_array.shape[1], {n_rows} != {n_rows + 1}" in str(context.exception), context.exception)
                    
                    with self.assertRaises(AssertionError) as context:
                        dst_array = np.zeros((n_columns, n_rows), dtype=dtype.to_pandas_dtype(), order='F')
                        jj.transpose_shuffled(src_array = src_array, dst_array = dst_array, row_indices = range(n_columns))
                    self.assertTrue(f"Expected destination array in a C (row-major) order" in str(context.exception), context.exception)

                    with self.assertRaises(AssertionError) as context:
                        dst_array = np.zeros((n_columns, n_rows), dtype=np.uint8, order='C')
                        jj.transpose_shuffled(src_array = src_array, dst_array = dst_array, row_indices = range(n_columns))
                    self.assertTrue(f"Source and destination arrays have diffrent datatypes, {src_array.dtype} != uint8" in str(context.exception), context.exception)

                    with self.assertRaises(AssertionError) as context:
                        dst_array = np.zeros((n_columns, n_rows), dtype=np.uint8, order='C')
                        jj.transpose_shuffled(src_array = src_array, dst_array = dst_array, row_indices = range(n_columns) - 1)
                    self.assertTrue(f"TODO" in str(context.exception), context.exception)
                    
                    with self.assertRaises(AssertionError) as context:
                        dst_array = np.zeros((n_columns, n_rows), dtype=np.uint8, order='C')
                        jj.transpose_shuffled(src_array = src_array, dst_array = dst_array, row_indices = [i - 1 for i in range(n_columns)])
                    self.assertTrue(f"TODO" in str(context.exception), context.exception)
                    
                    with self.assertRaises(AssertionError) as context:
                        dst_array = np.zeros((n_columns, n_rows), dtype=np.uint8, order='C')
                        jj.transpose_shuffled(src_array = src_array, dst_array = dst_array, row_indices = [i + 1 for i in range(n_columns)])
                    self.assertTrue(f"TODO" in str(context.exception), context.exception)

if __name__ == '__main__':
    unittest.main()
