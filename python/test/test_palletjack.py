import unittest
import tempfile

import palletjack as pj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import itertools as it
import pyarrow.fs as fs
import os
from numpy.lib.stride_tricks import as_strided

chunk_size = 3
n_row_groups = 2
n_columns = 5
n_rows = n_row_groups * chunk_size
current_dir = os.path.dirname(os.path.realpath(__file__))

def get_table():
    # Generate a random 2D array of floats using NumPy
    # Each column in the array represents a column in the final table
    data = np.random.rand(n_rows, n_columns)

    # Convert the NumPy array to a list of PyArrow Arrays, one for each column
    pa_arrays = [pa.array(data[:, i]) for i in range(n_columns)]

    # Optionally, create column names
    column_names = [f'column_{i}' for i in range(n_columns)]

    # Create a PyArrow Table from the Arrays
    return pa.Table.from_arrays(pa_arrays, names=column_names)

class TestPalletJack(unittest.TestCase):

    def test_read_column_chunk(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table()
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all(use_threads=False)
            # Create an array of zeros with shape (10, 10)
            np_array = np.zeros((n_rows, n_columns), dtype='f', order='F')

            # Display the original array
            print("Original array (Fortran order):")
            print(np_array)

            # Create a view of a subset of rows (e.g., rows 1 and 2)
            subset_view = np_array[1:4, :]

            # Display the view of the subset of rows
            print("\nSubset view of rows 1 and 2:")
            print(subset_view)

            # Verify if it's a view (changes in the view should affect the original array)
            subset_view[0, 0] = 99
            subset_view[2, 4] = 101
            
            print("\nModified subset view:")
            print(subset_view)

            print("\nOriginal array after modification:")
            print(np_array)

            pj.read_column_chunk(metadata = pr.metadata, parquet_path = path, np_array = subset_view, row_idx = 0, column_idx = 1, row_group_idx = 13)
            pj.read_column_chunk(metadata = pr.metadata, parquet_path = path, np_array = subset_view, row_idx = 0, column_idx = 2, row_group_idx = 13)
            pj.read_column_chunk(metadata = pr.metadata, parquet_path = path, np_array = np_array, row_idx = 0, column_idx = 1, row_group_idx = 13)
            pj.read_column_chunk(metadata = pr.metadata, parquet_path = path, np_array = np_array, row_idx = 0, column_idx = 2, row_group_idx = 13)

            # for c in range(n_columns):
            #     for rg in range(n_row_groups):                
            #         pj.read_column_chunk(metadata = pr.metadata, parquet_path = path, np_array = subset_view, row_idx = 0, column_idx = c, row_group_idx = rg)

            print (expected_data)
            print ()
            print (np_array)
            # self.assertEqual(np_array.all(), expected_data)

if __name__ == '__main__':
    unittest.main()
