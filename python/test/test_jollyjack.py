import unittest
import tempfile

import jollyjack as jj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import os

chunk_size = 3
n_row_groups = 2
n_columns = 5
n_rows = n_row_groups * chunk_size
current_dir = os.path.dirname(os.path.realpath(__file__))

def get_table():
    # Generate a random 2D array of floats using NumPy
    # Each column in the array represents a column in the final table
    data = np.random.rand(n_rows, n_columns).astype(np.float32)

    # Convert the NumPy array to a list of PyArrow Arrays, one for each column
    pa_arrays = [pa.array(data[:, i]) for i in range(n_columns)]
    schema = pa.schema([(f'column_{i}', pa.float32()) for i in range(n_columns)])
    # Create a PyArrow Table from the Arrays
    return pa.Table.from_arrays(pa_arrays, schema=schema)

class TestJollyJack(unittest.TestCase):
   
    def test_read_entire_table(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table()
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all(use_threads=False)
            # Create an array of zeros
            np_array = np.zeros((n_rows, n_columns), dtype='f', order='F')

            print("\nEmpty array:")
            print(np_array)
            row_begin = 0
            row_end = 0
            
            for rg in range(pr.metadata.num_row_groups):
                row_begin = row_end
                row_end = row_begin + pr.metadata.row_group(rg).num_rows
                subset_view = np_array[row_begin:row_end, :] 
                jj.read_into_numpy_f32(metadata = pr.metadata
                                       , parquet_path = path
                                       , np_array = subset_view
                                       , row_group_idx = rg
                                       , column_indices = range(pr.metadata.num_columns))

            print("\nArray with data read directly into it:")
            print(np_array)

            print("\nExpected data:")
            print (expected_data.to_pandas().to_records(index=False))
            self.assertTrue(np.array_equal(np_array, expected_data))

if __name__ == '__main__':
    unittest.main()
