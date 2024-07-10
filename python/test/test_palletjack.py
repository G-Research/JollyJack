import unittest
import tempfile

import palletjack as pj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import itertools as it
import pyarrow.fs as fs
import os

n_row_groups = 5
n_columns = 7
chunk_size = 1 # A row group per
current_dir = os.path.dirname(os.path.realpath(__file__))

def get_table():
    # Generate a random 2D array of floats using NumPy
    # Each column in the array represents a column in the final table
    data = np.random.rand(n_row_groups, n_columns)

    # Convert the NumPy array to a list of PyArrow Arrays, one for each column
    pa_arrays = [pa.array(int(x) for x in data[:, i]) for i in range(n_columns)]

    # Optionally, create column names
    column_names = [f'column_{i}' for i in range(n_columns)]

    # Create a PyArrow Table from the Arrays
    return pa.Table.from_arrays(pa_arrays, names=column_names)

class TestPalletJack(unittest.TestCase):

    def test_inmemory_index_data(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table()

            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            index_path = path + '.index'
            pj.generate_metadata_index(path, index_path)
            index_data1 = pj.generate_metadata_index(path)
            index_data2 = fs.LocalFileSystem().open_input_stream(index_path).readall()
            # Compare the actual output to the expected output
            self.assertEqual(index_data1, index_data2)
            
            pr = pq.ParquetReader()
            pr.open(path)
            metadata = pr.metadata
            c = metadata.row_group(0).column(0)
            data = pr.read_all(use_threads=False)
            data = data

if __name__ == '__main__':
    unittest.main()
