import unittest
import tempfile

import jollyjack as jj
import palletjack as pj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import platform
import os
from pyarrow import fs

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

    def test_read_numpy_column_mapping(self):

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = pa.float32())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')
            column_names = {f'column_{i}':n_columns- i - 1 for i in range(n_columns)}
            names = [c for c, t in column_names] if isinstance(column_names, dict) else [c.encode('utf8') for c in column_names] 
            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = range(n_row_groups)
                                , column_names = column_names
                                )

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            self.assertTrue(np.array_equal(np_array, expected_data), f"{np_array}\n{expected_data}")

if __name__ == '__main__':
    unittest.main()
