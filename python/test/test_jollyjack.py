import unittest
import tempfile

import jollyjack as jj
import palletjack as pj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import os

chunk_size = 3
n_row_groups = 2
n_columns = 5
n_rows = n_row_groups * chunk_size
current_dir = os.path.dirname(os.path.realpath(__file__))

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

    def test_read_entire_table(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows, n_columns)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all(use_threads=False)
            # Create an array of zeros
            np_array1 = np.zeros((n_rows, n_columns), dtype='f', order='F')

            row_begin = 0
            row_end = 0
            
            for rg in range(pr.metadata.num_row_groups):
                row_begin = row_end
                row_end = row_begin + pr.metadata.row_group(rg).num_rows
                subset_view = np_array1[row_begin:row_end, :] 
                jj.read_into_numpy (metadata = pr.metadata
                                       , parquet_path = path
                                       , np_array = subset_view
                                       , row_group_indices = [rg]
                                       , column_indices = range(pr.metadata.num_columns))

            self.assertTrue(np.array_equal(np_array1, expected_data))

            np_array2 = np.zeros((n_rows, n_columns), dtype='f', order='F')
            jj.read_into_numpy (metadata = pr.metadata
                                    , parquet_path = path
                                    , np_array = np_array2
                                    , row_group_indices = range(pr.metadata.num_row_groups)
                                    , column_indices = range(pr.metadata.num_columns))

            self.assertTrue(np.array_equal(np_array2, expected_data))
                
    def test_read_with_palletjack(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows, n_columns)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            index_path = path + '.index'
            pj.generate_metadata_index(path, index_path)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all()
            # Create an array of zeros
            np_array = np.zeros((n_rows, n_columns), dtype='f', order='F')

            row_begin = 0
            row_end = 0

            for rg in range(pr.metadata.num_row_groups):
                column_indices=list(range(n_columns))
                metadata = pj.read_metadata(index_path, row_groups=[rg], column_indices=column_indices)

                row_begin = row_end
                row_end = row_begin + metadata.num_rows
                subset_view = np_array[row_begin:row_end, :] 
                jj.read_into_numpy (metadata = metadata
                                       , parquet_path = path
                                       , np_array = subset_view
                                       , row_group_indices = [0]
                                       , column_indices = column_indices)

            self.assertTrue(np.array_equal(np_array, expected_data))

    def test_read_nonzero_column_offset(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = chunk_size, n_columns = n_columns)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            pr = pq.ParquetReader()
            pr.open(path)
            # Create an array of zeros
            cols = 2
            offset = n_columns - cols
            np_array = np.zeros((chunk_size, cols), dtype='f', order='F')

            jj.read_into_numpy (metadata = pr.metadata
                                    , parquet_path = path
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(offset, offset + cols))

            expected_data = pr.read_all(use_threads=False, column_indices = range(offset, offset + cols))
            self.assertTrue(np.array_equal(np_array, expected_data))

    def test_read_unsupported_column_types(self):
         with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = chunk_size, n_columns = n_columns, data_type = pa.bool_())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            pr = pq.ParquetReader()
            pr.open(path)
            # Create an array of zerosx
            np_array = np.zeros((chunk_size, n_columns), dtype='f', order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (metadata = pr.metadata
                                    , parquet_path = path
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(n_columns))

            self.assertTrue(f"Column 0 has unsupported data type: 0!" in str(context.exception), context.exception)

    def test_read_dtype(self):
        
        for dtype in [pa.float16(), pa.float32(), pa.float64()]:
            for (n_row_groups, n_columns, chunk_size) in [
                    (1, 1, 1),
                    (2, 2, 1),
                    (1, 1, 2),
                    (1, 1, 10),
                    (1, 1, 100),
                    (1, 1, 1_000), 
                    (1, 1, 10_000),
                    (1, 1, 100_000),
                    (1, 1, 1_000_000),
                    (1, 1, 10_000_000),
                    (1, 1, 10_000_001), # +1 to make sure it is not a result of multip,lication of a round number
                ]:
                
                with self.subTest((n_row_groups, n_columns, chunk_size, dtype)):
                    n_rows = n_row_groups * chunk_size
                    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
                        path = os.path.join(tmpdirname, "my.parquet")
                        table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
                        pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

                        pr = pq.ParquetReader()
                        pr.open(path)
                        # Create an array of zerosx
                        np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')

                        jj.read_into_numpy (metadata = pr.metadata
                                                , parquet_path = path
                                                , np_array = np_array
                                                , row_group_indices = range(n_row_groups)
                                                , column_indices = range(n_columns))

                        expected_data = pr.read_all().to_pandas().to_numpy()
                        self.assertTrue(np.array_equal(np_array, expected_data), f"{np_array}\n{expected_data}")

if __name__ == '__main__':
    unittest.main()
