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

    def test_read_entire_table(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows, n_columns)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            pr = pq.ParquetReader()
            pr.open(path)

            # Create an array of zeros
            np_array1 = np.zeros((n_rows, n_columns), dtype='f', order='F')

            row_begin = 0
            row_end = 0

            for rg in range(n_row_groups):
                row_begin = row_end
                row_end = row_begin + pr.metadata.row_group(rg).num_rows
                subset_view = np_array1[row_begin:row_end, :] 
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = subset_view
                                    , row_group_indices = [rg]
                                    , column_indices = range(pr.metadata.num_columns))

            expected_data = pr.read_all()
            self.assertTrue(np.array_equal(np_array1, expected_data))

            np_array2 = np.zeros((n_rows, n_columns), dtype='f', order='F')
            jj.read_into_numpy (source = path
                                , metadata = None
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

            # Create an array of zeros
            np_array = np.zeros((n_rows, n_columns), dtype='f', order='F')

            row_begin = 0
            row_end = 0

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
                                    , column_indices = column_indices)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all()
            self.assertTrue(np.array_equal(np_array, expected_data))

    def test_read_nonzero_column_offset(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = chunk_size, n_columns = n_columns)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an array of zeros
            cols = 2
            offset = n_columns - cols
            np_array = np.zeros((chunk_size, cols), dtype='f', order='F')

            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = [0]
                                , column_indices = range(offset, offset + cols))

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all(column_indices = range(offset, offset + cols))
            self.assertTrue(np.array_equal(np_array, expected_data))

    def test_read_unsupported_column_types(self):
         with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = chunk_size, n_columns = n_columns, data_type = pa.bool_())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an array of zerosx
            np_array = np.zeros((chunk_size, n_columns), dtype='f', order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(n_columns))

            self.assertTrue(f"Column 0 has unsupported data type: 0!" in str(context.exception), context.exception)

    def test_read_dtype_numpy(self):
        
        for pre_buffer in [False, True]:
            for use_threads in [False, True]:
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
                        
                        with self.subTest((n_row_groups, n_columns, chunk_size, dtype, pre_buffer, use_threads)):
                            n_rows = n_row_groups * chunk_size
                            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
                                path = os.path.join(tmpdirname, "my.parquet")
                                table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
                                pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

                                # Create an empty array
                                np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')

                                jj.read_into_numpy (source = path
                                                    , metadata = None
                                                    , np_array = np_array
                                                    , row_group_indices = range(n_row_groups)
                                                    , column_indices = range(n_columns)
                                                    , pre_buffer = pre_buffer
                                                    , use_threads = use_threads)

                                pr = pq.ParquetReader()
                                pr.open(path)
                                expected_data = pr.read_all().to_pandas().to_numpy()
                                self.assertTrue(np.array_equal(np_array, expected_data), f"{np_array}\n{expected_data}")

    def test_read_dtype_torch(self):
        
        if os_name == "Windows":
            # Code specific to Windows
            print("Not running on Windows because of issues with torch + numpy (https://github.com/marcin-krystianc/JollyJack/issues/15).")
            # Add your Windows-specific code here
            return
        
        if os_name != "Windows":
            import torch

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
                        (1, 1, 1_000_001),
                    ]:                

                    with self.subTest((n_row_groups, n_columns, chunk_size, dtype)):
                        n_rows = n_row_groups * chunk_size

                        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
                            path = os.path.join(tmpdirname, "my.parquet")
                            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
                            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

                            tensor = torch.zeros(n_columns, n_rows, dtype = numpy_to_torch_dtype_dict[dtype.to_pandas_dtype()]).transpose(0, 1)

                            jj.read_into_torch (source = path
                                                , metadata = None
                                                , tensor = tensor
                                                , row_group_indices = range(n_row_groups)
                                                , column_indices = range(n_columns))

                            pr = pq.ParquetReader()
                            pr.open(path)
                            expected_data = pr.read_all().to_pandas().to_numpy()
                            self.assertTrue(np.array_equal(tensor.numpy(), expected_data), f"{tensor.numpy()}\n{expected_data}")

    def test_read_numpy_column_names(self):

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = pa.float32())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')

            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = range(n_row_groups)
                                , column_names = [f'column_{i}' for i in range(n_columns)]
                                )

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            self.assertTrue(np.array_equal(np_array, expected_data), f"{np_array}\n{expected_data}")

    def test_read_torch_column_names(self):

        if os_name == "Windows":
            # Code specific to Windows
            print("Not running on Windows because of issues with torch + numpy (https://github.com/marcin-krystianc/JollyJack/issues/15).")
            # Add your Windows-specific code here
            return

        if os_name != "Windows":
            import torch

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = pa.float32())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an empty array
            tensor = torch.zeros(n_columns, n_rows, dtype = torch.float32).transpose(0, 1)

            jj.read_into_torch (source = path
                                , metadata = None
                                , tensor = tensor
                                , row_group_indices = range(n_row_groups)
                                , column_names = [f'column_{i}' for i in range(n_columns)]
                                )

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            self.assertTrue(np.array_equal(tensor.numpy(), expected_data), f"{tensor.numpy()}\n{expected_data}")

    def test_read_invalid_column(self):

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = pa.float32())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            pr = pq.ParquetReader()
            pr.open(path)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_names = [f'foo_bar_{i}' for i in range(n_columns)]
                                    )

            self.assertTrue(f"Column 'foo_bar_0' was not found!" in str(context.exception), context.exception)
                
            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_indices = [i + 1 for i in range(n_columns)]
                                    )

            self.assertTrue(f"Trying to read column index {n_columns} but row group metadata has only {n_columns} columns" in str(context.exception), context.exception)

    def test_read_filesystem(self):

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = pa.float32())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')

            with fs.LocalFileSystem().open_input_file(path) as f:
                jj.read_into_numpy (source = f
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_names = [f'column_{i}' for i in range(n_columns)]
                                    )

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            self.assertTrue(np.array_equal(np_array, expected_data), f"{np_array}\n{expected_data}")

    def test_read_invalid_row_group(self):

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = pa.float32())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [n_row_groups]
                                    , column_names = [f'column_{i}' for i in range(n_columns)]
                                    )
            
            self.assertTrue(f"Trying to read row group {n_row_groups} but file only has {n_row_groups} row groups" in str(context.exception), context.exception)

    def test_read_data_with_nulls(self):

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = pa.float32())
            df = table.to_pandas()
            df.iloc[0, 0] = np.nan
            table = pa.Table.from_pandas(df)

            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_names = [f'column_{i}' for i in range(n_columns)]
                                    )

            self.assertTrue(f"Unexpected end of stream. Column 0 contains null values?" in str(context.exception), context.exception)

    def test_read_not_enough_rows(self):

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = pa.float32())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an empty array
            np_array = np.zeros((n_rows + 1, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_names = [f'column_{i}' for i in range(n_columns)]
                                    )

            self.assertTrue(f"Expected to read {n_rows + 1} rows, but read only {n_rows}!" in str(context.exception), context.exception)

    def test_read_numpy_column_names_mapping(self):

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = pa.float32())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')
            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = range(n_row_groups)
                                , column_names = {f'column_{i}':n_columns - i - 1 for i in range(n_columns)}
                                )
        
            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            reversed_expected_data = expected_data[:, ::-1]
            self.assertTrue(np.array_equal(np_array, reversed_expected_data), f"\n{np_array}\n\n{reversed_expected_data}")
            
            np_array = np.zeros((n_rows, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')
            for c in range(n_columns):
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_names = {f'column_{c}':n_columns - c - 1}
                                    )

            self.assertTrue(np.array_equal(np_array, reversed_expected_data), f"\n{np_array}\n\n{reversed_expected_data}")

    def test_read_numpy_column_indices_mapping(self):

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = pa.float32())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')
            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = range(n_row_groups)
                                , column_indices = {i:n_columns - i - 1 for i in range(n_columns)}
                                )

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            reversed_expected_data = expected_data[:, ::-1]
            self.assertTrue(np.array_equal(np_array, reversed_expected_data), f"\n{np_array}\n\n{reversed_expected_data}")
             
            np_array = np.zeros((n_rows, n_columns), dtype=pa.float32().to_pandas_dtype(), order='F')
            for c in range(n_columns):
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_indices = {c : n_columns - c - 1}
                                    )

            self.assertTrue(np.array_equal(np_array, reversed_expected_data), f"\n{np_array}\n\n{reversed_expected_data}")

if __name__ == '__main__':
    unittest.main()
