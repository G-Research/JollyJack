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
from parameterized import parameterized


chunk_size = 3
n_row_groups = 2
n_columns = 5
n_rows = n_row_groups * chunk_size
current_dir = os.path.dirname(os.path.realpath(__file__))
supported_encodings = ['PLAIN', 'BYTE_STREAM_SPLIT']

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

class TestJollyJack(unittest.TestCase):

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True]))
    def test_read_entire_table(self, pre_buffer, use_threads, use_memory_map):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows, n_columns)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

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
                                    , column_indices = range(pr.metadata.num_columns)
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            expected_data = pr.read_all()
            self.assertTrue(np.array_equal(np_array1, expected_data))

            np_array2 = np.zeros((n_rows, n_columns), dtype='f', order='F')
            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array2
                                , row_group_indices = range(pr.metadata.num_row_groups)
                                , column_indices = range(pr.metadata.num_columns)
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map)

            self.assertTrue(np.array_equal(np_array2, expected_data))
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True]))
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
                                    , use_memory_map = use_memory_map)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all()
            self.assertTrue(np.array_equal(np_array, expected_data))
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True]))
    def test_read_nonzero_column_offset(self, pre_buffer, use_threads, use_memory_map):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = chunk_size, n_columns = n_columns)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            # Create an array of zeros
            cols = 2
            offset = n_columns - cols
            np_array = np.zeros((chunk_size, cols), dtype='f', order='F')

            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = [0]
                                , column_indices = range(offset, offset + cols)
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all(column_indices = range(offset, offset + cols))
            self.assertTrue(np.array_equal(np_array, expected_data))
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True]))
    def test_read_unsupported_column_types(self, pre_buffer, use_threads, use_memory_map):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = chunk_size, n_columns = n_columns, data_type = pa.bool_())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            # Create an array of zerosx
            np_array = np.zeros((chunk_size, n_columns), dtype='f', order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(n_columns)
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            self.assertTrue(f"Column[0] ('column_0') has unsupported data type: 0!" in str(context.exception), context.exception)

    @parameterized.expand(itertools.product([pa.float16(), pa.float32(), pa.float64()]))
    def test_read_unsupported_encoding_dictionary(self, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = chunk_size, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=True)

            # Create an array of zerosx
            np_array = np.zeros((chunk_size, n_columns), dtype=dtype.to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(n_columns))

            self.assertTrue(f"Cannot read column=0 due to unsupported_encoding=RLE_DICTIONARY!" in str(context.exception), context.exception)

    def test_read_unsupported_encoding_delta_byte_array(self):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = chunk_size, n_columns = n_columns, data_type = pa.float16())
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, column_encoding='DELTA_BYTE_ARRAY')

            # Create an array of zerosx
            np_array = np.zeros((chunk_size, n_columns), dtype=pa.float16().to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(n_columns))

            self.assertTrue(f"Cannot read column=0 due to unsupported_encoding=DELTA_BYTE_ARRAY!" in str(context.exception), context.exception)

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()], supported_encodings))
    def test_read_dtype_numpy(self, pre_buffer, use_threads, use_memory_map, dtype, encoding):

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

            with self.subTest((n_row_groups, n_columns, chunk_size, dtype, pre_buffer, use_threads, encoding)):
                n_rows = n_row_groups * chunk_size
                with tempfile.TemporaryDirectory() as tmpdirname:
                    path = os.path.join(tmpdirname, "my.parquet")
                    table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
                    pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False, column_encoding = encoding)

                    # Create an empty array
                    np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')

                    jj.read_into_numpy (source = path
                                        , metadata = None
                                        , np_array = np_array
                                        , row_group_indices = range(n_row_groups)
                                        , column_indices = range(n_columns)
                                        , pre_buffer = pre_buffer
                                        , use_threads = use_threads
                                        , use_memory_map = use_memory_map)

                    pr = pq.ParquetReader()
                    pr.open(path)
                    expected_data = pr.read_all().to_pandas().to_numpy()
                    self.assertTrue(np.array_equal(np_array, expected_data), f"{np_array}\n{expected_data}")
                    pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()], supported_encodings))
    def test_read_dtype_torch(self, pre_buffer, use_threads, use_memory_map, dtype, encoding):

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

            with self.subTest((n_row_groups, n_columns, chunk_size, dtype, encoding)):
                n_rows = n_row_groups * chunk_size

                with tempfile.TemporaryDirectory() as tmpdirname:
                    path = os.path.join(tmpdirname, "my.parquet")
                    table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
                    pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False, column_encoding = encoding)

                    tensor = torch.zeros(n_columns, n_rows, dtype = numpy_to_torch_dtype_dict[dtype.to_pandas_dtype()]).transpose(0, 1)

                    jj.read_into_torch (source = path
                                        , metadata = None
                                        , tensor = tensor
                                        , row_group_indices = range(n_row_groups)
                                        , column_indices = range(n_columns)
                                        , pre_buffer = pre_buffer
                                        , use_threads = use_threads
                                        , use_memory_map = use_memory_map)

                    pr = pq.ParquetReader()
                    pr.open(path)
                    expected_data = pr.read_all().to_pandas().to_numpy()
                    self.assertTrue(np.array_equal(tensor.numpy(), expected_data), f"{tensor.numpy()}\n{expected_data}")
                    pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_numpy_column_names(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')

            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = range(n_row_groups)
                                , column_names = [f'column_{i}' for i in range(n_columns)]
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            self.assertTrue(np.array_equal(np_array, expected_data), f"{np_array}\n{expected_data}")
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_torch_column_names(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            # Create an empty array
            tensor = torch.zeros(n_columns, n_rows, dtype = numpy_to_torch_dtype_dict[dtype.to_pandas_dtype()]).transpose(0, 1)

            jj.read_into_torch (source = path
                                , metadata = None
                                , tensor = tensor
                                , row_group_indices = range(n_row_groups)
                                , column_names = [f'column_{i}' for i in range(n_columns)]
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            self.assertTrue(np.array_equal(tensor.numpy(), expected_data), f"{tensor.numpy()}\n{expected_data}")
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_invalid_column(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_names = [f'foo_bar_{i}' for i in range(n_columns)]
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            self.assertTrue(f"Column 'foo_bar_0' was not found!" in str(context.exception), context.exception)
                
            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_indices = [i + 1 for i in range(n_columns)]
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            if pre_buffer:
                self.assertTrue(f"The file only has {n_columns} columns, requested metadata for column: {n_columns}" in str(context.exception), context.exception)
            else:
                self.assertTrue(f"Trying to read column index {n_columns} but row group metadata has only {n_columns} columns" in str(context.exception), context.exception)

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_filesystem(self, pre_buffer, use_threads, use_memory_map, dtype):
                
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype = dtype.to_pandas_dtype(), order='F')

            with fs.LocalFileSystem().open_input_file(path) as f:
                jj.read_into_numpy (source = f
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_names = [f'column_{i}' for i in range(n_columns)]
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            self.assertTrue(np.array_equal(np_array, expected_data), f"{np_array}\n{expected_data}")
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_invalid_row_group(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [n_row_groups]
                                    , column_names = [f'column_{i}' for i in range(n_columns)]
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            if pre_buffer:
                self.assertTrue(f"The file only has {n_row_groups} row groups, requested metadata for row group: {n_row_groups}" in str(context.exception), context.exception)
            else:
                self.assertTrue(f"Trying to read row group {n_row_groups} but file only has {n_row_groups} row groups" in str(context.exception), context.exception)

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()], supported_encodings))
    def test_read_data_with_nulls(self, pre_buffer, use_threads, use_memory_map, dtype, encoding):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            df = table.to_pandas()
            df.iloc[0, 0] = np.nan
            table = pa.Table.from_pandas(df)

            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False, column_encoding=encoding)
            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype = dtype.to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_names = [f'column_{i}' for i in range(n_columns)]
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            self.assertTrue(f"Unexpected end of stream" in str(context.exception), context.exception)

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_not_enough_rows(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            # Create an empty array
            np_array = np.zeros((n_rows + 1, n_columns), dtype=dtype.to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_names = [f'column_{i}' for i in range(n_columns)]
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            self.assertTrue(f"Expected to read {n_rows + 1} rows, but read only {n_rows}!" in str(context.exception), context.exception)

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_numpy_column_names_mapping(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = range(n_row_groups)
                                , column_names = {f'column_{i}':n_columns - i - 1 for i in range(n_columns)}
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map)
        
            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            reversed_expected_data = expected_data[:, ::-1]
            self.assertTrue(np.array_equal(np_array, reversed_expected_data), f"\n{np_array}\n\n{reversed_expected_data}")
            
            np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
            for c in range(n_columns):
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_names = {f'column_{c}':n_columns - c - 1}
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            self.assertTrue(np.array_equal(np_array, reversed_expected_data), f"\n{np_array}\n\n{reversed_expected_data}")
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_numpy_column_indices_mapping(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)
  
            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = range(n_row_groups)
                                , column_indices = {i:n_columns - i - 1 for i in range(n_columns)}
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map)

            pr = pq.ParquetReader()
            pr.open(path)
            expected_data = pr.read_all().to_pandas().to_numpy()
            reversed_expected_data = expected_data[:, ::-1]
            self.assertTrue(np.array_equal(np_array, reversed_expected_data), f"\n{np_array}\n\n{reversed_expected_data}")
            
            np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
            for c in range(n_columns):
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_indices = {c : n_columns - c - 1}
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            self.assertTrue(np.array_equal(np_array, reversed_expected_data), f"\n{np_array}\n\n{reversed_expected_data}")
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_numpy_column_indices_multi_mapping(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            pr = pq.ParquetReader()
            pr.open(path)

            column_data = pr.read_all(column_indices=[3]).to_pandas().to_numpy()
            expected_data = np.repeat(column_data, 3, axis=1)

            # Create an empty array
            np_array = np.zeros((n_rows, 3), dtype=dtype.to_pandas_dtype(), order='F')
            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = range(n_row_groups)
                                , column_indices = ((3, 0), (3, 1))
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map)

            jj.read_into_numpy (source = path
                    , metadata = None
                    , np_array = np_array
                    , row_group_indices = range(n_row_groups)
                    , column_indices = [[3, 2]]
                    , pre_buffer = pre_buffer
                    , use_threads = use_threads
                    , use_memory_map = use_memory_map)

            self.assertTrue(np.array_equal(np_array, expected_data), f"\n{np_array}\n\n{expected_data}")
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_numpy_column_names_multi_mapping(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            pr = pq.ParquetReader()
            pr.open(path)

            column_data = pr.read_all(column_indices=[3]).to_pandas().to_numpy()
            expected_data = np.repeat(column_data, 3, axis=1)

            # Create an empty array
            np_array = np.zeros((n_rows, 3), dtype=dtype.to_pandas_dtype(), order='F')
            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = range(n_row_groups)
                                , column_names = (('column_3', 0), ('column_3', 1))
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map)

            jj.read_into_numpy (source = path
                    , metadata = None
                    , np_array = np_array
                    , row_group_indices = range(n_row_groups)
                    , column_names = [['column_3', 2]]
                    , pre_buffer = pre_buffer
                    , use_threads = use_threads
                    , use_memory_map = use_memory_map)

            self.assertTrue(np.array_equal(np_array, expected_data), f"\n{np_array}\n\n{expected_data}")
            pr.close()

    # over 4GB
    @parameterized.expand([(200_000_000, pa.float16()), (1_100_000_000, pa.float32()), (550_000_000, pa.float64())])
    def test_read_large_array(self, chunk_size, dtype):

        n_row_groups = 1
        n_columns = 1
        n_rows = n_row_groups * chunk_size

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")

            # Create an array of consecutive float32 numbers with 1 million rows
            data = np.arange(n_rows, dtype=dtype.to_pandas_dtype())

            # Create a PyArrow table with a single column
            table = pa.table([data], names=['c0'])
            data = None

            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)
            table = None

            pr = pq.ParquetReader()
            pr.open(path)

            # Create an empty array
            np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')

            jj.read_into_numpy (source = path
                                , metadata = pr.metadata
                                , np_array = np_array
                                , row_group_indices = range(pr.metadata.num_row_groups)
                                , column_indices = range(n_columns)
                                )

            self.assertTrue(np.min(np_array) == 0)
            self.assertTrue(np.max(np_array) == n_rows-1)
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_not_enough_buffer(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows = n_rows, n_columns = n_columns, data_type = dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)

            # Create an empty array
            np_array = np.zeros((n_rows - 1, n_columns), dtype=dtype.to_pandas_dtype(), order='F')

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_indices = {0:n_columns}
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            self.assertTrue(f"Attempted to read {chunk_size} rows into location [0, {n_columns}], but that is beyond target's boundaries." in str(context.exception), context.exception)

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(n_row_groups)
                                    , column_indices = range(n_columns)
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map)

            self.assertTrue(f"Buffer overrun error: Attempted to read {chunk_size} rows into location [{chunk_size}, {n_columns - 1}], but there was space available for only {chunk_size - 1} rows." in str(context.exception), context.exception)

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_entire_table_with_slices(self, pre_buffer, use_threads, use_memory_map, dtype):

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

            # Create an array of zeros
            expected_data = pr.read_all()
            np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = range(pr.metadata.num_row_groups)
                                , column_indices = range(pr.metadata.num_columns)
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map
                                , row_ranges = [slice (x, x + 1) for x in range(n_rows)])

            self.assertTrue(np.array_equal(np_array, expected_data))

            pr.close()
 
    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_partial_table_with_slices(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows, n_columns, data_type=dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)
            pr = pq.ParquetReader()
            pr.open(path)

            # Create an array of zeros
            expected_data = pr.read_all()
            expected_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
            expected_array[:chunk_size] = expected_data[:chunk_size]
            row_ranges = [slice (0, chunk_size), ]
            np_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
            jj.read_into_numpy (source = path
                                , metadata = None
                                , np_array = np_array
                                , row_group_indices = [0]
                                , column_indices = range(pr.metadata.num_columns)
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map
                                , row_ranges = row_ranges)

            self.assertTrue(np.array_equal(np_array, expected_array))
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_entire_tensor_with_slices(self, pre_buffer, use_threads, use_memory_map, dtype):

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
            tensor = torch.zeros(n_columns, n_rows, dtype = numpy_to_torch_dtype_dict[dtype.to_pandas_dtype()]).transpose(0, 1)
            
            jj.read_into_torch (source = path
                                , metadata = None
                                , tensor = tensor
                                , row_group_indices = range(pr.metadata.num_row_groups)
                                , column_indices = range(pr.metadata.num_columns)
                                , pre_buffer = pre_buffer
                                , use_threads = use_threads
                                , use_memory_map = use_memory_map
                                , row_ranges = row_ranges)

            self.assertTrue(np.array_equal(tensor.numpy(), expected_array))
            pr.close()

    @parameterized.expand(itertools.product([False, True], [False, True], [False, True], [pa.float16(), pa.float32(), pa.float64()]))
    def test_read_with_slices_error_handling(self, pre_buffer, use_threads, use_memory_map, dtype):

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "my.parquet")
            table = get_table(n_rows, n_columns, data_type=dtype)
            pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)
            pr = pq.ParquetReader()
            pr.open(path)

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
                                    , row_ranges = [slice (0, 2 * chunk_size), ])
            self.assertTrue(f"Requested to read {2 * chunk_size} rows, but the current row group has only {chunk_size} rows" in str(context.exception), context.exception)

            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = range(pr.metadata.num_row_groups)
                                    , column_indices = range(pr.metadata.num_columns)
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map
                                    , row_ranges = [slice (0, chunk_size), slice (chunk_size, 2 * chunk_size - 1)])
            self.assertTrue(f"Requested to read {chunk_size - 1} rows, but the current row group has {chunk_size} rows" in str(context.exception), context.exception)

            with self.assertRaises(ValueError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(pr.metadata.num_columns)
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map
                                    , row_ranges = [slice (0, chunk_size, 2)])
            self.assertTrue(f"Row range 'slice(0, {chunk_size}, 2)' is not contiguous" in str(context.exception), context.exception)

            with self.assertRaises(ValueError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(pr.metadata.num_columns)
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map
                                    , row_ranges = [slice(None, chunk_size)])
            self.assertTrue(f"Row range 'slice(None, {chunk_size}, None)' is not a valid range" in str(context.exception), context.exception)

            with self.assertRaises(ValueError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(pr.metadata.num_columns)
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map
                                    , row_ranges = [slice(chunk_size, None)])
            self.assertTrue(f"Row range 'slice({chunk_size}, None, None)' is not a valid range" in str(context.exception), context.exception)

            with self.assertRaises(ValueError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(pr.metadata.num_columns)
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map
                                    , row_ranges = [slice(chunk_size, chunk_size - 1)])
            self.assertTrue(f"Row range 'slice({chunk_size}, {chunk_size - 1}, None)' is not a valid range" in str(context.exception), context.exception)
            
            with self.assertRaises(RuntimeError) as context:
                jj.read_into_numpy (source = path
                                    , metadata = None
                                    , np_array = np_array
                                    , row_group_indices = [0]
                                    , column_indices = range(pr.metadata.num_columns)
                                    , pre_buffer = pre_buffer
                                    , use_threads = use_threads
                                    , use_memory_map = use_memory_map
                                    , row_ranges = [slice(0, chunk_size), slice(0, chunk_size)])
            self.assertTrue(f"Expected to read 2 row ranges, but read only 1!" in str(context.exception), context.exception)

            pr.close()

    @parameterized.expand(itertools.product([pa.float16(), pa.float32(), pa.float64()], 
                                            [(1, 1), (5,6),(8, 8), (16, 16), 
                                             (32, 32), (32, 33), (33, 32), (64, 64), 
                                             (65, 64), (64, 65), (100, 200), (1000, 2000)]))
    def test_copy_to_numpy_row_major(self, dtype, n_rows_n_columns):

        n_rows = n_rows_n_columns[0]
        n_columns = n_rows_n_columns[1]
        
        src_array = get_table(n_rows, n_columns, data_type = dtype).to_pandas().to_numpy()
        dst_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='C')
        expected_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='C')
        np.copyto(expected_array, src_array)                   
        jj.copy_to_numpy_row_major(src_array = src_array, dst_array = dst_array, row_indices = range(n_rows))
        self.assertTrue(np.array_equal(expected_array, dst_array), f"{expected_array}\n!=\n{dst_array}")

        # Reversed rows
        jj.copy_to_numpy_row_major(src_array = src_array, dst_array = dst_array, row_indices = [n_rows - i - 1 for i in range(n_rows)])
        expected_array = expected_array[::-1, :]
        self.assertTrue(np.array_equal(expected_array, dst_array), f"{expected_array}\n!=\n{dst_array}")

        # Subsets
        src_view = src_array[1:(n_rows - 1), 1:(n_columns - 1)] 
        dst_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='C')
        dst_view = dst_array[1:(n_rows - 1), 1:(n_columns - 1)] 
        expected_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='C')
        np.copyto(expected_array, src_array)
        expected_array = expected_array[1:(n_rows - 1), 1:(n_columns - 1)]    
        jj.copy_to_numpy_row_major(src_array = src_view, dst_array = dst_view, row_indices = range(n_rows - 2))
        self.assertTrue(np.array_equal(expected_array, dst_view), f"{expected_array}\n!=\n{dst_view}")

        # Scattered rows
        dst_array = np.zeros((2 * n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='C')
        expected_array = np.zeros((2 * n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='C')
        expected_array[::2] = src_array # Copy to even indices (0, 2, ...)
        jj.copy_to_numpy_row_major(src_array = src_array, dst_array = dst_array, row_indices = [i * 2 for i in range(n_rows)])
        self.assertTrue(np.array_equal(expected_array, dst_array), f"{expected_array}\n!=\n{dst_array}")

    @parameterized.expand(itertools.product([pa.float16(), pa.float32(), pa.float64()], [1, 2], [1, 2]))
    def test_copy_to_torch_row_major(self, dtype, n_rows, n_columns):

        src_tensor = torch.tensor(get_table(n_rows, n_columns, data_type = dtype).to_pandas().to_numpy())
        dst_tensor = torch.zeros(n_rows, n_columns, dtype = numpy_to_torch_dtype_dict[dtype.to_pandas_dtype()])
        expected_tensor = src_tensor.clone().contiguous()   
        jj.copy_to_torch_row_major(src_tensor = src_tensor, dst_tensor = dst_tensor, row_indices = range(n_rows))
        self.assertTrue(torch.equal(expected_tensor, dst_tensor), f"{expected_tensor}\n!=\n{dst_tensor}")

        # Reversed rows
        jj.copy_to_torch_row_major(src_tensor = src_tensor, dst_tensor = dst_tensor, row_indices = [n_rows - i - 1 for i in range(n_rows)])
        expected_tensor = torch.flipud(expected_tensor)
        self.assertTrue(torch.equal(expected_tensor, dst_tensor), f"{expected_tensor}\n!=\n{dst_tensor}")

    @parameterized.expand(itertools.product([pa.float16(), pa.float32(), pa.float64()], [5, 50], [6, 60]))
    def test_copy_to_row_major_arg_validation(self, dtype, n_rows, n_columns):

        src_array = get_table(n_rows, n_columns, data_type = dtype).to_pandas().to_numpy()

        with self.assertRaises(AssertionError) as context:
            dst_array = np.zeros((n_rows, n_columns + 1), dtype=dtype.to_pandas_dtype(), order='C')
            jj.copy_to_numpy_row_major(src_array = src_array, dst_array = dst_array, row_indices = range(n_rows))
        self.assertTrue(f"src_array.shape[1] != dst_array.shape[1], {n_columns} != {n_columns + 1}" in str(context.exception), context.exception)

        with self.assertRaises(AssertionError) as context:
            dst_array = np.zeros((n_rows - 1, n_columns), dtype=dtype.to_pandas_dtype(), order='C')
            jj.copy_to_numpy_row_major(src_array = src_array, dst_array = dst_array, row_indices = range(n_rows))
        self.assertTrue(f"src_array.shape[0] > dst_array.shape[0], {n_rows} > {n_rows - 1}" in str(context.exception), context.exception)

        with self.assertRaises(AssertionError) as context:
            dst_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='F')
            jj.copy_to_numpy_row_major(src_array = src_array, dst_array = dst_array, row_indices = range(n_rows))
        self.assertTrue(f"Expected destination array in a C (row-major) order" in str(context.exception), context.exception)

        with self.assertRaises(AssertionError) as context:
            dst_array = np.zeros((n_rows, n_columns), dtype=np.uint8, order='C')
            jj.copy_to_numpy_row_major(src_array = src_array, dst_array = dst_array, row_indices = range(n_rows))
        self.assertTrue(f"Source and destination arrays have diffrent datatypes, {src_array.dtype} != uint8" in str(context.exception), context.exception)

        with self.assertRaises(AssertionError) as context:
            dst_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='C')
            jj.copy_to_numpy_row_major(src_array = src_array, dst_array = dst_array, row_indices = range(n_rows - 1))
        self.assertTrue(f"Unexpected len of row indices, {n_rows - 1} != {n_rows}" in str(context.exception), context.exception)

        with self.assertRaises(AssertionError) as context:
            dst_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='C')
            jj.copy_to_numpy_row_major(src_array = src_array, dst_array = dst_array, row_indices = [i - 1 for i in range(n_rows)])
        self.assertTrue(f"Row index = -1 is not in the expected range [0, {n_rows})" in str(context.exception), context.exception)

        with self.assertRaises(AssertionError) as context:
            dst_array = np.zeros((n_rows, n_columns), dtype=dtype.to_pandas_dtype(), order='C')
            jj.copy_to_numpy_row_major(src_array = src_array, dst_array = dst_array, row_indices = [i + 1 for i in range(n_rows)])
        self.assertTrue(f"Row index = {n_rows} is not in the expected range [0, {n_rows})" in str(context.exception), context.exception)

if __name__ == '__main__':
    unittest.main()
    #unittest.main(argv=['first-arg-is-ignored', '-k', 'TestJollyJack.test_read_unsupported_encoding_delta_byte_array'])
