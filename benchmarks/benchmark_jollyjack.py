import jollyjack as jj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import concurrent.futures
import humanize
import random
import torch
import time
import os

n_files = 10
row_groups = 1
n_columns = 7_000
n_columns_to_read = 1_000
chunk_size = 64_000
n_rows = row_groups * chunk_size

n_threads = 2
work_items = n_threads

parquet_path = "my.parquet"
jollyjack_numpy = None
arrow_numpy = None

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

def worker_arrow_row_group(use_threads, pre_buffer):

    for f in range(n_files):
        pr = pq.ParquetReader()
        pr.open(f"{parquet_path}{f}", pre_buffer=pre_buffer)

        column_indices_to_read = random.sample(range(0, n_columns), n_columns_to_read)
        table = pr.read_row_groups([row_groups-1], column_indices = column_indices_to_read, use_threads=use_threads)

def worker_jollyjack_numpy(pre_buffer, dtype):
        
    np_array = np.zeros((chunk_size, n_columns_to_read), dtype=dtype, order='F')
    
    for f in range(n_files):
        pr = pq.ParquetReader()
        pr.open(f"{parquet_path}{f}")
        
        column_indices_to_read = random.sample(range(0, n_columns), n_columns_to_read)
        jj.read_into_numpy(metadata = pr.metadata, parquet_path = f"{parquet_path}{f}", np_array = np_array
                                , row_group_indices = [row_groups-1], column_indices = column_indices_to_read, pre_buffer=pre_buffer)

def worker_jollyjack_torch(pre_buffer, dtype):

    tensor = torch.zeros(n_columns, n_rows, dtype = numpy_to_torch_dtype_dict[dtype.to_pandas_dtype()])
    tensor = tensor.transpose(0, 1)
                        
    for f in range(n_files):
        pr = pq.ParquetReader()
        pr.open(f"{parquet_path}{f}")
        
        column_indices_to_read = random.sample(range(0, n_columns), n_columns_to_read)
        jj.read_into_torch(metadata = pr.metadata, parquet_path = f"{parquet_path}{f}", tensor = tensor
                                , row_group_indices = [row_groups-1], column_indices = column_indices_to_read, pre_buffer=pre_buffer)

def genrate_data(n_rows, n_columns, path, compression, dtype):

    table = get_table(n_rows, n_columns, dtype)

    t = time.time()
    print(f"writing parquet file:{path}, columns={n_columns}, row_groups={row_groups}, rows={n_rows}, compression={compression}, dtype={dtype}")
    
    pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, compression=compression, store_schema=False)
    parquet_size = os.stat(path).st_size
    print(f"Parquet size={humanize.naturalsize(parquet_size)}")

    dt = time.time() - t
    print(f"finished writing parquet file in {dt:.2f} seconds")

def measure_reading(max_workers, worker):

    def dummy_worker():
        time.sleep(0.01)

    tt = []
    # measure multiple times and take the fastest run
    for _ in range(0, 3):
        # Create the pool and warm it up
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        dummy_items = [pool.submit(dummy_worker) for i in range(0, work_items)]
        for dummy_item in dummy_items: 
            dummy_item.result()

        # Submit the work
        t = time.time()
        work_results = [pool.submit(worker) for i in range(0, work_items)]
        for work_result in work_results:
            work_result.result()

        pool.shutdown(wait=True)
        tt.append(time.time() - t)

    return min (tt)

for compression, dtype in [(None, pa.float32()), ('snappy', pa.float32()), (None, pa.float16())]:
    
    print(f".")
    for f in range(n_files):
        path = f"{parquet_path}{f}"
        genrate_data(n_rows, n_columns, path = path, compression = compression, dtype = dtype)

    print(f".")
    for n_threads in [1, 2]:
        for pre_buffer in [False, True]:
            for use_threads in [False, True]:
                print(f"`ParquetReader.read_row_groups` n_threads:{n_threads}, use_threads:{use_threads}, pre_buffer:{pre_buffer}, dtype:{dtype}, compression={compression}, duration:{measure_reading(n_threads, lambda:worker_arrow_row_group(use_threads=use_threads, pre_buffer = pre_buffer)):.2f} seconds")

    print(f".")
    for n_threads in [1, 2]:
        for pre_buffer in [False, True]:
            print(f"`JollyJack.read_into_numpy` n_threads:{n_threads}, pre_buffer:{pre_buffer}, dtype:{dtype}, compression={compression}, duration:{measure_reading(n_threads, lambda:worker_jollyjack_numpy(pre_buffer, dtype.to_pandas_dtype())):.2f} seconds")

    print(f".")
    for n_threads in [1, 2]:
        for pre_buffer in [False, True]:
            print(f"`JollyJack.read_into_torch` n_threads:{n_threads}, pre_buffer:{pre_buffer}, dtype:{dtype}, compression={compression}, duration:{measure_reading(n_threads, lambda:worker_jollyjack_torch(pre_buffer, dtype.to_pandas_dtype())):.2f} seconds")
