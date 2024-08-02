import jollyjack as jj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import concurrent.futures
import humanize
import random
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

def get_table():
    # Generate a random 2D array of floats using NumPy
    # Each column in the array represents a column in the final table
    data = np.random.rand(n_rows, n_columns).astype(np.float32)

    # Convert the NumPy array to a list of PyArrow Arrays, one for each column
    pa_arrays = [pa.array(data[:, i]) for i in range(n_columns)]
    schema = pa.schema([(f'column_{i}', pa.float32()) for i in range(n_columns)])
    # Create a PyArrow Table from the Arrays
    return pa.Table.from_arrays(pa_arrays, schema=schema)
        
def worker_arrow_row_group(use_threads, pre_buffer):

    for f in range(n_files):
        pr = pq.ParquetReader()
        pr.open(f"{parquet_path}{f}", pre_buffer=pre_buffer)

        column_indices_to_read = random.sample(range(0, n_columns), n_columns_to_read)
        table = pr.read_row_groups([row_groups-1], column_indices = column_indices_to_read, use_threads=use_threads)

def worker_jollyjack_row_group(pre_buffer):
        
    np_array = np.zeros((chunk_size, n_columns_to_read), dtype='f', order='F')
    
    for f in range(n_files):
        pr = pq.ParquetReader()
        pr.open(f"{parquet_path}{f}")
        
        column_indices_to_read = random.sample(range(0, n_columns), n_columns_to_read)
        jj.read_into_numpy_f32(metadata = pr.metadata, parquet_path = f"{parquet_path}{f}", np_array = np_array
                                , row_group_idx = row_groups-1, column_indices = column_indices_to_read, pre_buffer=pre_buffer)

def genrate_data(table, path, compression):

    t = time.time()
    print(f"writing parquet file:{path}, columns={n_columns}, row_groups={row_groups}, rows={n_rows}")
    
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
        for i in range(0, work_items):
            pool.submit(worker)

        pool.shutdown(wait=True)
        tt.append(time.time() - t)

    return min (tt)

for compression in [None, 'snappy']:
    for f in range(n_files):
        table = get_table()
        path = f"{parquet_path}{f}"
        genrate_data(table = table, path = path, compression = compression)

    for n_threads in [1, 2]:
        for pre_buffer in [False, True]:
            for use_threads in [False, True]:
                print(f"`ParquetReader.read_row_groups` n_threads:{n_threads}, use_threads:{use_threads}, pre_buffer:{pre_buffer}, duration:{measure_reading(n_threads, lambda:worker_arrow_row_group(use_threads=use_threads, pre_buffer = pre_buffer)):.2f} seconds")

    for n_threads in [1, 2]:
        for pre_buffer in [False, True]:
            print(f"`JollyJack.read_into_numpy_f32` n_threads:{n_threads}, pre_buffer:{pre_buffer}, duration:{measure_reading(n_threads, lambda:worker_jollyjack_row_group(pre_buffer)):.2f} seconds")
