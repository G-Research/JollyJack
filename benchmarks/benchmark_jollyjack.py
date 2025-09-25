import jollyjack as jj
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
import concurrent.futures
import humanize
import random
import time
import sys
import os

n_threads = 2
n_files = 2
n_repeats = 10
row_groups = 2
n_columns = 7_000
n_columns_to_read = 3_000
chunk_size = 32_000
parquet_path = "my.parquet" if sys.platform.startswith('win') else "/tmp/my.parquet"

def generate_random_parquet(
    filename: str,
    n_columns: int,
    n_row_groups: int,
    chunk_size: int,
    dtype = pa.float32(),
    compression = None,
):
    print(".")
    print(f"Generating a Parquet file: {filename}, cols: {n_columns}, row_groups:{n_row_groups}, chunk_size:{chunk_size}, compression={compression}, dtype={dtype}")

    writer = None
    schema = pa.schema([ pa.field(f"col_{i}", dtype) for i in range(n_columns) ] )
    try:
        for i in range(n_row_groups):
            print(f"  Generating row group {i+1}/{n_row_groups}...")
            data = {f"col_{j}": np.random.uniform(-100, 100, size=chunk_size) for j in range(n_columns) }

            df = pd.DataFrame(data)
            table = pa.Table.from_pandas(df, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(filename, schema, use_dictionary=False, compression=compression, write_statistics=False, store_schema=False )

            print("  writing...:")
            writer.write_table(table)

        print("Parquet file generated successfully!")
    finally:
        if writer:
            writer.close()

def genrate_data(n_columns, n_row_groups, path, compression, dtype):

    t = time.time()
    generate_random_parquet (filename = path, n_columns = n_columns, n_row_groups = n_row_groups, chunk_size = chunk_size, compression = compression, dtype = dtype)
    parquet_size = os.stat(path).st_size

    dt = time.time() - t
    print(f"finished writing parquet file in {dt:.2f} seconds, size={humanize.naturalsize(parquet_size)}")

def worker_arrow_row_group(use_threads, pre_buffer, path):

    pr = pq.ParquetReader()
    pr.open(path, pre_buffer = pre_buffer)

    row_groups_to_read = random.sample(range(row_groups), 1)
    column_indices_to_read = random.sample(range(n_columns), n_columns_to_read)
    table = pr.read_row_groups(row_groups = row_groups_to_read, column_indices = column_indices_to_read, use_threads=use_threads)

def worker_jollyjack_numpy(use_threads, pre_buffer, dtype, path):
        
    np_array = np.zeros((chunk_size, n_columns_to_read), dtype=dtype, order='F')

    pr = pq.ParquetReader()
    pr.open(path)

    row_groups_to_read = random.sample(range(row_groups), 1)
    column_indices_to_read = random.sample(range(n_columns), n_columns_to_read)
    jj.read_into_numpy(source = path
                        , metadata = pr.metadata
                        , np_array = np_array
                        , row_group_indices = row_groups_to_read
                        , column_indices = column_indices_to_read
                        , pre_buffer = pre_buffer
                        , use_threads = use_threads)

def worker_jollyjack_copy_to_row_major(dtype, path):

    np_array = np.zeros((chunk_size, n_columns_to_read), dtype=dtype, order='F')
    dst_array = np.zeros((chunk_size, n_columns_to_read), dtype=dtype, order='C')
    row_indicies = list(range(chunk_size))
    random.shuffle(row_indicies)

    pr = pq.ParquetReader()
    pr.open(path)
    
    row_groups_to_read = random.sample(range(row_groups), 1)
    column_indices_to_read = random.sample(range(n_columns), n_columns_to_read)
    jj.read_into_numpy(source = path
                        , metadata = pr.metadata
                        , np_array = np_array
                        , row_group_indices = row_groups_to_read
                        , column_indices = column_indices_to_read
                        , pre_buffer = True
                        , use_threads = False)

    jj.copy_to_numpy_row_major(np_array, dst_array, row_indicies)

def worker_numpy_copy_to_row_major(dtype, path):

    np_array = np.zeros((chunk_size, n_columns_to_read), dtype=dtype, order='F')
    dst_array = np.zeros((chunk_size, n_columns_to_read), dtype=dtype, order='C')

    pr = pq.ParquetReader()
    pr.open(path)

    row_groups_to_read = random.sample(range(row_groups), 1)
    column_indices_to_read = random.sample(range(n_columns), n_columns_to_read)
    jj.read_into_numpy(source = path
                        , metadata = pr.metadata
                        , np_array = np_array
                        , row_group_indices = row_groups_to_read
                        , column_indices = column_indices_to_read
                        , pre_buffer = True
                        , use_threads = False)

    np.copyto(dst_array, np_array)        

def worker_jollyjack_torch(pre_buffer, dtype, path):

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

    tensor = torch.zeros(n_columns_to_read, chunk_size, dtype = numpy_to_torch_dtype_dict[dtype]).transpose(0, 1)

    pr = pq.ParquetReader()
    pr.open(path)    

    row_groups_to_read = random.sample(range(row_groups), 1)
    column_indices_to_read = random.sample(range(n_columns), n_columns_to_read)
    jj.read_into_torch(source = path
                        , metadata = pr.metadata
                        , tensor = tensor
                        , row_group_indices = row_groups_to_read
                        , column_indices = column_indices_to_read
                        , pre_buffer = pre_buffer
                        , use_threads = False)

def measure_reading(max_workers, worker):

    def dummy_worker():
        time.sleep(0.01)

    tt = []
    # measure multiple times and take the fastest run
    for _ in range(5):
        # Create the pool and warm it up
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        dummy_items = [pool.submit(dummy_worker) for i in range(0, n_threads)]
        for dummy_item in dummy_items: 
            dummy_item.result()

        # Submit the work
        t = time.time()
        work_results = []
        for _ in range(n_repeats):
            work_results.extend([pool.submit(worker, path = f"{parquet_path}{i}") for i in range(0, n_files)])

        for work_result in work_results:
            work_result.result()

        pool.shutdown(wait=True)
        tt.append(time.time() - t)

    return min (tt)

for compression, dtype in [(None, pa.float32()), ('snappy', pa.float32()), (None, pa.float16())]:
    
    print(f".")
    for f in range(n_files):
        path = f"{parquet_path}{f}"
        genrate_data(path = path, n_row_groups = row_groups, n_columns = n_columns, compression = compression, dtype = dtype)

    print(f".")
    for n_threads in [1, n_threads]:
        for pre_buffer in [False, True]:
            for use_threads in [False, True]:
                print(f"`ParquetReader.read_row_groups` n_threads:{n_threads}, use_threads:{use_threads}, pre_buffer:{pre_buffer}, dtype:{dtype}, compression={compression}, duration:{measure_reading(n_threads, lambda path:worker_arrow_row_group(use_threads = use_threads, pre_buffer = pre_buffer, path = path)):.2f} seconds")

    print(f".")
    for n_threads in [1, n_threads]:
        for pre_buffer in [False, True]:
            for use_threads in [False, True]:
                print(f"`JollyJack.read_into_numpy` n_threads:{n_threads}, use_threads:{use_threads}, pre_buffer:{pre_buffer}, dtype:{dtype}, compression={compression}, duration:{measure_reading(n_threads, lambda path:worker_jollyjack_numpy(use_threads, pre_buffer, dtype.to_pandas_dtype(), path = path)):.2f} seconds")

    print(f".")
    for n_threads in [1, n_threads]:
        for pre_buffer in [False, True]:
            print(f"`JollyJack.read_into_torch` n_threads:{n_threads}, pre_buffer:{pre_buffer}, dtype:{dtype}, compression={compression}, duration:{measure_reading(n_threads, lambda path:worker_jollyjack_torch(pre_buffer, dtype.to_pandas_dtype(), path = path)):.2f} seconds")

    print(f".")
    for jj_variant in [1, 2]:
        os.environ["JJ_copy_to_row_major"] = str(jj_variant)
        for n_threads in [1, n_threads]:
            print(f"`JollyJack.copy_to_row_major` n_threads:{n_threads}, dtype:{dtype}, compression={compression}, jj_variant={jj_variant} duration:{measure_reading(n_threads, lambda path:worker_jollyjack_copy_to_row_major(dtype.to_pandas_dtype(), path = path)):.2f} seconds")

    print(f".")
    for n_threads in [1, n_threads]:
        print(f"`numpy.copy_to_row_major` n_threads:{n_threads}, dtype:{dtype}, compression={compression}, duration:{measure_reading(n_threads, lambda path:worker_numpy_copy_to_row_major(dtype.to_pandas_dtype(), path)):.2f} seconds")
