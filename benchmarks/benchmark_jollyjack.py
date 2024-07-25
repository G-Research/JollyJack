import jollyjack as jj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pyarrow.fs as fs
import concurrent.futures
import time
import os

row_groups = 1
n_columns = 10_000
chunk_size = 32_000
n_rows = row_groups * chunk_size
work_items = 32
batch_size = 40
n_reads = 600

all_columns = list(range(n_columns))
all_row_groups = list(range(row_groups))
columns_batches = [all_columns[i:i+batch_size] for i in range(0, len(all_columns), batch_size)]
row_groups_batches = [all_row_groups[i:i+batch_size] for i in range(0, len(all_row_groups), batch_size)]

parquet_path = "my.parquet"
index_path = parquet_path + '.index'
np_array = np.zeros((chunk_size, n_columns), dtype='f', order='F')

def get_table():
    # Generate a random 2D array of floats using NumPy
    # Each column in the array represents a column in the final table
    data = np.random.rand(n_rows, n_columns).astype(np.float32)

    # Convert the NumPy array to a list of PyArrow Arrays, one for each column
    pa_arrays = [pa.array(data[:, i]) for i in range(n_columns)]
    schema = pa.schema([(f'column_{i}', pa.float32()) for i in range(n_columns)])
    # Create a PyArrow Table from the Arrays
    return pa.Table.from_arrays(pa_arrays, schema=schema)

def worker_arrow_row_group():
    
    pr = pq.ParquetReader()
    pr.open(parquet_path)
        
    for r in range(0, int(row_groups)):
        table = pr.read_row_groups([r], use_threads=False)
        table = table
        
def worker_jollyjack_row_group():
        
    pr = pq.ParquetReader()
    pr.open(parquet_path)
    
    for r in range(0, int(row_groups)):
        jj.read_into_numpy_f32(metadata = pr.metadata, parquet_path = parquet_path, np_array = np_array, row_group_idx = r, column_indices = all_columns)

def genrate_data(table):

    t = time.time()
    print(f"writing parquet file, columns={n_columns}, row_groups={row_groups}, rows={n_rows}")
    pq.write_table(table, parquet_path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, compression=None, store_schema=False)
    dt = time.time() - t
    print(f"finished writing parquet file in {dt:.2f} seconds")

    t = time.time()
    print("Generating metadata index")
    pj.generate_metadata_index(parquet_path, index_path)
    dt = time.time() - t
    print(f"Metadata index generated in {dt:.2f} seconds")

    parquet_size = os.stat(parquet_path).st_size
    index_size = os.stat(index_path).st_size
    index_size_percentage = 100 * index_size / parquet_size
    print(f"Parquet size={parquet_size}, index size={index_size}({index_size_percentage:.2f}%)")
    print("")

def measure_reading(max_workers, worker):

    def dummy_worker():
        time.sleep(0.01)

    tt = []
    # measure multiple times and take the fastest run
    for _ in range(0, 5):
        # Create the pool and warm it up 
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        dummy_items = [pool.submit(dummy_worker) for i in range(0, len(all_columns), batch_size)]
        for dummy_item in dummy_items: 
            dummy_item.result()

        # warm up the OS cache
        worker()

        # Submit the work
        t = time.time()
        for i in range(0, work_items):
            pool.submit(worker)

        pool.shutdown(wait=True)
        tt.append(time.time() - t)

    return min (tt)

table = get_table()
genrate_data(table)
index_data = fs.LocalFileSystem().open_input_stream(index_path).readall()

print(f"Reading a single row group using arrow (single-threaded) {measure_reading(1, worker_arrow_row_group):.2f} seconds")
print(f"Reading a single row group using palletjack (single-threaded) {measure_reading(1, worker_jollyjack_row_group):.2f} seconds")
print(".")

print(f"Reading a single row group using arrow (multi-threaded) {measure_reading(8, worker_arrow_row_group):.2f} seconds")
print(f"Reading a single row group using palletjack (multi-threaded) {measure_reading(8, worker_jollyjack_row_group):.2f} seconds")
print(".")
