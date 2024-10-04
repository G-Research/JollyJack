# JollyJack

## Features

- Reading parquet files directly into numpy arrays and torch tensors (fp16, fp32, fp64)
- Faster and requiring less memory than vanilla PyArrow
- Compatibility with [PalletJack](https://github.com/marcin-krystianc/PalletJack)

## Known limitations

- Data cannot contain null values

## Required

- pyarrow  ~= 17.0
 
JollyJack operates on top of pyarrow, making it an essential requirement for both building and using JollyJack. While our source package is compatible with recent versions of pyarrow, the binary distribution package specifically requires the latest major version of pyarrow.

##  Installation

```
pip install jollyjack
```

## How to use:

### Generating a sample parquet file:
```
import jollyjack as jj
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

from pyarrow import fs

chunk_size = 3
n_row_groups = 2
n_columns = 5
n_rows = n_row_groups * chunk_size
path = "my.parquet"

data = np.random.rand(n_rows, n_columns).astype(np.float32)
pa_arrays = [pa.array(data[:, i]) for i in range(n_columns)]
schema = pa.schema([(f'column_{i}', pa.float32()) for i in range(n_columns)])
table =  pa.Table.from_arrays(pa_arrays, schema=schema)
pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=True, store_schema=False, write_page_index=True)
```

### Generating a numpy array to read into:
```
# Create an array of zeros
np_array = np.zeros((n_rows, n_columns), dtype='f', order='F')
```

### Reading entire file into numpy array:
```
pr = pq.ParquetReader()
pr.open(path)

row_begin = 0
row_end = 0

for rg in range(pr.metadata.num_row_groups):
    row_begin = row_end
    row_end = row_begin + pr.metadata.row_group(rg).num_rows

    # To define which subset of the numpy array we want read into,
    # we need to create a view which shares underlying memory with the target numpy array
    subset_view = np_array[row_begin:row_end, :] 
    jj.read_into_numpy (source = path
                        , metadata = pr.metadata
                        , np_array = subset_view
                        , row_group_indices = [rg]
                        , column_indices = range(pr.metadata.num_columns))

# Alternatively
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy (source = f
                        , metadata = None
                        , np_array = np_array
                        , row_group_indices = range(pr.metadata.num_row_groups)
                        , column_indices = range(pr.metadata.num_columns))
```

### Reading columns in reversed order:
```
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy (source = f
                        , metadata = None
                        , np_array = np_array
                        , row_group_indices = range(pr.metadata.num_row_groups)
                        , column_indices = {i:pr.metadata.num_columns - i - 1 for i in range(pr.metadata.num_columns)})
```

### Reading column 3 into multiple destination columns
```
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy (source = f
                        , metadata = None
                        , np_array = np_array
                        , row_group_indices = range(pr.metadata.num_row_groups)
                        , column_indices = ((3, 0), (3, 1)))
```

### Generating a torch tensor to read into:
```
import torch
# Create a tesnsor and transpose it to get Fortran-style order
tensor = torch.zeros(n_columns, n_rows, dtype = torch.float32).transpose(0, 1)
```

### Reading entire file into the tensor:
```
pr = pq.ParquetReader()
pr.open(path)

jj.read_into_torch (source = path
                    , metadata = pr.metadata
                    , tensor = tensor
                    , row_group_indices = range(pr.metadata.num_row_groups)
                    , column_indices = range(pr.metadata.num_columns)
                    , pre_buffer = True
                    , use_threads = True)

print(tensor)
```

## Benchmarks:

| n_threads | use_threads | pre_buffer | dtype     | compression | PyArrow   | JollyJack |
|-----------|-------------|------------|-----------|-------------|-----------|-----------|
| 1         | False       | False      | float     | None        | **6.79s** | **3.55s** |
| 1         | True        | False      | float     | None        | **5.17s** | **2.32s** |
| 1         | False       | True       | float     | None        | **5.54s** | **2.76s** |
| 1         | True        | True       | float     | None        | **3.98s** | **2.66s** |
| 2         | False       | False      | float     | None        | **4.63s** | **2.33s** |
| 2         | True        | False      | float     | None        | **3.89s** | **2.36s** |
| 2         | False       | True       | float     | None        | **4.19s** | **2.61s** |
| 2         | True        | True       | float     | None        | **3.36s** | **2.39s** |
| 1         | False       | False      | float     | snappy      | **7.00s** | **3.56s** |
| 1         | True        | False      | float     | snappy      | **5.21s** | **2.23s** |
| 1         | False       | True       | float     | snappy      | **5.22s** | **3.30s** |
| 1         | True        | True       | float     | snappy      | **3.73s** | **2.84s** |
| 2         | False       | False      | float     | snappy      | **4.43s** | **2.49s** |
| 2         | True        | False      | float     | snappy      | **3.40s** | **2.42s** |
| 2         | False       | True       | float     | snappy      | **4.07s** | **2.63s** |
| 2         | True        | True       | float     | snappy      | **3.14s** | **2.55s** |
| 1         | False       | False      | halffloat | None        | **7.21s** | **1.23s** |
| 1         | True        | False      | halffloat | None        | **3.53s** | **0.71s** |
| 1         | False       | True       | halffloat | None        | **7.43s** | **1.96s** |
| 1         | True        | True       | halffloat | None        | **4.04s** | **1.52s** |
| 2         | False       | False      | halffloat | None        | **3.84s** | **0.64s** |
| 2         | True        | False      | halffloat | None        | **3.11s** | **0.57s** |
| 2         | False       | True       | halffloat | None        | **4.07s** | **1.17s** |
| 2         | True        | True       | halffloat | None        | **3.39s** | **1.14s** |
