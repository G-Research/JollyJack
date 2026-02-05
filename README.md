# JollyJack

JollyJack is a high-performance Parquet reader designed to load data directly
into NumPy arrays and PyTorch tensors with minimal overhead.

## Features

- Load Parquet straight into NumPy arrays or PyTorch tensors (fp16, fp32, fp64, int32, int64)
- Up to 6× faster and with lower memory use than vanilla PyArrow
- Compatibility with [PalletJack](https://github.com/marcin-krystianc/PalletJack)
- Optional io_uring + O_DIRECT backend for I/O-bound workloads

## Known limitations

- Data must not contain null values
- Destination NumPy arrays and PyTorch tensors must be column-major (Fortran-style) 

## Selecting a reader backend

By default, the reader uses the regular file API via
`parquet::ParquetFileReader`. In most cases, this is the recommended choice.

An alternative reader backend based on **io_uring** is also available. It can
provide better performance, especially for very large datasets and when used
together with **O_DIRECT**.

To enable the alternative backend, set the `JJ_READER_BACKEND` environment
variable to one of the following values:

- `io_uring` - Uses io_uring for async I/O with the page cache
- `io_uring_odirect` - Uses io_uring with O_DIRECT (bypasses the page cache)

## Performance tuning tips

JollyJack performance is primarily determined by I/O, threading,
and memory allocation behavior. The optimal configuration depends on whether
your workload is I/O-bound or memory-/CPU-bound.

### Threading strategy

- JollyJack can be safely called concurrently from multiple threads.
- Parallel reads usually improve throughput, but oversubscribing threads can cause contention and degrade performance.

### Reuse destination arrays

- Reusing NumPy arrays or PyTorch tensors avoids repeated memory allocation.
- While allocation itself is fast, it can trigger kernel contention and degrade performance.

### Large datasets (exceed filesystem cache)

For datasets larger than the available page cache, performance is typically I/O-bound.

Recommended configuration:

- `use_threads = True`, `pre_buffer = True`, `JJ_READER_BACKEND = io_uring_odirect`

This combination bypasses the page cache, reduces double buffering and allows deeper I/O queues via io_uring

### Small datasets (fit in filesystem cache)

For datasets that comfortably fit in RAM, performance is typically CPU- or memory-bound.

Recommended configuration:
- `use_threads = False`, `pre_buffer = False` and use the default reader backend (no io_uring)

## Requirements

- pyarrow  ~= 22.0.0
 
JollyJack builds on top of PyArrow. While the source package may work with
newer versions, the prebuilt binary wheels are built and tested against pyarrow 22.x.

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
schema = pa.schema([(f"column_{i}", pa.float32()) for i in range(n_columns)])
table = pa.Table.from_arrays(pa_arrays, schema=schema)
pq.write_table(
    table,
    path,
    row_group_size=chunk_size,
    use_dictionary=False,
    write_statistics=True,
    store_schema=False,
    write_page_index=True,
)
```

### Generating a NumPy array to read into:
```
# Create an array of zeros
np_array = np.zeros((n_rows, n_columns), dtype="f", order="F")
```

### Reading entire file into NumPy array:
```
pr = pq.ParquetReader()
pr.open(path)

row_begin = 0
row_end = 0

for rg in range(pr.metadata.num_row_groups):
    row_begin = row_end
    row_end = row_begin + pr.metadata.row_group(rg).num_rows

    # To define which subset of the NumPy array we want read into,
    # we need to create a view which shares underlying memory with the target NumPy array
    subset_view = np_array[row_begin:row_end, :]
    jj.read_into_numpy(
        source=path,
        metadata=pr.metadata,
        np_array=subset_view,
        row_group_indices=[rg],
        column_indices=range(pr.metadata.num_columns),
    )

# Alternatively
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy(
        source=f,
        metadata=None,
        np_array=np_array,
        row_group_indices=range(pr.metadata.num_row_groups),
        column_indices=range(pr.metadata.num_columns),
    )
```
### Reading columns in reverse order:
```
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy(
        source=f,
        metadata=None,
        np_array=np_array,
        row_group_indices=range(pr.metadata.num_row_groups),
        column_indices={
            i: pr.metadata.num_columns - i - 1 for i in range(pr.metadata.num_columns)
        },
    )
```

### Reading column 3 into multiple destination columns
```
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy(
        source=f,
        metadata=None,
        np_array=np_array,
        row_group_indices=range(pr.metadata.num_row_groups),
        column_indices=((3, 0), (3, 1)),
    )
```

### Sparse reading
```
np_array = np.zeros((n_rows, n_columns), dtype="f", order="F")
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy(
        source=f,
        metadata=None,
        np_array=np_array,
        row_group_indices=[0],
        row_ranges=[slice(0, 1), slice(4, 6)],
        column_indices=range(pr.metadata.num_columns),
    )
print(np_array)
```

### Using cache options
```
np_array = np.zeros((n_rows, n_columns), dtype="f", order="F")
cache_options = pa.CacheOptions(hole_size_limit=1024, range_size_limit=2048, lazy=True)
with fs.LocalFileSystem().open_input_file(path) as f:
    jj.read_into_numpy(
        source=f,
        metadata=None,
        np_array=np_array,
        row_group_indices=[0],
        row_ranges=[slice(0, 1), slice(4, 6)],
        column_indices=range(pr.metadata.num_columns),
        cache_options=cache_options,
        pre_buffer=True,
    )
print(np_array)
```

### Generating a PyTorch tensor to read into:
```
import torch

# Create a tensor and transpose it to get Fortran-style order
tensor = torch.zeros(n_columns, n_rows, dtype=torch.float32).transpose(0, 1)
```

### Reading entire file into the tensor:
```
pr = pq.ParquetReader()
pr.open(path)

jj.read_into_torch(
    source=path,
    metadata=pr.metadata,
    tensor=tensor,
    row_group_indices=range(pr.metadata.num_row_groups),
    column_indices=range(pr.metadata.num_columns),
    pre_buffer=True,
    use_threads=True,
)

print(tensor)
```
