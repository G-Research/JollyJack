# JollyJack

JollyJack is a high-performance Parquet reader designed to load data directly
into NumPy arrays and PyTorch tensors with minimal overhead.

## Features

- Load Parquet straight into NumPy arrays or PyTorch tensors (fp16, fp32, fp64, int32, int64)
- Up to 6× faster and with lower memory use than vanilla PyArrow
- Compatibility with [PalletJack](https://github.com/G-Research/PalletJack)
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

For datasets larger than the available page cache, performance is typically
I/O-bound. Enabling either `pre_buffer=True` or `prefetch_page_cache=True`
brings throughput close to the raw I/O ceiling.

Recommended configuration:

- `use_threads = True`, `prefetch_page_cache = True`, `pre_buffer = False`,
  with the default reader backend.

Both options reach near-identical throughput. `prefetch_page_cache` avoids the
temporary buffer copies that `pre_buffer` uses (see section below) and the
increased LLC miss rate.

### Small datasets (fit in filesystem cache)

For datasets that comfortably fit in RAM, performance is typically CPU- or
memory-bound.

Recommended configuration:

- `use_threads = True`, `prefetch_page_cache = True`, `pre_buffer = False`,
  with the default reader backend.

### Pre-buffering and `cache_options`

When `pre_buffer=True`, Arrow merges nearby column ranges and reads them into
temporary buffers. The default maximum merged range is 32 MB
(`range_size_limit`).

Arrow supports several memory allocators (mimalloc, jemalloc, system). With
mimalloc (the default on most platforms), allocations above
[~16 MB](https://github.com/microsoft/mimalloc/blob/75d69f4ab736ad9f56cdd76c7eb883f60ac48869/include/mimalloc/types.h#L205)
go straight to the OS (`mmap`/`munmap`) instead of the internal arena. This
means the memory cannot be reused between calls, and each call pays the cost
of mapping and zeroing fresh pages. Other allocators may behave similarly.

To avoid this, lower `range_size_limit` so that merged ranges fit inside the
allocator's arena:

```python
cache_options = pa.CacheOptions(
    hole_size_limit=8192,           # default
    range_size_limit=16*1024*1024,  # 16 MB, fits in mimalloc arena
    lazy=False,
)
jj.read_into_numpy(
    source=path,
    metadata=None,
    np_array=np_array,
    row_group_indices=[0],
    column_indices=range(n_columns),
    pre_buffer=True,
    cache_options=cache_options,
)
```

To debug allocator issues with mimalloc, run with `MIMALLOC_SHOW_STATS=1` and
`MIMALLOC_VERBOSE=1`. This prints allocation statistics at process exit.

### Pre-buffering and `ARROW_IO_THREADS`

When `pre_buffer=True`, Arrow dispatches reads to its IO thread pool,
configured via the `ARROW_IO_THREADS` environment variable (default: 8). 
Tuning this value may improve performance.

### Page cache prefetching with `prefetch_page_cache`

With `pre_buffer=True`, Arrow's IO thread pool allocates temporary buffers
and fills them on the IO thread's core. When worker threads on different
cores later consume those buffers, the data is cold in their caches,
causing LLC misses.

`prefetch_page_cache` provides an alternative: it calls
`posix_fadvise(POSIX_FADV_WILLNEED)` to tell the kernel to start loading
the relevant byte ranges into the page cache. Each worker thread then
reads directly via `pread` into its own locally-allocated buffer, keeping
data hot in its local CPU caches.

Two ways to use it:

**As a parameter on `read_into_numpy`:**

```python
jj.read_into_numpy(
    source=path,
    metadata=pr.metadata,
    np_array=np_array,
    row_group_indices=range(pr.metadata.num_row_groups),
    column_indices=range(pr.metadata.num_columns),
    prefetch_page_cache=True,
)
```

This is only useful for local or network-mounted file systems that have a
page cache. Remote file systems such as S3 will not benefit from this.

**As a standalone call** (when you want to prefetch ahead of time, e.g.
from a different thread):

```python
jj.prefetch_page_cache(
    source=path,
    metadata=pr.metadata,
    row_group_indices=range(pr.metadata.num_row_groups),
    column_indices=range(pr.metadata.num_columns),
)

jj.read_into_numpy(
    source=path,
    metadata=pr.metadata,
    np_array=np_array,
    row_group_indices=range(pr.metadata.num_row_groups),
    column_indices=range(pr.metadata.num_columns),
    pre_buffer=False,
)
```

## Requirements

- pyarrow ~= 24.0.0

JollyJack builds on top of PyArrow. While the source package may work with
newer versions, the prebuilt binary wheels are built and tested against pyarrow 24.x.

## Installation

```bash
pip install jollyjack
```

## How to use

### Generating a sample Parquet file
```python
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

### Generating a NumPy array to read into
```python
# Create an array of zeros
np_array = np.zeros((n_rows, n_columns), dtype="f", order="F")
```

### Reading an entire file into a NumPy array
```python
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
### Reading columns in reverse order
```python
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
```python
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
```python
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
```python
np_array = np.zeros((n_rows, n_columns), dtype="f", order="F")
cache_options = pa.CacheOptions(
    hole_size_limit=8192,           # default
    range_size_limit=16*1024*1024,  # 16 MB, fits in mimalloc arena
    lazy=False,
)
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

### Using page cache prefetching
```python
np_array = np.zeros((n_rows, n_columns), dtype="f", order="F")
pr = pq.ParquetReader()
pr.open(path)

# cache_options controls which byte ranges are prefetched into the page cache
cache_options = pa.CacheOptions(
    hole_size_limit=8192,
    range_size_limit=16*1024*1024,
    lazy=False,
)

# Prefetch and read in one call
jj.read_into_numpy(
    source=path,
    metadata=pr.metadata,
    np_array=np_array,
    row_group_indices=range(pr.metadata.num_row_groups),
    column_indices=range(pr.metadata.num_columns),
    cache_options=cache_options,
    prefetch_page_cache=True,
)

# Or prefetch separately, then read
jj.prefetch_page_cache(
    source=path,
    metadata=pr.metadata,
    row_group_indices=range(pr.metadata.num_row_groups),
    column_indices=range(pr.metadata.num_columns),
    cache_options=cache_options,
)
jj.read_into_numpy(
    source=path,
    metadata=pr.metadata,
    np_array=np_array,
    row_group_indices=range(pr.metadata.num_row_groups),
    column_indices=range(pr.metadata.num_columns),
    pre_buffer=False,
)
```

### Generating a PyTorch tensor to read into
```python
import torch

# Create a tensor and transpose it to get Fortran-style order
tensor = torch.zeros(n_columns, n_rows, dtype=torch.float32).transpose(0, 1)
```

### Reading an entire file into a PyTorch tensor
```python
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
