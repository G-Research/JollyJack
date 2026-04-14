import jollyjack as jj
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
import concurrent.futures
import threading
import humanize
import random
import time
import sys
import os

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchmarkSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JJB_")

    benchmark_mode: str = "CPU"
    n_files: int | None = None
    n_repeats: int | None = None
    purge_cache: bool | None = None
    worker_counts: list[int] = [1, 2]
    row_groups: int = 1
    n_columns: int = 7_000
    n_columns_to_read: int = 3_000
    chunk_size: int = 32_000
    measure_iterations: int = 5
    parquet_path: str = (
        "my.parquet" if sys.platform.startswith("win") else "/tmp/my.parquet"
    )
    benchmarks_to_run: set[str] = {"all"}

    @field_validator("worker_counts", mode="before")
    @classmethod
    def parse_worker_counts(cls, v):
        if isinstance(v, str):
            return [int(x) for x in v.split(",")]
        return v

    @field_validator("benchmarks_to_run", mode="before")
    @classmethod
    def parse_benchmarks(cls, v):
        if isinstance(v, str):
            return set(v.split(","))
        return v

    @model_validator(mode="after")
    def apply_mode_defaults(self):
        if self.benchmark_mode == "FILE_SYSTEM":
            if self.n_files is None:
                self.n_files = 6
            if self.n_repeats is None:
                self.n_repeats = 1
            if self.purge_cache is None:
                self.purge_cache = not sys.platform.startswith("win")
        elif self.benchmark_mode == "CPU":
            if self.n_files is None:
                self.n_files = 1
            if self.n_repeats is None:
                self.n_repeats = 6
            if self.purge_cache is None:
                self.purge_cache = False
        else:
            raise ValueError(f"Invalid benchmark_mode: {self.benchmark_mode}")
        return self


cfg = BenchmarkSettings()

column_indices_to_read = random.sample(range(cfg.n_columns), cfg.n_columns_to_read)
row_groups_to_read = random.sample(range(cfg.row_groups), 1)


def purge_file_from_cache(path: str):

    with open(path, "r") as f:
        fd = f.fileno()
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def generate_random_parquet(
    filename: str,
    n_columns: int,
    n_row_groups: int,
    chunk_size: int,
    dtype=pa.float32(),
    compression=None,
):
    print(".")
    print(
        f"Generating a Parquet file: {filename}, cols: {n_columns}, row_groups:{n_row_groups}, chunk_size:{chunk_size}, compression={compression}, dtype={dtype}"
    )

    writer = None
    schema = pa.schema([pa.field(f"col_{i}", dtype) for i in range(n_columns)])
    try:
        for i in range(n_row_groups):
            print(f"  Generating row group {i+1}/{n_row_groups}...")
            data = {
                f"col_{j}": np.random.uniform(-100, 100, size=chunk_size)
                for j in range(n_columns)
            }

            df = pd.DataFrame(data)
            table = pa.Table.from_pandas(df, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(
                    filename,
                    schema,
                    use_dictionary=False,
                    compression=compression,
                    write_statistics=False,
                    store_schema=False,
                )

            print("  writing...:")
            writer.write_table(table)

        print("Parquet file generated successfully!")
    finally:
        if writer:
            writer.close()


def generate_data(n_columns, n_row_groups, path, compression, dtype):

    t = time.time()
    generate_random_parquet(
        filename=path,
        n_columns=n_columns,
        n_row_groups=n_row_groups,
        chunk_size=cfg.chunk_size,
        compression=compression,
        dtype=dtype,
    )
    parquet_size = os.stat(path).st_size

    dt = time.time() - t
    print(
        f"finished writing parquet file in {dt:.2f} seconds, size={humanize.naturalsize(parquet_size)}"
    )


def worker_arrow_row_group(use_threads, pre_buffer, path):

    pr = pq.ParquetReader()
    pr.open(path, pre_buffer=pre_buffer)

    table = pr.read_row_groups(
        row_groups=row_groups_to_read,
        column_indices=column_indices_to_read,
        use_threads=use_threads,
    )
    np_array = table.to_pandas()


def get_thread_local_np_array(dtype):
    np_array = getattr(thread_local_data, "np_array", None)
    if np_array is None:
        np_array = np.empty(
            (cfg.chunk_size, cfg.n_columns_to_read), dtype=dtype, order="F"
        )
        # By writing all zeros we make sure that the memory is properly allocated and mapped to physical RAM (avoid Memory Allocation Contention)
        np_array[:] = 0
        thread_local_data.np_array = np_array
    return np_array


def worker_jollyjack_numpy(use_threads, pre_buffer, dtype, path):

    np_array = get_thread_local_np_array(dtype)

    jj.read_into_numpy(
        source=path,
        metadata=None,
        np_array=np_array,
        row_group_indices=row_groups_to_read,
        column_indices=column_indices_to_read,
        pre_buffer=pre_buffer,
        use_threads=use_threads,
    )


def worker_jollyjack_copy_to_row_major(dtype, path):

    np_array = np.zeros((cfg.chunk_size, cfg.n_columns_to_read), dtype=dtype, order="F")
    dst_array = np.zeros(
        (cfg.chunk_size, cfg.n_columns_to_read), dtype=dtype, order="C"
    )
    row_indices = list(range(cfg.chunk_size))
    random.shuffle(row_indices)

    pr = pq.ParquetReader()
    pr.open(path)

    jj.read_into_numpy(
        source=path,
        metadata=pr.metadata,
        np_array=np_array,
        row_group_indices=row_groups_to_read,
        column_indices=column_indices_to_read,
        pre_buffer=True,
        use_threads=False,
    )

    jj.copy_to_numpy_row_major(np_array, dst_array, row_indices)


def worker_numpy_copy_to_row_major(dtype, path):

    np_array = np.zeros((cfg.chunk_size, cfg.n_columns_to_read), dtype=dtype, order="F")
    dst_array = np.zeros(
        (cfg.chunk_size, cfg.n_columns_to_read), dtype=dtype, order="C"
    )

    pr = pq.ParquetReader()
    pr.open(path)

    jj.read_into_numpy(
        source=path,
        metadata=pr.metadata,
        np_array=np_array,
        row_group_indices=row_groups_to_read,
        column_indices=column_indices_to_read,
        pre_buffer=True,
        use_threads=False,
    )

    np.copyto(dst_array, np_array)


def worker_raw_bytes_read(dtype, path, read_metadata):

    np_array = get_thread_local_np_array(dtype)
    buf = np_array.reshape(-1, order="A").data

    with open(path, "rb") as f:
        if read_metadata:
            pr = pq.ParquetReader()
            pr.open(f)
            _ = pr.metadata

        fd = f.fileno()
        os.preadv(fd, [buf], 0)


def worker_jollyjack_torch(pre_buffer, dtype, path):

    import torch

    numpy_to_torch_dtype_dict = {
        np.bool: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
    }

    tensor = torch.zeros(
        cfg.n_columns_to_read, cfg.chunk_size, dtype=numpy_to_torch_dtype_dict[dtype]
    ).transpose(0, 1)

    pr = pq.ParquetReader()
    pr.open(path)

    jj.read_into_torch(
        source=path,
        metadata=pr.metadata,
        tensor=tensor,
        row_group_indices=row_groups_to_read,
        column_indices=column_indices_to_read,
        pre_buffer=pre_buffer,
        use_threads=False,
    )


def calculate_data_size(dtype):
    return cfg.chunk_size * cfg.n_columns_to_read * dtype.byte_width


def measure_reading(max_workers, worker):
    global thread_local_data
    tt = []
    data_set_bytes = 0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    thread_local_data = threading.local()

    # measure multiple times and take the fastest run
    for _ in range(cfg.measure_iterations):

        if cfg.purge_cache:
            for i in range(cfg.n_files):
                purge_file_from_cache(path=f"{cfg.parquet_path}{i}")

        # Submit the work
        t = time.time()
        work_results = []
        for _ in range(cfg.n_repeats):
            work_results.extend(
                [
                    pool.submit(worker, path=f"{cfg.parquet_path}{i}")
                    for i in range(0, cfg.n_files)
                ]
            )

        for work_result in work_results:
            work_result.result()

        tt.append(time.time() - t)
        data_set_bytes = len(work_results) * calculate_data_size(dtype=dtype)

    pool.shutdown(wait=True)

    tts = [f"{t:.2f}" for t in tt]
    tts = f"[{', '.join(tts)}]"
    throughput_gbps = data_set_bytes / min(tt) / (1024 * 1024 * 1024) * 8
    return f"{min(tt):.2f}s, {throughput_gbps:.2f} Gb/s -> {tts}"


print(f".")
print(f"pyarrow.version = {pa.__version__}")
print(f"jollyjack.version = {jj.__version__}")

print(f".")
for name, value in cfg.model_dump().items():
    print(f"{name} = {value}")
print(f".")

for compression, dtype in [
    (None, pa.float32()),
    ("snappy", pa.float32()),
    (None, pa.float16()),
]:

    for f in range(cfg.n_files):
        path = f"{cfg.parquet_path}{f}"
        generate_data(
            path=path,
            n_row_groups=cfg.row_groups,
            n_columns=cfg.n_columns,
            compression=compression,
            dtype=dtype,
        )

    print(f"....................................")
    print(f"dtype:{dtype}, compression={compression}:")
    print(f".")

    if {"all", "raw_bytes"} & cfg.benchmarks_to_run and not sys.platform.startswith(
        "win"
    ):
        print(f".")
        for n_workers in cfg.worker_counts:
            for read_metadata in [False, True]:
                print(
                    f"`raw_bytes_read` n_workers:{n_workers}, read_metadata:{read_metadata}, duration:{measure_reading(n_workers, lambda path: worker_raw_bytes_read(dtype.to_pandas_dtype(), path, read_metadata = read_metadata))}"
                )

    if {"all", "arrow"} & cfg.benchmarks_to_run:
        print(f".")
        for n_workers in cfg.worker_counts:
            for pre_buffer in [False, True]:
                for use_threads in [False, True]:
                    print(
                        f"`pq.read_row_groups` n_workers:{n_workers}, use_threads:{use_threads}, pre_buffer:{pre_buffer}, duration:{measure_reading(n_workers, lambda path:worker_arrow_row_group(use_threads = use_threads, pre_buffer = pre_buffer, path = path))}"
                    )

    if {"all", "jj_numpy"} & cfg.benchmarks_to_run:
        print(f".")
        for jj_reader in (
            [None]
            if sys.platform.startswith("win")
            else [None, "io_uring", "io_uring_odirect"]
        ):

            if jj_reader is None:
                os.environ.pop("JJ_READER_BACKEND", None)
            else:
                os.environ["JJ_READER_BACKEND"] = jj_reader

            print(f".")
            for n_workers in cfg.worker_counts:
                for pre_buffer in [False, True]:
                    for use_threads in [False, True]:
                        print(
                            f"`jj.read_into_numpy` jj_reader:{jj_reader}, n_workers:{n_workers}, use_threads:{use_threads}, pre_buffer:{pre_buffer}, duration:{measure_reading(n_workers, lambda path:worker_jollyjack_numpy(use_threads, pre_buffer, dtype.to_pandas_dtype(), path = path))}"
                        )

    if {"all", "jj_torch"} & cfg.benchmarks_to_run:
        print(f".")
        for n_workers in cfg.worker_counts:
            for pre_buffer in [False, True]:
                print(
                    f"`jj.read_into_torch` n_workers:{n_workers}, pre_buffer:{pre_buffer}, duration:{measure_reading(n_workers, lambda path:worker_jollyjack_torch(pre_buffer, dtype.to_pandas_dtype(), path = path))}"
                )

    if {"all", "copy_to_row_major"} & cfg.benchmarks_to_run:
        print(f".")
        for jj_variant in [1, 2]:
            os.environ["JJ_copy_to_row_major"] = str(jj_variant)
            for n_workers in cfg.worker_counts:
                print(
                    f"`jj.copy_to_row_major` n_workers:{n_workers}, jj_variant={jj_variant} duration:{measure_reading(n_workers, lambda path:worker_jollyjack_copy_to_row_major(dtype.to_pandas_dtype(), path = path))}"
                )

    if {"all", "np_copy"} & cfg.benchmarks_to_run:
        print(f".")
        for n_workers in cfg.worker_counts:
            print(
                f"`np.copy_to_row_major` compression={compression}, duration:{measure_reading(n_workers, lambda path:worker_numpy_copy_to_row_major(dtype.to_pandas_dtype(), path))}"
            )
