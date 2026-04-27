import jollyjack as jj

def read_into_torch(
    source,
    metadata,
    tensor,
    row_group_indices,
    column_indices=[],
    column_names=[],
    pre_buffer=False,
    use_threads=True,
    use_memory_map=False,
    cache_options=None,
    prefetch_page_cache=False,
):
    """
    Read parquet data directly into a tensor.

    Parameters
    ----------
    source : str, pathlib.Path, pyarrow.NativeFile, or file-like object
    metadata : FileMetaData, optional
    tensor : The tensor to read into. The shape of the tensor needs to match the number of rows and columns to be read.
    row_group_indices : list[int]
    row_ranges : list[slice], optional
        Specifies slices of destination rows to read into. Each slice defines a range
        of rows in the destination tensor where data should be written.
        Example: [slice(0, 100), slice(200, 300)] will read data into rows 0-99 and 200-299.
        If None, reads into all rows sequentially.
    column_indices : list[int] | dict[int, int] | Iterable[tuple[int, int]], optional
        Specifies the columns to read from the parquet file. Can be:
        - A list of column indices to read.
        - A dict mapping source column indices to target column indices in the tensor.
        - An iterable of tuples, where each tuple contains (source_index, target_index).
    column_names : list[str] | dict[str, int] | Iterable[tuple[str, int]], optional
        Specifies the columns to read from the parquet file by name. Can be:
        - A list of column names to read.
        - A dict mapping source column names to target column indices in the tensor.
        - An iterable of tuples, where each tuple contains (column_name, target_index).
    pre_buffer : bool, default False
    use_threads : bool, default True
    use_memory_map : bool, default False
    cache_options : arrow::io::CacheOptions, default None -> CCacheOptions.LazyDefaults()
    prefetch_page_cache : bool, default False
        When True, calls posix_fadvise(POSIX_FADV_WILLNEED) on the byte ranges for
        the requested columns and row groups before reading. This warms the OS page
        cache so that the subsequent pread calls find data already resident in memory.
        Use this when pre_buffer=True causes high LLC miss rates due to Arrow's
        IO thread pool allocating buffers on cores different from the worker threads.
        Only useful for local or network-mounted file systems that have a page cache.
        Remote file systems such as S3 will not benefit from this.

    Notes:
    -----
    Either column_indices or column_names must be provided, but not both.
    When using an iterable of tuples for column_indices or column_names,
    each tuple should contain exactly two elements: the source column (index or name)
    and the target column index in the tensor.
    """

    jj._read_into_torch(
        source,
        metadata,
        tensor,
        row_group_indices,
        column_indices,
        column_names,
        pre_buffer,
        use_threads,
        use_memory_map,
        prefetch_page_cache=prefetch_page_cache,
    )
    return

def read_into_numpy(
    source,
    metadata,
    np_array,
    row_group_indices,
    row_ranges=[],
    column_indices=[],
    column_names=[],
    pre_buffer=False,
    use_threads=True,
    use_memory_map=False,
    cache_options=None,
    prefetch_page_cache=False,
):
    """
    Read parquet data directly into a numpy array.
    NumPy array needs to be in a Fortran-style (column-major) order.

    Parameters
    ----------
    source : str, pathlib.Path, pyarrow.NativeFile, or file-like object
    metadata : FileMetaData, optional
    np_array : The array to read into. The shape of the array needs to match the number of rows and columns to be read.
    row_group_indices : list[int]
    row_ranges : list[slice], optional
        Specifies slices of destination rows to read into. Each slice defines a range
        of rows in the destination array where data should be written.
        Example: [slice(0, 100), slice(200, 300)] will read data into rows 0-99 and 200-299.
        If None, reads into all rows sequentially.
    column_indices : list[int] | dict[int, int] | Iterable[tuple[int, int]], optional
        Specifies the columns to read from the parquet file. Can be:
        - A list of column indices to read.
        - A dict mapping source column indices to target column indices in the array.
        - An iterable of tuples, where each tuple contains (source_index, target_index).
    column_names : list[str] | dict[str, int] | Iterable[tuple[str, int]], optional
        Specifies the columns to read from the parquet file by name. Can be:
        - A list of column names to read.
        - A dict mapping source column names to target column indices in the array.
        - An iterable of tuples, where each tuple contains (column_name, target_index).
    pre_buffer : bool, default False
    use_threads : bool, default True
    use_memory_map : bool, default False
    cache_options : pa.CacheOptions(), default None -> CCacheOptions.LazyDefaults()
    prefetch_page_cache : bool, default False
        When True, calls posix_fadvise(POSIX_FADV_WILLNEED) on the byte ranges for
        the requested columns and row groups before reading. This warms the OS page
        cache so that the subsequent pread calls find data already resident in memory.
        Use this when pre_buffer=True causes high LLC miss rates due to Arrow's
        IO thread pool allocating buffers on cores different from the worker threads.
        Only useful for local or network-mounted file systems that have a page cache.
        Remote file systems such as S3 will not benefit from this.

    Notes:
    -----
    Either column_indices or column_names must be provided, but not both.
    When using an iterable of tuples for column_indices or column_names,
    each tuple should contain exactly two elements: the source column (index or name)
    and the target column index in the numpy array.
    """

    jj._read_into_numpy(
        source,
        metadata,
        np_array,
        row_group_indices,
        column_indices,
        column_names,
        pre_buffer,
        use_threads,
        use_memory_map,
        cache_options,
        prefetch_page_cache=prefetch_page_cache,
    )
    return

def prefetch_page_cache(
    source,
    metadata,
    row_group_indices,
    column_indices=[],
    column_names=[],
    use_memory_map=False,
    cache_options=None,
) -> None:
    """
    Prefetch parquet byte ranges into the OS page cache.

    Calls posix_fadvise(POSIX_FADV_WILLNEED) on the byte ranges corresponding
    to the requested row groups and columns. The kernel starts reading those
    pages into the page cache asynchronously.

    With pre_buffer=True, Arrow's IO thread pool allocates temporary buffers and
    fills them on the IO thread's core. When worker threads on different cores
    later consume those buffers, the data is cold in their caches, causing high
    LLC miss rates.

    prefetch_page_cache avoids this: the page cache is warmed, and each worker
    thread reads via pread into its own locally-allocated buffer, keeping data
    hot in the worker's local CPU caches.

    Only useful for local or network-mounted file systems that have a page cache.
    Remote file systems such as S3 will not benefit from this.

    Parameters
    ----------
    source : str, pathlib.Path, pyarrow.NativeFile, or file-like object
        Path to the parquet file or an already-opened Arrow file handle.
    metadata : pyarrow.parquet.FileMetaData or None
        Parquet file metadata. If None, metadata is read from the file.
    row_group_indices : list[int]
        Indices of the row groups whose data should be prefetched.
    column_indices : list[int], optional
        Column indices to prefetch. Mutually exclusive with column_names.
    column_names : list[str], optional
        Column names to prefetch. Mutually exclusive with column_indices.
    use_memory_map : bool, default False
        Whether to memory-map the file.
    cache_options : pyarrow.CacheOptions, optional
        Controls how nearby byte ranges are coalesced before the fadvise call.
        Defaults to Arrow's lazy defaults (hole_size_limit=8192, range_size_limit=32MB).

    Notes
    -----
    Either column_indices or column_names must be provided, but not both.
    """
    ...

def copy_to_torch_row_major(src_tensor, dst_tensor, row_indices):
    """
    Copy source column-major tensor to a row-major tensor and shuffle its rows according to provided indices.

    Args:
        src_tensor (torch.Tensor): Source column-major tensor to be copied and shuffled.
        dst_tensor (torch.Tensor): Destination row-major tensor to store the result.
        row_indices (numpy.ndarray): Array of indices specifying the row permutation.

    Raises:
        AssertionError: If tensor shapes do not match or row_indices is invalid.
        RuntimeError: If row_indices has an invalid index.

    Example:
        >>> src = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        >>> dst = torch.zeros(2, 2, dtype=torch.int32)
        >>> indices = np.array([1, 0])
        >>> jj.copy_to_torch_row_major(src, dst, indices)
    """
    return

def copy_to_numpy_row_major(src_array, dst_array, row_indices):
    """
    Copy source column-major array to a row-major array and shuffle its rows according to provided indices.

    Args:
        src_array (numpy.ndarray): Source column-major array to be copied and shuffled.
        dst_array (numpy.ndarray): Destination row-major array to store the result.
        row_indices (numpy.ndarray): Array of indices specifying the row permutation.

    Raises:
        AssertionError: If array shapes do not match or row_indices is invalid.
        RuntimeError: If row_indices has an invalid index.

    Example:
        >>> src = np.array([[1, 2], [3, 4]], dtype=int, order='F')
        >>> dst = np.zeros((2, 2), dtype=int, order='C')
        >>> indices = np.array([1, 0])
        >>> jj.copy_to_numpy_row_major(src, dst, indices)
    """
    return
