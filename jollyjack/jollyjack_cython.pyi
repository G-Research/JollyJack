import jollyjack as jj

def read_into_torch (source, metadata, tensor, row_group_indices, column_indices = [], column_names = [], pre_buffer = False, use_threads = True, use_memory_map = False):
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

    Notes:
    -----
    Either column_indices or column_names must be provided, but not both.
    When using an iterable of tuples for column_indices or column_names, 
    each tuple should contain exactly two elements: the source column (index or name) 
    and the target column index in the numpy array.
    """

    jj._read_into_torch (source
                     , metadata
                     , tensor
                     , row_group_indices
                     , column_indices
                     , column_names
                     , pre_buffer
                     , use_threads
                     , use_memory_map
                     )
    return

def read_into_numpy (source, metadata, np_array, row_group_indices, row_ranges = [], column_indices = [], column_names = [], pre_buffer = False, use_threads = True, use_memory_map = False):
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

    Notes:
    -----
    Either column_indices or column_names must be provided, but not both.
    When using an iterable of tuples for column_indices or column_names, 
    each tuple should contain exactly two elements: the source column (index or name) 
    and the target column index in the numpy array.
    """

    jj._read_into_numpy (source
                     , metadata
                     , np_array
                     , row_group_indices
                     , column_indices
                     , column_names
                     , pre_buffer
                     , use_threads
                     , use_memory_map
                     )
    return

def copy_to_torch_row_major (src_tensor, dst_tensor, row_indices):
    """
    Copy source column-major array to a row-major array and shuffles its rows according to provided indices.
    
    Args:
        src_array (numpy.ndarray): Source column-major array to be copied and shuffled.
        dst_array (numpy.ndarray): Destination row-major array to store the result.
        row_indices (numpy.ndarray): Array of indices specifying the row permutation.

    Raises:
        AssertError: If array shapes do not match or row_indices is invalid.
        RuntimeError: If row_indices has invalid index.
        
    Example:
        >>> src = np.array([[1, 2], [3, 4]], dtype=int, order='F')
        >>> dst = np.zeros((2, 2), dtype=int, order='C')
        >>> indices = np.array([1, 0])
        >>> copy_to_row_major(src, dst, indices)
        array([[3, 4],
                [1, 2]])
   """
    return

def copy_to_numpy_row_major (src_array, dst_array, row_indices):
    """
    Copy source column-major array to a row-major array and shuffles its rows according to provided indices.
    
    Args:
        src_array (numpy.ndarray): Source column-major array to be copied and shuffled.
        dst_array (numpy.ndarray): Destination row-major array to store the result.
        row_indices (numpy.ndarray): Array of indices specifying the row permutation.

    Raises:
        AssertError: If array shapes do not match or row_indices is invalid.
        RuntimeError: If row_indices has invalid index.
        
    Example:
        >>> src = np.array([[1, 2], [3, 4]], dtype=int, order='F')
        >>> dst = np.zeros((2, 2), dtype=int, order='C')
        >>> indices = np.array([1, 0])
        >>> copy_to_row_major(src, dst, indices)
        array([[3, 4],
                [1, 2]])
   """
    return