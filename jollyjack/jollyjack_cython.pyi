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

def read_into_numpy (source, metadata, np_array, row_group_indices, column_indices = [], column_names = [], pre_buffer = False, use_threads = True, use_memory_map = False):
    """
    Read parquet data directly into a numpy array.
    NumPy array needs to be in a Fortran-style (column-major) order.

    Parameters
    ----------
    source : str, pathlib.Path, pyarrow.NativeFile, or file-like object
    metadata : FileMetaData, optional
    np_array : The array to read into. The shape of the array needs to match the number of rows and columns to be read.
    row_group_indices : list[int]
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
