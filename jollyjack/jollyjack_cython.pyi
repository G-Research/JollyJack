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
    column_indices : list[int] or dict[int, int], optional
        List of columns to read from parquet file or a mapping of columns to read from the parquet file into particulrar columns in the tensor.
    column_names : list[str] or dict[str, int], optional
        List of columns to read from parquet file or a mapping of columns to read from the parquet file into particulrar columns in the tensor.
    pre_buffer : bool, default False
    use_threads : bool, default True
    use_memory_map : bool, default False
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
    column_indices : list[int] or dict[int, int], optional
        List of columns to read from parquet file or a mapping of columns to read from the parquet file into particulrar columns in the array.

    column_names : list[str] or dict[str, int], optional
        List of columns to read from parquet file or a mapping of columns to read from the parquet file into particulrar columns in the array.

    pre_buffer : bool, default False
    use_threads : bool, default True
    use_memory_map : bool, default False
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
