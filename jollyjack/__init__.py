# Importing pyarrow is necessary to load the runtime libraries
import pyarrow
import pyarrow.parquet
from importlib.metadata import version, requires
from .jollyjack_cython import *

