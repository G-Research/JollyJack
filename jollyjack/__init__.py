# Importing pyarrow is necessary to load the runtime libraries
import pyarrow
import pyarrow.parquet
from importlib.metadata import version, requires
__version__ = version(__package__)  # Uses package metadata
dependencies = requires(__package__)
pyarrow_req = next((r for r in dependencies if r.startswith('pyarrow')), '')
from .jollyjack_cython import *

