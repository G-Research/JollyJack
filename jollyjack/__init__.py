# Importing pyarrow is necessary to load the runtime libraries

import pyarrow

# package/__init__.py
try:
    from importlib.metadata import version, requires
    __version__ = version(__package__)  # Uses package metadata
    dependencies = requires(__package__)
    print(dependencies)  # List of requirements strings

except ImportError:
    __version__ = "0.0.1"  # Fallback


try:
    from .jollyjack_cython import *
        
except ImportError as e:
    if "libarrow.so" in str(e):
        raise ImportError(f"Unable to load libarrow.so. This version of {__package__}={__version__} is built against arrow  ensure you have pyarrow==17.x installed. Current pyarrow version: {pyarrow.__version__}")
    else:
        raise
