#!/usr/bin/env python
import os
import sys

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import pyarrow
import numpy

include_dirs = [pyarrow.get_include(), numpy.get_include()]
library_dirs = pyarrow.get_library_dirs()

print ("include_dirs=", include_dirs)
print ("library_dirs=", library_dirs)

extra_compile_args = []
extra_link_args = []
debug = False,

if os.getenv('DEBUG', '') == 'ON':
    extra_compile_args = ["-O0", '-DDEBUG']
    extra_link_args = ["-debug:full"]
    debug = True,

# Define your extension
extensions = [
    Extension( "jollyjack.jollyjack_cython", ["jollyjack/jollyjack_cython.pyx", "jollyjack/jollyjack.cc"],
        include_dirs = include_dirs,  
        library_dirs = library_dirs,
        libraries=["arrow", "parquet"], 
        language = "c++",
        extra_compile_args = extra_compile_args + (['/std:c++17'] if sys.platform.startswith('win') else ['-std=c++17']),
        extra_link_args = extra_link_args,
    )
]

compiler_directives = {"language_level": 3, "embedsignature": True}
extensions = cythonize(extensions, compiler_directives=compiler_directives, gdb_debug=debug, emit_linenums=debug)

# Make default named pyarrow shared libs available.
pyarrow.create_library_symlinks()

setup(
    packages=["jollyjack"],
    package_dir={"": "."},
    zip_safe=False,
    ext_modules=extensions,
    project_urls={
        "Documentation": "https://github.com/G-Research/PalletJack",
        "Source": "https://github.com/G-Research/PalletJack",
    },
)
