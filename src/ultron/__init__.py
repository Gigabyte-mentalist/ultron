# ruff: noqa: I001 -- these two imports' *order* is load-bearing (each must
# run before anything else touches CUDA/creates a window); keep them as
# separate statements so no import-sorter ever merges or reorders them.
from ultron import _cuda_dlls  # noqa: F401  # must be imported first, see _cuda_dlls.py
from ultron import _dpi_awareness  # noqa: F401  # must run before any window, see its docstring

__version__ = "0.1.0"
