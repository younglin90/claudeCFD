"""Make the parent package directory importable as a flat collection of
top-level modules.  The package directory contains a dash in its name
(`solve_T-MLP-u`) so Python's `import` syntax can't reach it directly.

Tests do:
    from _pkgshim import setup_paths
    setup_paths()
    from mesh import build_structured_1d
    from equations import Advection
    ...
"""
import os
import sys


def setup_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)        # solve_T-MLP-u/
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
