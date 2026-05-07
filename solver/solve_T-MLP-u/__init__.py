"""solve_T-MLP-u — validation harness for the user's T-MLP-u
primitive high-order reconstruction technique.

Scope (set in collaboration with the user, 2026-05-07):
  - Dimensions: 1D and 2D
  - Grids: structured + unstructured
  - Governing equations: linear/nonlinear scalar advection + Euler
  - Reconstruction: first-order, classical TVD (minmod / van Leer / Superbee),
    MLP-u (Park-Yoon-Kim 2010, baseline), and T-MLP-u (user-supplied,
    extends MLP-u to unstructured grids and may be mixed with TVD).
  - Free tuning parameters: 0 (only physically/numerically meaningful
    constants — γ, CFL, fixed limiter formulae).

This package's directory name contains a dash ("solve_T-MLP-u") so it
cannot be imported with the usual `import solver.solve_T-MLP-u` syntax.
We sidestep the problem by inserting the package directory itself onto
sys.path on first import; siblings are then loadable via
`from mesh import Mesh1D` etc.  External callers should
either run scripts inside `tests/` (where the path shim is set up
automatically) or use::

    import importlib, sys
    pkg = importlib.import_module(
        "solver.solve_T-MLP-u",
        package=None,
    )
    # or simpler:
    import sys, os
    sys.path.insert(0, "<repo>/solver/solve_T-MLP-u")
    from solver import solve

Other solvers (`solver/He2024/`, `solver/five_eq_IMEX/`,
`solver/five_eq_IMEX_v2/`, `solver/denner_1d/`, …) MUST NOT be modified.
All work happens inside this directory.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
