# solver/denner_1d/solver_fraysse.py
# Re-export hub — backward-compatible imports for all Fraysse solver variants.
#
# Solver variants (each in its own module):
#   1. Fraysse Conservative  → fraysse_conservative.py (step_fraysse)
#   2. Fraysse Primitive     → fraysse_primitive.py    (step_fraysse_primitive)
#   3. Fraysse Primitive+THINC → fraysse_primitive.py  (step_fraysse_primitive with use_thinc=True)
#   4. He2024 5-equation     → he2024_solver.py        (step_he2024)
#
# Common functions (EOS, flux, ghost, THINC) → fraysse_common.py
#
# Usage (unchanged from before):
#   from solver.denner_1d.solver_fraysse import step_fraysse
#   from solver.denner_1d.solver_fraysse import step_fraysse_primitive
#   from solver.denner_1d.solver_fraysse import step_he2024

from .fraysse_common import *        # noqa: F401,F403
from .fraysse_conservative import *  # noqa: F401,F403
from .fraysse_primitive import *     # noqa: F401,F403
from .he2024_solver import *         # noqa: F401,F403
