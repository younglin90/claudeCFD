"""five_eq_IMEX — clean-room all-Mach IMEX 5-equation FVM solver.

Status: under construction.  See `docs/five_eq_all_mach_plan.md` for the
implementation roadmap (Phase 0 → 10).

Public API will live in `main.py` and a thin set of well-named helper modules
(`eos/`, `flux/`, `time_integrator/`, `primitive/`, `boundary/`).  The legacy
`solver/He2024/` tree is frozen for regression reference and must not be edited.
"""
