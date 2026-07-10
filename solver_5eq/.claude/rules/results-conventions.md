---
description: Result artifact + plotting conventions for solver_5eq/results
paths:
  - "results/**"
---

# Results conventions — solver_5eq

- **PNG save is mandatory.** Every validation run must call `matplotlib.use('Agg')`
  then `plt.savefig(...)` to a FIXED path, overwriting in place. Canonical path:
  `results/1D/{case_name}/diff_vs_exact.png`. **Never** create per-round filenames.
  Print `Plot saved: ...` after each run.
- Canonical 02/07 driver = `results/run_02_07_five_eq_imex.py` — a thin wrapper that
  loads `.codex-loop/verify_02_07_acceptance.py`. Prefer it over ad-hoc drivers.
- Legacy He2024 driver `results/round177_unified.py` is reference-only.
- Report artifacts (`plan_report.md`, `fix_report.md`, `qa_report.md`) are produced by the
  agent pipeline into `results/`; they may not exist until that stage runs.
