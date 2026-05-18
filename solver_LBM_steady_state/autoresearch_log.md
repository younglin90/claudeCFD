# Autoresearch Log — SCMK-LBM Steady-State Solver Optimization

**Goal**: Combined (speedup × accuracy) on 5 benchmarks
**Scope**: solver_scmk*.py, lbm_periodic.py
**Metric**: composite_score = mean(LBE_speedup) × accuracy_factor × convergence_factor
**Direction**: higher is better
**Verify**: `python3 verify_metric.py | tail -1`
**Iterations**: 15 (bounded)

## Iteration log

| # | change | composite | mean_speedup | worst_err | conv_frac | decision |
|---|---|---|---|---|---|---|
| 0 (baseline) | — (Phase-4 current) | 20.06 | 20.89 | 7.94e-3 | 1.00 | baseline |
