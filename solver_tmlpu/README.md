# TMLP-u isolated workspace

This folder contains the TMLP-u solver, tests, result artifacts, and previous
TMLP-u autoresearch state copied out of the main repository tree so Codex can
work here without sharing `autoresearch-results/` with Denner runs.

## Layout

- `solver/solve_T-MLP-u/` - TMLP-u code and tests
- `results/T-MLP-u/` - PNG, JSON, TSV, and VTK outputs
- `docs/plans/t_mlp_u_paper_verification.md` - paper verification plan
- `.codex-loop/verify_tmlpu_target_cases.py` - TMLP-u helper gate
- `autoresearch-archive/` - copied previous TMLP-u autoresearch artifacts

## Typical commands

```bash
MPLCONFIGDIR=/tmp/mpl python3 solver/solve_T-MLP-u/tests/tmlpu_autoresearch_verify.py \
  --n 50 \
  --workers 2 \
  --smooth-tvd bounded_cd \
  --smooth-face-increment tmlpu \
  --sharp-face-increment tmlpu \
  --sharp-tvd stacs \
  --vertex-mlp-augment \
  --plot results/T-MLP-u/tmplpu_bvd_candidate.png
```

For a fresh codex-autoresearch run, use this directory as the working
directory so its active artifacts are written to `solver_tmlpu/autoresearch-results/`.

