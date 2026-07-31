# AGENTS.md - solver_denner workspace

This directory is the isolated Denner solver workspace.

## Scope

- Active solver: `solver_denner/solver/denner_1d/`
- Legacy single-file Denner reference: `solver_denner/solver/denner2018_1d.py`
- Denner validation driver: `solver_denner/results/run_denner1d_17case.py`
- Denner notes: `solver_denner/docs/`

## Rules

- Modify Denner-related code here, not the repository-root legacy Denner path.
- Keep shared 1D validation specs in `validation/1D/` as the source of truth.
- The local `solver` package overlays `solver_denner/solver` and extends to the repo-level `solver/` package for shared utilities only.
- Results produced by the Denner driver are written under `solver_denner/results/1D/{case}/diff_vs_exact.png`.
- Do not make Denner validation pass criteria finite-only or visually permissive.
  For wave/shock cases such as 05_B, 07_B Air-Water, and 13_E, PASS must require
  quantitative agreement with exact/reference profiles. Numerical diffusion may
  be handled by comparing against a mildly smoothed exact/reference profile, but
  the criteria must still include field-wise correlation, scaled L2/L1 error,
  amplitude ratio, and peak/location checks where applicable.
- If a case looks very different from the exact/reference PNG, treat that as a
  validation failure and improve the solver or the exact/reference consistency;
  do not hide it by weakening thresholds.

## Autoresearch continuation guard

For this workspace, do not stop or emit a completion/final-success summary while the active autoresearch target is unmet.
Before any final/stopping answer for an autoresearch task, run:

```bash
python3 scripts/autoresearch_continue_guard.py
```

If it exits `10`, the target is not reached. Continue the loop or report only the current blocker/status; do not claim completion.
A stale `best_metric == 0`, `last_trial_metric == 0`, or `supervisor.terminal_reason == goal_reached` is not sufficient when `state.current_metrics.case14_failure_count != 0`. The stop guard also rejects solver-side `is_final_step`, feature tracker, or exact tracker gates.
Use `.codex-autoresearch/CONTINUATION_POLICY.md` as the local policy.

Current user override for the active foreground run: execute codex-autoresearch
directly in this chat with no subagents/custom agents, and use case-07 validation
mesh `N=800` (`DENNER_CASE07_N=800`,
`DENNER_CASE07_N_AIR_WATER=800`) unless the user explicitly changes it.

Codex Desktop on Windows cannot currently install the codex-autoresearch lifecycle
hooks in `C:\Users\user\.codex`; do not rely on a Windows Stop hook to force
continuation. In Desktop sessions, manually follow `.codex-autoresearch/CONTINUATION_POLICY.md`
and continue the foreground loop whenever `allowed_to_stop=false`.
