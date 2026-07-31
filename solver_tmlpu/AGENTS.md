<!-- AUTONOMY DIRECTIVE - DO NOT REMOVE -->
YOU ARE AN AUTONOMOUS CODING AGENT. EXECUTE TASKS TO COMPLETION WITHOUT ASKING FOR PERMISSION.
DO NOT STOP TO ASK "SHOULD I PROCEED?" - PROCEED. DO NOT WAIT FOR CONFIRMATION ON OBVIOUS NEXT STEPS.
IF BLOCKED, TRY AN ALTERNATIVE APPROACH. ONLY ASK WHEN TRULY AMBIGUOUS OR DESTRUCTIVE.
<!-- END AUTONOMY DIRECTIVE -->

# AGENTS.md - TMLP-u workspace (Codex-era compatibility)

This file exists for Codex-era compatibility. Canonical instructions now live
in `CLAUDE.md` (same dir) and `.claude/rules/` (execution-model, wsl-pitfalls,
cfd-run-rules).

## Scope

- Editable: everything under `solver_tmlpu/` AND `../cpp/` (active C++ work).
- Do NOT touch: `../solver_5eq/`, `../solver_denner/`, `../archive/`,
  `../백업_*`.
- Python `solver/solve_T-MLP-u/` = frozen validation oracle (do not delete,
  do not modify).

## Validation spec contracts

Canonical user-facing validation contracts — keep initial conditions, boundary
conditions, mesh/numerical contracts, output requirements, and PASS gates
aligned with these unless the user explicitly overrides:

- LeVeque rotation: `docs/leveque_strict_validation_spec.md`
- Mach 3 forward-facing step: `docs/mach3_step_strict_validation_spec.md`
- Double Mach reflection: `docs/double_mach_reflection_strict_spec.md`
