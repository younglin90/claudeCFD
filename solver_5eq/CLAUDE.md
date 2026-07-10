# CLAUDE.md — solver_5eq (5-equation multicomponent compressible FVM)

> **Response rule**: Always reply in caveman full mode (skill: caveman). Keep code / commits / error-strings / commands verbatim; write security warnings & irreversible-action confirmations in normal prose. Disable with "stop caveman".
> **Language**: Repo-wide C++ migration is in progress (see parent `../CLAUDE.md`). New numerics belong in `../cpp/`; this folder's Python is a **validation oracle + research code — never delete it**.

## Project overview

**5-equation (Allaire/Kapila) all-speed multicomponent compressible FVM solver** — 1D core with 2D/3D extension, unifying incompressible→compressible flow (IMEX, implicit pressure).

- Subproject of monorepo `claudeCFD/` (split out 2026-07-02, same as `solver_denner/`, `solver_tmlpu/`). **Git is managed at the repo root** — no local `.git` here.
- Python runs under WSL: `wsl.exe -d ubuntu`; this folder = `~/work/claude_code/claudeCFD/solver_5eq`.
- C++ migration tree = parent `../cpp/` (managed separately). Paper PDFs/summaries = `../papers/`.

## Project map

```
solver_5eq/
├── solver/
│   ├── five_eq_IMEX/        ← ★ ACTIVE (1D BE1-IMEX + 2D/3D nd_solver)
│   │                           entry main.py; also main_2d.py, main_3d.py, nd_solver.py,
│   │                           eos_facade.py, he2024_compat.py, explicit.py,
│   │                           energy_flux.py, limiters.py
│   ├── five_eq_IMEX_v2/     ← v2 redesign (docs/v2_round_*.md)
│   ├── He2024/              ← FROZEN. only eos_general.py, primitive_W.py used,
│   │                           loaded via five_eq_IMEX/he2024_compat.py (bypasses __init__)
│   └── __init__.py          ← empty on purpose — keep it empty
├── tests/                   ← unit tests test_*.py + v2_smoke/
├── validation/              ← FROZEN specs: 1D/ (26 cases), 2D/, 3D/, INDEX.md
├── results/                 ← drivers + artifacts: 1D/{case}/, 2D/, 3D/
│   ├── run_02_07_five_eq_imex.py   ← canonical 02/07 driver (loads .codex-loop verify)
│   ├── round177_unified.py         ← legacy He2024 driver (reference only)
│   └── run_2d_validations.py, run_2d3d_recommended.py, all_26_summary.md
├── docs/                    ← all 5eq specs/plans/diagnostics (32 files)
│   ├── five_eq_all_mach_plan.md          ← roadmap + changelog (SINGLE SOURCE OF TRUTH)
│   ├── five_eq_IMEX_current_formulation.md, SOLVER_DESIGN_GUIDE.md
│   └── nd_validation_plan.md, v2_round_1..7.md
├── .codex-loop/             ← 1D verification harness (canonical acceptance)
│   ├── verify_02_07_acceptance.py, verify_01_03_06 / 08_26 / 16_19 / 32_35 …
│   └── LOOP_STATE.md, oscillation_guards.py
└── .claude/
    ├── agents/              ← code_planner / code_maker / code_validator / unit_tester
    ├── commands/harness-1d-cfd.md
    └── rules/               ← scoped rules (execution-model, edit-boundaries, results-conventions)
```

## Active solver

- Entry point: `solver/five_eq_IMEX/main.py::solve(eos1, eos2, W0, dx, t_end, …)`
  - primitive W = (α₁, T₁, T₂, u, p); conservative U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁)
  - time integration default `be1` (single-stage BE, Abgrall-consistent). **`ARS222` is banned** (unstable, ρ(A)≈9.02).
  - key args: `imp_dissipation=0.02` (biharmonic), `schur=True`, `pe_projection_mode`, `pure_branch`, `alpha_pure_tol`.
- EOS: `solver/five_eq_IMEX/eos_facade.py::make_eos('ideal'|'sg'|'nasg', …)` — wraps He2024 Phase-1 output.
- 2D/3D: `main_2d.py`, `main_3d.py`, `nd_solver.py` (see `docs/nd_validation_plan.md`).

### Governing equations

```
∂(αᵢρᵢ)/∂t + ∂(αᵢρᵢu)/∂x = 0
∂(ρu)/∂t + ∂(ρu²+p)/∂x = 0
∂(ρE)/∂t + ∂((ρE+p)u)/∂x = 0
∂αᵢ/∂t + u·∂αᵢ/∂x = (αᵢ+Dᵢ)∂u/∂x   (Allaire: Dₖ=0; Kapila: D₁ = α₁α₂(ρ₂c₂²−ρ₁c₁²)/(α₂ρ₁c₁²+α₁ρ₂c₂²))
```

## Rules & preferences

- **Code style**: match surrounding Python. Stack is numpy/scipy/matplotlib only — do not add heavy deps. The Python here is a deterministic oracle: keep numerics reproducible.
- **Communication**: Korean, caveman full (see Response rule). Terse reports; verbatim commands/paths.
- **Edit boundaries**: editable = `five_eq_IMEX{,_v2}/`, `tests/`, `docs/`, `results/`, `.codex-loop/`; **frozen** = `solver/He2024/`, `validation/`. Detail → `.claude/rules/edit-boundaries.md`.
- **Result artifacts**: overwrite one fixed PNG path, never per-round filenames. Detail → `.claude/rules/results-conventions.md`.

## Capability boundaries (what you CAN do here)

- **Gate runner** (from repo root): `cd .. && python3 .agents/skills/benchmark-validate/scripts/run_and_compare.py` (add `--include-07` for the 07 gate).
- **Agent pipeline**: `.claude/agents/` (code_planner → code_maker → code_validator/unit_tester); harness command `/harness-1d-cfd`.
- **Repo-root tooling**: paper-search MCP (`../.agents/tools/paper-search-mcp`), paper PDFs (`../papers/`), benchmark-validate skill (`../.agents/skills/`).
- **Internal specs to consult**: `docs/five_eq_all_mach_plan.md` (roadmap SoT), `docs/SOLVER_DESIGN_GUIDE.md`, `docs/five_eq_IMEX_current_formulation.md`, `validation/INDEX.md`.

## Lessons learned / guardrails

- **`ARS222` unstable** (ρ(A)≈9.02) → banned; use `be1` only.
- **well_balanced α-jump blows up in ~10 steps** despite 1-step ρ(A)≈1.0008 — nonlinear instability, still open.
- **07-B acoustic 0/3 FAIL** (open, top priority). Argon-Air closest: `Liu=0.591` (single item over 0.50). Air-Water pressure blows up under `mode=contact`.
- **He2024 __init__ trap**: import EOS only through `he2024_compat.py`; never `import solver.He2024` (keep `solver/__init__.py` empty).
- **No per-round result filenames** — always overwrite the fixed path.

## Validation status (snapshot 2026-07-02 — `docs/` is the live source of truth)

Regression gates all PASS: uniform_flow / amplification_matrix (be1 ρ(A)≈1.0008 <1.005) / transport_eigenmode / 02A_nasg (p_rel_linf≈2.8e-15). Open issues tracked in `docs/five_eq_all_mach_plan.md`.

## Run commands (WSL)

```bash
# unit tests (run from any cwd — __file__ bootstrap)
python3 tests/test_uniform_flow.py
python3 tests/test_amplification_matrix.py

# 02/07 validation driver
python3 results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01
python3 results/run_02_07_five_eq_imex.py --case 07 --n07 50 --cfl07 0.1 --imp-dissipation 0.02 --pe-projection-mode contact --max-steps07 1000
```

## Roadmap

Single source of truth = `docs/five_eq_all_mach_plan.md`. Current: Phase 0-3 done (be1 adopted over unstable ARS222); Phase 4 (generalized Rhie-Chow), 7 (APEC), 8 (positivity), 10 (THINC-BVD) partial; Phase 5 (SLAU2), 6 (ACID) TBD as 07-B recovery candidates.

## Execution model

The main session is **Advisor**, subagents are **Workers**. Full delegation + verification protocol → `.claude/rules/execution-model.md`.

## Agent pipeline

`code_planner` (opus, plan → `results/plan_report.md`) → `code_maker` (edits only, no run → `results/fix_report.md`) → `unit_tester` / `code_validator` (run + judge → `results/qa_report.md`). Author and verify stay in separate passes — no self-approve in one context.
