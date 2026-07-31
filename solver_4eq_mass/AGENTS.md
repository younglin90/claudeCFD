# AGENTS.md - solver_4eq_mass workspace (MASS-FRACTION TRANSPORT VARIANT)

> **This workspace is the mass-fraction-transport variant, forked from `solver_denner` at the
> published-paper state.** The reference workspace `solver_denner` holds that paper state and
> must NOT be edited from here. Everything below still describes the same solver; the one
> addition is an opt-in transported-variable switch.
>
> **Switching the transported colour function**
>
> | | env | transported variable | status |
> |---|---|---|---|
> | volume fraction (baseline) | `ACID_YADV` unset | `alpha` | default; **byte-identical** to the paper build, 19/19 PASS |
> | mass fraction (variant) | `ACID_YADV=1` | `Y = alpha*rho_a/rho` | 15/19 PASS; `alpha` is recovered algebraically for all output |
>
> ```bash
> DENNER_ACID=1                ./build-cpp/cpp/denner_1d/denner1d_validate   # alpha path
> DENNER_ACID=1 ACID_YADV=1    ./build-cpp/cpp/denner_1d/denner1d_validate   # Y path
> ```
>
> Case definitions, reference solutions and pass/fail gates are unchanged and still expressed in
> `alpha`; only the solver's internal transported variable changes. Full derivation, A/B
> measurements and verdict: **`docs/YADV_RESEARCH.md`**.

Isolated Denner C++ solver workspace: a faithful Denner ACID (acoustically-conservative
interface discretisation, JCP 367 (2018) 192-234) pressure-based all-Mach 4-equation
two-phase FVM solver with NASG EOS.

## Communication

Inherit the parent `claudeCFD/CLAUDE.md` rules: caveman full mode, respond in Korean.

## Project Map

| Path | Role |
|------|------|
| `cpp/denner_1d/src/acid.cpp` | Core ACID solver: `compute_R` residual, coupled Newton, analytic Jacobian, globalization (~1600 lines) |
| `cpp/denner_1d/src/cases.cpp` | 10 benchmark case definitions + exact NASG Riemann references |
| `cpp/denner_1d/src/eos.cpp` | NASG equation of state |
| `cpp/denner_1d/src/numerics.cpp` | Numerics helpers |
| `cpp/denner_1d/src/solver.cpp` | Solver driver |
| `cpp/denner_1d/src/validation.cpp` | Pass/fail gates |
| `cpp/denner_1d/src/png.cpp` | PNG output |
| `cpp/denner_1d/include/denner1d/*.hpp` | Headers: `acid cases eos numerics png solver types validation` |
| `cpp/denner_1d/include/denner1d/types.hpp` | `SolverConfig` — ONE global parameter set (cfl 0.45, max_steps); per-case numeric knobs are BANNED (only problem definition differs per case) |
| `cpp/denner_1d/apps/denner1d_validate.cpp` | Runs cases, prints JSON metrics + pass/fail |
| `cpp/denner_1d/apps/denner1d_dump.cpp` | Dumps `x,alpha,p,u,rho,p_ref,u_ref,rho_ref` CSV to stdout |
| `cpp/denner_1d/apps/denner1d_run.cpp` | Single-case run |
| `cpp/denner_1d/tests/denner1d_unit.cpp` | Unit tests |
| `cpp/denner_1d/README.md` | Case notes |
| `build-cpp/cpp/denner_1d/` | Build output (binaries live here) |
| `validation/1D/` | 17 markdown case specs — source of truth |
| `solver/denner_1d/` | Python oracle — preserved, DO NOT DELETE |

Current state: 10/10 validation cases pass. case01 is machine-exact (`linf_p=0`).

## Build & Run

Run from Windows via `wsl.exe -d ubuntu bash -lc '...'` (WSL2 Ubuntu):

```bash
# build (reconfigure with: cmake -S . -B build-cpp)
cmake --build build-cpp -j8

# run validation (case ids: 01,02,04,05,07,13,14,15,24,25,26,27,28,30,31,33,34 (29,32 excluded: documented blockers))
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate --only 07,25
# prints per-case JSON: pass, corr_p, l2_p, amp_ratio_p, hf_p, ...

# dump a case profile to CSV (positional case id; capture via subprocess, not shell '>')
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_dump 25
```

## Rules

- Modify Denner C++ code under `cpp/denner_1d/` — NOT the Python oracle
  (`solver/denner_1d/`) or repo-root legacy.
- Keep `validation/1D/` specs as the source of truth.
- Validation PASS criteria must stay quantitative — never finite-only or visually
  permissive. Wave/shock cases must require field-wise correlation, scaled L2/L1
  error, amplitude ratio, and peak/location checks. If a case looks very different
  from the reference, that is a FAILURE — fix the solver or the reference
  consistency; do NOT weaken thresholds.
- Results PNGs overwrite a fixed path (no per-round filenames) and print
  `Plot saved: ...`.

## Rules pointers

- `.claude/rules/execution-model.md` — Advisor/Worker execution model.
- `.claude/rules/denner-pitfalls.md` — known pitfalls / guardrails.
