# Request to Claude — Find What Is Wrong and Propose Fixes (five_eq_IMEX)

You can access this repository code directly.
Please diagnose **what is still wrong** in `solver/five_eq_IMEX/` and propose concrete fixes.

## 1) Scope and constraints

- Target module: `solver/five_eq_IMEX/`
- Do not modify unrelated legacy paths unless strictly needed for reference.
- Keep `tests/test_uniform_flow.py` passing byte-exact.
- Focus on root-cause fixes, not parameter fishing.

## 2) Current measured state (latest)

### Spectral stability
From `python3 tests/test_amplification_matrix.py`:

- `ARS222 raw` ρ(A) = `8.1142`
- `be1 raw` ρ(A) = `1.0457`
- `be1 schur=True` ρ(A) = `1.0455`
- `be1 pe_correct=True` ρ(A) = `1.0471`

So be1 is much improved, but still slightly unstable (`ρ(A) > 1`).

### Dominant mode
From `python3 tests/test_transport_eigenmode.py`:

- dominant mode still pressure-dominant, `|λ| = 1.0457` (not explosive like before, but still > 1)

### 02-A run
From `python3 results/run_02A_new.py`:

- still fails with NaN at step `193` (improved from earlier failures around 18/174, but not solved)
- plot regenerated: `results/1D/02_A/diff_vs_exact.png`

### 07 run
From `run_07` in `results/run_01_07_validated.py`:

- subcase 1 FAIL
- subcase 2 FAIL
- subcase 3 FAIL
- overall FAIL
- plot regenerated: `results/all_26_plots/case_07_result.png`

## 3) Recent code changes already applied

Please review these first:

- `solver/five_eq_IMEX/residual.py`
  - implicit pressure-work changed to split form:
  - `div_pu = p * div_u + u * grad_p`

- `solver/five_eq_IMEX/pe_correction.py`
  - added `apply_pe_tangent_projection` (full 5-equation tangent projection)

- `solver/five_eq_IMEX/time_integrator.py` (`be1_step`)
  - `pe_project_explicit=True` default
  - `explicit_force_lo=True` default (explicit residual uses `force_lo=True`)
  - `imp_dissipation` default changed to `0.5`

These changes reduced ρ(A) from ~3.56 to ~1.046, but did not fully fix long-time robustness.

## 4) What I need from you (strict)

Please provide:

1. **Root-cause diagnosis** (equation-level + code-level mapping)
   - Why does `|λ|` remain slightly > 1?
   - Why does 02-A still hit NaN around step 193?
   - Which operator/split term is still mathematically inconsistent or under-damped?

2. **Specific wrong code locations**
   - file + function + line range
   - what is wrong there and why

3. **Concrete fix plan (PR-sized)**
   - prioritize minimal invasive sequence (PR1/PR2/...)
   - include expected impact on:
     - `ρ(A)` target: `< 1.02` (or at least `< 1.00 + O(dt)`)
     - 02-A survival: `>= 1000 steps`, finite

4. **Patch proposal level details**
   - exact formulas and discrete stencil changes
   - if Schur block is incomplete/incorrect, show the corrected reduced system
   - if positivity layer is missing, show exactly where to enforce and with what limiter/update ordering

5. **Validation checklist**
   - commands to run
   - pass/fail criteria

## 5) Strong preference on solution direction

Do **not** suggest broad hyper-parameter tuning first.

Prefer structural fixes such as:

- operator-consistent pressure-velocity coupling,
- consistent Schur reduction and back-substitution,
- positivity-preserving conservative update ordering,
- explicit/implicit split correction where mathematically justified.

## 6) Commands you should use for verification

```bash
python3 tests/test_uniform_flow.py
python3 tests/test_amplification_matrix.py
python3 tests/test_transport_eigenmode.py
python3 results/run_02A_new.py
python3 -c "import results.run_01_07_validated as r; print(r.run_07())"
```

## 7) Output format I want from you

- A) Root cause summary (max 15 lines)
- B) Critical findings list (severity order)
- C) PR plan (PR1/PR2/PR3 with file-level scope)
- D) Minimal patch snippets for PR1
- E) Expected metric changes table (`ρ(A)`, 02-A step count, 07 status)

