# Validation Index — claudeCFD Phase 1 Testing

**Project**: All-speed 1D multi-component compressible Euler solver  
**Current Status**: **FAIL (Round 7)** — Waiting for code fixes  
**Test Case**: Phase 1 — 1D Water-Air Advection (Abgrall)

---

## Quick Links

### For code_maker (Code Fixes)
1. **[CODE_FIXES_REQUIRED.md](results/CODE_FIXES_REQUIRED.md)** ← START HERE
   - Exact line numbers and code to change
   - 3 changes required in `solver/solver_1d.py`
   - ~40 lines, 30–45 minutes

2. **[VALIDATION_ROUND7_SUMMARY.md](results/VALIDATION_ROUND7_SUMMARY.md)**
   - Full analysis of why Phase 1 is failing
   - Root cause analysis (3 numerical defects)
   - Expected behavior vs. observed

### For code_validator (Validation Testing)
1. **[qa_report.md](results/qa_report.md)** ← Current round status
   - Pass/fail criteria for each test
   - Detailed breakdown of failures
   - Instructions for next round

2. **[results/1D/phase1_abgrall_water_air/](results/1D/phase1_abgrall_water_air/)**
   - Diagnostic plots (pressure/velocity/temperature profiles)
   - Detailed comparison graphs
   - Error metrics visualization

### Test Execution
- **Test script**: `results/tmp_test.py`
- **Specification**: `validation/1D/phase1_abgrall_water_air.md`
- **Run command**: `python3 results/tmp_test.py`

---

## Current Status (Round 7)

| Criterion | Result | Measured | Required | Status |
|-----------|--------|----------|----------|--------|
| Convergence (100 iter) | 100 steps, no divergence | ✓ | ✓ | **PASS** |
| Pressure uniformity | max\|(p-p₀)/p₀\| = 0.9999 | 9.9999e-01 | < 1e-02 | **FAIL** |
| Velocity uniformity | max\|u-u₀\| = 2.26 m/s | 2.26e+00 | < 1e-02 | **FAIL** |
| Energy conservation | \|(E-E₀)/E₀\| = 5.2e-3 | 5.2e-03 | < 1e-02 | **PASS** |
| Species bounds | 0 ≤ Yᵢ ≤ 1 | ✓ | ✓ | **PASS** |

**Overall**: **FAIL** (cannot proceed to Phase 2/3)

---

## Problem Summary

The Phase 1 test case is designed to validate that uniform flow fields are preserved during advection (a trivial problem with known exact solution: u = 1.0 m/s, p = 1e5 Pa everywhere, forever).

**Result after 100 time steps**:
- Pressure collapsed from 1e5 Pa to 1 Pa in some cells (9 orders of magnitude!)
- Velocity oscillates from -0.88 to +3.26 m/s (should be constant at 1.0)
- Temperature spikes to 330 K (should be 300 K)
- Physics is completely broken, though energy is somehow conserved

**Root Cause**: The Backward Euler implicit time integration solver has three critical numerical instabilities:

1. **Jacobian computed via forward finite differences** — numerically inaccurate for multi-scale problems
2. **Line search condition too permissive** — accepts steps that violate physical bounds
3. **No density positivity check in Newton loop** — invalid states propagate

---

## Round History

| Round | Status | Main Issue | Notes |
|-------|--------|------------|-------|
| 4 | FAIL | Negative pressure discovered | Initial validation |
| 5 | FAIL | Root cause identified: Jacobian instability | Specific fixes proposed |
| 6 | FAIL | Fixes not applied to code | Problem persists unchanged |
| **7** | **FAIL** | Same defects, code not modified | Identical output as Round 4, 5, 6 |

**Key observation**: The solver code has not been modified between Rounds 4–7. The same bugs produce the same failures.

---

## Next Steps

### Phase 1: code_maker Applies Fixes

**Time estimate**: 45 minutes

1. Open `solver/solver_1d.py`
2. Apply **Change 1**: Replace `_numerical_jacobian_fd()` (lines 200–219)
   - Switch from forward difference to central difference
   - Better numerical stability for multi-scale variables

3. Apply **Change 2 & 3**: Replace Newton iteration (lines 259–301)
   - Implement Armijo line search condition
   - Add density positivity check
   - Only accept steps that pass both checks

4. Quick test (5 minutes):
   ```bash
   python3 << 'EOF'
   from solver.solver_1d import run_solver_1d, build_water_air_ic
   import numpy as np
   case = build_water_air_ic()
   result = run_solver_1d(case)
   # Pressure/velocity should be within tolerance
   EOF
   ```

### Phase 2: code_validator Runs Round 8

**Time estimate**: 30 minutes

1. Confirm code_maker has applied all three fixes
2. Run: `python3 results/tmp_test.py`
3. Check that all 5 criteria PASS
4. If PASS: Create report and proceed to Phase 2
5. If FAIL: Diagnose and report additional issues

---

## File Structure

```
claudeCFD/
├── solver/
│   ├── solver_1d.py                    ← NEEDS FIXES
│   ├── flux_allspeed.py                (working)
│   ├── eos/
│   │   ├── nasg.py                     (working)
│   │   └── ideal.py                    (working)
│   └── ...
├── validation/
│   └── 1D/
│       └── phase1_abgrall_water_air.md ← Test specification
├── results/
│   ├── tmp_test.py                     ← Test script
│   ├── qa_report.md                    ← Current status
│   ├── CODE_FIXES_REQUIRED.md          ← For code_maker
│   ├── VALIDATION_ROUND7_SUMMARY.md    ← For code_validator
│   └── 1D/
│       └── phase1_abgrall_water_air/
│           ├── round6_profiles.png     ← Diagnostic plot
│           └── ...
├── CLAUDE.md                           ← Project rules
└── VALIDATION_INDEX.md                 ← This file
```

---

## Key Documents

| Document | Purpose | Audience | Length |
|----------|---------|----------|--------|
| **CODE_FIXES_REQUIRED.md** | Exact code changes with line numbers | code_maker | 8.4 KB |
| **VALIDATION_ROUND7_SUMMARY.md** | Full analysis and root cause | code_validator | 12 KB |
| **qa_report.md** | Current round pass/fail status | Both | 8.5 KB |
| **CLAUDE.md** | Project rules and constraints | Both | 5 KB |
| **validation/1D/phase1_abgrall_water_air.md** | Test specification | Both | 2 KB |

---

## Testing Workflow

### Running the Test

```bash
cd /home/younglin90/work/claude_code/claudeCFD
python3 results/tmp_test.py
```

### Expected Output (After Fixes)

```
OVERALL VERDICT: PASS

1. Convergence (100 iter completed):
   n_steps=100, converged=True
   Status: PASS

2. Pressure uniformity max|Δp/p₀|:
   < 1e-02
   Status: PASS

3. Velocity uniformity max|Δu|:
   < 1e-02 m/s
   Status: PASS

4. Energy conservation |ΔE/E₀|:
   < 1e-02
   Status: PASS

5. Species bounds 0 ≤ Yᵢ ≤ 1:
   Status: PASS
```

### Current Output (Before Fixes)

```
OVERALL VERDICT: FAIL

1. Convergence (100 iter completed):
   n_steps=100, converged=True
   Status: PASS

2. Pressure uniformity max|Δp/p₀|:
   9.999900e-01  ← HUGE FAILURE
   Status: FAIL

3. Velocity uniformity max|Δu|:
   2.260385e+00 m/s  ← HUGE FAILURE
   Status: FAIL

4. Energy conservation |ΔE/E₀|:
   5.213832e-03
   Status: PASS

5. Species bounds 0 ≤ Yᵢ ≤ 1:
   Status: PASS
```

---

## Constraints

Per **CLAUDE.md**:
- Code modifications must be in `solver/` only (not validation/ or results/)
- No changes to test specification or PASS criteria
- Must use only Python + NumPy/SciPy (no C extensions)
- Backward Euler implicit integration is mandatory

---

## Questions?

- **For code_maker**: See `CODE_FIXES_REQUIRED.md` (exact line numbers and code)
- **For code_validator**: See `VALIDATION_ROUND7_SUMMARY.md` (full analysis)
- **General**: See `CLAUDE.md` (project rules)

---

**Last updated**: 2026-04-08  
**Validator**: code_validator  
**Status**: Awaiting code_maker fixes  
**Est. Round 8 completion**: 1 hour after fixes applied

