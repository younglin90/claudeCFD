# CODE VALIDATOR — Validation Report (Round 2)

**Date**: 2026-03-31 16:46 UTC  
**Status**: AWAITING code_maker FIX  
**Validator Agent**: code_validator (passive testing mode)

---

## Validation Attempt Summary

code_validator successfully initialized validation pipeline and ran pre-flight checks on the CFD solver (APEC flux, multicomponent EOS) but **encountered 2 critical runtime bugs** that prevent proceeding with full 1D validation.

### Results
- **1D Case A** (Ideal Gas, Smooth Interface): FAIL — Negative density at t≈0.004s
- **1D Case C** (Air-Water NASG): FAIL — Shape mismatch in cons_to_prim
- **1D Case B** (SRK): NOT TESTED (blocked by Case A failure)

**Recommendation**: Fix bugs in solver and revalidate before 2D/3D testing.

---

## Bug Report

### Bug #1: Negative Density in TVD-RK3 Integration

**File**: `solver/solve.py` (line 252, `_tvd_rk3_step`)

**Error**:
```
ValueError: cons_to_prim: non-positive density rho=-0.0005501362868886371
```

**Test Case**:
- Case A (Ideal Gas): 2-species (γ₁=1.4/M₁=28, γ₂=1.66/M₂=4)
- Domain: [0, 1], periodic BC
- Grid: 101 cells, Δx = 1/101
- Time: TVD-RK3, CFL=0.6, t_end=0.5s
- Initial: tanh interface at x=0.5, u₀=1.0, p₀=0.9 (non-dimensional)

**Failure Point**: Step 800, t ≈ 0.004s, cells m ≈ 20-50

**Root Cause (Hypothesis)**:
The TVD-RK3 temporal integration is producing negative density at intermediate stages. The most likely culprits:

1. Flux divergence computation producing overshooting:
   ```
   RHS[m] = -(F_right - F_left) / dx
   ```
   If |RHS[m, 0]| (density RHS) is very large, then:
   ```
   ρ^(1) = ρ^n + dt * RHS[m, 0] < 0  (at Stage 1)
   ```

2. APEC flux returning incorrect magnitude or sign
3. Periodic boundary condition corrupting ghost cells
4. Very small initial densities (ρ ≈ 1e-5 kg/m³) amplifying numerical errors

**Impact**: BLOCKS Case A validation completely.

---

### Bug #2: Shape Mismatch in cons_to_prim for Multi-species

**File**: `solver/utils.py` (line 540, `cons_to_prim`)

**Error**:
```
ValueError: cons_to_prim: U has 3 elements, expected 4 for 2 species
(U=[rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{N-1}])
```

**Test Case**:
- Case C (Air-Water): 2-species system (Air: IdealGas, Water: NASG)
- Grid: 50 cells
- Failure: First flux computation in initialization

**Root Cause (Hypothesis)**:
Array indexing or boundary condition application is corrupting the shape of the state vector. For N=2 species, `U` should be:
```
U = [rho, rho*u, rho*E, rho*Y_1]  (4 elements)
```

But cons_to_prim is receiving U with only 3 elements. This could happen if:

1. Initial U is constructed incorrectly
2. `_apply_bc` (boundary condition) is returning wrong shape
3. Array slicing in flux.py is accidentally reducing dimensionality

**Impact**: BLOCKS Case C validation completely.

---

## Required Fixes

### Priority 1 (CRITICAL): Negative Density Issue

**Locations to modify**:
1. `solver/solve.py` — _tvd_rk3_step, _forward_euler_step
2. `solver/flux.py` — apec_flux
3. `solver/utils.py` — cons_to_prim

**Actions**:
1. Add defensive checks to ensure ρ > 0 after each TVD-RK3 stage:
   ```python
   # After Stage 1, 2, 3
   U1[:, 0] = np.maximum(U1[:, 0], 1e-30)  # Ensure ρ > 0
   ```

2. Add debug logging to identify exact flux divergence magnitude:
   ```python
   # In _spatial_rhs, after computing F_left, F_right
   print(f"F_left[0]={F_left[0]:.3e}, F_right[0]={F_right[0]:.3e}, "
         f"divergence={-(F_right[0]-F_left[0])/dx:.3e}")
   ```

3. Verify flux values are finite:
   ```python
   assert np.all(np.isfinite(F_left)), f"Non-finite F_left at step {step}"
   assert np.all(np.isfinite(F_right)), f"Non-finite F_right at step {step}"
   ```

4. Review APEC flux formula against `docs/APEC_flux.md` to ensure implementation is correct.

### Priority 2 (HIGH): Shape Mismatch Issue

**Locations to modify**:
1. `solver/solve.py` — _spatial_rhs, _apply_bc
2. `solver/flux.py` — apec_flux
3. `solver/utils.py` — cons_to_prim
4. `pipeline/quick_1d_test.py` — Initial U construction

**Actions**:
1. Add shape assertions in _spatial_rhs:
   ```python
   for m in range(N_cells):
       assert U_ext[m].shape[0] == 2 + len(eos_list), \
           f"Wrong shape: {U_ext[m].shape}"
   ```

2. Add logging in _apply_bc to verify output shape:
   ```python
   U_ext = _apply_bc(...)
   assert U_ext.shape == (N_cells + 2, 2 + len(eos_list)), \
       f"_apply_bc returned wrong shape: {U_ext.shape}"
   ```

3. Add shape validation in cons_to_prim with debug output:
   ```python
   if U.shape[0] != 2 + N:
       print(f"DEBUG: U.shape={U.shape}, N={N}, expected shape=(2+N,)=(4,)")
       print(f"DEBUG: eos_list length={len(eos_list)}")
       raise ValueError(...)
   ```

4. Verify initial U in quick_1d_test.py:
   ```python
   assert U.shape == (n_cells, 4), f"Initial U wrong shape: {U.shape}"
   ```

---

## Validation Artifacts

### Key Documents Generated

| File | Purpose |
|------|---------|
| `qa_report.md` | **PRIMARY** — Full QA report with test results, error analysis, and code_maker instructions |
| `fix_report.md` | Detailed debug analysis and recommended fixes |
| `VALIDATOR_STATUS.txt` | Protocol status and next actions |
| `README_VALIDATOR.md` | This document — Overview and communication |

### Test Script

**Location**: `pipeline/quick_1d_test.py`

This script contains two test functions:
1. `test_ideal_gas_simple()` — Case A pre-flight test
2. `test_water_air_simple()` — Case C pre-flight test

**To retest** (after fixes):
```bash
python3 pipeline/quick_1d_test.py
```

Expected output if fixes work:
```
SUMMARY
======
Ideal Gas: PASS
Water-Air: PASS

Total: 2/2 PASS
```

---

## Validation Protocol

### Current Phase: Phase 1 — Bug Identification ✓ COMPLETED

- ✓ Initialized validation infrastructure
- ✓ Ran pre-flight tests on Cases A and C
- ✓ Identified and documented 2 critical bugs
- ✓ Generated detailed reports for code_maker

### Next Phase: Phase 2 — Code Fixes (PENDING)

code_maker must:
1. Read `qa_report.md` (main report) and `fix_report.md` (debug details)
2. Fix bugs in solver code
3. Test with `python3 pipeline/quick_1d_test.py`
4. If PASS: `touch pipeline/code_ready.flag` to signal completion
5. Report status

### Phase 3: Revalidation (BLOCKED)

code_validator will:
1. Wait for `pipeline/code_ready.flag`
2. Rerun quick tests
3. If PASS: Run full 1D suite (Cases A, B, C)
4. Generate comprehensive results report

### Phase 4: Full 1D Validation (BLOCKED)

**Full Test Suite** (only if Phase 3 PASS):
- **Case A**: Ideal Gas, n_cells=501, t_end=8.0s (8 flow-throughs)
- **Case B**: SRK CH₄/N₂, transcritical, multiple grid resolutions
- **Case C**: Air/Water NASG, basic run

All results saved to `results/1D/{CaseName}/`

### Phase 5: 2D/3D Validation (BLOCKED)

**ABSOLUTE RULE**: No 2D/3D testing until 1D is 100% PASS.

---

## File Locations

### Critical Files for code_maker

| Path | Content |
|------|---------|
| `/home/younglin90/work/claude_code/claudeCFD/pipeline/qa_report.md` | **READ THIS** — Main QA report |
| `/home/younglin90/work/claude_code/claudeCFD/pipeline/fix_report.md` | Debug details and fix guidance |
| `/home/younglin90/work/claude_code/claudeCFD/pipeline/quick_1d_test.py` | Test script for validation |
| `/home/younglin90/work/claude_code/claudeCFD/pipeline/code_ready.flag` | **CREATE THIS** after fixes pass tests |

### Solver Code to Modify

| Path | Purpose |
|------|---------|
| `solver/solve.py` | Time integration (TVD-RK3, Forward Euler) |
| `solver/flux.py` | APEC flux computation |
| `solver/utils.py` | Primitive/conservative variable conversion |
| `solver/eos/*.py` | EOS implementations (unlikely to need changes) |

---

## How to Signal Completion

Once code_maker has fixed the bugs:

```bash
# After verifying fixes with:
python3 /home/younglin90/work/claude_code/claudeCFD/pipeline/quick_1d_test.py

# If output shows "PASS" for both cases, create the flag:
touch /home/younglin90/work/claude_code/claudeCFD/pipeline/code_ready.flag

# git commit (recommended):
cd /home/younglin90/work/claude_code/claudeCFD/
git add solver/
git commit -m "Fix negative density and shape mismatch bugs in TVD-RK3 and cons_to_prim"
```

---

## Contact / Escalation

If bugs are unclear or require clarification, code_maker can:
1. Review detailed error messages in `qa_report.md`
2. Check debug analysis in `fix_report.md`
3. Look at test code in `pipeline/quick_1d_test.py`
4. Add `verbose=True` to solver runs to see detailed logs

---

**Generated by**: code_validator (CFD solver validation agent)  
**Date**: 2026-03-31 16:46 UTC  
**Status**: Awaiting code_maker fixes
