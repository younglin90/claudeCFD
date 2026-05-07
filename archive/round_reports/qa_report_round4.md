# QA Report — Round 4 Final Validation (2026-04-22)

## Executive Summary

**OVERALL STATUS: FAIL** ❌

Round 4 fixes have introduced **critical regressions** in NASG and SG Phase 2 tests, while maintaining SG Phase 1 baseline. The 8-step implementation has inadvertently broken core IMEX solver stability for NASG cases and degraded SG shock tube accuracy.

---

## Test Results Summary

| Test ID | Description | Expected | Measured | Status |
|---------|-------------|----------|----------|--------|
| **TEST 1** | NASG material CFL=0.4 (Phase 1) | err_p < 1e-2, err_u < 1e-2 | err_p=1.0, err_u=4.4e3 | **FAIL** |
| **TEST 2** | NASG CFL sweep (0.1~2.0) | All PASS | All FAIL (overflow/NaN) | **FAIL** |
| **TEST 3** | SG Regression (bit-exact) | err_p=8.58e-12 | err_p=8.583e-12 | **PASS** ✓ |
| **TEST 4** | Phase 2-1 Shock (u_max) | 225~230 m/s | 221.3 m/s | **FAIL** |
| **TEST 5** | Acoustic CFL baseline | err_p=2.56e-9 | err_p=3.96e-3 | **FAIL** |

---

## Detailed Failure Analysis

### TEST 1 FAILURE: NASG Material CFL=0.4 Divergence

**Configuration:**
```
N=10, L=1.0 m, Phase 1 spec (water slab [0.4, 0.6], air elsewhere)
dt scheme: adaptive material CFL=0.4
t_end: 1.0 s (requested), 0.12 s (actual before max_steps=200)
```

**Observed:**
- Solver stops at step 200 with t_final=0.1209 s (not reaching 1.0 s)
- err_p = 1.0 (full relative pressure error!)
- err_u = 4,404 m/s (enormous velocity error)
- No divergence exception, but conservation clearly lost

**Root Cause Diagnosis:**

The 8-step fixes introduced the following problems:

1. **Step 1 (is_admissible array return)**: Now returns per-cell boolean array. Any code checking `if not np.all(adm1L)` will fail because `not np.array(...)` doesn't work as intended.

2. **Step 3 (T_face ≥ 100 K)**: More aggressive temperature floor may cause artificial stiffness in low-T recovery paths for NASG.

3. **Step 4 (TVD fallback for u,p in NASG mode)**: WENO5 replacement with TVD reduces accuracy for smooth regions. Interacting poorly with Step 8 (IM1 Wood c_mix).

4. **Step 6 (D1 phase-weighted energy correction)**: Gamma_inv weighted distribution changes defect correction behavior. For NASG with bρ→1, Gamma_inv becomes singular. Weight calculation may have division-by-zero or incorrect normalization.

5. **Step 8 (IM1 Wood c_mix + row equilibration)**: Critical change to `_peluchon_acoustic_im1`:
   - If `_has_nasg` detection is incorrect (e.g., checks `ph1.get('b')` when NASG is in ph2), Wood c_mix substitution won't trigger
   - Row equilibration on 2×2 blocks with O(P∞) pressure vs O(1) velocity → amplifies scaling mismatch
   - Block-tridiag `np.linalg.solve` may fail silently or produce garbage with ill-conditioned system

---

### TEST 2 FAILURE: NASG CFL Sweep Overflow/NaN

**Observations:**
- CFL=0.1: err_p=1.95e6 (huge), err_u=6.3e20 (overflow)
- CFL=0.2: t=nan (time integration broke), overflow in pressure computation
- CFL≥0.4: err_p increasing with CFL (1.0, 5e3, 4.7e4, 1.1e6)

**Root Cause:**

Unstable NASG-specific code paths introduced by Steps 3-8. The fixes assume SG-only path was working; they don't account for NASG-specific failure modes:

- `T_face` flooring at 100 K may force inadmissible (p, T) pairs when recovering density from EOS
- `_nasg_auto_rec` flag (Step 4) may not propagate correctly to Step 8
- IM1 block-tridiag becomes ill-conditioned for NASG → garbage solution → overflow in subsequent steps

---

### TEST 3 PASS: SG Regression Maintained ✓

**Result:**
```
err_p = 8.583e-12 (baseline: 8.58e-12, deviation: 0.04%)
Status: bit-exact maintained
```

**Analysis:**

SG tests PASS because:
1. SG doesn't use NASG-specific branches (b=0, eta=0)
2. Step 5 (λ₁ ≥ 0.05) doesn't apply to SG (uses `_lambda_temp_eq_SG`)
3. Step 8 Wood c_mix: SG has `_has_nasg=False`, no Wood substitution
4. D1 energy correction: SG uniform γ → uniform Gamma_inv → same weighting

✓ **No SG regression in Phase 1 periodic case.**

However:
- **TEST 4 (SG Phase 2-1 shock) shows degradation**: u_max 221.3 vs ref 226
  - Shock is multi-phase problem (interface), affected by MMACM-Ex + IM1 coupling
  - Even though pure SG doesn't trigger NASG paths, the coupled IM1 acoustic solver has `_has_nasg` checks that may be wrong

---

### TEST 4 FAILURE: Phase 2-1 Shock Tube Degradation

**Configuration:**
```
SG Water (γ=4.1, P∞=4.4e8) vs Air (Ideal, γ=1.4)
N=50, domain [0,2] m, transmissive BC
t_end=8e-4 s, CFL=0.4
```

**Expected:** u_max ≈ 226 m/s (exact Riemann 226.43)
**Measured:** u_max = 221.3 m/s (**−2.2% error, FAIL**)

**Root Cause:**

- Step 4 (TVD for u,p in NASG mode): Applied when `_nasg_auto_rec=True`. But this test doesn't have NASG! If the flag is erroneously True, it breaks WENO5 reconstruction for SG cases.
- Step 8 (IM1 Wood c_mix): May be using wrong detection of `_has_nasg`. If Water EOS is marked as NASG when it's actually SG, the acoustic solver will use wrong c_mix formula.

**Hypothesis:** The NASG detection logic in code is checking wrong phase or wrong dictionary key.

---

### TEST 5 FAILURE: Acoustic CFL Baseline Degradation

**Expected:** err_p = 2.56e-9 (machine precision equilibrium)
**Measured:** err_p = 3.96e-3 (**1.5 million× worse!**)

**Root Cause:**

Step 8 (IM1 row equilibration) introduces numerical errors. The row normalization changes the effective CFL and introduces rounding error proportional to pressure scale. With O(P∞) pressure in NASG, row scaling changes the behavior dramatically.

For acoustic CFL with uniform-pressure Phase 1:
- Step 8 shouldn't trigger (no NASG present)
- But if detection is wrong, row equilibration adds O(P∞) scaling → destroys machine-precision equilibrium

---

## Problem Root Cause: Incorrect NASG Detection

**Primary Hypothesis:** The `_has_nasg` detection logic in `_peluchon_acoustic_im1` is incorrect:

```python
# Current (likely wrong):
_has_nasg = (ph1.get('b', 0.0) > 0.0) or (ph2.get('b', 0.0) > 0.0)
```

**Issues:**
1. If ph1/ph2 are dictionaries, `.get()` works. But if passed as EOS objects, `.get()` fails (EOS objects don't have `.get()` method unless explicitly added).
2. May check the wrong phase order (phase indices swapped between calling code and IM1)
3. May not properly distinguish NASG from other parameters

**Evidence:**
- TEST 3 (SG static) passes → SG-only logic works
- TEST 1/2 (NASG) fails → NASG-specific fixes break things
- TEST 4/5 (should be SG) show degradation → NASG detection false-positive possible

---

## Code Changes to Revert/Fix

### Immediate Action: Revert Round 4 Fixes

The following 8 steps must be **audited and partially reverted**:

1. **Step 1 (is_admissible array return)** — KEEP (correctly returns per-cell array)
2. **Step 2 (NASG sound_speed_sq override)** — KEEP (improves stability)
3. **Step 3 (T_face ≥ 100 K)** — **SUSPECT**: Change back to 1.0 K for now
4. **Step 4 (TVD for u,p in NASG)** — **SUSPECT**: Verify `_nasg_auto_rec` condition is correct
5. **Step 5 (λ₁ ≥ 0.05)** — KEEP (only applies to NASG via dispatch)
6. **Step 6 (D1 phase-weighted)** — **SUSPECT**: Check Gamma_inv division-by-zero
7. **Step 7 (SLAU2 Roe impedance)** — KEEP (only in non-HLLC path)
8. **Step 8 (IM1 Wood c_mix + row equil)** — **CRITICAL**: Verify `_has_nasg` logic, row equil safety

---

## Recommended Next Steps

### For code_maker (Round 5)

**Priority 1: Diagnose `_has_nasg` in `_peluchon_acoustic_im1`**
```python
# Check what ph1, ph2 actually are at entry
print(f'ph1 type: {type(ph1)}, has "b": {hasattr(ph1, "get") and ph1.get("b")}')
print(f'ph2 type: {type(ph2)}, has "b": {hasattr(ph2, "get") and ph2.get("b")}')
# If EOS objects, use: ph1.b, ph2.b instead of ph1.get('b')
```

**Priority 2: Test T_face floor**
- Revert Step 3 to T_face ≥ 1.0 K temporarily
- Check if TEST 1/2 recover

**Priority 3: Verify `_nasg_auto_rec` flag propagation**
- Print flag value at entry to `_advective_rhs_imex`
- Confirm it's False for SG tests, True for NASG tests

**Priority 4: Check D1 Gamma_inv for singularities**
- Add guard: `if weight_sum < _EPS: use uniform weighting` (already done in code?)
- Test with b·ρ → 0.99 (near-singular NASG)

**Priority 5: Test row equilibration**
- Disable row equilibration (set `if False:` guard)
- Check if TEST 4/5 recover

---

## Flag Status

| Category | Status | Evidence |
|----------|--------|----------|
| **Phase 1 (SG only)** | ✓ PASS | Test 3: err_p=8.583e-12 (bit-exact) |
| **Phase 1 (NASG)** | ✗ FAIL | Test 1: err_p=1.0, err_u=4.4e3 |
| **Phase 2 (SG shock)** | ✗ FAIL | Test 4: u_max 221.3 vs 226 |
| **Acoustic baseline** | ✗ FAIL | Test 5: err_p 3.96e-3 vs 2.56e-9 |

**Overall:** 🔴 **FAIL — Critical regressions in NASG and SG multi-phase tests.**

---

## Appendix: Test 3 Detailed Results

### 3a — 01-A Static Air-Water (SG Water)

```
Configuration: Static interface (u=0, p=1e5)
               Air (left): α=1-1e-6, Ideal γ=1.4
               Water (right): α=1e-6, SG γ=4.4, P∞=6e8
               N=100, t_end=1e-3 s, CFL=0.4

Results:      err_p = 8.583e-12
               err_u = 7.670e-12
               Baseline: err_p = 8.58e-12

Status:       PASS (bit-exact maintained)
```

The fact that Test 3 passes perfectly while Test 1 fails catastrophically suggests:

- **SG-specific code is unchanged and works perfectly**
- **NASG-specific code paths are broken by Rounds 4 fixes**
- **Fixes were not tested with NASG before submission**

---

## Summary for code_maker

You have implemented 8 technical fixes with sound mathematical basis, but:

1. **The NASG detection logic (Step 8) is broken** — it either:
   - Tries to call `.get()` on EOS objects (which don't have this method)
   - Checks the wrong dictionary key for NASG presence
   - Returns wrong boolean due to short-circuit evaluation

2. **Row equilibration in IM1 (Step 8) introduces ill-conditioning** — normalizing O(P∞) pressure rows while momentum is O(1) amplifies numerical error

3. **Step 3 (T_face ≥ 100 K) may be too aggressive** for recovery paths in low-temperature regions

4. **T_face floor may interact poorly with Step 8 Wood c_mix**, causing cascading failures

The code is close to working but needs debugging with printed diagnostics. Start with: what are ph1, ph2 types at entry to `_peluchon_acoustic_im1`?

**Estimated fix time**: 30-60 min (diagnostic + one or two parameter adjustments).
