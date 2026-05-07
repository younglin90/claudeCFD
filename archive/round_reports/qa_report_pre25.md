# QA Report — Test A Phase 1 NASG Full Validation

**Date**: 2026-04-22  
**Validator**: code_validator  
**Status**: **FAIL**

---

## Test Specification

**Case**: 02-A Test A — 2-phase Water-Air PE Advection (Abgrall 1996)

| Parameter | Value |
|-----------|-------|
| Domain | [0, 1] m, periodic BC |
| N | 10 cells |
| Water | NASG (x ∈ [0.4, 0.6], α=1) |
| Air | Ideal (elsewhere, α=0) |
| u₀, p₀, T₀ | 1.0 m/s, 1×10⁵ Pa, 300 K (uniform) |
| Duration | t_end = 1.0 s (100 fixed steps @ 0.01s/step) |
| EOS | NASG (γ=1.187, P∞=7.028e8, b=6.61e-4, kv=3610, η=-1.177788e6) |
| Solver | `solve_IMEX` (Peluchon IM1 + advective split) |
| Configuration | alpha_scheme='tvd', use_mmacm_ex=True, use_apec=True, use_compression=True |

---

## PASS Criteria

| Item | Criterion | Result | Status |
|------|-----------|--------|--------|
| 100 iteration completion | t_end ≈ 1.0 s | t_final = 0.781 s (78.2%) | ✗ FAIL |
| Pressure preservation | err_p = max\|(p−p₀)/p₀\| < 1e-2 | **0.1816** (18.16%) | ✗ FAIL |
| Velocity preservation | err_u = max\|u−u₀\| < 1e-2 | **0.0385** (3.85%) | ✗ FAIL |
| Energy conservation | \|ΔE/E\| < 1e-2 | 1.938e-14 | ✓ PASS |
| Volume fraction bounds | 0 ≤ α ≤ 1 | min=9.6e-6, max=0.997 | ✓ PASS |

---

## Execution Results

### Run 1: 100 steps (CFL-based adaptive dt)
- **Duration**: 0.00157 s (1.57 ms)
- **Steps**: 100
- **Final pressure**: p = [99,650 Pa, 100,000 Pa] (range 350 Pa)
- **Final velocity**: u = 1.000 m/s
- **err_p**: 3.532e-3 (0.35%)
- **err_u**: 1.590e-4 (0.016%)
- **Verdict**: PASS (but only 1.57 ms simulation)

**Issue**: CFL-based dt (~1.5e-5 s) → 100 steps = 1.57 ms ≠ 1.0 s

### Run 2: t_end=1.0s (10,000 max_steps)
- **Duration**: 0.781 s (78.1% of target)
- **Steps**: 10,000 (max reached)
- **Final pressure**: p = [117,800 Pa, 118,200 Pa] (range 400 Pa)
- **Final velocity**: u_min=0.9955 m/s, u_max=1.0385 m/s
- **err_p**: **0.1816** (18.16% — **CRITICAL**)
- **err_u**: **0.03847** (3.847% — **CRITICAL**)
- **Verdict**: FAIL

---

## Failure Analysis

### Pressure Evolution (Run 2)
```
Step 1:    p_range = [100000, 100000]
Step 100:  p_range = [99600, 100000]   (0.4% variance)
Step 300:  p_range = [94200, 195000]   ← SPIKE! (oscillation)
Step 500:  p_range = [70900, 1920000]  ← CATASTROPHIC (u_max=35.29 m/s)
Step 600:  p_range = [61500, 106000]   (recovery)
Step 700:  p_range = [1, 106000]       ← COLLAPSE (p→1 Pa)
Step 800:  p_range = [1, 114000]       (p_min still 1 Pa)
Step 900:  p_range = [35900, 3470000]  ← SPIKE (u_max=2.7 m/s)
Step 1000: p_range = [1, 162000]       (oscillation continues)
Step 2000: p_range = [99900, 101000]   (stabilization)
Step 5000: p_range = [114000, 118000]  (converged, but err_p=0.14)
Step 10000: p_range = [117800, 118200] (final err_p=0.1816)
```

### Velocity Evolution (Run 2)
```
Step 1-100:   u ≈ 1.0000 m/s (stable)
Step 300:     u_max = 1.2 m/s (oscillation onset)
Step 500:     u_max = 35.3 m/s ← CRITICAL INSTABILITY
Step 600:     u_max = 3.4 m/s (recovery)
Step 800:     u_max = 1.68 m/s
Step 1000:    u_max = 10.9 m/s ← SECONDARY SPIKE
Step 2000+:   u → 1.0±0.2 m/s (oscillations dampen)
Step 10000:   u → 1.0 m/s (but err_u=0.0385 remains)
```

### Root Cause

**Observation**: Early-time pressure spikes → solve_IMEX is NOT preserving PE (pressure-velocity equilibrium).

**Hypothesis** (unconfirmed, for code_planner):
1. **NASG EOS issue**: Direct formula for pressure may have cancellation errors or NASG parameter mismatch
   - NASG γ=2.35 in unit_tester, but spec uses γ=1.187 — **possible source**
2. **IM1 acoustic coupling**: Block-tridiagonal may be unstable for NASG stiffness (P∞=7e8)
3. **Peluchon split error**: Phase 2-1 (SG Water) succeeded, but NASG may require different acoustic damping
4. **Parameter sensitivity**: CFL=0.4 may be too aggressive for low-speed advection + stiff EOS

---

## Comparison: Unit Test vs Full Test

| Metric | unit_tester (10 steps, t=0.163ms) | Full Test (10k steps, t=0.781s) |
|--------|-------------------------------------|--------------------------------|
| EOS | NASG (γ=2.35) | NASG (γ=1.187) |
| err_p | 2.017e-4 | **0.1816** |
| err_u | 4.691e-7 | **0.0385** |
| Status | PASS | **FAIL** |

**Key difference**: unit_tester stopped at t=0.163ms before instability could grow.

---

## Step-by-Step Trace (Critical Region: Steps 300-1000)

| Step | t (s) | Mach | p_min | p_max | u_max | Issue |
|------|-------|------|-------|-------|-------|-------|
| 300 | 5.07e-3 | 0.044 | 94.2k | 195k | 1.20 | Pressure oscillation starts |
| 400 | 6.83e-3 | 0.069 | 55.4k | 953k | 3.31 | Large pressure variance |
| 500 | 9.05e-3 | 0.113 | 70.9k | 1.92M | **35.30** | **CRITICAL: velocity spike** |
| 600 | 1.49e-2 | 0.063 | 61.5k | 106k | 3.37 | Oscillation recovery |
| 700 | 2.55e-2 | 0.083 | **1.00** | 106k | 1.97 | **p_min collapse** |
| 800 | 3.72e-2 | 0.063 | **1.00** | 114k | 1.68 | p_min remains at 1 Pa |
| 900 | 4.92e-2 | 1.486 | 35.9k | 3.47M | 2.75 | **Mach spike to 1.5** |
| 1000 | 6.29e-2 | 10.888 | **1.00** | 162k | 10.87 | **Mach = 10.9 (transonic)** |
| 2000 | 1.51e-1 | 0.080 | 99.8k | 101k | 1.03 | Stabilization |
| 10000 | 7.82e-1 | 0.070 | 117.8k | 118.2k | 1.038 | Converged (err=18%) |

---

## Diagnostic Output

**Energy conservation** (excellent):
```
ΔE/E = -1.938e-14  (machine precision) ✓
```

**α (volume fraction)** (correct):
```
α_min = 9.602e-6  (≥0) ✓
α_max = 0.9968    (≤1) ✓
```

**Final state (t=0.781s)**:
```
p = [117800 Pa, 118200 Pa]  — expected [100000, 100000]
u = [0.9955, 1.0385] m/s    — expected [1.0, 1.0]
T = [varying]               — expected [300, 300] K
```

---

## Conclusion

### Test A Result: **FAIL**

The `solve_IMEX` solver **fails to preserve pressure-velocity equilibrium** for the NASG water-air case, despite achieving energy conservation and volume fraction correctness.

**Critical failures**:
1. ✗ Pressure preservation (err_p=18.16% >> 1%)
2. ✗ Velocity preservation (err_u=3.85% >> 1%)
3. ✗ t_end coverage (78.2% < 99%)

**Why unit_tester masked the failure**:
- Unit test ran only 10 steps (1.57 ms) before instabilities amplified
- NASG EOS may be marginally stable at short times but accumulates drift

**Recommendation for code_planner** (Phase 1 NASG completion failure):

### Issue 1: EOS Parameter Mismatch
**File**: `results/unit_tests/test_nasg_phase1_debug.py`, Line 28
```python
nasg_water = NASGEOS(gamma=2.35, pinf=1e9, kv=943.8, b=6.61e-4, eta=-1167e3)
```

**Spec** (`validation/1D/02_A_PE_advection_unified.md`, Line 51):
```
Water NASG: γ=1.187, P∞=7.028e8, b=6.61e-4, kv=3610, η=−1.177788e6
```

**Discrepancy**: γ=2.35 (unit test) vs γ=1.187 (spec) — **100% mismatch!**

→ **Action**: Unit test and full validation are using **wrong NASG parameters**. Spec should be the ground truth. Correct unit_tester parameters to match spec.

### Issue 2: Acoustic Coupling Instability
Pressure spikes to 1.92e6 Pa (19× nominal) at step 500 suggest IM1 acoustic solver is **unstable for NASG stiffness** (P∞=7e8).

→ **Action**: 
1. Check IM1 eigenvalue stability for NASG stiffness (P∞/ρ/c² ratio)
2. Consider reducing CFL from 0.4 to 0.2 for NASG cases
3. Audit Peluchon IM1 block-tridiagonal coefficients for NASG numerics

### Issue 3: Long-time PE Drift
Even after oscillations damp (step 2000+), pressure converges to **1.18e5 Pa** (18% high), not equilibrium value of 1e5 Pa.

→ **Action**: Examine `cons_to_prim` NASG pressure formula for cumulative rounding error or EOS cancellation in mixture pressure solve.

---

## Files Generated

- **Validation script**: `/home/younglin90/work/claude_code/claudeCFD/results/tmp_test_02A.py` (100-step run)
- **Full test script**: `/home/younglin90/work/claude_code/claudeCFD/results/tmp_test_02A_v2.py` (1.0s run)
- **Plot (100-step)**: `/home/younglin90/work/claude_code/claudeCFD/results/cat_A_exact/02A_abgrall_nasg.png`
- **Plot (1.0s)**: `/home/younglin90/work/claude_code/claudeCFD/results/cat_A_exact/02A_abgrall_nasg_1s.png`

---

## Next Steps

**Phase 1 NASG validation is BLOCKED.** 

Recommend:
1. **Fix NASG parameters** (γ=1.187, P∞=7.028e8, η=−1.177788e6) in both unit_tester and full validation
2. **Re-run unit test** to confirm parameter fix stabilizes 10-step run
3. **Audit IM1 for NASG** (acoustic wave equation coefficients, CFL selection)
4. **Attempt Phase 1 with SG EOS** (which is known to work) as a baseline comparison

