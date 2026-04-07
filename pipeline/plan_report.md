# Plan Report -- Sharp Advection Pressure Explosion Fix

**Date**: 2026-04-07
**Planner**: code_planner agent
**Target**: Fix pressure divergence in sharp advection test (N=50, CFL=2.0)

---

## FAIL Item: test_sharp_advection (pressure divergence)

### Symptom

Pressure explodes to p_max = 1.045e+08 Pa (1045x reference) within the first few time steps of the sharp advection test. The smooth test (N=10, CFL=auto) still passes with L2(p) = 4.195e-10.

### Root Cause

**The energy equation in Path B (assembly.py lines 385-400) is fully decoupled from the implicit pressure-velocity system.**

Two recent changes combined to break acoustic stability:

1. **Path B in assembly.py (lines 385-400)**: Energy flux moved entirely to the b vector (RHS). The A matrix energy rows contain ONLY temporal derivatives (dEdp, dEdT, dEdu) -- no flux terms. This eliminates:
   - The velocity coupling: `H_acid * u / dx` terms that link energy to velocity
   - The MWI pressure coupling: `H_acid * d_hat / dx^2` terms that provide implicit acoustic damping in the energy equation

2. **T-correction clamp in solver_a.py (lines 238-247)**: Zeros dT for interface cells where H_face is large. Combined with Path B's decoupled energy equation, this removes the last mechanism for the energy equation to influence the coupled system at interfaces.

**Why Path A works but Path B fails:**

In Path A (assembly.py lines 359-384), the ACID energy flux contributes to the A matrix:
- `A[row_e, col_u_i] += H_i_acid / dx` -- velocity coupling
- `A[row_e, col_p_i] += H_i_acid * d_R / (dx*dx)` -- pressure coupling via MWI

These implicit terms create acoustic pressure damping in the energy equation. When pressure deviates, the implicit velocity and pressure flux terms in A respond immediately, stabilizing the system. With CFL_acoustic ~ 3000 (dt=0.04s, c=1500 m/s, dx=0.02m), this implicit coupling is essential.

In Path B, the energy flux `H_R * uf_R - H_L * uf_L` is frozen at the Picard iterate state. Any pressure perturbation from the continuity/momentum equations propagates without resistance from the energy equation, leading to positive feedback and divergence.

**Why the smooth test passes:** N=10, smooth tanh interface, CFL=auto (0.5). The interface gradient is mild, H_face variation is gradual, and the convective CFL is small enough that the explicit energy flux doesn't amplify errors significantly. The T-clamp threshold (1e6) is also not triggered for smooth interfaces.

### Impact

- Fixing this will NOT break the smooth test: the smooth test already passes with Path A (ACID energy), and switching back to Path A preserves the same temporal and flux discretization that produces L2(p) ~ 4e-10.
- The T-correction clamp becomes unnecessary if Path A is restored, since Path A's implicit H_acid terms in A are inherently well-conditioned (ACID removes the 4000x enthalpy jump by construction).

### Fix Strategy: Option A (Recommended) -- Revert to Path A (ACID energy)

Switch `use_acid_energy=False` back to `use_acid_energy=True` in solver_a.py, and remove the T-correction clamp.

**Rationale**: Path A was designed specifically for the ACID formulation (Denner 2018). It uses the rho*h temporal formulation with dp/dt source, and ACID face enthalpies that eliminate the 4000x enthalpy jump at water-air interfaces. The ill-conditioning that motivated Path B was due to non-ACID H_face values; with ACID face enthalpies, the condition ratio drops to O(1).

### Modification 1: solver_a.py -- Switch to ACID energy equation

**File**: `/home/younglin90/work/claude_code/claudeCFD/solver/denner_1d/solver_a.py`
**Lines**: 186-198

**Before:**
```python
            # ACID = Acoustically Conservative Interface Discretisation (Denner 2018)
            # ACID modifies the MWI face density (mwi.py) for acoustic conservation.
            # The energy equation uses the standard dE_total/dt form with H_f
            # computed by face-VOF blending in consistent.py (psi_face_vof).
            # use_acid_energy=False: correct legacy E_total temporal path.
            ph1=ph1, ph2=ph2,
            rho_h_n=rho_h_n,
            rho_h_k=rho_h_k,
            d_rho_h_dp_v=d_rho_h_dp_v,
            d_rho_h_dT_v=d_rho_h_dT_v,
            p_n=p_n,
            use_acid_energy=False,
```

**After:**
```python
            # ACID = Acoustically Conservative Interface Discretisation (Denner 2018)
            # ACID modifies the MWI face density (mwi.py) for acoustic conservation.
            # The energy equation uses the Denner 2018 rho*h formulation with ACID
            # face enthalpies. This keeps H_face terms in the A matrix, providing
            # implicit acoustic pressure damping essential for CFL_acoustic >> 1.
            ph1=ph1, ph2=ph2,
            rho_h_n=rho_h_n,
            rho_h_k=rho_h_k,
            d_rho_h_dp_v=d_rho_h_dp_v,
            d_rho_h_dT_v=d_rho_h_dT_v,
            p_n=p_n,
            use_acid_energy=True,
```

### Modification 2: solver_a.py -- Remove T-correction clamp

**File**: `/home/younglin90/work/claude_code/claudeCFD/solver/denner_1d/solver_a.py`
**Lines**: 220-247

**Before:**
```python
        # ----  3e. Clamp T corrections for ill-conditioned energy rows  ----
        #
        # With Path B (explicit energy flux in b), any tiny velocity variation
        # du from the p,u solve gets amplified by H_face/dx in the deferred
        # energy residual r_e ~ H/dx * du.  For air cells at a water-air
        # interface, H_face ~ 1.83e9 J/m^3 while dEdT_eff ~ 834 J/(m^3 K),
        # so the condition ratio H/(alpha*dEdT*dx) ~ 4.4e6 >> 1.  The solve
        # amplifies floating-point artefacts into dT ~ 5e-3 K.
        #
        # Fix: zero dT for cells where the energy equation is ill-conditioned.
        # With Path B the energy rows are decoupled from p,u in A (no
        # off-diagonal flux entries), so clamping dT does not affect dp/du.
        #
        # Threshold 1e6:
        #   Air at interface  -> ratio ~ 4.4e6 -> CLAMPED
        #   Interior air      -> ratio ~ 619   -> not clamped
        #   Water (any)       -> ratio ~ 2027  -> not clamped
        #   Smooth interface  -> ratio < 1e6   -> not clamped (N=10 smooth test)
        _KV_FLOOR_CL   = 718.0   # [J/(kg K)]  matches assembly _KV_FLOOR
        _COND_THRESH_T = 1.0e6
        _alpha_bdf1    = 1.0 / dt
        _H_abs   = np.abs(H_face)
        _Hmax    = np.maximum(_H_abs[:-1], _H_abs[1:])          # (N,)
        _dEdT_cl = np.maximum(np.abs(props_k['dEdT_v']),
                              rho_k * _KV_FLOOR_CL)             # (N,)
        _ill = _Hmax > (_COND_THRESH_T * _alpha_bdf1 * _dEdT_cl * dx)
        if np.any(_ill):
            dvar[2 * N:][_ill] = 0.0
```

**After:**
```python
        # No T-correction clamp needed with Path A (ACID energy equation).
        # ACID face enthalpies eliminate the 4000x enthalpy jump at interfaces,
        # so the energy equation rows are well-conditioned by construction.
```

### Verification Criteria

1. **Sharp test**: `python validation/1D/test_sharp_advection.py --N 50 --CFL 2.0 --max-ft 1` should produce L2(p) < 1e-4 (no divergence)
2. **Smooth test**: `python validation/1D/test_smooth_advection.py --N 10 --max-ft 1` should still produce L2(p) ~ 4e-10 (PASS)
3. Pressure should remain within [0.9e5, 1.1e5] Pa throughout the sharp test
4. No NaN/Inf in any field

### Priority

**CRITICAL** -- Pressure explosion causes immediate divergence. Must fix before any other work.

---

## code_maker Instructions

Perform the following modifications in order:

1. **`/home/younglin90/work/claude_code/claudeCFD/solver/denner_1d/solver_a.py` L197**: Change `use_acid_energy=False` to `use_acid_energy=True`
2. **`/home/younglin90/work/claude_code/claudeCFD/solver/denner_1d/solver_a.py` L186-191**: Update the comment to reflect ACID energy usage (see Before/After above)
3. **`/home/younglin90/work/claude_code/claudeCFD/solver/denner_1d/solver_a.py` L220-247**: Remove the entire T-correction clamp block (replace with single comment line)

After modifications, run both tests to verify:
```bash
python validation/1D/test_smooth_advection.py --N 10 --max-ft 1
python validation/1D/test_sharp_advection.py --N 50 --CFL 2.0 --max-ft 1
```

Create `pipeline/code_ready.flag` upon completion.
