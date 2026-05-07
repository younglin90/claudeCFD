# Validation Report

Generated: 2026-05-03T12:23:05.495897+00:00

| gate | status | returncode | wall_s |
|---|---:|---:|---:|
| uniform_flow | PASS | 0 | 0.46 |
| amplification_matrix | PASS | 0 | 2.49 |
| transport_eigenmode | PASS | 0 | 0.64 |
| 02A_nasg | PASS | 0 | 6.39 |

## uniform_flow

```text
Phase 3 uniform-flow regression


=== uniform W (Ideal air + SG water) ===
  [OK] L_E[α₁ρ₁]                 max|·| = 0.000e+00
  [OK] L_E[α₂ρ₂]                 max|·| = 0.000e+00
  [OK] L_E[ρu]                   max|·| = 0.000e+00
  [OK] L_E[ρE]                   max|·| = 0.000e+00
  [OK] L_E[α]                    max|·| = 0.000e+00
  [OK] grad_p                    max|·| = 0.000e+00
  [OK] div_pu                    max|·| = 0.000e+00
  [OK] div_u                     max|·| = 0.000e+00
  one ARS222 step: err(p)=1.99e-13, err(u)=0.00e+00, err(T1)=5.97e-11, err(T2)=0.00e+00, err(α)=0.00e+00
   *** test_uniform_air_water: PASS

=== uniform W (Ideal air + NASG water, 02-A like) ===
  [OK] L_E[α₁ρ₁]                 max|·| = 0.000e+00
  [OK] L_E[α₂ρ₂]                 max|·| = 0.000e+00
  [OK] L_E[ρu]                   max|·| = 0.000e+00
  [OK] L_E[ρE]                   max|·| = 0.000e+00
  [OK] L_E[α]                    max|·| = 0.000e+00
  [OK] grad_p                    max|·| = 0.000e+00
  [OK] div_pu                    max|·| = 0.000e+00
  [OK] div_u                     max|·| = 0.000e+00
   *** test_uniform_nasg_air: PASS

--------------------------------------------------------------------
All tests passed.
```

## amplification_matrix

```text
Amplification spectral radius around α-jump PE state, N=8, dt=3.7e-05
--------------------------------------------------------------------------
  integrator                 ρ(A)                                top 3 |λ|
--------------------------------------------------------------------------
  ARS222 raw           8.8316e+00            8.832e+00 8.832e+00 5.908e+00
  be1 raw              1.0009e+00            1.001e+00 1.001e+00 1.000e+00
  be1 schur=True       1.0009e+00            1.001e+00 1.001e+00 1.000e+00
  be1 pe_correct=True     1.0009e+00            1.001e+00 1.001e+00 1.000e+00
```

## transport_eigenmode

```text
Transport eigenmode analysis (be1, N=8, dt=3.7e-05)
α profile: [0.999 0.999 0.999 0.001 0.001 0.999 0.999 0.999]
------------------------------------------------------------------------------

Mode 0:  λ = 9.9930e-01-5.6918e-02j, |λ| = 1.0009
  α₁    |max|=0.000  pattern: [········]
  T₁    |max|=0.000  pattern: [········]
  T₂    |max|=0.000  pattern: [········]
  u     |max|=0.000  pattern: [········]
  p     |max|=1.000  pattern: [+--+·--+]

Mode 1:  λ = 9.9930e-01+5.6918e-02j, |λ| = 1.0009
  α₁    |max|=0.000  pattern: [········]
  T₁    |max|=0.000  pattern: [········]
  T₂    |max|=0.000  pattern: [········]
  u     |max|=0.000  pattern: [········]
  p     |max|=1.000  pattern: [+--+·--+]

Mode 2:  λ = 9.9990e-01-3.2835e-02j, |λ| = 1.0004
  α₁    |max|=0.000  pattern: [········]
  T₁    |max|=0.000  pattern: [········]
  T₂    |max|=0.000  pattern: [········]
  u     |max|=0.000  pattern: [········]
  p     |max|=1.000  pattern: [·++··--·]
```

## 02A_nasg

```text
Plot saved: results/1D/02_A/diff_vs_exact.png
ACCEPT 02_A_NASG pass=True p_rel_linf=2.619e-15 u_abs_linf=1.090e-13 alpha_range=1.000 rho_range=1.000 corr_alpha=1.000 corr_rho=1.000 alpha_l1=0.000 rho_l1=0.000 finite=True complete=True
CASE_JSON {"admissible": true, "alpha_l1_ratio": 1.976663951229393e-14, "alpha_range_ratio": 1.0000000000004354, "case": "02_A_NASG", "complete": true, "corr_alpha": 1.0, "corr_rho": 0.9999999999999998, "finite": true, "hf_oscillation_ok": true, "hf_sharp_cells": 70, "hf_smooth_cells": 26, "p_hf_ok": true, "p_rel_linf": 2.6193447411060335e-15, "p_sharp_overshoot": 2.6193447411060335e-15, "p_sharp_turns": 0, "p_sharp_tv_excess": 1.8480932340025903e-14, "p_smooth_hf_max": 2.1827872842550278e-16, "p_smooth_hf_rms": 9.435820748917802e-17, "p_smooth_local_hf_max": 2.1827872842550278e-16, "p_smooth_local_turns": 0, "p_smooth_local_tv_excess": 1.0186340659856795e-15, "pass": true, "rho_hf_ok": true, "rho_l1_ratio": 2.082851713854131e-14, "rho_range_ratio": 1.0000000000004587, "rho_sharp_overshoot": 4.552500390191189e-13, "rho_sharp_turns": 1, "rho_sharp_tv_excess": 1.2512340587401366e-12, "rho_smooth_hf_max": 1.279832631314336e-15, "rho_smooth_hf_rms": 6.152461125605406e-16, "rho_smooth_local_hf_max": 1.3442580382161395e-12, "rho_smooth_local_turns": 0, "rho_smooth_local_tv_excess": 1.0081713242016122e-11, "steps": 100, "u_abs_linf": 1.0902390101819037e-13, "u_hf_ok": true, "u_sharp_overshoot": 1.0902390101819037e-13, "u_sharp_turns": 0, "u_sharp_tv_excess": 1.8141044222375058e-12, "u_smooth_hf_max": 5.484501741648273e-14, "u_smooth_hf_rms": 2.496693586479318e-14, "u_smooth_local_hf_max": 5.484501741648273e-14, "u_smooth_local_turns": 0, "u_smooth_local_tv_excess": 3.084199562408685e-13, "wall": 3.268183946609497}
0
```
