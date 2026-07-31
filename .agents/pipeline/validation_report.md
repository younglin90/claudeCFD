# Validation Report

Generated: 2026-07-02T13:08:25.545380+00:00

| gate | status | returncode | wall_s |
|---|---:|---:|---:|
| uniform_flow | PASS | 0 | 0.44 |
| amplification_matrix | PASS | 0 | 2.94 |
| transport_eigenmode | PASS | 0 | 0.61 |
| 02A_nasg | PASS | 0 | 7.46 |

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
ACCEPT 02_A_NASG pass=True p_rel_linf=2.765e-15 u_abs_linf=4.219e-14 alpha_range=1.000 rho_range=1.000 corr_alpha=1.000 corr_rho=1.000 alpha_l1=0.000 rho_l1=0.000 finite=True complete=True
CASE_JSON {"admissible": true, "alpha_l1_ratio": 1.0415823806325853e-14, "alpha_range_ratio": 1.0000000000003681, "case": "02_A_NASG", "complete": true, "corr_alpha": 1.0, "corr_rho": 1.0, "finite": true, "hf_oscillation_ok": true, "hf_sharp_cells": 70, "hf_smooth_cells": 26, "p_hf_ok": true, "p_rel_linf": 2.764863893389702e-15, "p_sharp_overshoot": 2.764863893389702e-15, "p_sharp_turns": 0, "p_sharp_tv_excess": 2.35741026699543e-14, "p_smooth_hf_max": 1.4551915228366853e-16, "p_smooth_hf_rms": 8.911191772433732e-17, "p_smooth_local_hf_max": 1.4551915228366853e-16, "p_smooth_local_turns": 0, "p_smooth_local_tv_excess": 1.0186340659856795e-15, "pass": true, "rho_hf_ok": true, "rho_l1_ratio": 1.9979168283870188e-14, "rho_range_ratio": 1.0000000000006906, "rho_sharp_overshoot": 6.155517186264049e-13, "rho_sharp_turns": 1, "rho_sharp_tv_excess": 1.7071335272015081e-12, "rho_smooth_hf_max": 7.214734069744388e-14, "rho_smooth_hf_rms": 2.710709052953442e-14, "rho_smooth_local_hf_max": 7.577916072420976e-11, "rho_smooth_local_turns": 0, "rho_smooth_local_tv_excess": 3.0613644952381946e-10, "steps": 100, "u_abs_linf": 4.218847493575595e-14, "u_hf_ok": true, "u_sharp_overshoot": 4.218847493575595e-14, "u_sharp_turns": 0, "u_sharp_tv_excess": 1.2333467580560864e-12, "u_smooth_hf_max": 5.995204332975845e-14, "u_smooth_hf_rms": 2.1348163344530445e-14, "u_smooth_local_hf_max": 5.995204332975845e-14, "u_smooth_local_turns": 0, "u_smooth_local_tv_excess": 2.8888003100746573e-13, "wall": 3.917734146118164}
0
```
