# Validation Report

Generated: 2026-04-28T12:49:36.269690+00:00

| gate | status | returncode | wall_s |
|---|---:|---:|---:|
| uniform_flow | PASS | 0 | 0.51 |
| amplification_matrix | PASS | 0 | 2.55 |
| transport_eigenmode | PASS | 0 | 0.62 |
| 02A_nasg | PASS | 0 | 4.82 |

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
02_A NASG: status=PASS t=1.00000e+00 step=100 err_p=2.548e-08 err_u=7.848e-07 finite=True wall=2.38s
Plot saved: results/1D/02_A/diff_vs_exact.png
```
