# validation/1D Target Spec Map for Denner 1D Autoresearch

Date: 2026-05-18  
Source of truth: `../validation/1D/*.md`  
Runner: `results/run_denner1d_17case.py`

This document records the mesh/time/initial-condition settings that should be used for the new multi-case target. The goal is to stop carrying accidental overrides from the previous case-07-focused run.

| Case | Spec mesh/time | Initial condition / BC summary | Runner status after this check |
|---|---:|---|---|
| 01_A | N=100, dt_fixed=0.01, t_end=1.0 | [0,1], transmissive, air left/water right at x=0.5, p=1e5, T=293, u=0 | matched |
| 02_A | N=100, dt_fixed=0.01, t_end=1.0 | [0,1] periodic, water band x in [0.4,0.6], p=1e5, T=300, u=1 | matched |
| 04_B | main spec N=500, CFL≈0.4, t_end=2.3e-3 | pure air sinusoidal inlet, u0=1, p0=1e5, f=2000, du=0.01 | matched |
| 05_B | current acceptance N=400, CFL=0.4, t_end=5.1e-4 | pure water sinusoidal inlet, u0=1, p0=1e5, f=6000, du=0.01 | matched |
| 07_B | spec N=400, L=1.5, acoustic CFL≈0.4 | Gaussian velocity/pressure pulse IC, u0=0, p0=1e5, subcase-specific interface/t_end | runner default N=400; previous PASS used explicit N=800 override |
| 13_E | N=400, CFL=0.30, t_end=6.7e-4 | [0,2], interface x=0.5, high-pressure air left p=1e9, low-pressure water right p=1e4, T=300, u=0 | fixed: default 800 -> 400 |
| 14_E | N=400, CFL=0.25, t_end=2.29e-4 | [0,1], interface x=0.7, high-pressure water left p=1e9, low-pressure air right p=1e5; rho_water=1000, rho_air=50; u=0 | fixed: default 800 -> 400 |
| 15_E | N=400, CFL=0.01, t_end=9.5e-4 | [0,1], uniform air-water mixture alpha_air=0.055, p=1e5, u=-100/+100 about x=0.5 | matched |
| 24_H | N=800, CFL in active verifier 0.10, shock target x=0.8 | homogeneous air-water mixture shock, psi_water in {0,0.25,0.5,0.75,1}, x_shock0=0.1, Ms=10 | matched |
| 25_H | verification default N=400, adaptive CFL, t_end=t_hit+2.42e-4 | Mach-10 air shock at x=0.25 interacting with water interface x=0.50 | matched |

Important note on case 07:

- `../validation/1D/07_B_acoustic_reflection_transmission.md` says common N=400.
- The previous paper-grade Air-Water pass used `DENNER_CASE07_N_AIR_WATER=800` to meet the stricter exact-peak >=90% gate.
- For this new multi-case baseline, do not pass that environment override unless explicitly checking the previous case-07 PASS baseline. This makes the baseline faithful to `../validation/1D` and avoids hiding runtime cost.
