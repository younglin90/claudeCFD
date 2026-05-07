# Codex Loop State

## Objective

Improve `solver/five_eq_IMEX` until 02-A and 07-B satisfy the strict-but-diffusion-aware acceptance verifier.

## Current Result

Goal reached: `false`, `acceptance_failures=1`.

## This Round

- Verified 07-B Air-Water setup:
  - left material: ideal air, `rho=1.157`, `c=347.8`, `gamma=1.4`, `kv=717.5`.
  - right material: stiffened-gas water, `rho=998`, `c=1344.6`, `gamma=4.1`, `pinf=4.4e8`, `kv=474.2`.
  - `bc_l=reflective`, `bc_r=transmissive`, `x_intf=0.5`, `x_src=0.1`, `sigma=0.014`, `t_end=1.63e-3`.
  - initial pulse is on the air side with `u_peak=0.02`; `p=P0+Z_air*u`; `T1` is adjusted by the isentropic acoustic `dT/dp` coefficient.
- Implemented an optional D1 acoustic split path:
  - `L_E,alpha = div(alpha u) - alpha div(u)`.
  - `L_I,alpha = -D1^n div(u*)`.
- Added CLI access to enable that split with `--kapila-acoustic-source07`; default remains the stable explicit D1 path.
- Added oscillation rejection to `results/run_02_07_five_eq_imex.py`, so profile-only Air-Water with visible interface ringing no longer prints PASS.

## Verification

- `MPLCONFIGDIR=/tmp/mpl python3 tests/test_acoustic_RT_diagnostic.py`: PASS.
- `MPLCONFIGDIR=/tmp/mpl python3 .codex-loop/verify_02_07_acceptance.py`: FAIL, final scalar `1`.

## Key Metrics

- 02-A NASG: PASS, `p_rel_linf=2.548e-08`, `u_abs_linf=7.848e-07`.
- 07-B Air-Water: FAIL, `L2p=3.927e-01`, `Lip=1.496e+00`, `corr_p=0.66`, `p_alt=0.69/0.34`, `profile=true`, `osc=false`.
- 07-B Helium-Air: PASS.
- 07-B Argon-Air: PASS.

## Rejected Trials

- D1 acoustic split: unstable on Air-Water, terminates with `dt_below_min` and huge residuals.
- D1 off / Allaire: stable but worse pressure profile and stronger ringing.
- He2024 m1-style harmonic narrow-band acoustic impedance: worsened Air-Water in the current five_eq_IMEX pressure/projection split and was reverted.

## Next Task

The next mechanism should be a face-level He2024/MMACM-style coupled `G_alpha` correction in the explicit flux path. D1 source splitting alone is not sufficient without a matching pressure/alpha coupling block.
