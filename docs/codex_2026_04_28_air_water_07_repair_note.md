# 2026-04-28 Air-Water 07-B repair note

## Current result

02-A NASG remains PASS after the optional Air-Water changes:

```bash
python3 results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01
# 02_A NASG: status=PASS err_p=3.576e-07 err_u=7.886e-06 finite=True
```

Best Air-Water profile run so far:

```bash
python3 results/run_02_07_five_eq_imex.py \
  --case 07 --subcase07 Air-Water --n07 200 --cfl07 0.4 \
  --imp-dissipation 0.1 --imp-dissipation-form acoustic_riemann \
  --pe-projection-mode interface_explicit \
  --alpha-floor07 1e-5 --pure-branch07 --energy-alpha-pure-tol07 1e-5 \
  --implicit-include-explicit-residual07 --kapila-closure07 --max-steps07 5000
```

Metrics:

```text
status=FAIL
L2p=4.139e-01 Lip=1.560e+00
L2u=1.225e-01 Liu=8.910e-01
frac_p=0.84 frac_u=0.97
corr_p=0.60 corr_u=0.29
finite=True term=None
```

Plot:

```text
results/1D/07_B/diff_vs_exact.png
```

## What improved

1. The previous Air-Water failure was not only interface exact/sign. A single-phase air diagnostic showed the acoustic pulse stayed near the source.
2. Root cause: the stage Newton residual accepted `L_E` but did not include it. For acoustic diagnostics this makes pressure/acoustic propagation inconsistent with explicit mass/energy coupling.
3. Added optional `implicit_include_explicit_residual` so 07 can solve `R=(U-U_n)/dt + L_E + L_I` without changing the default 02-A stable path.
4. Added `acoustic_riemann` implicit p/u face state. This restores impedance-aware acoustic transmission and reduces interface ringing compared with central faces.
5. Added acoustic face smoothing through `imp_dissipation` for the `acoustic_riemann` form.
6. Added runner knobs for `always_explicit`, `imp_dissipation_form`, `energy_alpha_pure_tol07`, `implicit_include_explicit_residual07`, and `kapila_closure07`.
7. Changed 07 plot to show `p-P0`, so pressure perturbation profile is visible.

## Remaining blocker

The Air-Water pressure profile now has the correct sign/location trend and pressure correlation reaches `corr_p=0.60`, but the transmitted pressure amplitude is over-damped and the reflected velocity amplitude/phase is still too weak:

```text
pressure: L2p 0.414 > 0.30, Lip 1.56 > 0.50
velocity: L2u 0.123 OK, Liu 0.891 > 0.50, corr_u 0.29 < 0.50
```

The current operator behaves like an overly diffusive acoustic/interface solver. Further work should focus on reducing amplitude damping without reintroducing water-side pressure ringing.

## Next targeted options

1. Replace simple face-space binomial smoothing with a sensor-limited smoothing that acts only on high-frequency pressure ringing in the water side.
2. Add a second-order acoustic time update or ARS222-safe path for the acoustic block; BE1 is too diffusive for 07-B's narrow Gaussian.
3. Build a dedicated linearized acoustic interface test that checks pressure/velocity R/T coefficients before the full five-equation nonlinear update.
4. Revisit PE projection as a local interface energy correction rather than tangent projection of conservative components, because global/always projection freezes acoustic waves and narrow projection leaves ringing.
