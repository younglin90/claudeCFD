# Research Cycle Log

Append one entry per research-improvement cycle.

## 2026-04-28 PE-Normal Residual Diagnostic

- Selected item: pressure oscillation at diffuse interface.
- Change: extended `tests/test_stationary_contact.py` with a 07-like Kapila + positivity LO + ACID diagnostic and projection check.
- Result: active 07-like path has `raw max|p_U·L_E| = 0.0` and projected value `0.0`; diffuse-interface PE-normal injection is not the current limiting mechanism.
- Validation: stationary-contact diagnostic PASS, amplification_matrix PASS (`ARS222=8.8316`, `BE1=1.0009`), 02-A NASG PASS (`err_p=3.576e-7`, `err_u=7.886e-6`), 07-B Air-Water profile PASS (`L2p=0.3927`, `Lip=1.496`, `L2u=0.1225`, `Liu=0.8872`, `corr_p=0.66`, `corr_u=0.29`), mandatory benchmark bundle PASS.
- Next: target acoustic wave validation, specifically 07-B pressure/velocity amplitude and phase mismatch.

## 2026-04-28 07-B Acoustic Smoothing Ablation

- Selected item: acoustic wave validation / pressure oscillation at diffuse interface.
- Attempted changes: near-pure impedance replacement, interface-local smoothing, pressure-only smoothing, and interface-excluding smoothing in `solver/five_eq_IMEX/residual.py::_acoustic_riemann_faces`.
- Result: every retained-code candidate either slightly worsened 07-B or strongly increased water-side pressure ringing. All failed solver changes were reverted; baseline global p/u acoustic smoothing is still the least-bad current option.
- Validation after revert: 02-A NASG PASS (`err_p=3.576e-7`, `err_u=7.886e-6`), 07-B Air-Water profile PASS (`L2p=0.3927`, `Lip=1.496`, `L2u=0.1225`, `Liu=0.8872`, `corr_p=0.66`, `corr_u=0.29`), mandatory benchmark bundle PASS.
- Numerical conclusion: the visible 07-B pressure oscillation is not fixed by face-smoothing placement. It likely comes from the acoustic block lacking a dedicated interface-correct R/T projection or from the BE1 pressure-only eigenmode (`|λ|≈1.0009`) rather than from diffuse-interface PE-normal residual injection.
- Next: add a dedicated linear acoustic interface diagnostic that extracts reflected/transmitted amplitudes and phase at the interface before full 07 metric aggregation; then target the pressure-velocity/acoustic projection operator rather than smoothing.

## 2026-04-28 Air-Water Acoustic R/T Diagnostic

- Selected item: acoustic wave validation / Wood mixture sound speed consistency.
- Change: added `tests/test_acoustic_RT_diagnostic.py`, a fast diagnostic for the active implicit acoustic Riemann impedance and closed-form Air-Water R/T coefficients at `alpha_floor=1e-5`.
- Result: the current active water-side impedance is `1.263012e6` versus pure water `1.341939e6` (`0.941184` ratio). The pressure coefficient is essentially unchanged: `R_p=0.999360` active versus `0.999400` pure, and `T_p=1.999360` active versus `1.999400` pure.
- Conclusion: the proposed P1 mechanism, where Wood/Kapila alpha-floor contamination collapses the Air-Water reflection coefficient to O(0.5), is not supported by the active code. This also explains why the previous near-pure impedance patch did not improve 07-B.
- Validation: new diagnostic PASS, stationary-contact PASS, 02-A NASG PASS (`err_p=3.576e-7`, `err_u=7.886e-6`), 07-B Air-Water profile PASS (`L2p=0.3927`, `Lip=1.496`, `L2u=0.1225`, `Liu=0.8872`, `corr_p=0.66`, `corr_u=0.29`), mandatory benchmark bundle PASS.
- Next: do not pursue H1 as the main fix. The next likely causes are D1/source lagging, pressure-work/acoustic energy coupling, BE1 acoustic dispersion, or boundary/interface measurement effects.
