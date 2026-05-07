# Sprint Contract: 2026-04-28 07-B Acoustic Ablation

## Goal

Identify the cause of 07-B Air-Water interface pressure oscillation and only retain a solver change if it improves validation without damaging 02-A or mandatory gates.

## Selected Mechanism

Acoustic wave validation / pressure oscillation at diffuse interface.

The previous diagnostic showed the active Kapila + positivity + ACID explicit path has zero stationary-contact `p_U dot L_E`. Therefore this round targeted the acoustic face state and pressure-velocity coupling rather than another PE tangent projection.

## Trials

Four small `acoustic_riemann` face-state changes were tested:

- near-pure acoustic impedance replacement;
- interface-local acoustic smoothing only;
- pressure-only smoothing;
- smoothing everywhere except the direct alpha-jump face.

## Outcome

No solver modification was retained. Each trial either slightly worsened the 07-B profile or strongly increased water-side pressure ringing. The code was restored to the baseline global pressure/velocity acoustic smoothing.

## Validation

Final baseline-restored validation:

- `python3 -m py_compile solver/five_eq_IMEX/residual.py` PASS
- `python3 tests/test_stationary_contact.py` PASS
- `python3 tests/test_amplification_matrix.py` PASS
- `python3 results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01` PASS
- 07-B Air-Water profile gate PASS_PROFILE, strict still false
- mandatory benchmark bundle PASS

## Conclusion

The 07-B interface pressure oscillation is not fixed by simple acoustic smoothing placement or near-pure impedance replacement. The next bounded repair should add a linear acoustic interface diagnostic and then target the pressure-velocity/acoustic projection operator using measured R/T amplitude and phase errors.
