# Autonomous research log — P1-P6 faithful physics validation

Context: P1-P6 made the candido solver physically faithful to Candido & Pascoa 2023
(committed: e1eb8f2 OpenMP, 9cf0c48 P1-P6). This log journals the autonomous validation of
the new physics against the paper, and the regression re-baselining.

## Key tension discovered

The faithful electric Courant limit (dt <= 0.1*tau_e, P5) makes dt ~20x smaller than the
prior non-physical configuration (electric Courant 1.0 / unlimited dt). Reaching the paper
morphology times (0.4-0.9 ms) therefore needs ~1100 steps instead of ~52. This is the
faithfulness-vs-affordability tradeoff the original authors avoided by keeping these options
off. Consequence: the long-window morphology regression assertion
(`final_time*tau_h*1e3 >= 0.9`) fails at the fixed 52-step budget.

## Plan

1. [done] Commit P1-P6 + OpenMP.
2. Validation experiment A: CaE=0.25 (paper validation case), faithful physics, increasing
   step counts. Verify long-run stability (mass/charge/divergence bounded over hundreds of
   steps) and measure morphology silhouette + current + electric Courant.
3. Validation experiment B: CaE sweep {0.26,0.32,0.42} for whipping radial asymmetry.
4. Regression re-baseline: decouple the morphology smoke diagnostic from the faithful
   electric-relaxation dt (run it on the hydrodynamic dt to reach the paper window
   affordably), keep production default faithful. Re-run full candido regression.
5. Update solver_vs_candido_gap_analysis.md with results.

## Findings

### Experiment A — CaE=0.25, 300 steps, faithful P1-P6 (nx=12 box, ~0.25 ms reached)

Stability (the headline result): over 300 steps the faithful physics stays **bounded and
conservative** — alpha mass drift 2.3e-14, max divergence 9.9e-13, **charge-budget residual
2.1e-13**, potential residual 8.3e-12. The conservative dimensional charge model (P2) holds
charge conservation long-term, unlike the prior non-conservative clamp.

Dynamics: electric Courant = 0.1 (faithful), dt = 2.96e-3, electric force 74.5, charge +/-8e-8,
max velocity 1.59 (jet accelerating). **final radial asymmetry = 0.0303** at CaE 0.25 — well
above the prior coarse-run values (~1e-4..1e-3) and below the 0.05 whipping threshold, which
is consistent with CaE 0.25 being the paper's stable single-droplet regime. The new physics
produces genuine, physically-scaled asymmetric dynamics where the old configuration produced
essentially none.

Caveat: tip displacement reads 0 (the meniscus-top metric); dynamics so far are charge/asymmetry
growth rather than gross axial tip motion at 0.25 ms (cone formation in the paper peaks ~0.4 ms).
nx=12 box is far coarser than the paper's ~11M-cell / 2 um mesh, so absolute morphology is not
expected to match 1.1% — the value is the qualitative trend + the conservation/stability.

Time-series (every 30 steps, nondim time): the evolution is smooth, monotonic and physical:

| step | time | radial_asym | max_vel | total_I | max_alpha | mass |
|---|---|---|---|---|---|---|
| 0   | 0.000 | 0.00005 | 0.000 | 0       | 0.688 | 0.4254 |
| 90  | 0.267 | 0.00581 | 0.290 | 2.25e-7 | 0.696 | 0.4254 |
| 180 | 0.533 | 0.01820 | 0.507 | 2.26e-7 | 0.719 | 0.4254 |
| 270 | 0.800 | 0.02843 | 1.036 | 2.26e-7 | 0.754 | 0.4254 |
| 300 | 0.889 | 0.03034 | 1.589 | 2.26e-7 | 0.767 | 0.4254 |

Radial asymmetry grows monotonically (no blow-up), the jet velocity accelerates (a marked
jump after step ~270 suggests cone-tip ejection onset), the total current is essentially
constant (2.25e-7, consistent with the paper's weak voltage/time dependence of average
current), the interface sharpens (max alpha up), and **mass is exactly conserved throughout**.
This is qualitatively the paper's behaviour and a large improvement over the prior
configuration (asymmetry ~1e-4 and current blow-ups documented in candido_3d_method_gap.md).


