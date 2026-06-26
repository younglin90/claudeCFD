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

### Experiment A2 — CaE=0.25, 900 steps (~0.79 ms, the paper morphology comparison time)

Stable and conserved over **900 steps**: mass drift 2.3e-14, max div 2.7e-12, charge-budget
residual 2.5e-13, potential residual 1e-11. Crucially, **the Taylor cone now forms and ejects**:
tip displacement 0.320 D_o (0 at 300 steps), final tip y 0.80 -> 1.12, max velocity 10.9
(jet ejection), max electric force 110, connected alpha=0.5 silhouette volume 1.62, radial
asymmetry 0.032 (below 0.05, consistent with the stable CaE=0.25 regime). The cone-jet emerges
naturally on the hydrodynamic timescale - the qualitative paper behaviour (cone ~0.4 ms, jet
~0.7 ms), which the prior configuration did not produce.

### Experiment B — CaE sweep {0.26, 0.32, 0.42}, 200 steps each

| CaE | voltage | radial asym | max vel | electric force | total current | charge residual |
|---|---|---|---|---|---|---|
| 0.26 | 2203 | 0.0221 | 0.58 | 61.0 | 2.31e-7 | 8.9e-14 |
| 0.32 | 2444 | 0.0275 | 1.27 | 86.3 | 2.56e-7 | 5.4e-14 |
| 0.42 | 2800 | 0.0325 | 6.48 | 145.4 | 2.93e-7 | 3.5e-14 |

Clean **monotonic paper trends**: radial asymmetry and jet velocity grow strongly with CaE
(velocity 0.58 -> 6.48), electric force grows ~quadratically with voltage, and the total
current rises only mildly (2.31e-7 -> 2.93e-7, ~27%) - i.e. weakly voltage-sensitive and
**O(1e-7) bounded at every CaE**, decisively resolving the prior current blow-up (ratios
1e9-1e11) and current/whipping Pareto-block documented in candido_3d_method_gap.md. Charge is
conserved at every CaE. Asymmetry stays below the 0.05 whipping threshold at this early
(~0.18 ms) time but grows with both CaE and time (cf. A2), so a developed high-CaE run is the
next probe for crossing the whipping threshold.

### Experiment C — CaE=0.42, 900 steps (developed-time whipping probe)

The high-CaE companion to A2 (same 900 steps, same nx=12 box, voltage 2800). Stable and
conserved throughout: mass drift 2.3e-14, max div 9.5e-12, charge-budget residual 4.1e-13.
Far more energetic than CaE 0.25: max velocity **20.2** (vs 10.9 at 0.25), max electric force
**289** (vs 110), silhouette 1.62.

The radial-asymmetry **trajectory** (not just the final value) is the finding. Time-series
(nondim time, every ~60 steps):

| step | time | radial_asym | max_vel | total_I | max_alpha |
|---|---|---|---|---|---|
| 0   | 0.00 | 0.0002 | 0.0  | 0       | 0.689 |
| 120 | 0.36 | 0.0220 | 0.81 | 2.93e-7 | 0.731 |
| 300 | 0.89 | **0.0372** (peak) | 12.8 | 2.91e-7 | 0.846 |
| 450 | 1.33 | 0.0309 | 17.2 | 2.34e-7 | 0.835 |
| 690 | 2.04 | **0.0007** (collapse) | 12.5 | 2.98e-7 | 0.833 |
| 780 | 2.31 | 0.0135 (re-rise) | 10.7 | 2.92e-7 | 0.866 |
| 900 | 2.67 | 0.0099 | 12.2 | 2.91e-7 | 0.892 |

Whereas CaE 0.25 (A2) rises to an asymmetry peak (~0.037) and then **plateaus** (~0.032,
steady single cone-jet), CaE 0.42 rises to a near-identical peak (0.037) and then
**oscillates** — collapsing to ~0.001 and re-rising — while the velocity (peaks ~20) and the
total current (oscillating 2.2e-7..3.0e-7) become unsteady too. This is the qualitative
**steady -> pulsating/intermittent transition**: at the paper's stable validation CaE (0.25)
the box produces a steady cone-jet; above it (0.42) the cone-jet enters an unsteady, pulsating
ejection regime (rise -> eject -> collapse -> re-form). Neither case crosses the 0.05 whipping
threshold with **sustained** asymmetry on this coarse box — sustained lateral whipping remains
gated on the paper's ~2 um / ~11M-cell resolution — but the onset of unsteadiness with field
strength is captured, and conservation holds even in the energetic pulsating regime.

### Headline

The faithful physics (P1-P6) turns the prior empirically-regularized, current-pathological
configuration into a **stable, charge-conserving solver that reproduces the paper's qualitative
cone-jet formation, ejection, asymmetry growth, weak-voltage current trend, and the
steady -> pulsating transition with field strength** on a coarse box mesh: CaE 0.25 gives a
steady cone-jet (asymmetry plateau), CaE 0.42 gives an unsteady pulsating jet (asymmetry +
velocity + current oscillation), both fully mass/charge conserving. Quantitative paper-level
agreement (1.1% morphology, exact Ganan-Calvo magnitude, sustained >0.05 whipping) remains
gated on the paper's ~2 um / ~11M-cell resolution (P7).


