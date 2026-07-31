# Defect Electrospray Screening Results

These are first-pass calculations for ordering the planned 3D defect study.
They combine Candido-scale dimensional force/current proxies with the current reduced 3D EHD non-axisymmetric response stepper.
Do not interpret these rows as resolved droplet-breakup DNS.

Reference field E0 = 2.671252e+06 V/m; reference CaE = 1.567252e-01.

| Difficulty | Case | Local CaE | E local [V/m] | Force ratio | q/m proxy [C/kg] | 3D offset | Plume proxy [deg] | Class |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | normal sharp tip | 0.1567 | 2.671e+06 | 0.0637 | 1.656e+02 | 0.0000 | 2.54 | stable cone-jet likely |
| 2 | 2 deg bent tip | 0.1663 | 2.751e+06 | 0.0648 | 1.706e+02 | 0.0192 | 5.39 | stable cone-jet likely |
| 2 | 5 deg bent tip | 0.1828 | 2.885e+06 | 0.0683 | 1.789e+02 | 0.0321 | 7.83 | asymmetric cone-jet risk |
| 2 | 10 deg bent tip | 0.2109 | 3.099e+06 | 0.0720 | 1.921e+02 | 0.0641 | 13.29 | whipping/multi-jet risk |
| 3 | single bump h=0.02Do | 0.1896 | 2.938e+06 | 0.0663 | 1.822e+02 | 0.0192 | 5.39 | stable cone-jet likely |
| 3 | single bump h=0.05Do | 0.2371 | 3.286e+06 | 0.0694 | 2.037e+02 | 0.0192 | 5.93 | stable cone-jet likely |
| 3 | single bump h=0.10Do | 0.3295 | 3.873e+06 | 0.0696 | 2.402e+02 | 0.0449 | 12.87 | asymmetric cone-jet risk |
| 4 | split/notched tip d=0.02Do | 0.2182 | 3.152e+06 | 0.0674 | 1.955e+02 | 0.0321 | 7.83 | asymmetric cone-jet risk |
| 4 | split/notched tip d=0.05Do | 0.2985 | 3.686e+06 | 0.0703 | 2.286e+02 | 0.0567 | 13.74 | whipping/multi-jet risk |
| 4 | split/notched tip d=0.10Do | 0.4529 | 4.541e+06 | 0.0736 | 2.816e+02 | 0.0710 | 22.30 | whipping/multi-jet risk |
| 5 | mild oxidized rough tip | 0.1828 | 2.885e+06 | 0.0668 | 1.342e+02 | 0.0192 | 5.75 | stable cone-jet likely |
| 5 | moderate oxidized rough tip | 0.2073 | 3.072e+06 | 0.0674 | 1.048e+02 | 0.0449 | 9.72 | asymmetric cone-jet risk |
| 5 | severe oxidized rough tip | 0.2649 | 3.473e+06 | 0.0667 | 6.460e+01 | 0.0641 | 13.81 | whipping/multi-jet risk |

Recommended next step: run the lowest-risk C0/C1/C2 cases with a resolved geometry mesh before spending time on split-tip breakup.
