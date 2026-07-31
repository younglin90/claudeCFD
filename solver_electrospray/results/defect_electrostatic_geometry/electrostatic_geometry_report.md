# 3D Defect Electrostatic Geometry Screening

This calculation embeds simplified 3D emitter defects on a Cartesian grid and solves a Laplace electrostatic problem.
It ranks cases by local field enhancement and high-field centroid offset near the tip.

| Difficulty | Case | Max E [V/m] | Enhancement | CaE(Emax) | High-field offset [um] | Lateral asymmetry | Risk |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | normal sharp tip | 4.819e+06 | 1.80 | 0.510 | 0.98 | 0.382 | whipping/multi-jet risk |
| 2 | 2 deg bent tip | 4.992e+06 | 1.87 | 0.547 | 8.07 | 0.382 | whipping/multi-jet risk |
| 2 | 5 deg bent tip | 4.820e+06 | 1.80 | 0.510 | 29.43 | 0.380 | whipping/multi-jet risk |
| 2 | 10 deg bent tip | 4.862e+06 | 1.82 | 0.519 | 57.14 | 0.379 | whipping/multi-jet risk |
| 3 | single bump h=0.02Do | 5.760e+06 | 2.16 | 0.729 | 6.26 | 0.393 | whipping/multi-jet risk |
| 3 | single bump h=0.05Do | 5.760e+06 | 2.16 | 0.729 | 6.26 | 0.393 | whipping/multi-jet risk |
| 3 | single bump h=0.10Do | 6.415e+06 | 2.40 | 0.904 | 4.06 | 0.415 | whipping/multi-jet risk |
| 4 | split/notched tip d=0.02Do | 4.848e+06 | 1.82 | 0.516 | 0.00 | 0.380 | whipping/multi-jet risk |
| 4 | split/notched tip d=0.05Do | 4.848e+06 | 1.82 | 0.516 | 0.00 | 0.380 | whipping/multi-jet risk |
| 4 | split/notched tip d=0.10Do | 4.848e+06 | 1.82 | 0.516 | 0.00 | 0.380 | whipping/multi-jet risk |
| 5 | mild oxidized rough tip | 4.171e+06 | 1.56 | 0.382 | 0.00 | 0.355 | asymmetric cone-jet risk |
| 5 | moderate oxidized rough tip | 3.200e+06 | 1.20 | 0.225 | 0.00 | 0.301 | stable cone-jet likely |
| 5 | severe oxidized rough tip | 1.907e+06 | 0.71 | 0.080 | 0.40 | 0.188 | stable cone-jet likely |

The highest-risk geometry cases should be advanced to reduced 3D EHD time-history calculations.
