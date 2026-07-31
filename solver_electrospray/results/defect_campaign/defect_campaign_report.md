# Electrospray Tip-Defect Calculation Campaign

This report combines three calculation levels: dimensional defect screening, 3D electrostatic geometry, and stabilized reduced 3D EHD drift response.
The campaign score is only a prioritization metric for the next simulations, not a validated physical instability threshold.

## Priority Order

| Rank | Difficulty | Case | Score | Field enh. | High-field offset (um) | Final offset | Growth | Reduced class |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 4 | split/notched tip d=0.10Do | 0.801 | 1.82 | 0.00 | 0.1273 | 1.74 | whipping-growth candidate |
| 2 | 2 | 10 deg bent tip | 0.789 | 1.82 | 57.14 | 0.0851 | 1.50 | whipping-growth candidate |
| 3 | 3 | single bump h=0.10Do | 0.631 | 2.40 | 4.06 | 0.0758 | 1.55 | whipping-growth candidate |
| 4 | 4 | split/notched tip d=0.05Do | 0.628 | 1.82 | 0.00 | 0.0871 | 1.54 | whipping-growth candidate |
| 5 | 5 | severe oxidized rough tip | 0.561 | 0.71 | 0.40 | 0.0885 | 1.56 | whipping-growth candidate |
| 6 | 2 | 5 deg bent tip | 0.530 | 1.80 | 29.43 | 0.0495 | 1.43 | asymmetric-growth candidate |
| 7 | 5 | moderate oxidized rough tip | 0.477 | 1.20 | 0.00 | 0.0673 | 1.38 | whipping-growth candidate |
| 8 | 4 | split/notched tip d=0.02Do | 0.459 | 1.82 | 0.00 | 0.0509 | 1.47 | asymmetric-growth candidate |
| 9 | 3 | single bump h=0.05Do | 0.438 | 2.16 | 6.26 | 0.0319 | 1.72 | asymmetric-growth candidate |
| 10 | 3 | single bump h=0.02Do | 0.406 | 2.16 | 6.26 | 0.0263 | 1.41 | asymmetric-growth candidate |
| 11 | 2 | 2 deg bent tip | 0.385 | 1.87 | 8.07 | 0.0258 | 1.38 | stable reduced response |
| 12 | 5 | mild oxidized rough tip | 0.355 | 1.56 | 0.00 | 0.0288 | 1.55 | asymmetric-growth candidate |
| 13 | 1 | normal sharp tip | 0.277 | 1.80 | 0.98 | 0.0219 | 1.18 | stable reduced response |

## Calculation Interpretation

- Start manuscript figures from the normal, 10 deg bent, 0.10Do bump, 0.10Do split, and severe oxidized cases.
- Bump defects mainly amplify local electric field; bending mainly shifts the high-field centroid; split defects dominate combined whipping risk.
- Oxidation reduces local electrostatic enhancement in the geometry solve, but rough asymmetric drift still increases reduced plume/offset risk.
- Projection-enabled long-time 3D whipping remains a solver-development target, because the coarse PIMPLE-style reduced run became non-finite before 40 steps.
