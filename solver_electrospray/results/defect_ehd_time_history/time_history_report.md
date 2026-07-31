# Reduced 3D Defect EHD Time-History Results

These calculations run the current reduced 3D EHD stepper for each defect severity in a stabilized drift-response mode.
The initial lateral drift scales with defect asymmetry and field enhancement so the coarse 3D mesh can rank non-axisymmetric growth.
They are not resolved cone-jet whipping or droplet-breakup DNS.

A projection-enabled PIMPLE-style trial on this same coarse mesh became non-finite after about 13-16 steps, so pressure projection is disabled here.
That failure is a solver-readiness finding: resolved 3D whipping needs a bounded Maxwell/charge update, smaller physical timestep control, and sharper VOF/interface handling.

| Difficulty | Case | Steps | Initial offset | Final offset | Growth | Max offset | Max pass metric | Gas leakage | Class |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | normal sharp tip | 100 | 0.0186 | 0.0219 | 1.18 | 0.0219 | 3.285e-05 | 0.000e+00 | stable reduced response |
| 2 | 2 deg bent tip | 100 | 0.0186 | 0.0258 | 1.38 | 0.0258 | 6.793e-05 | 0.000e+00 | stable reduced response |
| 2 | 5 deg bent tip | 100 | 0.0348 | 0.0495 | 1.43 | 0.0495 | 1.610e-04 | 0.000e+00 | asymmetric-growth candidate |
| 2 | 10 deg bent tip | 100 | 0.0566 | 0.0851 | 1.50 | 0.0851 | 3.262e-04 | 0.000e+00 | whipping-growth candidate |
| 3 | single bump h=0.02Do | 100 | 0.0186 | 0.0263 | 1.41 | 0.0263 | 7.256e-05 | 0.000e+00 | asymmetric-growth candidate |
| 3 | single bump h=0.05Do | 100 | 0.0186 | 0.0319 | 1.72 | 0.0319 | 1.219e-04 | 0.000e+00 | asymmetric-growth candidate |
| 3 | single bump h=0.10Do | 100 | 0.0489 | 0.0758 | 1.55 | 0.0758 | 2.966e-04 | 0.000e+00 | whipping-growth candidate |
| 4 | split/notched tip d=0.02Do | 100 | 0.0348 | 0.0509 | 1.47 | 0.0509 | 1.759e-04 | 0.000e+00 | asymmetric-growth candidate |
| 4 | split/notched tip d=0.05Do | 100 | 0.0566 | 0.0871 | 1.54 | 0.0871 | 3.493e-04 | 0.000e+00 | whipping-growth candidate |
| 4 | split/notched tip d=0.10Do | 100 | 0.0730 | 0.1273 | 1.74 | 0.1273 | 7.367e-04 | 0.000e+00 | whipping-growth candidate |
| 5 | mild oxidized rough tip | 100 | 0.0186 | 0.0288 | 1.55 | 0.0288 | 9.506e-05 | 0.000e+00 | asymmetric-growth candidate |
| 5 | moderate oxidized rough tip | 100 | 0.0489 | 0.0673 | 1.38 | 0.0673 | 2.039e-04 | 0.000e+00 | whipping-growth candidate |
| 5 | severe oxidized rough tip | 100 | 0.0566 | 0.0885 | 1.56 | 0.0885 | 3.656e-04 | 0.000e+00 | whipping-growth candidate |
