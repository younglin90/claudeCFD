# Charge-Free Dielectric Maxwell-Stress Droplet History

This table is generated from the executable coupled 2D dielectric droplet gate with zero free charge and Maxwell-stress divergence enabled.
The values are reduced-kernel deformation diagnostics and must not be described as a resolved full two-phase Navier-Stokes droplet benchmark.

| Step | Deformation D | Max Maxwell acceleration | Max capillary acceleration | Projection ratio | Max invariant violation | Status |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | PASS |
| 1 | 7.825202e-05 | 6.926858e+03 | 1.389883e-02 | 1.289901e-14 | 7.540203e-07 | PASS |
| 2 | 2.353573e-04 | 6.924165e+03 | 1.389795e-02 | 7.866983e-14 | 1.336943e-06 | PASS |
| 3 | 4.714765e-04 | 6.918889e+03 | 1.389653e-02 | 3.395750e-13 | 1.820437e-06 | PASS |
| 4 | 7.861710e-04 | 6.911116e+03 | 1.389467e-02 | 5.302527e-13 | 3.160868e-06 | PASS |
| 5 | 1.179166e-03 | 6.900923e+03 | 1.389240e-02 | 7.054001e-13 | 4.867022e-06 | PASS |

The final deformation is 1.179166e-03; all deformation increments are positive: True.
Acceptance requires monotone deformation, zero free-charge body acceleration, active Maxwell/capillary/projection terms, and max invariant violation <= 1e-5.

## Voltage-Squared Scaling Check

The same coupled stepper is rerun at V=40 and V=80. Maxwell-stress acceleration and final deformation should scale with the applied voltage squared.

| Quantity | V=40 | V=80 | Observed ratio | Expected ratio | Status |
|---|---:|---:|---:|---:|---:|
| Final deformation D | 2.954006e-04 | 1.179166e-03 | 3.991751e+00 | 4.000000e+00 | PASS |
| Max Maxwell acceleration | 1.731715e+03 | 6.926858e+03 | 4.000000e+00 | 4.000000e+00 | PASS |

## Grid-Refinement Check

| Grid | Initial D | Final D | Increment from previous grid | Status |
|---:|---:|---:|---:|---:|
| 16 | 0.000000e+00 | 8.605147e-04 | 0.000000e+00 | PASS |
| 20 | 0.000000e+00 | 1.086392e-03 | 2.258770e-04 | PASS |
| 24 | 0.000000e+00 | 1.179166e-03 | 9.277391e-05 | PASS |

The medium-to-fine deformation increment divided by the coarse-to-medium increment is 4.107276e-01.

## Timestep-Refinement Check

| dt | Steps | Final D | Relative delta | Status |
|---:|---:|---:|---:|---:|
| 2.000000e-04 | 5 | 1.179166e-03 | 0.000000e+00 | PASS |
| 1.000000e-04 | 10 | 1.083780e-03 | 8.089217e-02 | PASS |
