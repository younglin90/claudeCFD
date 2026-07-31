# Coupled 2D Droplet Deformation Grid Refinement

This table is generated from the executable coupled 2D EHD droplet deformation observable used by the reduced verification suite.

| Grid | Initial D | Final D | Delta D | Max invariant violation | Gas charge leakage | Max charge relaxation fraction | Max projection ratio | Max pressure correction norm | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.000000e+00 | 6.831441e-02 | 6.831441e-02 | 4.145089e-04 | 0.000000e+00 | 0.000000e+00 | 2.717230e-02 | 4.664598e+05 | PASS |
| 20 | 0.000000e+00 | 7.089112e-02 | 7.089112e-02 | 7.866258e-05 | 0.000000e+00 | 0.000000e+00 | 2.669297e-02 | 5.859077e+05 | PASS |
| 24 | 0.000000e+00 | 7.211345e-02 | 7.211345e-02 | 8.761913e-05 | 0.000000e+00 | 1.250000e-01 | 2.761681e-02 | 7.047147e+05 | PASS |

The medium-to-fine deformation increment divided by the coarse-to-medium increment is 4.743749e-01.

Acceptance requires monotone deformation growth under refinement, refinement ratio <= 0.6, max invariant violation <= 6e-4, gas charge leakage equal to zero, projection ratio below one, and non-zero pressure correction in each projected coupled step. Charge loss is reported separately because this case includes implicit free-charge relaxation.
