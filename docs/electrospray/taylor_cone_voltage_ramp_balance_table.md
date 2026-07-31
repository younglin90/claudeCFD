# Taylor-Cone Voltage-Ramp Balance

This table is generated from the executable static Taylor-cone voltage-ramp balance case used by the reduced electrospray verification suite.

| Voltage fraction | Voltage | Normal electric field | Maxwell pressure | Capillary pressure | Normalized balance residual | Status |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 2.347935e-01 | 1.000000e+00 | RAMP |
| 0.25 | 3.028469e-02 | 1.211387e-01 | 1.467460e-02 | 2.347935e-01 | 9.375000e-01 | RAMP |
| 0.50 | 6.056937e-02 | 2.422775e-01 | 5.869838e-02 | 2.347935e-01 | 7.500000e-01 | RAMP |
| 0.75 | 9.085406e-02 | 3.634162e-01 | 1.320714e-01 | 2.347935e-01 | 4.375000e-01 | RAMP |
| 1.00 | 1.211387e-01 | 4.845550e-01 | 2.347935e-01 | 2.347935e-01 | 1.182127e-16 | PASS |

Acceptance requires a monotone electric-field increase, a monotone force-residual decrease, and a final normalized balance residual below 1e-12.
