# 3D defect DNS production matrix smoke report

This smoke run exercises all manuscript production cases with the guarded 3D EHD stepper.
The grid is intentionally coarse; use the exported case matrix for production mesh/timestep refinement.

| case | defect | steps | final V | max pass metric | q/m | plume angle | status |
|---|---:|---:|---:|---:|---:|---:|---|
| P0_normal | normal | 8 | 5.000e-02 | 8.680e-06 | 6.979e-07 | 3.442e-02 | completed |
| P1_bent_5deg | bent | 8 | 1.250e-02 | 5.684e-06 | 3.074e-07 | 4.408e+00 | completed |
| P1_bent_10deg | bent | 8 | 1.250e-02 | 5.586e-06 | 4.437e-07 | 9.111e+00 | completed |
| P2_bump_005Do | bump | 8 | 5.000e-02 | 8.680e-06 | 6.979e-07 | 3.442e-02 | completed |
| P2_bump_010Do | bump | 8 | 5.000e-02 | 8.680e-06 | 6.979e-07 | 3.442e-02 | completed |
| P3_split_005Do | split | 8 | 1.250e-02 | 8.594e-06 | 2.053e-07 | 6.758e-03 | completed |
| P3_split_010Do | split | 8 | 1.250e-02 | 8.594e-06 | 2.053e-07 | 6.758e-03 | completed |
| P4_oxidized_moderate | oxidized | 8 | 5.000e-02 | 8.680e-06 | 6.964e-07 | 3.442e-02 | completed |
| P4_oxidized_severe | oxidized | 8 | 5.000e-02 | 8.680e-06 | 6.955e-07 | 3.442e-02 | completed |
