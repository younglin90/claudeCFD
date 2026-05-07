# QA Report — Test A 5-Setting Validation

## Summary

| Setting | Status | err_p | err_u | wall_time | t_final | Comment |
|---------|--------|-------|-------|-----------|---------|----------|
| 1 | FAIL | 3.2596e+21 | 1.3137e+15 | 1.324s | 0.0117s | OK |
| 2 | FAIL | nan | nan | 0.097s | 0.0000s | NaN/Inf in results |
| 3 | FAIL | 7.5043e+28 | 9.8113e+30 | 0.400s | 0.0124s | OK |
| 4 | FAIL | nan | nan | 0.032s | 0.0000s | NaN/Inf in results |
| 5 | FAIL | nan | nan | 0.232s | 0.0000s | NaN/Inf in results |

## Details

### Setting 1
- CFL: 0.1
- use_material_cfl: True
- alpha_scheme: cicsam
- iterative_im1: True
- Status: FAIL
- err_p: 3.259569e+21 (limit 1e-2)
- err_u: 1.313703e+15 (limit 1e-2)
- wall_time: 1.324s
- t_final: 0.011668s
- comment: OK

### Setting 2
- CFL: 0.2
- use_material_cfl: True
- alpha_scheme: cicsam
- iterative_im1: True
- Status: FAIL
- err_p: nan (limit 1e-2)
- err_u: nan (limit 1e-2)
- wall_time: 0.097s
- t_final: 0.000000s
- comment: NaN/Inf in results

### Setting 3
- CFL: 0.1
- use_material_cfl: True
- alpha_scheme: cicsam
- iterative_im1: False
- Status: FAIL
- err_p: 7.504301e+28 (limit 1e-2)
- err_u: 9.811337e+30 (limit 1e-2)
- wall_time: 0.400s
- t_final: 0.012393s
- comment: OK

### Setting 4
- CFL: 0.1
- use_material_cfl: True
- alpha_scheme: tvd
- iterative_im1: True
- Status: FAIL
- err_p: nan (limit 1e-2)
- err_u: nan (limit 1e-2)
- wall_time: 0.032s
- t_final: 0.000000s
- comment: NaN/Inf in results

### Setting 5
- CFL: 0.1
- use_material_cfl: True
- alpha_scheme: mstacs
- iterative_im1: True
- Status: FAIL
- err_p: nan (limit 1e-2)
- err_u: nan (limit 1e-2)
- wall_time: 0.232s
- t_final: 0.000000s
- comment: NaN/Inf in results

