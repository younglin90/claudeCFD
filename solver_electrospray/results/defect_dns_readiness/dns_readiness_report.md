# Guarded 3D Defect DNS Readiness Runs

These are PIMPLE-enabled guarded 3D runs for the representative defect cases.
Passing this readiness level means the case can run with adaptive dt/retry and bounded force/projection increments; it is still not a resolved breakup DNS.

| Case | Steps | Min dt | Max retry | Max pass metric | Offset growth | Droplets | Breakup events | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| normal sharp tip | 30 | 2.500e-04 | 2 | 9.797e-06 | 1.00 | 1 | 0 | completed |
| 10 deg bent tip | 30 | 2.500e-04 | 2 | 9.662e-06 | 1.00 | 1 | 0 | completed |
| single bump h=0.10Do | 30 | 1.250e-04 | 3 | 9.903e-06 | 1.00 | 1 | 0 | completed |
| split/notched tip d=0.10Do | 30 | 2.500e-04 | 2 | 9.902e-06 | 1.00 | 1 | 0 | completed |
| severe oxidized rough tip | 30 | 2.500e-04 | 2 | 9.663e-06 | 1.00 | 1 | 0 | completed |

## Remaining DNS Blockers

- Increase mesh resolution and replace the column proxy with actual emitter/nozzle geometry.
- Validate PLIC curvature on spherical/cylindrical interfaces before using breakup diameters as manuscript evidence.
- Add true open outflow and droplet exit accounting before plume divergence and q/m distributions are claimed.