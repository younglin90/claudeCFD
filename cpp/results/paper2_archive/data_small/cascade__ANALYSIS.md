# Speed-up cascade — analysis (final)

Headline figure: `Fig_cascade_final.*`, built by `cascade_final.py` from
`cascade_2d_perrep.csv` + `cascade_3d_perrep.csv` (written by `parse_2d.py` and
`parse_hllc_3d.py`). Reduced table: `cascade_final.csv`.
Companion: `Fig_composition.*` (absolute time composition).
Superseded: `Fig_cascade.*`, `Fig_cascade_v2.*`, `cascade_2d.csv`, `cascade_3d.csv`.

## What is being measured

The closed form replaces two stages of the reconstruction — the cell shift (Newton on
tanh → probit identity) and the face value (quadrature → closed form). Everything else,
the quadratic surface fit, MUSCL, BVD selection, flux, MOOD and time integration, is
identical between the two schemes. The gain therefore dilutes at every level of
aggregation, and the cascade is the honest way to present it.

## Reduction procedure

Repeats are contaminated: an unrelated workload shared the machine, unpinned across all
64 logical cpus, while the 2D sweep was pinned to 0–22. Taking a minimum over repeats is
biased, because the S1 minimum and the S2 minimum land in windows of different quietness
(double Mach: S1 repeat 1 ran at 164 s against 209/203 s for the other two, which alone
drives its control to 0.79).

Instead, for every case:

1. pair S1 and S2 by repeat index,
2. form the ratio inside each pair,
3. divide by the **shared-work ratio of the same pair**,
4. take the median over pairs.

The shared bucket (2D: MUSCL + BVD selection; 3D: the o2-LSQ gradient) executes identical
instructions in both schemes, so its ratio estimates how much faster the machine happened
to be during that pair. Dividing by it cancels the drift that step 2 cannot see.

## Result

| case | dim | pairs | kernel | accelerated stages | reconstruction | solver |
|---|---|---|---|---|---|---|
| LeVeque | 2D | 3 | 14.4× | 1.43× | 1.27× | 1.24× |
| shock mixing | 2D | 3 | 14.4× | 1.98× | 1.34× | — |
| shock vortex | 2D | 3 | 14.4× | 1.61× | 1.19× | 1.17× |
| Mach 3 step | 2D | 2 | 14.4× | 1.15× | 1.05× | 1.05× |
| configuration 3 | 2D | 3 | 14.4× | 1.71× | 1.27× | 1.35× |
| double Mach | 2D | 3 | 14.4× | 1.13× | 1.04× | 1.04× |
| deformation | 3D | 2 | 80× | 1.52× | 1.18× | 1.07× |
| spherical blast | 3D | 2 | 80× | 1.58× | 1.32× | 1.19× |
| two cylinder | 3D | 2 | 80× | 1.68× | 1.44× | 1.24× |

`shock_mixing_bench` prints no per-scheme `[WALL]` line, so that case stops at the
reconstruction level.

Solver level: **raw 0.96–1.44×, corrected 1.04–1.35×**. The two raw values below unity
(Mach 3 0.96, double Mach 0.97) sit inside the noise and move above break-even once
their own controls (0.912, 0.921) are divided out. Machine-speed control across the nine
cases: 0.833–1.146.

**Do not quote raw solver-level numbers without the control.** Either report the
reconstruction level, where the ratio is well clear of the noise, or repeat on a quiet
machine.

## Condition audit (all gates pass)

- S1 and S2 command lines differ in exactly one token: `THINCQQ_TANH=1` ↔ `THINCQQ_GAUSS=1`.
  Everything else — `BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1
  BVD_BETA_L=1.4 BVD_BETA_S=0.8`, mesh, `CFD_MAXSTEP`, `taskset` — is identical.
- Newton iterations are reported for every S1 run and for no S2 run.
- `cells`, `steps` and `flux` are single-valued within each case.
- `p_min > 0` in all eight Euler runs.
- `recon_calls=1500` in all six 2D benches (500 steps × 3 Runge-Kutta stages). A single
  value across benches that run different numbers of schemes proves `CHENG3_PROF` counts
  only the cheng3 path, so the profile is attributable to the scheme under test even in
  `leveque_bench`, which also runs mlp_u1 and T-MLP-u-L.
- The profile sum never exceeds the wall of the scheme it is attributed to (91–97 %).

## Repeat spread

3D, re-measured under HLLC on cores 24–46, is an order of magnitude cleaner than the
contended 2D sweep:

| case / scheme | n | wall spread | reconstruction spread |
|---|---|---|---|
| two cylinder S1 | 2 | 0.24 % | 1.56 % |
| two cylinder S2 | 2 | 3.06 % | 5.28 % |
| spherical blast S1 | 2 | 1.59 % | 3.35 % |
| spherical blast S2 | 2 | 9.69 % | 19.86 % |
| deformation S1 | 2 | 4.93 % | 5.60 % |
| deformation S2 | 2 | 3.06 % | 5.34 % |
| 2D (all cases) | 2–3 | 6.4–58.0 % | 6.3–46.9 % |

## Notes and caveats

- The 3D Euler cases were re-run with `N3_FLUX=1` (HLLC, the paper flux and the flux of
  the 2026-07-13 reference logs). The first pass had used the bench default RHLLC; those
  logs are superseded.
- The deformation case runs the scalar `advect3d` driver, which has no Riemann flux at
  all — the flux question does not arise there. It is recorded as `flux=-1`.
- `3d_sphere_s1_r1` was killed nine seconds in during an earlier clean-up and produced no
  data; it was re-run separately, after the main driver released the cores.
- The 2D `geom` bucket lumps the LSQ gradient together with the cell shift, so cases where
  LSQ dominates that bucket understate the cell-shift gain. 3D separates the two.
- Kernel-level numbers (14.4×, 80×) come from the isolated RDTSC micro-benchmark in
  `../gauss_paper_v3/`, not from these runs.
- `mach3_bench` freezes under `HLLC_PVRS=1` — the reflective wall emits no flux and the
  solution stalls at `rho[1.4000,1.4000]`. These runs use `M3_FLUX=hll`; the frozen
  attempt is kept as `rep1/2d_mach3_s*.FROZEN-HLLCPVRS.log` and excluded from the
  aggregation.
- 3D runs are step-capped profiling runs (`CFD_MAXSTEP=100`); the physics state is not
  meaningful and no VTK was written. Metrics lines are persisted in every `3d_*.log`.
