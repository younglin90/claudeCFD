# paper2 §3.12 — 2D speedup cascade: total solver wall (re-measured 2026-07-28)

Data: `cpp/results/paper2_cascade/rep{1,2,3}/2d_<case>_<scheme>.log` (full stdout, `### `-prefixed provenance header per run). The top-level `2d_<case>_<scheme>.log` is a copy of the MEDIAN-wall repeat of that pair. `cascade_2d.csv` holds the machine-readable table.

## 1. What was added to the apps

Total wall time was missing from every 2D bench except two. Added to `cpp/apps/*.cpp` ONLY (no solver header touched); `CHENG3_PROF` stage output untouched.

| app | change | key |
|---|---|---|
| `leveque_bench.cpp` | `<chrono>`; `run()` gained a `label` arg and now times `solve_adv2d` and prints; `[WALL] TOTAL` at end of `main` | `[WALL] <label> wall=%.3fs steps=%d` |
| `mach3_bench.cpp` | same pattern around `solve_euler2d` in `run()` + `[WALL] TOTAL` | same |
| `double_mach_bench.cpp` | same | same |
| `config3_bench.cpp` | same | same |
| `validation_smoke.cpp` | ALREADY timed the solve (table column `wall(s)`); added one keyed line so the key matches the 3D benches | `[WALL] <case> wall=...s` |
| `shock_mixing_bench.cpp` | **NO CHANGE — already printed `wall=51.0s`** in its status line, same `wall=` key as the 3D benches | `status=... wall=%.1fs` |

`[WALL] <label>` = wall of ONE time-integration loop. `[WALL] TOTAL` = whole app (several schemes per app), so TOTAL is NOT the cascade number — use the labelled line of the run that actually uses the cheng3 recon (`BVD` / `T-MLP-u`).

### Diff summary (file:line, post-edit numbering)

`cpp/` is NOT tracked by git (`git status` reports `?? cpp/apps/`), so no `git diff` exists. Added lines, verbatim:

```
leveque_bench.cpp:10       #include <chrono>
leveque_bench.cpp:30       run(...) signature += `, const char* label = nullptr`
leveque_bench.cpp:35,37-38 t0_ around solve_adv2d + printf("[WALL] %s wall=%.3fs steps=%d")
leveque_bench.cpp:76       t_all_ start (before the mlp_u1 run)
leveque_bench.cpp:91-92    printf("[WALL] TOTAL wall=%.3fs")
leveque_bench.cpp:71,76,81 call sites pass labels "mlp_u1" / "T-MLP-u-L" / "BVD"
mach3_bench.cpp:16,29,38,41-42,125,154-155        same pattern around solve_euler2d
  call-site labels: "mlp_u1", "T-MLP-u-L", "BVD", "C_line"
double_mach_bench.cpp:15,72,86,89-90,177,203-204  same pattern
  call-site labels: "mlp_u1", "T-MLP-u-L", "ORDER-2", "BVD"
config3_bench.cpp:19,61,64,69-70,129,138-139      same pattern
  call-site labels: "mlp_u1", "T-MLP-u"
validation_smoke.cpp:107   ONE line added (it already timed the solve at :96-98):
                           printf("[WALL] %s wall=%.3fs steps=%d", C.name, wall, r.n_steps)
shock_mixing_bench.cpp     UNCHANGED
```
No `include/cfd/*.hpp` was touched. Rebuild verified by md5 change of every binary (leveque_bench ad1f98->1b68ba, mach3_bench 905153->e943a4, double_mach_bench b9aaaa->107cbf, config3_bench 015d13->947b51, validation_smoke b3f74b->9129fb; shock_mixing_bench md5 unchanged = d1a50b, as expected since its source was not edited).

## 2. Conditions

- `CFD_MAXSTEP=500 CHENG3_PROF=1` (as in the original 2026-07-13 `paper2_recon_profile` run)
- S1 `BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1`
- S2 `BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_GAUSS=1`
- cores `taskset -c 0-22:2` = 12 PHYSICAL cores, `OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores` (verified: the 12 OMP threads sat on cpus 0,2,...,22; the parallel 3D agent sat on cpu 24+, no physical-core overlap)
- 3 repeats per (case, scheme); mach3 has 2 (see §6)

### Exact commands
```
leveque      taskset -c 0-22:2 env OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores CFD_MAXSTEP=500 CHENG3_PROF=1  BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1 ./leveque_bench 200
shockmixing  taskset -c 0-22:2 env OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores CFD_MAXSTEP=500 CHENG3_PROF=1  BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1 ./shock_mixing_bench 400 80
shockvortex  taskset -c 0-22:2 env OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores CFD_MAXSTEP=500 CHENG3_PROF=1 VS_ONLY=16 VS_NSCALE=4 BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1 ./validation_smoke
mach3        taskset -c 0-22:2 env OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores CFD_MAXSTEP=500 CHENG3_PROF=1 M3_MESH=uniform M3_BVD_ONLY=1 M3_FLUX=hll M3_CFL=0.3 BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1 ./mach3_bench 480 160 4.0
config3      taskset -c 0-22:2 env OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores CFD_MAXSTEP=500 CHENG3_PROF=1 C3_FLUX=llf C3_BC=dirichlet C3_SKIP_MLP=1 BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1 ./config3_bench 280 0.8
doublemach   taskset -c 0-22:2 env OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores CFD_MAXSTEP=500 CHENG3_PROF=1 DM_MESH2D=/home/younglin90/work/claude_code/claudeCFD/cpp/results/paper_final/meshes/dmr_wc_paper.mesh2d DM_SINGLE=1 BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1 ./double_mach_bench 240 60 0.2
```

## 3. Cascade table — median over repeats (s)

| case | scheme | MUSCL | geom | face | THINC | BVD_sel | recon sum | solver wall | min wall | recon/wall | app total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| leveque | S1 | 5.8 | 9.2 | 10.7 | 20.0 | 2.4 | 28.1 | **30.6** | 26.4 | 0.919 | 47.1 |
| leveque | S2 | 5.8 | 5.6 | 7.9 | 13.5 | 2.1 | 21.5 | **24.1** | 22.9 | 0.894 | 40.4 |
| shockmixing | S1 | 31.4 | 20.6 | 20.2 | 40.9 | 2.3 | 76.5 | **82.6** | 72.2 | 0.926 | - |
| shockmixing | S2 | 34.5 | 7.5 | 13.4 | 20.8 | 2.6 | 59.1 | **66.0** | 60.4 | 0.899 | - |
| shockvortex | S1 | 60.2 | 22.6 | 30.2 | 52.9 | 5.2 | 118.6 | **125.0** | 118.3 | 0.948 | - |
| shockvortex | S2 | 73.6 | 12.2 | 22.0 | 34.2 | 5.3 | 113.1 | **120.2** | 101.7 | 0.938 | - |
| mach3 | S1 | 42.2 | 12.4 | 14.5 | 26.9 | 4.4 | 73.5 | **76.0** | 71.4 | 0.967 | 134.5 |
| mach3 | S2 | 46.6 | 10.0 | 15.7 | 25.7 | 4.8 | 77.1 | **79.8** | 71.5 | 0.966 | 131.8 |
| config3 | S1 | 60.5 | 41.1 | 28.4 | 69.5 | 5.1 | 135.1 | **138.1** | 121.4 | 0.976 | 138.3 |
| config3 | S2 | 56.8 | 13.7 | 22.2 | 35.9 | 4.6 | 93.6 | **96.3** | 95.9 | 0.971 | 96.5 |
| doublemach | S1 | 203.8 | 51.0 | 67.2 | 118.2 | 26.4 | 349.0 | **360.9** | 283.4 | 0.967 | 361.7 |
| doublemach | S2 | 229.8 | 47.3 | 65.0 | 113.7 | 27.0 | 370.2 | **382.4** | 350.3 | 0.967 | 383.0 |

`recon sum` = MUSCL + THINC + BVD_sel  (THINC = geom + face).

## 3b. THE CASCADE — S1(tanh) / S2(GAUSS) speedup at each aggregation layer

Analogue of the 3D deform chain (80x kernel -> 4.6x stage -> 1.37x recon -> 1.29x solver). Finest layer available from `CHENG3_PROF` in 2D is `geom` (the cell-D / moment kernel where GAUSS replaces the tanh Newton solve); `face` is the shared quadrature, so it dilutes.

| case | geom (kernel) | THINC (stage) | recon (all) | solver (total) |
|---|---|---|---|---|
| leveque | 1.64x | 1.47x | 1.31x | 1.27x (min-based 1.15x) |
| shockmixing | 2.77x | 1.96x | 1.29x | 1.25x (min-based 1.20x) |
| shockvortex | 1.86x | 1.55x | 1.05x | 1.04x (min-based 1.16x) |
| mach3 | 1.24x | 1.05x | 0.95x | 0.95x (min-based 1.00x) |
| config3 | 3.00x | 1.93x | 1.44x | 1.43x (min-based 1.27x) |
| doublemach | 1.08x | 1.04x | 0.94x | 0.94x (min-based 0.81x) |

The geom and THINC columns come from within-run counters and are solid. The recon and solver columns divide two DIFFERENT processes' walls, so they carry the full contention noise of §5 — on the Euler cases (mach3, doublemach) the S1/S2 solver ratio is ~1.0 and is buried in that noise. Note this is not new: the 2026-07-13 breakdown also had S1 ~ S2 on the Euler benches (mach3 THINC 16.3 vs 16.0 s; doublemach 60.6 vs 56.1 s). GAUSS's win is large only where the THINC stage dominates (leveque advection, shockmixing, config3).

## 4. The layer that matters: recon → solver dilution (WITHIN-run ratio)

Absolute walls are noisy on this machine (§6), but `recon_sum / solver_wall` is measured inside a single process, so it is contention-robust. This is the last hop of the cascade.

| case | scheme | rep1 | rep2 | rep3 | median | spread |
|---|---|---|---|---|---|---|
| leveque | S1 | 0.921 | 0.911 | 0.919 | 0.919 | 1.1% |
| leveque | S2 | 0.893 | 0.895 | 0.894 | 0.894 | 0.3% |
| shockmixing | S1 | 0.926 | 0.921 | 0.926 | 0.926 | 0.6% |
| shockmixing | S2 | 0.899 | 0.895 | 0.900 | 0.899 | 0.6% |
| shockvortex | S1 | 0.948 | 0.949 | 0.947 | 0.948 | 0.2% |
| shockvortex | S2 | 0.872 | 0.941 | 0.938 | 0.938 | 7.6% |
| mach3 | S1 | - | 0.966 | 0.968 | 0.967 | 0.2% |
| mach3 | S2 | - | 0.965 | 0.967 | 0.966 | 0.3% |
| config3 | S1 | 0.911 | 0.976 | 0.978 | 0.976 | 7.0% |
| config3 | S2 | 0.971 | 0.971 | 0.972 | 0.971 | 0.0% |
| doublemach | S1 | 0.966 | 0.968 | 0.967 | 0.967 | 0.2% |
| doublemach | S2 | 0.966 | 0.968 | 0.967 | 0.967 | 0.2% |

## 5. Per-repeat solver wall (timing reliability)

| case | scheme | rep1 | rep2 | rep3 | median | min | spread |
|---|---|---|---|---|---|---|---|
| leveque | S1 | 26.4 | 32.6 | 30.6 | 30.6 | 26.4 | **20.7%** |
| leveque | S2 | 22.9 | 28.5 | 24.1 | 24.1 | 22.9 | **22.1%** |
| shockmixing | S1 | 72.2 | 83.9 | 82.6 | 82.6 | 72.2 | **14.7%** |
| shockmixing | S2 | 60.4 | 66.0 | 68.2 | 66.0 | 60.4 | **12.0%** |
| shockvortex | S1 | 118.3 | 125.0 | 127.7 | 125.0 | 118.3 | **7.6%** |
| shockvortex | S2 | 160.7 | 120.2 | 101.7 | 120.2 | 101.7 | **46.3%** |
| mach3 | S1 | - | 80.6 | 71.4 | 76.0 | 71.4 | **12.1%** |
| mach3 | S2 | - | 88.2 | 71.5 | 79.8 | 71.5 | **20.9%** |
| config3 | S1 | 186.7 | 121.4 | 138.1 | 138.1 | 121.4 | **43.9%** |
| config3 | S2 | 102.0 | 96.3 | 95.9 | 96.3 | 95.9 | **6.2%** |
| doublemach | S1 | 283.4 | 370.1 | 360.9 | 360.9 | 283.4 | **25.7%** |
| doublemach | S2 | 398.2 | 382.4 | 350.3 | 382.4 | 350.3 | **12.7%** |

## 6. Honest limitations

1. **A foreign, UNPINNED workload shared the machine the whole time.** From 19:02 a third party's jobs (`SEMO_Robuster_*`, `SEMO_SPH_*`, plus cmake/gmake/cc1plus and 3 pytest processes) ran with affinity `0-63`, i.e. free to land on cpus 0-22. loadavg went 16 -> 68 and settled around 25-35 (see §7). My 12 threads were pinned, but pinning does not stop unpinned threads from time-slicing the same cpus, and the SMT siblings (cpus 1,3,...,23) were never mine. Repeat-to-repeat spread is therefore **7-46%** and the absolute walls should be treated as upper bounds with ~20% error bars. The within-run ratios in §4 are the trustworthy part.
2. **rep1's S2 half is additionally self-contaminated**: between 19:14 and 19:22 I ran mach3 diagnostics on the same core set. `shockvortex_s2` rep1 (160.7 s vs 101.7/120.2 in rep2/3) and `doublemach_s2` rep1 (398.2 s) are that artefact. Later diagnostics were moved to cpus 48-62.
3. **mach3 could not be run with the literal S1/S2 recipe.** `HLLC_PVRS=1` (part of the documented recipe) makes the Mach-3 reflective wall emit no flux, so the uniform inflow state is preserved exactly: after 500 steps the field is `rho[1.4000,1.4000] p_min=1.0000 max|drho|=0.0000`. Isolated at 120x40/300 steps: baseline -> `rho[0.3818,5.8629]`; `+HLLC_HLLBLEND=0` -> `rho[0.3837,5.8944]`; `+HLLC_PVRS=1` -> **frozen**. The 2026-07-13 log shows a developed field, so that run cannot have used HLLC+PVRS either. mach3 was therefore re-run with its canonical flux `M3_FLUX=hll M3_CFL=0.3` (`paper_final/paper_runner.sh`), which reproduces the old `p_min` to 3 digits (0.3079 vs 0.3083). The frozen-field logs are kept as `rep1/2d_mach3_s{1,2}.FROZEN-HLLCPVRS.log` and are NOT in the CSV. mach3 hence has 2 repeats.
4. **The 2026-07-13 absolute stage times do not reproduce** (§8): the C++ recon code changed after that date (2D analytic-moment cache, 2026-07-26; recon perf work) and the old run's thread count is unrecorded — the old runner script no longer exists anywhere in the repo, so the case settings had to be reverse-engineered from the log headers. What DOES reproduce is the case setup (cells, grid, t_end, flux id, steps) and the physics diagnostics — see §8.
5. `config3` is run with `C3_FLUX=llf` because the old log header says `flux=0` = `FLUX_LLF`. That is NOT the S1/S2 paper flux (HLLC), and LLF is cheaper than HLLC, so config3's `recon/wall` fraction is biased HIGH relative to a paper-recipe run. Kept for comparability with the old breakdown; flag it if the cascade figure is meant to show the paper configuration.
6. `shockmixing` exits rc=1 (status=INCOMPLETE) — expected, `CFD_MAXSTEP=500` stops it at t=8.92/120. Same as the old run.

## 7. loadavg per run

| case | scheme | rep | loadavg before | loadavg after |
|---|---|---|---|---|
| leveque | S1 | rep1 | 16.39 | 21.48 |
| leveque | S1 | rep2 | 28.52 | 34.27 |
| leveque | S1 | rep3 | 27.87 | 32.06 |
| leveque | S2 | rep1 | 26.76 | 27.74 |
| leveque | S2 | rep2 | 26.39 | 28.87 |
| leveque | S2 | rep3 | 25.41 | 28.08 |
| shockmixing | S1 | rep1 | 21.48 | 25.30 |
| shockmixing | S1 | rep2 | 34.27 | 30.50 |
| shockmixing | S1 | rep3 | 32.06 | 33.96 |
| shockmixing | S2 | rep1 | 27.74 | 30.76 |
| shockmixing | S2 | rep2 | 28.87 | 28.58 |
| shockmixing | S2 | rep3 | 28.08 | 28.97 |
| shockvortex | S1 | rep1 | 25.30 | 27.42 |
| shockvortex | S1 | rep2 | 30.50 | 26.15 |
| shockvortex | S1 | rep3 | 33.96 | 34.64 |
| shockvortex | S2 | rep1 | 30.76 | 36.71 |
| shockvortex | S2 | rep2 | 28.58 | 26.71 |
| shockvortex | S2 | rep3 | 28.97 | 30.08 |
| mach3 | S1 | rep2 | 26.15 | 24.17 |
| mach3 | S1 | rep3 | 34.64 | 27.43 |
| mach3 | S2 | rep2 | 26.71 | 30.70 |
| mach3 | S2 | rep3 | 30.08 | 30.30 |
| config3 | S1 | rep1 | 67.55 | 42.64 |
| config3 | S1 | rep2 | 24.17 | 21.27 |
| config3 | S1 | rep3 | 27.43 | 28.45 |
| config3 | S2 | rep1 | 33.63 | 39.93 |
| config3 | S2 | rep2 | 30.70 | 28.13 |
| config3 | S2 | rep3 | 30.30 | 28.30 |
| doublemach | S1 | rep1 | 42.64 | 26.76 |
| doublemach | S1 | rep2 | 21.27 | 26.39 |
| doublemach | S1 | rep3 | 28.45 | 25.41 |
| doublemach | S2 | rep1 | 39.93 | 40.25 |
| doublemach | S2 | rep2 | 28.13 | 27.87 |
| doublemach | S2 | rep3 | 28.30 | 30.52 |

## 8. Reproduction check vs the ORIGINAL 2026-07-13 breakdown

### 8a. Case setup + physics diagnostics (this is the real check)

(`%` signs below are literal percentages.)

| case | old | new | verdict |
|---|---|---|---|
| leveque | `N=200 cells=160000`, BVD `L1=8.9325e-02 cone_pk=0.968` | `N=200 cells=160000`, BVD `L1=8.9319e-02 cone_pk=0.969` | same case; 7e-5 relative drift (code changed) |
| shockmixing | `cells=64000 grid=400x80 flux=2 cfl=0.40`, `t=8.9218 rho[0.9557,1.9756]` | same header, `t=8.9231 rho[0.9559,1.9697]` | same case, small drift |
| shockvortex | `100x50 steps=500 t=0.096/0.350 rho_min=9.15e-01 p_min=8.84e-01` | identical to all printed digits | **exact match** (confirms `VS_ONLY=16 VS_NSCALE=4`, default RHLLC flux) |
| mach3 | `cells=129024 grid=480x160 mesh=unstructured_2d t_end=4.00` | identical header (confirms `M3_MESH=uniform 480 160 4.0`); flux had to change, see §6.3 | setup match, flux differs |
| config3 | `cells=156800 t_end=0.800 flux=0`, `ens=29.33266 rho_min=0.1378 p_min=0.0290` | identical header, `ens=28.97663 rho_min=0.1378 p_min=0.0290` | same case, 1.2% ens drift |
| doublemach | `cells=370421 nodes=186212`, BVD `rho[1.400,18.239] p_min=1.0000` | identical header, `rho[1.399,18.181] p_min=0.9991` | same case, small drift |

### 8b. Stage times old -> new (median)

| case | scheme | MUSCL | THINC | geom | face | BVD_sel | THINC/MUSCL old->new |
|---|---|---|---|---|---|---|---|
| leveque | S1 | 4.2->5.8 (1.37x) | 11.3->20.0 (1.76x) | 5.3->9.2 (1.74x) | 6.0->10.7 (1.78x) | 1.6->2.4 (1.48x) | 2.69 -> 3.45 |
| leveque | S2 | 4.1->5.8 (1.41x) | 9.2->13.5 (1.47x) | 3.9->5.6 (1.46x) | 5.4->7.9 (1.47x) | 1.6->2.1 (1.33x) | 2.24 -> 2.33 |
| shockmixing | S1 | 23.7->31.4 (1.32x) | 21.2->40.9 (1.93x) | 10.5->20.6 (1.96x) | 10.7->20.2 (1.89x) | 1.9->2.3 (1.20x) | 0.89 -> 1.30 |
| shockmixing | S2 | 24.1->34.5 (1.43x) | 12.5->20.8 (1.66x) | 4.9->7.5 (1.52x) | 7.6->13.4 (1.76x) | 1.9->2.6 (1.34x) | 0.52 -> 0.60 |
| shockvortex | S1 | 41.7->60.2 (1.44x) | 28.2->52.9 (1.87x) | 12.3->22.6 (1.84x) | 15.9->30.2 (1.90x) | 3.9->5.2 (1.34x) | 0.68 -> 0.88 |
| shockvortex | S2 | 42.2->73.6 (1.74x) | 20.7->34.2 (1.65x) | 8.0->12.2 (1.52x) | 12.7->22.0 (1.73x) | 3.9->5.3 (1.36x) | 0.49 -> 0.46 |
| mach3 | S1 | 29.7->42.2 (1.42x) | 16.3->26.9 (1.65x) | 7.0->12.4 (1.77x) | 9.2->14.5 (1.57x) | 3.3->4.4 (1.34x) | 0.55 -> 0.64 |
| mach3 | S2 | 30.3->46.6 (1.54x) | 16.0->25.7 (1.61x) | 6.6->10.0 (1.51x) | 9.3->15.7 (1.67x) | 3.3->4.8 (1.44x) | 0.53 -> 0.55 |
| config3 | S1 | 52.0->60.5 (1.16x) | 37.8->69.5 (1.84x) | 22.2->41.1 (1.85x) | 15.6->28.4 (1.82x) | 3.8->5.1 (1.34x) | 0.73 -> 1.15 |
| config3 | S2 | 48.8->56.8 (1.16x) | 20.6->35.9 (1.74x) | 8.4->13.7 (1.63x) | 12.2->22.2 (1.82x) | 3.6->4.6 (1.26x) | 0.42 -> 0.63 |
| doublemach | S1 | 116.4->203.8 (1.75x) | 60.6->118.2 (1.95x) | 28.3->51.0 (1.80x) | 32.2->67.2 (2.08x) | 11.5->26.4 (2.29x) | 0.52 -> 0.58 |
| doublemach | S2 | 120.2->229.8 (1.91x) | 56.1->113.7 (2.03x) | 24.0->47.3 (1.97x) | 32.1->65.0 (2.03x) | 12.0->27.0 (2.25x) | 0.47 -> 0.49 |

mach3's row compares different fluxes (old row = whatever the 2026-07-13 runner used, new row = `M3_FLUX=hll M3_CFL=0.3`), so its ratios are not a like-for-like comparison.

Pattern: EVERY stage is 1.2-2.1x slower than 2026-07-13, and the slowdown is roughly uniform across MUSCL / geom / face / BVD_sel. That is the signature of a machine-level difference (fewer cores + contention), not of one stage regressing — the old run's thread count is not recorded but the project default for paper-resolution 2D is 24 physical cores, i.e. 2x what was available here. The residual detail is that S1's tanh THINC scaled slightly worse (1.61-1.95x) than S2's GAUSS THINC (1.47-1.74x), consistent with the transcendental-heavy tanh path suffering more from SMT-sibling contention. **Conclusion: the stage breakdown is internally consistent and the case setups reproduce exactly, but the absolute 2026-07-13 numbers are NOT reproducible on this machine state — do not mix old and new absolute seconds in one figure.**

