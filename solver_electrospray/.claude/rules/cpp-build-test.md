---
description: C++ solver build/run/test conventions, OpenMP safety contract, and the known pre-existing test failure.
paths:
  - "include/**"
  - "apps/*.cpp"
  - "tests/**"
  - "CMakeLists.txt"
---

# C++ build, run, and test

All builds and runs happen in **WSL2 ubuntu** (`wsl.exe -d ubuntu bash -c '...'`). The
finite-volume core under `include/` is header-only; the application lives in
`include/electrospray/` and `apps/`.

- **Build:** `cd <project> && cmake --build build -j$(nproc)`. OpenMP is enabled in
  `CMakeLists.txt` (`-fopenmp`, gomp). Configure is automatic when `CMakeLists.txt` changes.
- **Run a case:** `./build/electrospray_case_runner --case X.json --output-dir Y`.
  Also `--print-defaults` (emits solver struct defaults as JSON — the GUI's single source of
  truth) and `--case-dir <openfoam-case>`.
- **Tests:** `ctest --test-dir build -R electrospray` (fast subset) and
  `ctest --test-dir build -R candido` (physics regression: `test_candido_*`).

## OpenMP safety contract (`include/fvm/Parallel.hpp`)

`FVM_PARALLEL_FOR` expands to `#pragma omp parallel for` only under `_OPENMP`. Apply it **only**
to loops that write **disjoint** per-cell outputs (`out[ci] = f(ci)`). **Never** apply it to:
- face-scatter loops that write both `x[f.owner]` and `x[f.neighbour]` (data race), or
- reduction loops (`sum += ...`, `max = max(...)`, counters, `push_back`).

Verify any parallelization is **bit-identical at `OMP_NUM_THREADS=1` and `=8`** against a serial
baseline (no reductions touched ⇒ same result at any thread count). Current speed-up ≈ 2.3× at 8
threads; runtime is dominated by the Eigen Krylov solves, which are not OpenMP-parallelized.

## Known PRE-EXISTING test failure — do not assume you broke it

`ctest test_candido_cone_jet_smoke3d` (a.k.a. #46) FAILS on the current tree; the other candido
and electrospray tests pass (20/21). Root cause: a long-window hydrodynamic diagnostic asserts the
run reaches ~0.9 ms, but task #14's adaptive electric-force CFL (`useElectricForceTimeStepLimit`,
default on) throttles dt so it under-shoots; disabling that limiter reaches the window but blows up
mass conservation. It is a calibration tension, not a flag toggle, and is left for a physics
decision by the maintainer. Prove any regression against the passing 20, not against #46.

## Cone-tip blow-up fix

Fine/defect tips diverge without the adaptive electric-force CFL: `dt = min(dtBase,
electricForceTimeStepSafety * min sqrt(rho*cbrt(V)/|F|))`, default safety 0.05, gated by
`useElectricForceTimeStepLimit`. Keep it on for production/GUI defaults.
