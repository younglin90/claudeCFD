# CLAUDE.md — solver_electrospray

C++ 3D leaky-dielectric VOF electrohydrodynamic (EHD) cone-jet solver reproducing **Candido & Páscoa 2023** (Phys. Fluids 35, 052110). Drives a top-tier paper on **geometric tip defects** (D1 blunting / D2 tilt / D3 protrusion) in miniaturized electrospray thrusters. Work-session cwd = this directory.

> Parent-workspace rules (language/C++ migration, PNG-save policy, GitHub) live in `claudeCFD/CLAUDE.md` and `.claude/rules/*` — do not duplicate them here.

## Project Map

| Path | Role |
| --- | --- |
| `include/fvm/` | Header-only finite-volume core (see key headers below). |
| `include/fvm/Mesh3D.hpp` | Unstructured polyhedral mesh + geometry. |
| `include/fvm/FieldOperators3D.hpp` | Least-squares gradients, div/lap. |
| `include/fvm/VofTransport3D.hpp` | Geometric volume-matched PLIC / isoAdvector interface advection. |
| `include/fvm/SurfaceTension3D.hpp` | Balanced-force CSF + PLIC-quadric curvature. |
| `include/fvm/EHDCoupling3D.hpp` | Maxwell force, charge coupling. |
| `include/fvm/Electrostatics3D.hpp` | Variable-coefficient Poisson. |
| `include/fvm/RhieChow3D.hpp`, `PressureVelocityCoupling3D.hpp` | Projection / pressure-velocity coupling. |
| `include/fvm/Parallel.hpp` | OpenMP macro layer. |
| `include/fvm/OpenFoamPolyMeshReader3D.hpp`, `OpenFoamFieldReader3D.hpp` | External OpenFOAM mesh/field IO. |
| `include/electrospray/CandidoTaylorConeJet3D.hpp` | **Main solver**: setup struct, options struct, `runCandidoConeJetSmoke3D` time loop. |
| `include/electrospray/BoundaryConditionSet.hpp` | Named-patch BC roles + paper-default BCs. |
| `include/electrospray/{Diagnostics,MaterialProperties,Validation,Benchmarks,ApplicationModels,AxisymmetricElectrostatics}.hpp` | App-layer support. |
| `apps/electrospray_case_runner.cpp` | Main runner; consumes case JSON, external OpenFOAM meshes, named-patch BCs. |
| `apps/electrospray_gui_server.py` + `apps/gui/index.html` | Local web GUI: case → mesh → per-patch BC → run → history charts (86 form fields; every solver param inputtable, defaults SSOT'd to `--print-defaults`). |
| `apps/generate_resolved_nozzle_mesh.py`, `generate_nozzle_cad.py` | Parametric emitter mesh generators (defects + tip grading; no OpenFOAM install needed). |
| `apps/electrospray_mesh_validate.cpp`, `electrospray_validation_cli.cpp`, `electrospray_profile.cpp` | Mesh validate / validation CLI / profiling. |
| `apps/plot_*.py` | Figure generation. |
| `apps/_gui_selftest.py`, `_gui_browser_test.py` | GUI regression harnesses (each starts+stops its own server in one call). |
| `tests/` | ctest suite (59 files): `test_candido_*.cpp` (physics), `test_openfoam_*.cpp` (mesh/field IO), `test_*.py` (GUI + case runner). |
| `docs/electrospray/` | Design + validation docs and `paper/` (JCP draft). |
| `build/` | WSL CMake build dir (binaries). |
| `runs/` | GUI / case outputs. |
| `results/` | Figures / PNGs. |
| `papers/`, `reference/` | Literature. |
| `CMakeLists.txt` | Root build config (OpenMP enabled). |

Consult these internal design/API docs first: `docs/electrospray/mesh_setup_guide.md`, `p7_resolved_nozzle_design.md`, `conejet_blowup_fix.md`, `paper_tip_defect_design.md`, `candido_3d_method_gap.md`, `validation_report.md`.

## Build & Run

All commands run inside WSL2 ubuntu: `wsl.exe -d ubuntu bash -c '...'`.

| Task | Command |
| --- | --- |
| Build | `cmake --build build -j$(nproc)` |
| Run case | `./build/electrospray_case_runner --case X.json --output-dir Y` |
| Case flags | `--print-defaults`, `--case-dir <openfoam-case>` |
| Tests | `ctest --test-dir build -R electrospray` and `-R candido` |
| GUI | `python3 apps/electrospray_gui_server.py --port 8765` |

GUI note: the harness SIGTERM-kills backgrounded/detached servers. Use the self-contained `apps/_gui_selftest.py` / `apps/_gui_browser_test.py` (spawn+kill their own server in one foreground call). Playwright chromium is cached at `~/.cache/ms-playwright/chromium-1208/...` — launch via `executable_path`.

## Core Rules

- **Runs: SMOKE ONLY (< ~15 min).** Do NOT launch long/production simulations — the maintainer runs the real paper meshes. Verify the pipeline (loads, BCs consumed, stable + conserving, defect changes result), not converged physics.
- **Git branch = `t_mlp_u_paper_verification`** (NOT `main`). This subproject is **mostly UNTRACKED** in claudeCFD — commit only the specific touched files, never `git add` the whole subproject. Use `git commit -F <msgfile>` (PowerShell→wsl quoting breaks inline `-m` with parens). Untracked files: `git checkout -- <file>` is a no-op — revert manually.
- **Plots:** `matplotlib.use('Agg')` + fixed `savefig` path (overwrite; NO per-round filenames); print `Plot saved: ...`.
- **Language:** C++ for the solver (WSL, OpenMP; OpenACC later). Python for tooling and validation oracles — do NOT delete the Python oracles.
- **Response tone:** caveman full mode is the parent-workspace default (`caveman` skill) unless the user says "stop caveman".

## Capability Boundaries — the solver CAN do these (don't wrongly refuse)

- Web GUI end-to-end (case → mesh → per-patch BC → run → summary table + history charts); every solver parameter is inputtable (86 GUI fields; the 76 solver-owned ones match `--print-defaults` with zero drift).
- Generate resolved-nozzle meshes with defects (D1/D2/D3) and tip grading, no OpenFOAM install.
- Import external OpenFOAM polyMesh + honor named-patch BCs (`use_named_patch_boundary_conditions`).
- Validation CLI, profiling, artifact generation, headless-browser GUI E2E.

## Lessons Learned / Guardrails

- **OpenMP:** only disjoint-write per-cell loops; never face-scatter or reduction loops; verify bit-identical at 1 and 8 threads. (detail in `.claude/rules/cpp-build-test.md`)
- **Known PRE-EXISTING failure:** `test_candido_cone_jet_smoke3d` (#46) — force-limit calibration tension; 20/21 otherwise green; do NOT assume you broke it. (detail in the rule file)
- Cone-tip blow-up is fixed by the adaptive electric-force CFL (`electricForceTimeStepSafety=0.05`); keep it on.
- Background servers get SIGTERM'd by the harness; `pkill -f electrospray_gui_server` inside a `bash -c` self-matches the wrapper shell (exit 15) — avoid it.
- A tip defect is invisible below ~3–4 wall cells → use tip-graded meshes.

---
See `.claude/rules/execution-model.md` and `.claude/rules/cpp-build-test.md`.
