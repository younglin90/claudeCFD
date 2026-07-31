# solver_tmlpu — T-MLP-u / THINC-QQ-BVD research workspace (C++ active)

## Response rules

- **응답 규칙**: 항상 caveman full 모드로 응답 (skill: caveman). 코드/커밋/에러문자열/명령어는 verbatim, 보안경고·되돌릴수없는작업은 정상문체. 끄기: "stop caveman".
- Autonomy: per `docs/tmlpu_autonomy_charter.md` — decide/execute/log without asking, EXCEPT long validation runs need user approval first.

## Project map

Active development lives in `../cpp/` (C++17, WSL2 ubuntu, OpenMP):

- `cpp/include/cfd/reconstruct_bvd.hpp` — 2D recon: `reconstruct_cheng3` = S1/S2/S3 (MUSCL-THINC/QQ-BVD); ~2600 lines, env-flag driven
- `cpp/include/cfd/reconstruct3d_unstr.hpp` — 3D unstructured recon (same schemes)
- `cpp/include/cfd/{solver2d,solver_euler2d,solver_euler3d,bvd_mood2d,...}.hpp` — solvers
- `cpp/apps/*.cpp` — benches: `leveque_bench`, `config3_bench`, `shock_mixing_bench`, `validation_smoke` (shockvortex `VS_ONLY=16`), `mach3_bench`, `double_mach_bench`, `octant3d_unstr_bench`, `deform3d_unstr_bench`, `langseth3d_unstr_bench`
- `cpp/build/` — cmake build dir (build INSIDE WSL: `cd cpp/build && cmake .. && make -j16`)
- `cpp/results/paper_final/` — S1 paper suite results + `meshes/` (paper meshes: `mach3_paper.mesh2d`, `dmr_wc_paper.mesh2d`, `oct_h*.umsh`, ...); `paper_runner.sh` = S1-only suite script
- `cpp/results/paper2_3_data/` — S2/S3 results (`S3cap` = current S3 [0.8,1.4], `S3nocap` = diagnostics)
- **`cpp/results/paper2_archive/`** — paper2 figure registry: `README.md` (per-case recipe + regen order), `scripts/<case>/`, `figures/`, `data_small/`, `origin/*.opju`, `MANIFEST.tsv` (index of all 47 VTK dumps). **Start here for any paper2 figure work.**
- `cpp/MIGRATION.md` — C++ migration status

This dir (`solver_tmlpu/`):

- Python legacy `solver/solve_T-MLP-u/` = validation oracle only (do not delete)
- `docs/` = validation specs + research docs
- `results/T-MLP-u/` = python-era outputs

Validation spec contracts (canonical, keep aligned):

- `docs/leveque_strict_validation_spec.md`
- `docs/mach3_step_strict_validation_spec.md`
- `docs/double_mach_reflection_strict_spec.md`

Long-term memory: auto-memory `MEMORY.md` (loaded per session) holds the evolving HARD rules; this file holds the stable subset.

## Schemes (paper2/3 definitions + exact recipes)

- **S1 (baseline)** = MUSCL(MLP-u2) + THINC/QQ tanh 2-beta (**beta_l=1.4/beta_s=0.8 in BOTH 2D and 3D** — code default is 1.4/0.8; the old "3D 1.6/0.8" was a stale doc, corrected 2026-07-12 by 2D-vs-3D recon audit which confirmed 3D is a faithful extension: MLP-u2 verbatim, tanh/GAUSS/BVD identical) + BVD min-TBV pick.
  Recipe: `BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1`
  (`THINCQQ_TANH` mandatory: 2D no-op, forces tanh in 3D where default is GAUSS)
- **S2 (paper2, speed)** = S1 with GAUSS probit closed-form sigmoid (no Newton cell-D, no quadrature face): same recipe minus `THINCQQ_TANH` plus `THINCQQ_GAUSS=1`. ~2.2x faster 2D, ~2.7x 3D, results ≈ S1.
- **S3 (paper3, accuracy)** = per-cell per-VARIABLE beta* in [0.8,1.4] (L2GN search on GAUSS closed-form TBV; option B: actual recon at beta* uses EXACT tanh).
  Recipe: `BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_GAUSS=1 THINCQQ_BETASTAR=1 THINCQQ_BSTAR_EXACT=1 THINCQQ_BSTAR_WIDE=1`
  NOTE 3D: do NOT add `THINCQQ_TANH` (it disables beta*); option-B tanh recon is automatic (opt-out `THINCQQ_BSTAR_GAUSSRECON=1`).

## Known pitfalls (guardrails — each one bit us)

- `BVD_CHENG3=1` is REQUIRED for any `THINCQQ_*` flag to have effect (without it `RECON_BVD` routes to the old ABVD path silently; identical results across schemes = the tell). `leveque_bench`'s "BVD" line is a hardcoded BJ/O2 BVD, not cheng3 — don't smoke-test THINCQQ with it.
- 3D default sigmoid = GAUSS; 2D default = tanh (OPPOSITE). 3D S1 needs `THINCQQ_TANH=1`; 3D S3 must NOT have it.
- WSL: inline shell vars (`$X`) in `wsl.exe bash -lc "..."` get mangled — ALWAYS write a script file to `/tmp/mbq/` and run `bash /tmp/x.sh`. (see `.claude/rules/wsl-pitfalls.md`)
- WSL 9p stale build: headers edited via UNC path may compile stale — touch header + rebuild + md5sum binary before trusting results.
- Run policy details: `.claude/rules/cfd-run-rules.md`

## Build & run (capability boundary)

- Build: `wsl.exe -d ubuntu`; `cd cpp/build && make <bench> -j16`
- Run pattern: `taskset -c <even cpus> env OMP_NUM_THREADS=N OMP_PROC_BIND=close OMP_PLACES=cores <FLAGS> ./bench <args>`. CPU = Threadripper 5975WX 32 physical cores (even cpu ids 0,2,...,62 = physical). Optimal for paper-res 2D = 24 physical cores (measured knee; 32 gives ~0 gain, bandwidth-bound).
- Profilers: `CHENG3_PROF=1` (2D recon breakdown to stderr), `E3D_PROF=1` (3D), `BOPT_DIAG=1` (beta* diag: `BSTAR_EXACT beta*: min/mean/max`), `CFD_MAXSTEP=N` caps steps for smokes.
- PDF reading: ALWAYS `tools pdf_to_md.py --backend marker` (raw fitz garbles equations).
- Result figures: after EVERY calc render figure + upload to tmpfiles.org ONLY (`curl -4`, 3x retry) + show link. Both x-z and x-y slices for 3D.

## Execution model

See `.claude/rules/execution-model.md` (Advisor/Worker split, reporting timing, advice handling).
