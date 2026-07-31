# paper2 archive — figures, scripts, Origin projects, dump index

Everything needed to regenerate any figure in `papers_draft/paper2_GAUSSQQ_draft.docx`.

**S1** = exact tanh THINC/QQ (Newton cell-D + Duffy-Gauss face quadrature).
**S2** = GAUSS/probit closed form. **S3** = per-cell per-variable beta\*.
paper2 compares S1 vs S2; S3 dumps are present for several cases but belong to paper3.

## Layout

| Path | Holds |
|---|---|
| `figures/` | every final PNG/PDF, renamed `<case>__<original>` |
| `scripts/<case>/` | the generating scripts, ORIGINAL filenames preserved |
| `data_small/` | the numbers behind the figures: every CSV/TSV/MD/log/`meta.txt` under 6 MB, renamed `<case>__<relpath>` |
| `origin/` | Origin projects (`.opju`) |
| `MANIFEST.tsv` | index of all 47 VTK/VTU dumps: case, scheme, size, mtime, absolute path |

Scripts keep their original filenames on purpose: each `fig_*.py` imports its `load_*.py`
sibling by module name, so renaming or flattening breaks the import.

The 7.5 GB of dumps stay where they were written. They are indexed here, not copied — two
copies of a dump means two versions of the same truth. Use `MANIFEST.tsv` to locate them.

## How to regenerate a figure

Run from inside `scripts/<case>/`. Each 2D case is a two-stage pipeline: a `load_*.py` (or
`diag_*.py`) reads the VTK dumps and writes an interpolated-grid cache, then `fig_*.py`
consumes the cache. Ordering matters; the figure script does not build its own cache.

```bash
cd scripts/config3
python3 load_config3.py          # -> _cache/cfg3_cache.npz  (cell-wise fields + symmetry)
python3 diag_config3_vort.py     # -> _cache/cfg3_grid.npz   (interpolated grid)
python3 fig_cfg3_diff.py         # -> _cache/figures/Fig_cfg3_diff.{png,pdf}
python3 fig_config3_sep.py
```

Other 2D cases are one stage shorter: `load_<case>.py` then `fig_<case>.py`.
3D cases have no cache stage and need a virtual display:

```bash
cd scripts/deform3d && xvfb-run -a python3 fig_deform.py E     # E = the chosen camera
cd scripts/twocyl3d && xvfb-run -a python3 fig_2cyl.py
cd scripts/sphere3d && xvfb-run -a python3 fig_sphere.py
```

Regenerated output lands in `<case>/_cache/figures/`, NOT in `figures/` — so a re-run can
never silently overwrite the archived version. Copy it over deliberately, keeping the
`<case>__` prefix.

`_cache/` is derived data and safe to delete; verified 2026-07-30 that the config3 chain
rebuilds from the VTK dumps alone with the cache removed. `_paths.py` in each case directory
resolves the results root on either Linux or Windows and owns the cache location; the
scripts originally hardcoded a per-session scratchpad path, which is why it exists.

## Cases

Common to every 2D case: `BC_FACE_RECON=1 BVD_CHENG3=1 BVD_CANDFLAG=1`, plus
`THINCQQ_GAUSS=1` for S2 only (S1 = flag absent in 2D, where tanh is the default).

`BVD_CHENG3=1` is REQUIRED — without it `RECON_BVD` silently routes to the old ABVD path
and every scheme returns identical results.

| Case | Data dir (under `cpp/results/`) | Bench + args | beta_l/beta_s | Notes |
|---|---|---|---|---|
| leveque | `paper2_3_data/leveque/{s1,s2,s3,s3wide}` | `leveque_bench 200` | 1.6 / 0.8 | 160k tri; L1 3.18e-3 (S1) vs 3.24e-3 (S2) |
| config3 | `paper2_3_data/config3/{s1,s2,s3}` | `config3_bench 280 0.8` | 1.6 / 0.8 | `C3_FLUX=llf C3_MOOD=1 C3_BC=dirichlet C3_SKIP_MLP=1` |
| shockvortex | `paper2_3_data/shockvortex/{s1,s2,s3}` | `validation_smoke` | 1.6 / 0.8 | `VS_ONLY=16 VS_FLUX=rhllc VS_RECON=bvd VS_NSCALE=4` |
| shockmixing | `paper2_3_data/shockmixing/{s1,s2,s3,...}` | shock_mixing_bench | 1.6 / 0.8 | extra S3 variants (`s3_pervar`, `s3argmin`, ...) are paper3 |
| mach3 | `paper2_3_data/mach3/{s1,s2}` | `mach3_bench 480 160 4` | 1.6 / 0.8 | `M3_FLUX=hll M3_INT=2 M3_CFL=0.3 M3_MESH=uniform M3_BVD_ONLY=1` |
| doublemach | `paper2_dmr_papermesh/{s1,s2}` | `double_mach_bench` | **1.4** / 0.8 | `DM_FLUX=rhllc DM_SINGLE=1`, mesh `dmr_wc_paper.mesh2d`; S1 needs `THINCQQ_TANH=1` here |
| 2cyl (3D) | `paper2_3d_final/{s1,s2}_2cyl` | `langseth3d_unstr_bench` | 1.4 / 0.8 | `LANG_CASE=vortex N3_FLUX=1 BVD_PPFLOOR=0.1 N3_T=0.7`, mesh `lang_vortex_nx112.umsh` |
| sphere (3D) | `paper2_3d_final/{s1,s2}_sphere` | `langseth3d_unstr_bench` | 1.4 / 0.8 | `N3_FLUX=1 BVD_PPFLOOR=0.1 N3_T=0.8`, mesh `lang_nx100.umsh`; figure uses `out_t0.8000.vtk` |
| deform (3D) | `paper2_3d_deform_cfl/{s1,s2}_ret_cfl025_nomood` | `deform3d_unstr_bench` | 1.4 / 0.8 | CFL 0.25, `ADV_NOMOOD=1`, 4920 steps, mesh `def_nx76.umsh` (2,633,856 tets) |
| deform T/2 | `paper2_3d_deform_final/{s1,s2}/out.vtk_t1.5000.vtk` | same | 1.4 / 0.8 | **MOOD ON** — see caveats |
| cascade | `paper2_cascade/` | see `ANALYSIS.md` there | — | 4-level speedup cascade (kernel / stage / recon / solver) |
| steepness | `gauss_paper_v3/` | `bench_2d.cpp`, `bench_3d.cpp` | beta ~ U(0.5,5) | Fig. 1 + Table 1 defect ensemble, 1e6 samples/shape |

3D flag traps: 3D default sigmoid is **GAUSS**, the opposite of 2D. So 3D S1 requires
`THINCQQ_TANH=1`, and 3D S3 must NOT have it (it disables beta\*).

## Figure to script map

| Figure (in `figures/`) | Script |
|---|---|
| `leveque__Fig_lev_field.*` | `scripts/leveque/fig_lev_a.py` |
| `leveque__Fig_lev_error.*` | `scripts/leveque/fig_lev_f.py` |
| `leveque__Fig_leveque_S1S2.*` | `scripts/leveque/fig_leveque.py` (6-panel original) |
| `config3__Fig_cfg3_density_{tanh,closed}.*` | `scripts/config3/fig_config3_sep.py` |
| `config3__Fig_cfg3_diff.*` | `scripts/config3/fig_cfg3_diff.py` |
| `shockvortex__Fig_sv_*` | `scripts/shockvortex/fig_shockvortex.py` |
| `shockmixing__Fig_sm_*` | `scripts/shockmixing/fig_shockmixing.py` |
| `mach3__Fig_m3_*` | `scripts/mach3/fig_mach3.py` |
| `doublemach__Fig_dmp_*` | `scripts/doublemach/fig_dmr_paper.py` |
| `3d__Fig_2cyl_{tanh,closed}.png` | `scripts/twocyl3d/fig_2cyl.py` |
| `3d__Fig_sph_{tanh,closed,diff}.png` | `scripts/sphere3d/fig_sphere.py` |
| `deform3d__Fig_def_{tanh,closed}.png` | `scripts/deform3d/fig_deform.py` (view `E`) |
| `cascade__Fig_cascade_final.*` | `scripts/cascade/cascade_final.py` |
| `cascade__Fig_composition.*` | `scripts/cascade/build_composition_fig.py` |
| `steepness__Fig_v3_steepness.*` | `scripts/steepness/plot_2d.py` |

Diagnostic-only, not in the paper: `config3/diag_config3.py` (symmetry residual),
`config3/diag_config3_vort.py` (vortex-core asymmetry measurement),
`cascade/{analyze,control_check,mksummary2,parse_2d,parse_hllc_3d}.py`,
`leveque/origin_{prep,fig1}.py` (Origin export path).

## Environment

3D renders need pyvista offscreen: `xvfb-run -a python3 <script>`.
2D renders are matplotlib Agg. Origin projects open in Origin; the API cannot set contour
palettes or area fills, so those were done by extracting polylines in Python (see
`leveque/origin_fig1.py`).

## Caveats carried into the paper

1. **beta_l is not uniform across cases.** doublemach and all 3D cases use 1.4; leveque,
   config3, shockvortex, shockmixing and mach3 use 1.6. Must be stated per case or unified.
2. **deform T/2 vs T mismatch.** The T frames (`paper2_3d_deform_cfl`) are MOOD off; the
   T/2 frames (`paper2_3d_deform_final`) are MOOD on. Either state it in the caption or
   re-run two T/2 frames with MOOD off.
3. **deform wall time.** Use the 500-step controlled measurement in
   `paper2_3d_deform_walltime` (S2 is 1.13x faster). The full-run 1.37x is contaminated by
   machine contention and must not be quoted.
4. **Appendix A** says the moment-matching constant c was fixed by least squares; the code
   hardcodes c = pi/2 (`GC = 1.5707963267948966`). Text needs correcting.
5. `mach3_bench` freezes under `HLLC_PVRS=1` (reflective wall emits no flux), hence
   `M3_FLUX=hll` for that case.

## Upload helper

Figures are reviewed via tmpfiles.org. The id is ALPHANUMERIC, not numeric:

```bash
curl -4 -s -m 90 -F "file=@FIG.png" https://tmpfiles.org/api/v1/upload | tr -d '\\' \
  | grep -oE 'tmpfiles\.org/[A-Za-z0-9]+/[A-Za-z0-9_.-]+'
```

`curl -4` is mandatory (IPv6 POST hangs from WSL). Insert `/dl/` for a direct link.
