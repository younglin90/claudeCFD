# Disk cleanup plan — claudeCFD (survey 2026-07-30)

## STATUS 2026-07-30: Tiers 0, 1 and the cache sweep EXECUTED. Tiers 2-3 still planned only.

**Total reclaimed: 25 GB** — disk went 226 GB used -> 201 GB used (730 -> 755 GB free).

| Step | Reclaimed |
|---|---|
| Tier 0 (garbage, builds, stray dumps, worktrees) | 0.3 GB |
| Tier 1 (octant probes) | 2.8 GB |
| Cache sweep (`pip` 7.2, `vscode-cpptools` 7.8, `npm/_cacache` 4.8, `_npx` 0.8, rest) | 22 GB |

Cache sweep notes:
- `npm cache clean --force` **exits 0 without clearing anything** here; `_cacache` still held
  4.8 GB afterwards. Delete `~/.npm/{_cacache,_npx}` directly, npm recreates them.
- `~/.cache/dotslash` dirs are mode `dr-x------` (owned, but no write bit) so `rm -rf` fails with
  Permission denied. `chmod -R u+w` first; no sudo needed.
- **Still there, deliberately: `~/.cache/huggingface` (15 GB) + `~/.cache/datalab` (3.3 GB).**
  These are marker's model weights, and this project's PDF rule is
  `pdf_to_md.py --backend marker`. Deleting them means a multi-GB re-download on the next PDF.
  Also kept: `~/.cache/ms-playwright` (622 MB, browser binaries), `~/.bun` and `~/.nvm`
  (runtimes, not caches).

Log files: `/tmp/mbq/cleanup_t0t1.log`, `/tmp/mbq/cleanup_cache.log`, `/tmp/mbq/cleanup_cache2.log`.
Verified afterwards: 23 bench binaries still in `cpp/build`, no mangled paths remain, the
paper2 archive still regenerates the deform figure.

Two candidates from the original plan turned out to be wrong and were NOT deleted:

1. **`solver_tmlpu/C:` was not garbage.** Its `g3dpilot` payload holds 12 `lvl*.bin` files that
   exist nowhere else, and its `agg_3d.csv` / `raw_3d_subsample.csv` / `summary_3d.md` differ from
   the `gauss_paper_v3` versions. Moved to `cpp/results/gauss_paper_v3/pilot_3d/` (16 files, 14 MB).
2. **`/tmp` is live.** A SEMO release-gate sweep (pid 955409, up to 5400 s) is building in
   `/tmp/build-integration-final-20260730-141059`. `/tmp` was left untouched; it clears on reboot.

Also rescued before deleting: 4 unique `.jsonl` from the mangled dirs ->
`solver_denner/autoresearch-results/rescued_from_mangled_path/`, and all octant probe
metadata/logs/CSV -> `cpp/results/_octant_probes_metadata/` (17 files, 13 MB), so the record of
those runs survives without the 2.78 GB of field data.

`vtx_bvd.txt` and `vtx_mlpu1.txt` were kept in `cpp/`: unlike the other 18 stray dumps, they have
no copy anywhere under `cpp/results`.

Every tier below lists the verification to run first.

## The important finding first

```
C:\            1.1T  1.1T   6.5G 100%  /mnt/c      <- full
/dev/sdd      1007G  267G   689G  28%  /           <- WSL ext4, plenty free
D:\            7.3T  1.2T   6.1T  17%  /mnt/d      <- 6.1 TB free
```

WSL files are not on C: directly; they live inside the ext4 VHDX, which is *stored* on C:.
That file only ever grows. **Deleting files inside WSL will not return one byte to C: until the
VHDX is compacted.** So the order matters:

1. delete inside WSL (tiers below)
2. `wsl --shutdown` from Windows
3. compact the VHDX (PowerShell as admin):

```powershell
Optimize-VHD -Path "$env:LOCALAPPDATA\Packages\<distro-pkg>\LocalState\ext4.vhdx" -Mode Full
```

`Optimize-VHD` needs Hyper-V tools. Without them, use `diskpart`:
`select vdisk file="..."` then `compact vdisk`.

Locate the file first: `(Get-ChildItem $env:LOCALAPPDATA\Packages -Recurse -Filter ext4.vhdx).FullName`

**Bigger lever than any deletion:** with 6.1 TB free on D:, moving the bulk results tree (or the
whole distro, via `wsl --export` / `--import --version 2` to a D: path) frees far more C: space
than the ~14 GB below. Deletion and relocation are independent; do both if C: stays tight.

## Tier 0 — garbage, zero risk (~300 MB + 6.5 GB of /tmp)

Artifacts of the known WSL inline-path-mangling bug: a Windows path became a literal directory
name. These are not real data.

| Path | Size |
|---|---|
| `claudeCFD/Microsoft.PowerShell.CoreFileSystem::wsl.localhostUbuntu...solver_denner` | 15 MB |
| `claudeCFD/Microsoft.PowerShell.CoreFileSystem::\wsl.localhostUbuntu...solver_denner` | 1.7 MB |
| `claudeCFD/solver_denner/Microsoft.PowerShell.Core*` (3 more) | ? |
| `claudeCFD/solver_tmlpu/C:` | 14 MB |
| `claudeCFD/solver_tmlpu/home` | 44 KB |

Check first: `find "<path>" -type f | head -20` — confirm the contents are duplicates of the
real tree, not the only copy of something.

Rebuildable build trees (keep `build/`, the one in daily use):

| Path | Size |
|---|---|
| `cpp/build-release` | 89 MB |
| `cpp/build_aocc` | 41 MB |
| `cpp/build_cl` | 27 MB |
| `cpp/build_dev` | 5.1 MB |

Stray bench output written into `cpp/` because a bench was run from that cwd: 21 files,
**118 MB** (`cmp_bvd.txt.vtk`, `lev_*.txt.vtk`, `m3_*.txt.vtk`, `config3_*.txt.vtk`, ...). None
are referenced by `paper2_archive`; all are superseded by the per-case dirs under `cpp/results/`.
Check: `grep -rl "cmp_bvd.txt.vtk" cpp/results cpp/apps` before removing.

Prunable git worktrees, registered but gone from disk:

```bash
cd claudeCFD/solver_denner && git worktree prune -v      # 3 prunable + /tmp/solver_denner_head_check
```

`/tmp` = **6.5 GB**, of which ~4.5 GB is `autotessell-wheel-*` and `build-*` from a different
project. Ephemeral by definition; clears on reboot anyway.

## Tier 1 — dropped case, safe to delete (2.78 GB)

`octant3d` (Hoppe Case-5 / Case-7) was **dropped from paper2**. Confirmed from each dir's
`meta.txt`: `./build/octant3d_unstr_bench`, `case=Hoppe-Case5-spiral(all-J,Fig9)`. These are
mesh-resolution probes for a case that is no longer in any manuscript.

| Path | Size | Contents |
|---|---|---|
| `cpp/results/paper2_3d_nx120` | 976 MB | octant case5, `oct_nx120` 10.4 M tets |
| `cpp/results/paper2_3d_nx115` | 863 MB | octant c7 |
| `cpp/results/paper2_3d_nx75` | 657 MB | octant, octant_case1 |
| `cpp/results/paper2_3d_nx64` | 287 MB | octant case5 + case7 |

Check first: confirm no figure in `papers_draft/` or `paper2_archive/figures/` derives from them
(`grep -rl "nx120\|nx115\|octant" papers_draft cpp/results/paper2_archive/scripts`). The BC
findings from these runs are already recorded in memory, so the conclusions survive deletion.

## Tier 2 — superseded by a later run (~2.3 GB)

| Path | Size | Superseded by |
|---|---|---|
| `paper2_3d_deform_mood` | 530 MB | `paper2_3d_deform_cfl` (MOOD off, CFL 0.25 — the version in the paper) |
| `paper2_3d_langseth` | 470 MB | `paper2_3d_final/{s1,s2}_sphere` (nx100 vs nx75, beta_l 1.4 vs 1.6) |
| `paper2_3d_langseth_vortex` | 370 MB | `paper2_3d_final/{s1,s2}_2cyl` |
| `gauss_paper` | 271 MB | `gauss_paper_v3` |
| `gauss_paper_v2` | 240 MB | `gauss_paper_v3` |
| `*_oldbc` (4 dirs) | 172 MB | current-BC runs; memory marks old-BC dumps superseded |
| `paper2_3d_deform_final/{s1,s2}/out.vtk` | 274 MB | the T frame now comes from `deform_cfl`; only `out.vtk_t1.5000.vtk` is still used |

**Caveat on the last row:** the T/2 frames in `paper2_3d_deform_final` are the ones the current
figure uses, so that directory must NOT be deleted wholesale — only its `out.vtk` (the T frame).
And if the MOOD-on/MOOD-off mismatch is fixed by re-running T/2 with MOOD off, this directory
becomes deletable and `deform_mood` may be needed as a comparison. **Resolve that open item
before touching either.**

Deleting anything in this tier invalidates rows in `paper2_archive/MANIFEST.tsv` — regenerate it
afterwards.

## Tier 3 — unused frames inside runs that are still in use (~1.8 GB)

These are real paper runs, but the figures only read some frames. `fig_sphere.py` reads
`out_t0.8000.vtk`; `fig_2cyl.py` reads `out_t0.7000.vtk`.

| Path | Size | Note |
|---|---|---|
| `paper2_3d_final/{s1,s2}_sphere/out_t0.5000.vtk` | 764 MB | intermediate, unused |
| `paper2_3d_final/{s1,s2}_2cyl/out_t{0.1,0.3,0.5}000.vtk` | 1.05 GB | intermediate, unused |

Lowest priority and highest regret risk: if a reviewer asks for time evolution, these are the
only record and each is a multi-hour run. Recommend keeping unless C: is critical, or moving
them to D: rather than deleting.

## Tier 4 — needs your decision

| Path | Size | Question |
|---|---|---|
| `~/work/claude_code/claudeCFD_tmlpu_autoresearch_bg` | 625 MB | git worktree, branch `t_mlp_u_autoresearch_bg`, last commit **2026-05-11**. Dormant 2.5 months. Delete only if that branch is merged or abandoned — check with `git branch --merged` and `git log master..t_mlp_u_autoresearch_bg --oneline` |
| `claudeCFD/solver_LBM_steady_state` | 14 GB | a different sub-project; the single largest item in the tree. Its own results need their own review |
| `~/.cache` | 35 GB | not project data; `~/.cache/pip`, browser/build caches. Usually safe, fully regenerable |
| `~/.local` | 26 GB | contains installed python packages — do NOT bulk delete |
| `~/.npm` | 5.6 GB | `npm cache clean --force` reclaims it |

## Recommended order

1. **Tier 0** now (garbage + rebuildable + `/tmp`), ~6.8 GB, no verification burden beyond the greps.
2. **Tier 1** after the octant grep confirms nothing cites it, 2.78 GB.
3. `npm cache clean --force` + `~/.cache` review, up to 40 GB — largest easy win, and it is not project data.
4. **Tier 2** only after the deform MOOD open item is resolved; regenerate `MANIFEST.tsv` after.
5. **Compact the VHDX.** Without this, steps 1-4 change nothing on C:.
6. If C: is still tight: relocate the distro or the results tree to D: (6.1 TB free).
7. **Tier 3** last, or never — prefer moving to D:.

Project-related total: **~7 GB safe (tiers 0-1)**, ~9 GB more at increasing risk (tiers 2-3).
Non-project: **~40 GB** in caches, which is the bigger and easier win.
