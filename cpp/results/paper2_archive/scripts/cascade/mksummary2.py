#!/usr/bin/env python3
"""Emit summary_2d.md from paper2_cascade logs + the old paper2_recon_profile logs."""
import os, re, glob, statistics

ROOT = "/home/younglin90/work/claude_code/claudeCFD/cpp/results/paper2_cascade"
OLD = "/home/younglin90/work/claude_code/claudeCFD/cpp/results/paper2_recon_profile"
CASES = ["leveque", "shockmixing", "shockvortex", "mach3", "config3", "doublemach"]
SCHEMES = ["s1", "s2"]
CHENG3_LABEL = {"leveque": "BVD", "shockmixing": "shockmixing", "shockvortex": "shockvortex_2d",
                "mach3": "BVD", "config3": "T-MLP-u", "doublemach": "BVD"}

RE_PROF = re.compile(r"\[CHENG3_PROF\]\s+recon_calls=(\d+)\s+MUSCL=([\d.]+)s\s+THINC=([\d.]+)s\s+"
                     r"\(geom=([\d.]+)s face=([\d.]+)s\)\s+BVD_sel=([\d.]+)s")
RE_WALL = re.compile(r"^\[WALL\] (\S+) wall=([\d.]+)s")
RE_SMWALL = re.compile(r"status=\S+\s+wall=([\d.]+)s steps=(\d+)")


def parse(path):
    d = {"walls": {}, "total": None, "prof": None, "lb": None, "la": None, "cmd": None}
    for ln in open(path, errors="replace"):
        m = RE_PROF.search(ln)
        if m:
            c, mu, th, g, fa, bv = m.groups()
            d["prof"] = dict(calls=int(c), muscl=float(mu), thinc=float(th),
                             geom=float(g), face=float(fa), bvd=float(bv))
        m = RE_WALL.match(ln)
        if m:
            if m.group(1) == "TOTAL":
                d["total"] = float(m.group(2))
            else:
                d["walls"][m.group(1)] = float(m.group(2))
        m = RE_SMWALL.search(ln)
        if m:
            d["walls"]["shockmixing"] = float(m.group(1))
        if ln.startswith("### loadavg_before:"):
            d["lb"] = ln.split(":", 1)[1].strip().split()[0]
        if ln.startswith("### loadavg_after:"):
            d["la"] = ln.split(":", 1)[1].strip().split()[0]
        if ln.startswith("### cmd:"):
            d["cmd"] = ln.split(":", 1)[1].strip()
    return d


def old_prof(case, sch):
    p = os.path.join(OLD, "%s_%s.log" % (case, sch))
    return parse(p)["prof"] if os.path.exists(p) else None


reps = sorted(glob.glob(os.path.join(ROOT, "rep*")))
repnames = [os.path.basename(r) for r in reps]
data = {}
for case in CASES:
    for sch in SCHEMES:
        runs = []
        for r in reps:
            p = os.path.join(r, "2d_%s_%s.log" % (case, sch))
            if os.path.exists(p):
                d = parse(p)
                if d["prof"] and CHENG3_LABEL[case] in d["walls"]:
                    d["rep"] = os.path.basename(r)
                    d["recon"] = d["prof"]["muscl"] + d["prof"]["thinc"] + d["prof"]["bvd"]
                    d["wall"] = d["walls"][CHENG3_LABEL[case]]
                    runs.append(d)
        if runs:
            data[(case, sch)] = runs


def med(v):
    return statistics.median(v) if v else float("nan")


def spread(v):
    return (max(v) - min(v)) / statistics.mean(v) * 100 if len(v) > 1 else 0.0


L = []
A = L.append
A("# paper2 §3.12 — 2D speedup cascade: total solver wall (re-measured 2026-07-28)\n")
A("Data: `cpp/results/paper2_cascade/rep{1,2,3}/2d_<case>_<scheme>.log` "
  "(full stdout, `### `-prefixed provenance header per run). The top-level "
  "`2d_<case>_<scheme>.log` is a copy of the MEDIAN-wall repeat of that pair. "
  "`cascade_2d.csv` holds the machine-readable table.\n")

A("## 1. What was added to the apps\n")
A("Total wall time was missing from every 2D bench except two. Added to `cpp/apps/*.cpp` ONLY "
  "(no solver header touched); `CHENG3_PROF` stage output untouched.\n")
A("| app | change | key |")
A("|---|---|---|")
A("| `leveque_bench.cpp` | `<chrono>`; `run()` gained a `label` arg and now times `solve_adv2d` "
  "and prints; `[WALL] TOTAL` at end of `main` | `[WALL] <label> wall=%.3fs steps=%d` |")
A("| `mach3_bench.cpp` | same pattern around `solve_euler2d` in `run()` + `[WALL] TOTAL` | same |")
A("| `double_mach_bench.cpp` | same | same |")
A("| `config3_bench.cpp` | same | same |")
A("| `validation_smoke.cpp` | ALREADY timed the solve (table column `wall(s)`); added one keyed "
  "line so the key matches the 3D benches | `[WALL] <case> wall=...s` |")
A("| `shock_mixing_bench.cpp` | **NO CHANGE — already printed `wall=51.0s`** in its status line, "
  "same `wall=` key as the 3D benches | `status=... wall=%.1fs` |")
A("")
A("`[WALL] <label>` = wall of ONE time-integration loop. `[WALL] TOTAL` = whole app "
  "(several schemes per app), so TOTAL is NOT the cascade number — use the labelled line of the "
  "run that actually uses the cheng3 recon (`BVD` / `T-MLP-u`).\n")
A("### Diff summary (file:line, post-edit numbering)\n")
A("`cpp/` is NOT tracked by git (`git status` reports `?? cpp/apps/`), so no `git diff` exists. "
  "Added lines, verbatim:\n")
A("```")
A("leveque_bench.cpp:10       #include <chrono>")
A("leveque_bench.cpp:30       run(...) signature += `, const char* label = nullptr`")
A("leveque_bench.cpp:35,37-38 t0_ around solve_adv2d + printf(\"[WALL] %s wall=%.3fs steps=%d\")")
A("leveque_bench.cpp:76       t_all_ start (before the mlp_u1 run)")
A("leveque_bench.cpp:91-92    printf(\"[WALL] TOTAL wall=%.3fs\")")
A("leveque_bench.cpp:71,76,81 call sites pass labels \"mlp_u1\" / \"T-MLP-u-L\" / \"BVD\"")
A("mach3_bench.cpp:16,29,38,41-42,125,154-155        same pattern around solve_euler2d")
A("  call-site labels: \"mlp_u1\", \"T-MLP-u-L\", \"BVD\", \"C_line\"")
A("double_mach_bench.cpp:15,72,86,89-90,177,203-204  same pattern")
A("  call-site labels: \"mlp_u1\", \"T-MLP-u-L\", \"ORDER-2\", \"BVD\"")
A("config3_bench.cpp:19,61,64,69-70,129,138-139      same pattern")
A("  call-site labels: \"mlp_u1\", \"T-MLP-u\"")
A("validation_smoke.cpp:107   ONE line added (it already timed the solve at :96-98):")
A("                           printf(\"[WALL] %s wall=%.3fs steps=%d\", C.name, wall, r.n_steps)")
A("shock_mixing_bench.cpp     UNCHANGED")
A("```")
A("No `include/cfd/*.hpp` was touched. Rebuild verified by md5 change of every binary "
  "(leveque_bench ad1f98->1b68ba, mach3_bench 905153->e943a4, double_mach_bench b9aaaa->107cbf, "
  "config3_bench 015d13->947b51, validation_smoke b3f74b->9129fb; shock_mixing_bench md5 "
  "unchanged = d1a50b, as expected since its source was not edited).\n")

A("## 2. Conditions\n")
A("- `CFD_MAXSTEP=500 CHENG3_PROF=1` (as in the original 2026-07-13 `paper2_recon_profile` run)")
A("- S1 `BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1`")
A("- S2 `BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_GAUSS=1`")
A("- cores `taskset -c 0-22:2` = 12 PHYSICAL cores, `OMP_NUM_THREADS=12 OMP_PROC_BIND=close "
  "OMP_PLACES=cores` (verified: the 12 OMP threads sat on cpus 0,2,...,22; the parallel 3D agent "
  "sat on cpu 24+, no physical-core overlap)")
A("- 3 repeats per (case, scheme); mach3 has 2 (see §6)\n")
A("### Exact commands\n```")
for case in CASES:
    for sch in SCHEMES:
        if (case, sch) in data:
            A("%-12s %s" % (case, data[(case, sch)][0]["cmd"]))
            break
A("```\n")

A("## 3. Cascade table — median over repeats (s)\n")
A("| case | scheme | MUSCL | geom | face | THINC | BVD_sel | recon sum | solver wall | "
  "min wall | recon/wall | app total |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|")
for case in CASES:
    for sch in SCHEMES:
        if (case, sch) not in data:
            continue
        rs = data[(case, sch)]
        p = lambda k: med([r["prof"][k] for r in rs])
        w = med([r["wall"] for r in rs])
        tot = [r["total"] for r in rs if r["total"] is not None]
        A("| %s | %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | **%.1f** | %.1f | %.3f | %s |" % (
            case, sch.upper(), p("muscl"), p("geom"), p("face"), p("thinc"), p("bvd"),
            med([r["recon"] for r in rs]), w, min(r["wall"] for r in rs),
            med([r["recon"] / r["wall"] for r in rs]),
            ("%.1f" % med(tot)) if tot else "-"))
A("")
A("`recon sum` = MUSCL + THINC + BVD_sel  (THINC = geom + face).\n")

A("## 3b. THE CASCADE — S1(tanh) / S2(GAUSS) speedup at each aggregation layer\n")
A("Analogue of the 3D deform chain (80x kernel -> 4.6x stage -> 1.37x recon -> 1.29x solver). "
  "Finest layer available from `CHENG3_PROF` in 2D is `geom` (the cell-D / moment kernel where "
  "GAUSS replaces the tanh Newton solve); `face` is the shared quadrature, so it dilutes.\n")
A("| case | geom (kernel) | THINC (stage) | recon (all) | solver (total) |")
A("|---|---|---|---|---|")
for case in CASES:
    if (case, "s1") not in data or (case, "s2") not in data:
        continue
    a, b = data[(case, "s1")], data[(case, "s2")]
    g = med([r["prof"]["geom"] for r in a]) / med([r["prof"]["geom"] for r in b])
    t = med([r["prof"]["thinc"] for r in a]) / med([r["prof"]["thinc"] for r in b])
    rc = med([r["recon"] for r in a]) / med([r["recon"] for r in b])
    w = med([r["wall"] for r in a]) / med([r["wall"] for r in b])
    wmin = min(r["wall"] for r in a) / min(r["wall"] for r in b)
    A("| %s | %.2fx | %.2fx | %.2fx | %.2fx (min-based %.2fx) |" % (case, g, t, rc, w, wmin))
A("")
A("The geom and THINC columns come from within-run counters and are solid. The recon and solver "
  "columns divide two DIFFERENT processes' walls, so they carry the full contention noise of §5 — "
  "on the Euler cases (mach3, doublemach) the S1/S2 solver ratio is ~1.0 and is buried in that "
  "noise. Note this is not new: the 2026-07-13 breakdown also had S1 ~ S2 on the Euler benches "
  "(mach3 THINC 16.3 vs 16.0 s; doublemach 60.6 vs 56.1 s). GAUSS's win is large only where the "
  "THINC stage dominates (leveque advection, shockmixing, config3).\n")

A("## 4. The layer that matters: recon → solver dilution (WITHIN-run ratio)\n")
A("Absolute walls are noisy on this machine (§6), but `recon_sum / solver_wall` is measured "
  "inside a single process, so it is contention-robust. This is the last hop of the cascade.\n")
A("| case | scheme | " + " | ".join(repnames) + " | median | spread |")
A("|---|---|" + "---|" * (len(repnames) + 2))
for case in CASES:
    for sch in SCHEMES:
        if (case, sch) not in data:
            continue
        rs = {r["rep"]: r["recon"] / r["wall"] for r in data[(case, sch)]}
        v = list(rs.values())
        A("| %s | %s | %s | %.3f | %.1f%% |" % (
            case, sch.upper(),
            " | ".join(("%.3f" % rs[n]) if n in rs else "-" for n in repnames),
            med(v), spread(v)))
A("")

A("## 5. Per-repeat solver wall (timing reliability)\n")
A("| case | scheme | " + " | ".join(repnames) + " | median | min | spread |")
A("|---|---|" + "---|" * (len(repnames) + 3))
for case in CASES:
    for sch in SCHEMES:
        if (case, sch) not in data:
            continue
        rs = {r["rep"]: r["wall"] for r in data[(case, sch)]}
        v = list(rs.values())
        A("| %s | %s | %s | %.1f | %.1f | **%.1f%%** |" % (
            case, sch.upper(),
            " | ".join(("%.1f" % rs[n]) if n in rs else "-" for n in repnames),
            med(v), min(v), spread(v)))
A("")

A("## 6. Honest limitations\n")
A("1. **A foreign, UNPINNED workload shared the machine the whole time.** From 19:02 a third "
  "party's jobs (`SEMO_Robuster_*`, `SEMO_SPH_*`, plus cmake/gmake/cc1plus and 3 pytest "
  "processes) ran with affinity `0-63`, i.e. free to land on cpus 0-22. loadavg went "
  "16 -> 68 and settled around 25-35 (see §7). My 12 threads were pinned, but pinning does not "
  "stop unpinned threads from time-slicing the same cpus, and the SMT siblings (cpus 1,3,...,23) "
  "were never mine. Repeat-to-repeat spread is therefore **7-46%** and the absolute walls should "
  "be treated as upper bounds with ~20% error bars. The within-run ratios in §4 are the "
  "trustworthy part.")
A("2. **rep1's S2 half is additionally self-contaminated**: between 19:14 and 19:22 I ran mach3 "
  "diagnostics on the same core set. `shockvortex_s2` rep1 (160.7 s vs 101.7/120.2 in rep2/3) and "
  "`doublemach_s2` rep1 (398.2 s) are that artefact. Later diagnostics were moved to cpus 48-62.")
A("3. **mach3 could not be run with the literal S1/S2 recipe.** `HLLC_PVRS=1` (part of the "
  "documented recipe) makes the Mach-3 reflective wall emit no flux, so the uniform inflow state "
  "is preserved exactly: after 500 steps the field is `rho[1.4000,1.4000] p_min=1.0000 "
  "max|drho|=0.0000`. Isolated at 120x40/300 steps: baseline -> `rho[0.3818,5.8629]`; "
  "`+HLLC_HLLBLEND=0` -> `rho[0.3837,5.8944]`; `+HLLC_PVRS=1` -> **frozen**. The 2026-07-13 log "
  "shows a developed field, so that run cannot have used HLLC+PVRS either. mach3 was therefore "
  "re-run with its canonical flux `M3_FLUX=hll M3_CFL=0.3` (`paper_final/paper_runner.sh`), which "
  "reproduces the old `p_min` to 3 digits (0.3079 vs 0.3083). The frozen-field logs are kept as "
  "`rep1/2d_mach3_s{1,2}.FROZEN-HLLCPVRS.log` and are NOT in the CSV. mach3 hence has 2 repeats.")
A("4. **The 2026-07-13 absolute stage times do not reproduce** (§8): the C++ recon code changed "
  "after that date (2D analytic-moment cache, 2026-07-26; recon perf work) and the old run's "
  "thread count is unrecorded — the old runner script no longer exists anywhere in the repo, so "
  "the case settings had to be reverse-engineered from the log headers. What DOES reproduce is "
  "the case setup (cells, grid, t_end, flux id, steps) and the physics diagnostics — see §8.")
A("5. `config3` is run with `C3_FLUX=llf` because the old log header says `flux=0` = `FLUX_LLF`. "
  "That is NOT the S1/S2 paper flux (HLLC), and LLF is cheaper than HLLC, so config3's "
  "`recon/wall` fraction is biased HIGH relative to a paper-recipe run. Kept for comparability "
  "with the old breakdown; flag it if the cascade figure is meant to show the paper configuration.")
A("6. `shockmixing` exits rc=1 (status=INCOMPLETE) — expected, `CFD_MAXSTEP=500` stops it at "
  "t=8.92/120. Same as the old run.\n")

A("## 7. loadavg per run\n")
A("| case | scheme | rep | loadavg before | loadavg after |")
A("|---|---|---|---|---|")
for case in CASES:
    for sch in SCHEMES:
        for r in data.get((case, sch), []):
            A("| %s | %s | %s | %s | %s |" % (case, sch.upper(), r["rep"], r["lb"], r["la"]))
A("")

A("## 8. Reproduction check vs the ORIGINAL 2026-07-13 breakdown\n")
A("### 8a. Case setup + physics diagnostics (this is the real check)\n")
A("(`%` signs below are literal percentages.)\n")
A("| case | old | new | verdict |")
A("|---|---|---|---|")
A("| leveque | `N=200 cells=160000`, BVD `L1=8.9325e-02 cone_pk=0.968` | `N=200 cells=160000`, "
  "BVD `L1=8.9319e-02 cone_pk=0.969` | same case; 7e-5 relative drift (code changed) |")
A("| shockmixing | `cells=64000 grid=400x80 flux=2 cfl=0.40`, `t=8.9218 rho[0.9557,1.9756]` | "
  "same header, `t=8.9231 rho[0.9559,1.9697]` | same case, small drift |")
A("| shockvortex | `100x50 steps=500 t=0.096/0.350 rho_min=9.15e-01 p_min=8.84e-01` | "
  "identical to all printed digits | **exact match** (confirms `VS_ONLY=16 VS_NSCALE=4`, "
  "default RHLLC flux) |")
A("| mach3 | `cells=129024 grid=480x160 mesh=unstructured_2d t_end=4.00` | identical header "
  "(confirms `M3_MESH=uniform 480 160 4.0`); flux had to change, see §6.3 | setup match, "
  "flux differs |")
A("| config3 | `cells=156800 t_end=0.800 flux=0`, `ens=29.33266 rho_min=0.1378 p_min=0.0290` | "
  "identical header, `ens=28.97663 rho_min=0.1378 p_min=0.0290` | same case, 1.2% ens drift |")
A("| doublemach | `cells=370421 nodes=186212`, BVD `rho[1.400,18.239] p_min=1.0000` | identical "
  "header, `rho[1.399,18.181] p_min=0.9991` | same case, small drift |")
A("")
A("### 8b. Stage times old -> new (median)\n")
A("| case | scheme | MUSCL | THINC | geom | face | BVD_sel | THINC/MUSCL old->new |")
A("|---|---|---|---|---|---|---|---|")
for case in CASES:
    for sch in SCHEMES:
        if (case, sch) not in data:
            continue
        o = old_prof(case, sch)
        if not o:
            continue
        rs = data[(case, sch)]
        n = {k: med([r["prof"][k] for r in rs]) for k in ("muscl", "thinc", "geom", "face", "bvd")}
        A("| %s | %s | %.1f->%.1f (%.2fx) | %.1f->%.1f (%.2fx) | %.1f->%.1f (%.2fx) | "
          "%.1f->%.1f (%.2fx) | %.1f->%.1f (%.2fx) | %.2f -> %.2f |" % (
              case, sch.upper(),
              o["muscl"], n["muscl"], n["muscl"] / o["muscl"],
              o["thinc"], n["thinc"], n["thinc"] / o["thinc"],
              o["geom"], n["geom"], n["geom"] / o["geom"],
              o["face"], n["face"], n["face"] / o["face"],
              o["bvd"], n["bvd"], n["bvd"] / o["bvd"],
              o["thinc"] / o["muscl"], n["thinc"] / n["muscl"]))
A("")
A("mach3's row compares different fluxes (old row = whatever the 2026-07-13 runner used, new row "
  "= `M3_FLUX=hll M3_CFL=0.3`), so its ratios are not a like-for-like comparison.\n")
A("Pattern: EVERY stage is 1.2-2.1x slower than 2026-07-13, and the slowdown is roughly uniform "
  "across MUSCL / geom / face / BVD_sel. That is the signature of a machine-level difference "
  "(fewer cores + contention), not of one stage regressing — the old run's thread count is not "
  "recorded but the project default for paper-resolution 2D is 24 physical cores, i.e. 2x what "
  "was available here. The residual detail is that S1's tanh THINC scaled slightly worse "
  "(1.61-1.95x) than S2's GAUSS THINC (1.47-1.74x), consistent with the transcendental-heavy tanh "
  "path suffering more from SMT-sibling contention. **Conclusion: the stage breakdown is "
  "internally consistent and the case setups reproduce exactly, but the absolute 2026-07-13 "
  "numbers are NOT reproducible on this machine state — do not mix old and new absolute seconds "
  "in one figure.**\n")

open(os.path.join(ROOT, "summary_2d.md"), "w").write("\n".join(L) + "\n")
print("wrote summary_2d.md, %d lines" % len(L))
