#!/usr/bin/env python3
"""Build cascade_2d.csv from cpp/results/paper2_cascade/rep*/2d_<case>_<scheme>.log"""
import os, re, glob, statistics

ROOT = "/home/younglin90/work/claude_code/claudeCFD/cpp/results/paper2_cascade"
CASES = ["leveque", "shockmixing", "shockvortex", "mach3", "config3", "doublemach"]
SCHEMES = ["s1", "s2"]
# which [WALL] label corresponds to the cheng3 (S1/S2) solve in each bench
CHENG3_LABEL = {
    "leveque": "BVD", "shockmixing": None, "shockvortex": "shockvortex_2d",
    "mach3": "BVD", "config3": "T-MLP-u", "doublemach": "BVD",
}

RE_PROF = re.compile(
    r"\[CHENG3_PROF\]\s+recon_calls=(\d+)\s+MUSCL=([\d.]+)s\s+THINC=([\d.]+)s\s+"
    r"\(geom=([\d.]+)s face=([\d.]+)s\)\s+BVD_sel=([\d.]+)s")
RE_WALL = re.compile(r"^\[WALL\] (\S+) wall=([\d.]+)s(?: steps=(\d+))?")
RE_SMWALL = re.compile(r"status=\S+\s+wall=([\d.]+)s steps=(\d+)")


def parse(path):
    d = {"walls": {}, "total": None, "prof": None, "load_before": None, "load_after": None}
    for ln in open(path, errors="replace"):
        m = RE_PROF.search(ln)
        if m:
            calls, muscl, thinc, geom, face, bvd = m.groups()
            d["prof"] = dict(calls=int(calls), muscl=float(muscl), thinc=float(thinc),
                             geom=float(geom), face=float(face), bvd=float(bvd))
        m = RE_WALL.match(ln)
        if m:
            lbl, w, st = m.groups()
            if lbl == "TOTAL":
                d["total"] = float(w)
            else:
                d["walls"][lbl] = (float(w), int(st) if st else 0)
        m = RE_SMWALL.search(ln)
        if m:
            d["walls"]["shockmixing"] = (float(m.group(1)), int(m.group(2)))
        if ln.startswith("### loadavg_before:"):
            d["load_before"] = ln.split(":", 1)[1].strip()
        if ln.startswith("### loadavg_after:"):
            d["load_after"] = ln.split(":", 1)[1].strip()
    return d


def cheng3_wall(case, d):
    lbl = CHENG3_LABEL[case]
    if lbl is None:
        lbl = "shockmixing"
    if lbl in d["walls"]:
        return d["walls"][lbl][0]
    return None


rows = []
reps = sorted(glob.glob(os.path.join(ROOT, "rep*")))
for case in CASES:
    for sch in SCHEMES:
        per = []
        for r in reps:
            p = os.path.join(r, "2d_%s_%s.log" % (case, sch))
            if os.path.exists(p):
                per.append((os.path.basename(r), parse(p)))
        if not per:
            continue
        pw = [cheng3_wall(case, d) for _, d in per if cheng3_wall(case, d) is not None]
        prof = [d["prof"] for _, d in per if d["prof"]]
        tot = [d["total"] for _, d in per if d["total"] is not None]
        def med(v):
            return statistics.median(v) if v else float("nan")
        def spread(v):
            return (max(v) - min(v)) / statistics.mean(v) * 100 if len(v) > 1 and statistics.mean(v) else 0.0
        rows.append(dict(
            case=case, scheme=sch, nrep=len(per),
            muscl=med([p["muscl"] for p in prof]), geom=med([p["geom"] for p in prof]),
            face=med([p["face"] for p in prof]), thinc=med([p["thinc"] for p in prof]),
            bvd_sel=med([p["bvd"] for p in prof]),
            recon_sum=med([p["muscl"] + p["thinc"] + p["bvd"] for p in prof]),
            solver_wall=med(pw), solver_wall_spread=spread(pw), wall_min=min(pw) if pw else float("nan"),
            app_total=med(tot) if tot else float("nan"),
            recon_spread=spread([p["muscl"] + p["thinc"] + p["bvd"] for p in prof]),
            walls_all=";".join("%s=%.3f" % (r, cheng3_wall(case, d)) for r, d in per
                               if cheng3_wall(case, d) is not None),
            load_before=per[0][1]["load_before"], load_after=per[-1][1]["load_after"],
        ))

hdr = ["case", "scheme", "nrep", "muscl_s", "geom_s", "face_s", "thinc_s", "bvd_sel_s",
       "recon_sum_s", "recon_spread_pct", "solver_wall_s", "solver_wall_min_s",
       "solver_wall_spread_pct", "app_total_s", "recon_frac_pct", "walls_all_reps",
       "loadavg_before", "loadavg_after"]
out = os.path.join(ROOT, "cascade_2d.csv")
with open(out, "w") as f:
    f.write(",".join(hdr) + "\n")
    for r in rows:
        frac = 100.0 * r["recon_sum"] / r["solver_wall"] if r["solver_wall"] else float("nan")
        f.write("%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,%.2f,%.3f,%.1f,%s,%s,%s\n" % (
            r["case"], r["scheme"], r["nrep"], r["muscl"], r["geom"], r["face"], r["thinc"],
            r["bvd_sel"], r["recon_sum"], r["recon_spread"], r["solver_wall"], r["wall_min"],
            r["solver_wall_spread"], r["app_total"], frac, r["walls_all"],
            r["load_before"], r["load_after"]))
print("wrote", out)
for r in rows:
    print("%-12s %-3s rep=%d recon=%.2fs wall=%.2fs (spread %.1f%%) [%s]" % (
        r["case"], r["scheme"], r["nrep"], r["recon_sum"], r["solver_wall"],
        r["solver_wall_spread"], r["walls_all"]))
