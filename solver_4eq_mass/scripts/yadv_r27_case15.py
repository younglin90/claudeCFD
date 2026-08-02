#!/usr/bin/env python3
"""Round 27 -- case15 per-predicate + mass/floor census across the standard config battery.

docs/YADV_ROUND_27_PLAN.md sect.3/4. Re-implements validation.cpp's case15 gate
(validation.cpp:684-730), rel_scale/correlation/accumulate (:18-51,317-359), in Python,
fed by denner1d_dump's own columns, plus a domain mass balance and pressure-floor census.
"""
import csv
import io
import json
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.path.join(ROOT, "build-cpp/cpp/denner_1d/denner1d_dump")
VALIDATE = os.path.join(ROOT, "build-cpp/cpp/denner_1d/denner1d_validate")
NAN_RE = re.compile(r"-?nan")

ACID_ENV_VARS = (
    "ACID_YADV", "ACID_YADV_ALPHA_IMPLICIT", "ACID_YADV_ALPHA_IMPLICIT_T",
    "ACID_NO_AJAC", "ACID_RHIST", "ACID_BLK_STEP",
    "ACID_YADV_RECON", "ACID_RECON", "ACID_YADV_RESYNC", "ACID_RESYNC",
    "ACID_YADV_HREINIT", "ACID_RINIT", "ACID_RCELL",
    "ACID_STALL_ACCEPT", "ACID_STALL_ACCEPT_MAX", "ACID_TSAT", "ACID_AJAC_BLK",
    "ACID_PROJ_UNTIL", "ACID_ADRIFT", "ACID_RECON_NULL",
    "ACID_F3", "ACID_YADV_F3", "ACID_TEND_SCALE", "ACID_DBG", "ACID_MBAL",
    "ACID_NFEAS", "ACID_YADV_ALPHA_IMPLICIT_CAV",
)

CONFIGS = [
    ("A", "OFF", {}),
    ("B", "ON", {"ACID_YADV": "1"}),
    ("C", "ON+IMPLICIT", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1"}),
    ("D", "ON+IMPLICIT+FD", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1", "ACID_NO_AJAC": "1"}),
    ("E", "OFF+FD", {"ACID_NO_AJAC": "1"}),
    ("F", "ON+IMPLICIT+T", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1",
                             "ACID_YADV_ALPHA_IMPLICIT_T": "1"}),
    ("G", "ON+FD", {"ACID_YADV": "1", "ACID_NO_AJAC": "1"}),
]


def base_env(overlay=None):
    env = dict(os.environ)
    for k in ACID_ENV_VARS:
        env.pop(k, None)
    env["DENNER_ACID"] = "1"
    if overlay:
        env.update(overlay)
    return env


def dump(cid, overlay=None):
    r = subprocess.run([DUMP, cid], capture_output=True, text=True, env=base_env(overlay), cwd=ROOT)
    reader = csv.DictReader(io.StringIO(r.stdout))
    cols = {k: [] for k in ("x", "alpha", "p", "u", "rho", "p_ref", "u_ref", "rho_ref")}
    for row in reader:
        for k in cols:
            v = row[k]
            cols[k].append(float("nan") if "nan" in v.lower() else float(v))
    return cols, r.stderr


def rel_scale(ref):
    return max(max(ref) - min(ref), 1.0)


def correlation(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    den = math.sqrt(va * vb)
    if den <= 1.0e-300:
        return 1.0
    return cov / den


def accumulate(got, ref):
    scale = rel_scale(ref)
    n = len(got)
    l2 = math.sqrt(sum(((got[i] - ref[i]) / scale) ** 2 for i in range(n)) / n)
    corr = correlation(got, ref)
    return l2, corr


def jump_stats(u, x):
    nn = len(u)
    central = abs(u[nn // 2] - u[nn // 2 - 1])
    jmax, tv = 0.0, 0.0
    for i in range(1, nn):
        if x[i] < 0.35 or x[i] > 0.65:
            continue
        j = abs(u[i] - u[i - 1])
        jmax = max(jmax, j)
        tv += j
    conc = jmax / max(tv, 1e-300)
    return central, jmax, conc


def tv_of(v):
    return sum(abs(v[i] - v[i - 1]) for i in range(1, len(v)))


def case15_gate(x, p, u, rho, p_ref, u_ref, rho_ref):
    cj, mj, cc = jump_stats(u, x)
    cj_r, mj_r, cc_r = jump_stats(u_ref, x)
    smooth_ok = cj <= max(8.0, 1.10 * cj_r) and mj <= max(8.0, 1.10 * mj_r) and \
                cc <= max(0.04, 1.10 * cc_r)
    p_osc = max(0.0, tv_of(p) - tv_of(p_ref)) / max(tv_of(p_ref), 1.0)
    r_osc = max(0.0, tv_of(rho) - tv_of(rho_ref)) / max(tv_of(rho_ref), 1.0e-6)
    osc_ok = p_osc < 0.02 and r_osc < 0.04
    l2_p, corr_p = accumulate(p, p_ref)
    l2_u, corr_u = accumulate(u, u_ref)
    l2_rho, corr_rho = accumulate(rho, rho_ref)
    pass_ = (corr_p >= 0.93 and corr_u >= 0.998 and corr_rho >= 0.99 and
             l2_p <= 0.18 and l2_u <= 0.06 and l2_rho <= 0.05 and smooth_ok and osc_ok)
    return dict(cj=cj, mj=mj, cc=cc, cj_r=cj_r, mj_r=mj_r, cc_r=cc_r, smooth_ok=smooth_ok,
                p_osc=p_osc, r_osc=r_osc, osc_ok=osc_ok, l2_p=l2_p, corr_p=corr_p,
                l2_u=l2_u, corr_u=corr_u, l2_rho=l2_rho, corr_rho=corr_rho, pass_=pass_)


def sweep():
    N = 400
    dx = 1.0 / N
    for cfg, label, overlay in CONFIGS:
        d, stderr = dump("15", overlay)
        g = case15_gate(d["x"], d["p"], d["u"], d["rho"], d["p_ref"], d["u_ref"], d["rho_ref"])
        M = sum(d["rho"]) * dx
        M_ref = sum(d["rho_ref"]) * dx
        nfloor = sum(1 for v in d["p"] if v <= 1.0 + 1e-12)
        print(f"[{cfg}] {label:16s} pass={g['pass_']!s:5s} l2_p={g['l2_p']:.5f} l2_u={g['l2_u']:.5f} "
              f"l2_rho={g['l2_rho']:.5f} corr_p={g['corr_p']:.6f} corr_u={g['corr_u']:.6f} "
              f"corr_rho={g['corr_rho']:.6f} cj={g['cj']:.3f} mj={g['mj']:.3f} cc={g['cc']:.5f} "
              f"smooth_ok={g['smooth_ok']} osc_ok={g['osc_ok']} M={M:.3f} M_ref={M_ref:.3f} "
              f"nfloor={nfloor}/{N}")


def tend_scale():
    for sigma in (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00):
        for cfg, overlay in (("B", {"ACID_YADV": "1"}), ("C", {"ACID_YADV": "1",
                             "ACID_YADV_ALPHA_IMPLICIT": "1"})):
            ov = dict(overlay, ACID_TEND_SCALE=str(sigma))
            d, _ = dump("15", ov)
            N = len(d["p"])
            dx = 1.0 / N
            M = sum(d["rho"]) * dx
            minp = min(d["p"])
            nfloor = sum(1 for v in d["p"] if v <= 1.0 + 1e-12)
            almin, almax = min(d["alpha"]), max(d["alpha"])
            print(f"sigma={sigma:.2f} [{cfg}] M={M:.3f} min_p={minp:.4e} "
                  f"alpha=[{almin:.4f},{almax:.4f}] nfloor={nfloor}")


def overlays():
    combos = [
        ("plain B", {"ACID_YADV": "1"}),
        ("B+F3", {"ACID_YADV": "1", "ACID_YADV_F3": "1"}),
        ("B+RECON", {"ACID_YADV": "1", "ACID_YADV_RECON": "1"}),
        ("B+RESYNC", {"ACID_YADV": "1", "ACID_YADV_RESYNC": "1"}),
        ("B+HREINIT", {"ACID_YADV": "1", "ACID_YADV_HREINIT": "1"}),
        ("C (=B+IMPLICIT)", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1"}),
        ("B+CAV (round 28)", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT_CAV": "1"}),
    ]
    N = 400
    dx = 1.0 / N
    for label, overlay in combos:
        d, _ = dump("15", overlay)
        g = case15_gate(d["x"], d["p"], d["u"], d["rho"], d["p_ref"], d["u_ref"], d["rho_ref"])
        M = sum(d["rho"]) * dx
        M_ref = sum(d["rho_ref"]) * dx
        nfloor = sum(1 for v in d["p"] if v <= 1.0 + 1e-12)
        print(f"{label:16s} l2_rho={g['l2_rho']:.5f} corr_rho={g['corr_rho']:.6f} "
              f"l2_p={g['l2_p']:.5f} corr_p={g['corr_p']:.6f} M={M:.3f} M_ref={M_ref:.3f} "
              f"nfloor={nfloor}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "sweep":
        sweep()
    elif cmd == "tend":
        tend_scale()
    elif cmd == "overlays":
        overlays()
    else:
        print("usage: yadv_r27_case15.py [sweep|tend|overlays]")
