#!/usr/bin/env python3
# Wiggle metric for the Denner ACID acoustic cases.
# Dumps a case via denner1d_dump (DENNER_ACID=1), parses CSV, counts slope-sign
# reversals of p in x=[0.45,0.95] with eps=1e-3 Pa, prints reversals + p2p + amp/corr.
# NEVER uses shell '>' redirection (flaky under wsl.exe): capture via subprocess.
import subprocess, sys, os

# Locate denner1d_dump relative to this script (main tree: build-cpp; agent worktree:
# build-wt); override with env WIG07_DUMP.
_here = os.path.dirname(os.path.abspath(__file__))
_cands = [os.environ.get("WIG07_DUMP", "")] + [
    os.path.join(_here, d, "cpp", "denner_1d", "denner1d_dump")
    for d in ("build-cpp", "build-wt", "build")
]
DUMP = next((p for p in _cands if p and os.path.isfile(p)), _cands[1])

def dump(case, extra_env=None):
    env = dict(os.environ)
    env["DENNER_ACID"] = "1"
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([DUMP, case], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.stderr.write("DUMP FAIL %s: %s\n" % (case, r.stderr[-500:]))
        sys.exit(2)
    rows = []
    for ln in r.stdout.splitlines():
        if ln.startswith("x,") or not ln.strip():
            continue
        parts = ln.split(",")
        if len(parts) < 8:
            continue
        rows.append([float(v) for v in parts])
    return rows  # x,alpha,p,u,rho,p_ref,u_ref,rho_ref

def reversals(xs, ps, lo, hi, eps):
    seg = [(x, p) for x, p in zip(xs, ps) if lo <= x <= hi]
    cnt = 0
    prev_sign = 0
    for i in range(len(seg) - 1):
        d = seg[i + 1][1] - seg[i][1]
        if abs(d) < eps:
            continue
        s = 1 if d > 0 else -1
        if prev_sign != 0 and s != prev_sign:
            cnt += 1
        prev_sign = s
    ph = [p for _, p in seg]
    p2p = (max(ph) - min(ph)) if ph else 0.0
    return cnt, p2p

def corr(a, b):
    n = len(a)
    ma = sum(a) / n; mb = sum(b) / n
    num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da = (sum((a[i]-ma)**2 for i in range(n)))**0.5
    db = (sum((b[i]-mb)**2 for i in range(n)))**0.5
    return num/(da*db) if da*db > 0 else 0.0

def amp_ratio(num, ref):
    an = max(num)-min(num); ar = max(ref)-min(ref)
    return an/ar if ar > 0 else float('inf')

def analyze(case, extra_env=None, lo=0.45, hi=0.95, eps=1e-3):
    rows = dump(case, extra_env)
    xs = [r[0] for r in rows]
    ps = [r[2] for r in rows]; us = [r[3] for r in rows]
    pref = [r[5] for r in rows]; uref = [r[6] for r in rows]
    cnt, p2p = reversals(xs, ps, lo, hi, eps)
    return {
        "case": case,
        "reversals": cnt,
        "p2p_Pa": p2p,
        "amp_ratio_p": amp_ratio(ps, pref),
        "amp_ratio_u": amp_ratio(us, uref),
        "corr_p": corr(ps, pref),
        "corr_u": corr(us, uref),
    }

if __name__ == "__main__":
    cases = sys.argv[1:] or ["07"]
    for cs in cases:
        m = analyze(cs)
        print("case%-3s reversals=%-3d p2p=%.4g Pa  amp_p=%.4f amp_u=%.4f corr_p=%.5f corr_u=%.5f"
              % (m["case"], m["reversals"], m["p2p_Pa"], m["amp_ratio_p"], m["amp_ratio_u"],
                 m["corr_p"], m["corr_u"]))
