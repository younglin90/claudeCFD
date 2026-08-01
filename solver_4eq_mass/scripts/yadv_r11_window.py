#!/usr/bin/env python3
"""Phase 3a Stage 2 -- ACID_TEND_SCALE observation-window sweep.

Companion to scripts/yadv_rh2.py (round 10, kept untouched -- its numbers must stay
reproducible). yadv_rh2.py's completion guard hardcodes a FINAL_TIME dict and infers
t_last from the last *printed* (every-200-steps) "ACID step" line -- both are too
imprecise once ACID_TEND_SCALE is scaling the run. This script instead reads the
solver's own "ACID done case=... step=... t=... of ..." line (round 11 Stage 1 sect.3.5,
ACID_DBG-gated) for an exact, scale-agnostic end state, and treats a "STALLED:" line
(round 11 Stage 1) as an authoritative null-run marker instead of the IC-match heuristic.

Three measurements (docs/YADV_ROUND_11_PLAN.md sect.4.4):
  (a) front-position-vs-window sweep for cases 24/34 under +ALPHA_IMPLICIT -- locates the
      shock front by argmax|dp/dx|, samples a plateau window derived from x_front (not a
      fixed box), fits Vs_front from x_front vs scaled t_end, and compares to Vs(mass).
  (b) stall-bracketing sweep for 24/plain, 34/plain, 33/+IMPLICIT -- brackets the step at
      which the retry loop first stalls.
  (c) control: 33/plain at scale 1.0, must reproduce YADV_RESEARCH.md sect.20.2/20.3's
      +8.808e-01 / Vs/Vs_ref=0.5355.

Root derived from __file__ (yadv_r9_sweep.py / yadv_rh2.py's pattern), not hardcoded.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build-cpp", "cpp", "denner_1d")
DUMP = os.path.join(BUILD, "denner1d_dump")

# Same NASG constants as yadv_rhcheck.py / yadv_rh2.py.
GA, KVA, PIA, ETA_A = 1.4, 720.25, 0.0, 0.0
GW, KVW, PIW, ETA_W = 4.1, 474.2, 4.4e8, 0.0

_ACID_DONE_RE = re.compile(r"ACID done case=(\S+) step=(\d+) t=([\-0-9.eE+]+) of ([\-0-9.eE+]+)")
_STALLED_RE = re.compile(
    r"STALLED: case=(\S+) no admissible step at dt=([\-0-9.eE+]+) after (\d+) retries, "
    r"step (\d+), t=([\-0-9.eE+]+) of ([\-0-9.eE+]+)")

BASE_FINAL_TIME = {"24": 0.7 / 6426.761, "33": 0.7 / 5456.494, "34": 0.7 / 8201.394}


def env_for(overlay, dbg=True):
    env = dict(os.environ, DENNER_ACID="1")
    for k in ("ACID_YADV", "ACID_YADV_ALPHA_IMPLICIT", "ACID_YADV_ALPHA_IMPLICIT_T",
              "ACID_NO_AJAC", "ACID_DBG", "ACID_TEND_SCALE"):
        env.pop(k, None)
    if dbg:
        env["ACID_DBG"] = "1"
    env.update(overlay)
    return env


def run(case, overlay, scale=None):
    """Runs denner1d_dump; returns (rows, stderr_text). overlay sets solver env vars."""
    ov = dict(overlay)
    if scale is not None:
        ov["ACID_TEND_SCALE"] = repr(scale)
    env = env_for(ov)
    r = subprocess.run([DUMP, case], capture_output=True, text=True, env=env, cwd=ROOT)
    rows = [[float(v) for v in ln.split(",")] for ln in r.stdout.strip().splitlines()[1:]]
    return rows, r.stderr


def parse_status(stderr_text):
    """Returns dict: stalled(bool), reason, cell, step, t, t_end (scale-agnostic, from the
    solver's own ACID_DBG lines) -- or None fields if neither line was found (shouldn't happen
    with ACID_DBG=1)."""
    m = _STALLED_RE.search(stderr_text)
    if m:
        return {"stalled": True, "case": m.group(1), "dt": float(m.group(2)),
                "retries": int(m.group(3)), "step": int(m.group(4)),
                "t": float(m.group(5)), "t_end": float(m.group(6))}
    m = _ACID_DONE_RE.search(stderr_text)
    if m:
        return {"stalled": False, "case": m.group(1), "step": int(m.group(2)),
                "t": float(m.group(3)), "t_end": float(m.group(4))}
    return {"stalled": None, "step": None, "t": None, "t_end": None}


def T_of(p, rho, al):
    return (al * (p + PIA) / (KVA * (GA - 1.0))
            + (1.0 - al) * (p + PIW) / (KVW * (GW - 1.0))) / rho


def preshock_state(case):
    """Robust pre-shock (p0, rho0) from the OFF path's rightmost (always-undisturbed) cell --
    same approach as yadv_rh2.py, copied (not imported) so this file stays independent."""
    rows, _ = run(case, {}, scale=None)
    last = rows[-1]
    return last[2], last[4]  # p, rho


def rh_residual(p0, rho0, u0, p1, rho1, u1):
    if rho1 == rho0:
        return None, None  # window landed in undisturbed state (rho1==rho0) -- caller skips it
    Vs = rho1 * (u1 - u0) / (rho1 - rho0) + u0
    mom = (p1 - p0) - rho0 * (Vs - u0) * (u1 - u0)
    return Vs, mom


def front_position(rows):
    """argmax|dp/dx| -- the shock front's grid position."""
    best_i, best_g = 0, -1.0
    for i in range(1, len(rows) - 1):
        dx = rows[i + 1][0] - rows[i - 1][0]
        if dx <= 0:
            continue
        g = abs((rows[i + 1][2] - rows[i - 1][2]) / dx)
        if g > best_g:
            best_g, best_i = g, i
    return rows[best_i][0], best_i


def window_state(rows, x_front, lo=0.05, hi=0.20):
    """Median state over [x_front-hi, x_front-lo] -- BEHIND the front (the shock propagates
    left-to-right into undisturbed material, so post-shock/processed material is at smaller x,
    not larger x -- an earlier version of this sampled ahead of the front by mistake and always
    landed in the still-undisturbed region). Window derived from the measured front position,
    not a fixed box (sect.20.2's flagged flaw)."""
    sel = [r for r in rows if x_front - hi <= r[0] <= x_front - lo]
    if not sel:
        return None
    sel.sort(key=lambda r: r[2])
    return sel[len(sel) // 2]


def sweep_a(case, overlay, scales):
    """Front-position-vs-window sweep. Returns list of dicts, one per scale."""
    p0, rho0 = preshock_state(case)
    out = []
    for s in scales:
        rows, err = run(case, overlay, scale=s)
        st = parse_status(err)
        if st["stalled"]:
            out.append({"scale": s, "stalled": True, **st})
            continue
        xf, i_front = front_position(rows)
        win = window_state(rows, xf)
        if win is None:
            out.append({"scale": s, "stalled": False, "no_window": True, "t_end": st["t_end"]})
            continue
        rho1, p1, u1 = win[4], win[2], win[3]
        Vs, mom = rh_residual(p0, rho0, 0.0, p1, rho1, u1)
        if Vs is None:
            out.append({"scale": s, "stalled": False, "undisturbed_window": True, "x_front": xf,
                        "t_end": st["t_end"]})
            continue
        out.append({"scale": s, "stalled": False, "x_front": xf, "t_end": st["t_end"],
                     "p_post": p1, "rho_post": rho1, "u_post": u1, "Vs_mass": Vs, "mom": mom})
    return out, p0, rho0


def linfit(xs, ys):
    """Least-squares slope/intercept + R^2 (no numpy dependency)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def sweep_b(case, overlay, scales):
    """Stall-bracketing sweep. Returns list of dicts, one per scale."""
    out = []
    for s in scales:
        rows, err = run(case, overlay, scale=s)
        st = parse_status(err)
        out.append({"scale": s, **st})
    return out


def main():
    print("Phase 3a Stage 2 -- ACID_TEND_SCALE window sweep\n")

    print("=== (a) front-position-vs-window sweep: cases 24/34, +ALPHA_IMPLICIT ===")
    a_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
    for case in ("24", "34"):
        overlay = {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1"}
        results, p0, rho0 = sweep_a(case, overlay, a_scales)
        print(f"\ncase {case}: preshock p0={p0:.5e} rho0={rho0:.4f}")
        print("| scale | status | x_front | t_end | p_post | rho_post | u_post | Vs(mass) | mom resid (rel) |")
        print("|---|---|---|---|---|---|---|---|---|")
        clean = []
        for r in results:
            if r.get("stalled"):
                print(f"| {r['scale']} | STALLED reason=? step={r.get('step')} | -- | {r.get('t_end')} | -- | -- | -- | -- | -- |")
                continue
            if r.get("no_window"):
                print(f"| {r['scale']} | no window found | -- | {r['t_end']} | -- | -- | -- | -- | -- |")
                continue
            if r.get("undisturbed_window"):
                print(f"| {r['scale']} | window still undisturbed (rho1==rho0) | {r['x_front']:.4f} "
                      f"| {r['t_end']:.6e} | -- | -- | -- | -- | -- |")
                continue
            print(f"| {r['scale']} | ok | {r['x_front']:.4f} | {r['t_end']:.6e} | {r['p_post']:.5e} "
                  f"| {r['rho_post']:.2f} | {r['u_post']:.1f} | {r['Vs_mass']:.1f} "
                  f"| {r['mom']/max(abs(r['p_post']),1):+.3e} |")
            clean.append(r)
        if len(clean) >= 3:
            fit = linfit([r["t_end"] for r in clean], [r["x_front"] for r in clean])
            if fit:
                slope, intercept, r2 = fit
                Vs_ref = 0.7 / final_time_of(case)
                print(f"Vs_front (linear fit of x_front vs t_end) = {slope:.1f}  (R^2={r2:.6f}, n={len(clean)})")
        if clean:
            last = clean[-1]
            print(f"largest-scale-inside-domain residual (scale={last['scale']}): "
                  f"mom={last['mom']:+.3e} ({last['mom']/max(abs(last['p_post']),1):+.3e} rel)")

    print("\n=== (b) stall-bracketing sweep ===")
    b_scales = [0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]
    for case, overlay, label in (
        ("24", {"ACID_YADV": "1"}, "24/plain"),
        ("34", {"ACID_YADV": "1"}, "34/plain"),
        ("33", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1"}, "33/+IMPLICIT"),
    ):
        results = sweep_b(case, overlay, b_scales)
        print(f"\n{label}:")
        print("| scale | status | step | t | t_end |")
        print("|---|---|---|---|---|")
        first_stall_step = None
        last_clean_step = None
        for r in results:
            if r["stalled"] is True:
                print(f"| {r['scale']} | STALLED (retries={r.get('retries')}) | {r['step']} | {r['t']:.4e} | {r['t_end']:.4e} |")
                if first_stall_step is None:
                    first_stall_step = r["step"]
            elif r["stalled"] is False:
                print(f"| {r['scale']} | clean (hit scaled t_end) | {r['step']} | {r['t']:.4e} | {r['t_end']:.4e} |")
                last_clean_step = r["step"]
            else:
                print(f"| {r['scale']} | (no ACID_DBG line found) | -- | -- | -- |")
        if first_stall_step is not None and last_clean_step is not None:
            print(f"bracket: clean through step {last_clean_step}, first STALLED at step {first_stall_step}")

    print("\n=== (c) control: 33/plain at scale 1.0 ===")
    p0, rho0 = preshock_state("33")
    rows, err = run("33", {"ACID_YADV": "1"}, scale=1.0)
    st = parse_status(err)
    if st["stalled"]:
        print("UNEXPECTED: 33/plain stalled at scale 1.0 -- control run itself is broken")
    else:
        undis = [r for r in rows if r[2] < 1.5 * p0]
        i_front = rows.index(undis[0])
        pre = undis[len(undis) // 2]
        post = rows[max(i_front - 60, 0)]
        Vs, mom = rh_residual(pre[2], pre[4], pre[3], post[2], post[4], post[3])
        Vs_ref = 0.7 / final_time_of("33")
        print(f"p_post={post[2]:.5e} rho_post={post[4]:.2f} u_post={post[3]:.1f} "
              f"Vs={Vs:.1f} Vs/Vs_ref={Vs/Vs_ref:.4f} "
              f"mom={mom:+.3e} ({mom/max(abs(post[2]),1):+.3e} rel)")
        print("expect (YADV_RESEARCH.md sect.20.2/20.3): Vs/Vs_ref=0.5355, mom_rel=+8.81e-01")


def final_time_of(case):
    return BASE_FINAL_TIME[case]


if __name__ == "__main__":
    main()
