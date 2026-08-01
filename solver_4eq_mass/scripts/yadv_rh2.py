#!/usr/bin/env python3
"""Phase 3a Stage 0 -- Rankine-Hugoniot check with a NULL-RUN guard.

yadv_rhcheck.py (kept byte-identical, not touched) has no guard against a SILENTLY STALLED
run: acid.cpp's per-step retry loop does `if (!stepped) break;` on exhausting all dt-halving
retries WITHOUT setting `diverged`, so solve_case_acid returns a finite state -- sometimes
still the pristine initial condition -- and it gets scored as a normal completed run. Round 10
found (docs/YADV_PHASE3_PLAN.md) that this is exactly what happened to the round-9 "case33 RH
closes to machine precision under +ALPHA_IMPLICIT" finding and to round 3's original "24/34
close to 1e-13": both were yadv_rhcheck.py's undisturbed-cell search locking onto a stalled
run's pristine IC and measuring cases.cpp's own closure-(A) construction against itself.

Root derived from __file__ (yadv_r9_sweep.py's pattern), not a hardcoded worktree path.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build-cpp", "cpp", "denner_1d")
DUMP = os.path.join(BUILD, "denner1d_dump")

# Same NASG constants as yadv_rhcheck.py (b=0, eta=0 for both phases in the 24/33/34 pair).
GA, KVA, PIA, ETA_A = 1.4, 720.25, 0.0, 0.0
GW, KVW, PIW, ETA_W = 4.1, 474.2, 4.4e8, 0.0

# final_time = 0.7 / Vs, Vs = compute_case24_shock(...).Vs from cases.cpp (Ms=10 * Wood mixture
# sound speed at alpha_pre). Reproduced independently via ACID_DBG traces this round; matches
# YADV_RESEARCH.md sect.11.2's Vs table (6426.761 / 5456.494 / 8201.394) to 4 sig figs.
FINAL_TIME = {"24": 0.7 / 6426.761, "33": 0.7 / 5456.494, "34": 0.7 / 8201.394}

CASES = ("24", "33", "34")
CONFIGS = [
    ("plain", {"ACID_YADV": "1"}),
    ("+IMPLICIT", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1"}),
]


def env_for(overlay):
    env = dict(os.environ, DENNER_ACID="1")
    for k in ("ACID_YADV", "ACID_YADV_ALPHA_IMPLICIT", "ACID_YADV_ALPHA_IMPLICIT_T",
              "ACID_NO_AJAC", "ACID_DBG"):
        env.pop(k, None)
    env.update(overlay)
    return env


def dump_rows(case, overlay):
    env = env_for(overlay)
    out = subprocess.run([DUMP, case], capture_output=True, text=True, env=env, cwd=ROOT).stdout
    return [[float(v) for v in ln.split(",")] for ln in out.strip().splitlines()[1:]]


def last_step_fraction(case, overlay):
    """Fraction of final_time actually reached, via ACID_DBG's last printed step."""
    env = env_for(overlay)
    env["ACID_DBG"] = "1"
    r = subprocess.run([DUMP, case], capture_output=True, text=True, env=env, cwd=ROOT)
    t_last = 0.0
    for line in r.stderr.splitlines():
        if line.startswith("ACID step"):
            for tok in line.split():
                if tok.startswith("t="):
                    t_last = float(tok[2:])
    return t_last / FINAL_TIME[case]


def T_of(p, rho, al):
    return (al * (p + PIA) / (KVA * (GA - 1.0))
            + (1.0 - al) * (p + PIW) / (KVW * (GW - 1.0))) / rho


def h_of(p, rho, al):
    T = T_of(p, rho, al)
    ra = (p + PIA) / (KVA * T * (GA - 1.0))
    rb = (p + PIW) / (KVW * T * (GW - 1.0))
    ha = GA * KVA * T + ETA_A
    hb = GW * KVW * T + ETA_W
    Y = al * ra / rho
    return Y * ha + (1.0 - Y) * hb, T, Y


def rh_residual(p0, rho0, u0, p1, rho1, u1):
    Vs = rho1 * (u1 - u0) / (rho1 - rho0) + u0
    mom = (p1 - p0) - rho0 * (Vs - u0) * (u1 - u0)
    return Vs, mom


def preshock_state(case):
    """Robust pre-shock (p0, rho0) from the OFF path's rightmost (always-undisturbed) cell --
    avoids re-deriving the NASG closure-(A) construction independently; the OFF path always
    completes and this project's own dumps confirm the rightmost cells stay pristine IC."""
    rows = dump_rows(case, {})  # OFF: ACID_YADV unset
    last = rows[-1]
    return last[2], last[4]  # p, rho


def plateau_window(rows, lo=0.3, hi=0.6):
    """Median of a fixed safe window: excludes the near-inflow Y-contact region and the
    domain's outer ~15% (boundary-reflection contamination observed in exited-shock runs)."""
    sel = [r for r in rows if lo <= r[0] <= hi]
    if not sel:
        return None
    sel.sort(key=lambda r: r[2])
    mid = sel[len(sel) // 2]
    return mid


def main():
    print("Phase 3a Stage 0 -- RH check with null-run guard\n")
    print("| case | config | status | p_post | rho_post | u_post | Vs(mass) | Vs/Vs_ref "
          "| momentum resid (rel) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for case in CASES:
        p0, rho0 = preshock_state(case)
        for name, overlay in CONFIGS:
            frac = last_step_fraction(case, overlay)
            rows = dump_rows(case, overlay)
            # void guard
            min_p = min(r[2] for r in rows)
            min_rho = min(r[4] for r in rows)
            ic_match = sum(1 for r in rows if abs(r[2] - p0) <= 1e-6 * max(p0, 1.0)
                           and abs(r[3]) < 1e-9) / len(rows)
            stalled = frac < 0.9 or ic_match > 0.90
            if stalled:
                print(f"| {case} | {name} | **NULL RUN** (t/t_end={frac:.4f}, "
                      f"IC-match={ic_match:.2f}, min_p={min_p:.2f}, min_rho={min_rho:.2e}) "
                      "| -- | -- | -- | -- | -- | -- |")
                continue
            # completed: try the in-domain undisturbed-cell search first (works when the shock
            # is still inside the domain, e.g. case33/plain); else fall back to the analytic
            # pre-shock state + a safe plateau window (handles an exited shock cleanly).
            undis = [r for r in rows if r[2] < 1.5 * p0]
            if undis:
                i_front = rows.index(undis[0])
                pre = undis[len(undis) // 2]
                post = rows[max(i_front - 60, 0)]
                mode = "in-domain"
            else:
                pre = [None, 1.0, p0, 0.0, rho0]  # x,alpha,p,u,rho -- alpha unused here
                post = plateau_window(rows)
                mode = "exited (analytic pre-shock + plateau window)"
                if post is None:
                    print(f"| {case} | {name} | completed but no plateau window found | "
                          "-- | -- | -- | -- | -- | -- |")
                    continue
            rho_pre, p_pre, u_pre = pre[4], pre[2], pre[3]
            rho_post, p_post, u_post = post[4], post[2], post[3]
            Vs, mom = rh_residual(p_pre, rho_pre, u_pre, p_post, rho_post, u_post)
            Vs_ref = 0.7 / FINAL_TIME[case]
            print(f"| {case} | {name} | completed ({mode}) | {p_post:.5e} | {rho_post:.2f} "
                  f"| {u_post:.1f} | {Vs:.1f} | {Vs/Vs_ref:.4f} "
                  f"| {mom:+.3e} ({mom/max(abs(p_post),1):+.2e}) |")


if __name__ == "__main__":
    main()
