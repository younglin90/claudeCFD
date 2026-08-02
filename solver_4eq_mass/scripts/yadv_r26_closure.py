#!/usr/bin/env python3
"""Round 26 -- closure(A) vs closure(B) reachability analysis for cases 24/33/34.

docs/YADV_ROUND_26_PLAN.md. All EOS/case constants below are transcribed from
cpp/denner_1d/src/{cases,eos}.cpp -- see the comment beside each for the source line.
Zero C++ changes; this script is the round's only new file (plan sect.6, gate G0/G5).
"""
import argparse
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

# ACID_ENV_VARS purge list, copied from yadv_r9_sweep.py's base_env() pattern (round 22 hygiene).
ACID_ENV_VARS = (
    "ACID_YADV", "ACID_YADV_ALPHA_IMPLICIT", "ACID_YADV_ALPHA_IMPLICIT_T",
    "ACID_NO_AJAC", "ACID_RHIST", "ACID_BLK_STEP",
    "ACID_YADV_RECON", "ACID_RECON", "ACID_YADV_RESYNC", "ACID_RESYNC",
    "ACID_YADV_HREINIT", "ACID_RINIT", "ACID_RCELL",
    "ACID_STALL_ACCEPT", "ACID_STALL_ACCEPT_MAX", "ACID_TSAT", "ACID_AJAC_BLK",
    "ACID_PROJ_UNTIL", "ACID_ADRIFT", "ACID_RECON_NULL",
    "ACID_F3", "ACID_YADV_F3", "ACID_TEND_SCALE", "ACID_DBG",
)


def base_env(overlay=None):
    env = dict(os.environ)
    for k in ACID_ENV_VARS:
        env.pop(k, None)
    env["DENNER_ACID"] = "1"  # .claude/rules/denner-pitfalls.md -- mandatory, else silent 11/19
    if overlay:
        env.update(overlay)
    return env


# ---------------------------------------------------------------------------------------------
# EOS -- transcribed from cpp/denner_1d/include/denner1d/types.hpp:8-14 (struct Phase) and
# cpp/denner_1d/src/eos.cpp:11-13 (air_phase), cases.cpp:446 (denner_water), cases.cpp:110-111
# (rho_air_ref/rho_water_ref). Field order: (gamma, pinf, b, kv, eta).
# ---------------------------------------------------------------------------------------------
AIR = dict(gamma=1.4, pinf=0.0, b=0.0, kv=720.25, eta=0.0)
WATER = dict(gamma=4.1, pinf=4.4e8, b=0.0, kv=474.2, eta=0.0)
RHO_AIR_REF = 1.1574
RHO_WATER_REF = 998.0
MS = 10.0


def phase_rho(p, T, ph):
    # eos.cpp:24-29 phase_props: rho = (p+pinf) / (kv*(gamma-1)*T + b*(p+pinf))
    gm1 = ph["gamma"] - 1.0
    A = ph["kv"] * T * gm1 + ph["b"] * (p + ph["pinf"])
    return (p + ph["pinf"]) / A


def phase_h(p, T, ph):
    # eos.cpp:29: h = gamma*kv*T + b*p + eta
    return ph["gamma"] * ph["kv"] * T + ph["b"] * p + ph["eta"]


def phase_cp(ph):
    return ph["gamma"] * ph["kv"]  # eos.cpp:35, constant (b=0 phases here)


def mixture_density(p, T, alpha, a, b):
    return alpha * phase_rho(p, T, a) + (1.0 - alpha) * phase_rho(p, T, b)


def temperature_for_mixture_density_pressure(p, rho, alpha, a, b):
    # cases.cpp:38-51, bisection (reproduced exactly, same tolerance behaviour)
    lo, hi = 1.0e-6, 1.0
    while mixture_density(p, hi, alpha, a, b) > rho and hi < 1.0e9:
        hi *= 2.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if mixture_density(p, mid, alpha, a, b) > rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def closure_a_shock(alpha_air):
    """cases.cpp:105-151 compute_case24_shock, transcribed verbatim (alpha_post = alpha_pre)."""
    a, b = AIR, WATER
    p_pre, u_pre = 1.0e5, 0.0
    alpha_pre = alpha_air
    rho_pre = alpha_pre * RHO_AIR_REF + (1.0 - alpha_pre) * RHO_WATER_REF
    T_pre = temperature_for_mixture_density_pressure(p_pre, rho_pre, alpha_pre, a, b)
    ra_pre = phase_rho(p_pre, T_pre, a)
    rb_pre = phase_rho(p_pre, T_pre, b)
    rho_pre = alpha_pre * ra_pre + (1.0 - alpha_pre) * rb_pre
    cpa_pre, cpb_pre = phase_cp(a), phase_cp(b)
    inv_gm1 = alpha_pre / (a["gamma"] - 1.0) + (1.0 - alpha_pre) / (b["gamma"] - 1.0)
    gamma_mix = 1.0 + 1.0 / inv_gm1
    cp_mix = (alpha_pre * ra_pre * cpa_pre + (1.0 - alpha_pre) * rb_pre * cpb_pre) / rho_pre
    c_pre = math.sqrt((gamma_mix - 1.0) * cp_mix * T_pre)
    Vs = MS * c_pre
    Pihat = ((gamma_mix - 1.0) / gamma_mix) * rho_pre * cp_mix * T_pre - p_pre
    pr = 1.0 + (2.0 * gamma_mix / (gamma_mix + 1.0)) * (MS * MS - 1.0) * (1.0 + Pihat / p_pre)
    p_post = pr * p_pre
    G = (gamma_mix + 1.0) / (gamma_mix - 1.0)
    pratio = (p_post + Pihat) / (p_pre + Pihat)
    rho_post = rho_pre * (G * pratio + 1.0) / (G + pratio)
    u_post = Vs * (1.0 - rho_pre / rho_post)
    alpha_post = alpha_pre
    T_post = temperature_for_mixture_density_pressure(p_post, rho_post, alpha_post, a, b)
    Y_pre = alpha_pre * ra_pre / rho_pre
    ra_post = phase_rho(p_post, T_post, a)
    Y_post = alpha_post * ra_post / rho_post
    return dict(p_pre=p_pre, u_pre=u_pre, T_pre=T_pre, alpha_pre=alpha_pre, rho_pre=rho_pre,
                Y_pre=Y_pre, Vs=Vs, gamma_mix=gamma_mix, cp_mix=cp_mix, c_pre=c_pre,
                p_post=p_post, u_post=u_post, T_post=T_post, alpha_post=alpha_post,
                rho_post=rho_post, Y_post=Y_post)


# ---------------------------------------------------------------------------------------------
# Closure (B): Y-held mixture Hugoniot, plan sect.2.1 Eq.B1-B2. Both phases have b=0 so
# v(p,T) = kv*(gamma-1)*T/(p+pinf) and h(p,T) is linear in T at fixed p -- the Hugoniot is
# explicit, not iterative.
# ---------------------------------------------------------------------------------------------
def mix_coeffs(Y, a, b):
    cpbar = Y * a["gamma"] * a["kv"] + (1.0 - Y) * b["gamma"] * b["kv"]
    cvbar = Y * a["kv"] + (1.0 - Y) * b["kv"]
    Ka = Y * (a["gamma"] - 1.0) * a["kv"]
    Kb = (1.0 - Y) * (b["gamma"] - 1.0) * b["kv"]
    qbar = Y * a["eta"] + (1.0 - Y) * b["eta"]
    return cpbar, cvbar, Ka, Kb, qbar


def S_of_p(p, Ka, Kb, a, b):
    return Ka / (p + a["pinf"]) + Kb / (p + b["pinf"])


def hugoniot_b(p0, T0, Y, p1, a, b):
    """Given upstream (p0,T0,Y) and downstream pressure p1, solve the Y-held RH jump for
    (T1,v1,u1,Vs) via the explicit Eq.B1-B2 form (plan sect.2.1)."""
    cpbar, cvbar, Ka, Kb, qbar = mix_coeffs(Y, a, b)
    S0 = S_of_p(p0, Ka, Kb, a, b)
    v0 = T0 * S0 + (Y * a["b"] + (1.0 - Y) * b["b"])
    S1 = S_of_p(p1, Ka, Kb, a, b)
    bbar = Y * a["b"] + (1.0 - Y) * b["b"]
    # h1 - h0 = 0.5*(p1-p0)*(v0+v1), v1 = T1*S1 + bbar, h1 = cpbar*T1 + qbar
    # cpbar*T1 - 0.5*(p1-p0)*S1*T1 = h0 + 0.5*(p1-p0)*(v0+bbar) - qbar
    h0 = cpbar * T0 + qbar
    rhs = h0 + 0.5 * (p1 - p0) * (v0 + bbar) - qbar
    denom = cpbar - 0.5 * (p1 - p0) * S1
    T1 = rhs / denom
    v1 = T1 * S1 + bbar
    dv = v0 - v1
    if dv <= 0.0 or p1 <= p0:
        return None
    mdot2 = (p1 - p0) / dv
    mdot = math.sqrt(mdot2)
    Vs = mdot * v0
    u1 = Vs * (1.0 - v1 / v0)
    rho1 = 1.0 / v1
    return dict(T1=T1, v1=v1, rho1=rho1, u1=u1, Vs=Vs, mdot=mdot, S1=S1, cpbar=cpbar,
                cvbar=cvbar, Ka=Ka, Kb=Kb)


def find_star_pressure(pL, TL, YL, uL, pR, TR, YR, uR, a, b):
    """Two-shock Riemann star-state solve, plan sect.2.3. Left state moves right shock relative
    to itself (already moving at uL); right state is at rest (uR=0)."""
    def uL_star(p):
        r = hugoniot_b(pL, TL, YL, p, a, b)
        if r is None:
            return None
        # shock moving LEFTWARD relative to the *L state in the lab frame is not needed here --
        # the left state drives a RIGHT-moving shock into itself from ahead (fresh gas = *L,
        # already-shocked reference = L). Velocity change across a right-facing shock with
        # upstream state L (velocity uL) and downstream *L: u*L = uL - (shock-frame deltau)
        return uL - r["u1"]

    def uR_star(p):
        r = hugoniot_b(pR, TR, YR, p, a, b)
        if r is None:
            return None
        return uR + r["u1"]

    lo = max(pL, pR) * 1.0000001
    hi = lo * 10.0
    for _ in range(200):
        vhi_l, vhi_r = uL_star(hi), uR_star(hi)
        if vhi_l is None or vhi_r is None:
            hi *= 2.0
            continue
        if vhi_l < vhi_r:
            break
        hi *= 2.0
    else:
        raise RuntimeError("star pressure bracket search failed")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        vl, vr = uL_star(mid), uR_star(mid)
        if vl is None or vr is None:
            lo = mid
            continue
        if vl > vr:
            lo = mid
        else:
            hi = mid
    pstar = math.sqrt(lo * hi)
    rL = hugoniot_b(pL, TL, YL, pstar, a, b)
    rR = hugoniot_b(pR, TR, YR, pstar, a, b)
    ustar_l = uL - rL["u1"]
    ustar_r = uR + rR["u1"]
    return dict(p=pstar, u_from_L=ustar_l, u_from_R=ustar_r, rL=rL, rR=rR)


def case_alpha(cid):
    return {"24": 0.5, "33": 0.75, "34": 0.25}[cid]


def stage0(verbose=True):
    """P0: reproduce plan sect.3.1 and 3.2 to >=5 sig figs, plus internal consistency asserts."""
    a, b = AIR, WATER
    results = {}
    ok = True
    for cid in ("24", "33", "34"):
        A = closure_a_shock(case_alpha(cid))
        t_end = 0.7 / A["Vs"]
        # closure(B) evaluated AT the reference shock speed Vs_ref -- solve for p1 such that
        # the resulting Vs matches A['Vs'] (bisection on p1).
        def vs_of_p(p1):
            r = hugoniot_b(A["p_pre"], A["T_pre"], A["Y_pre"], p1, a, b)
            return r["Vs"] if r else None
        lo, hi = A["p_pre"] * 1.0001, A["p_pre"] * 1.0e8
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            vs = vs_of_p(mid)
            if vs is None or vs < A["Vs"]:
                lo = mid
            else:
                hi = mid
        p1_match = math.sqrt(lo * hi)
        Bm = hugoniot_b(A["p_pre"], A["T_pre"], A["Y_pre"], p1_match, a, b)

        # exact Riemann solution: left state = closure-A post-shock (Y=Y_L=A['Y_post']),
        # right state = pre-shock (Y=Y_pre)
        star = find_star_pressure(A["p_post"], A["T_post"], A["Y_post"], A["u_post"],
                                   A["p_pre"], A["T_pre"], A["Y_pre"], A["u_pre"], a, b)
        rel_u_mismatch = abs(star["u_from_L"] - star["u_from_R"]) / max(abs(star["u_from_R"]), 1e-30)

        rho_star_L = star["rL"]["rho1"]
        rho_star_R = star["rR"]["rho1"]
        S_L = A["u_post"] - star["rL"]["Vs"]  # left-shock lab-frame speed (moving left, negative)
        S_R = A["u_pre"] + star["rR"]["Vs"]   # == star["rR"]["Vs"] since u_pre=0
        x_leftshock = 0.1 + S_L * t_end
        x_contact = 0.1 + star["u_from_L"] * t_end
        x_leadshock = 0.1 + S_R * t_end

        results[cid] = dict(A=A, t_end=t_end, Vs_ref=A["Vs"], B_at_Vsref=Bm, star=star,
                             rel_u_mismatch=rel_u_mismatch, rho_star_L=rho_star_L,
                             rho_star_R=rho_star_R, S_L=S_L, S_R=S_R,
                             x_leftshock=x_leftshock, x_contact=x_contact,
                             x_leadshock=x_leadshock)
        if verbose:
            print(f"case {cid}: Vs_ref={A['Vs']:.4f} t_end={t_end:.6e} rho_pre={A['rho_pre']:.5f} "
                  f"T_pre={A['T_pre']:.4f} Y_pre={A['Y_pre']:.7e}")
            print(f"  (A) p_post={A['p_post']:.7e} rho_post={A['rho_post']:.5f} "
                  f"u_post={A['u_post']:.4f} T_post={A['T_post']:.3f} Y_post={A['Y_post']:.7e}")
            print(f"  (B)@Vsref p_post={p1_match:.7e} rho_post={Bm['rho1']:.4f} "
                  f"u_post={Bm['u1']:.4f} T_post={Bm['T1']:.3f}")
            print(f"  star: p*={star['p']:.6e} u*_L={star['u_from_L']:.3f} "
                  f"u*_R={star['u_from_R']:.3f} rel_mismatch={rel_u_mismatch:.3e}")
            print(f"  rho*_L={rho_star_L:.3f} rho*_R={rho_star_R:.3f} S_L={S_L:.1f} S_R={S_R:.1f} "
                  f"S_R/Vs_ref={S_R/A['Vs']:.4f}")
            print(f"  x_leftshock={x_leftshock:.4f} x_contact={x_contact:.4f} "
                  f"x_leadshock={x_leadshock:.4f}")
        if rel_u_mismatch > 1e-9:
            ok = False
            print(f"  P0 FAIL case {cid}: star-state velocity mismatch {rel_u_mismatch:.3e}")
    print("P0", "PASS" if ok else "FAIL")
    return results, ok


# ---------------------------------------------------------------------------------------------
# Gate re-implementation -- validation.cpp:18-26 (rel_scale), :49 (correlation degenerate
# branch), :317-359 (accumulate), :445-460 (gradient_peak_x), :469-505 (case24_spec_pass)
# ---------------------------------------------------------------------------------------------
def rel_scale(ref):
    return max(max(ref) - min(ref), 1.0)


def correlation(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
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
    linf = max(abs((got[i] - ref[i]) / scale) for i in range(n))
    return l2, corr, linf


def gradient_peak_x(field, x, lo_frac, hi_frac):
    n = len(field)
    lo = int(lo_frac * n)
    hi = int(hi_frac * n)
    best_i, best_g = lo, -1.0
    for i in range(max(lo, 1), min(hi, n - 1)):
        g = abs(field[i + 1] - field[i - 1])
        if g > best_g:
            best_g = g
            best_i = i
    return x[best_i]


def case24_spec_pass_py(x, p, u, rho, p_ref, u_ref, rho_ref):
    n = len(x)
    dx = x[1] - x[0]
    x_shock = gradient_peak_x(p_ref, x, 0.05, 0.99)
    rho_post = rho_ref[0]
    rho_pre = rho_ref[-1]
    jump = max(abs(rho_post - rho_pre), 1.0)
    lo, hi = 0.005, x_shock - max(10.0 * dx, 0.03)
    dip = hump = s2 = 0.0
    cnt = 0
    for i in range(n):
        if not (lo < x[i] < hi):
            continue
        dip = max(dip, (rho_post - rho[i]) / jump)
        hump = max(hump, (rho[i] - rho_post) / jump)
        s2 += (rho[i] - rho_ref[i]) ** 2
        cnt += 1
    if cnt <= 0:
        return dict(pass_=False)
    plateau_l2 = math.sqrt(s2 / cnt) / jump
    plateau_ok = dip <= 0.02 and hump <= 0.01 and plateau_l2 <= 0.015
    l2_p, corr_p, linf_p = accumulate(p, p_ref)
    l2_u, corr_u, linf_u = accumulate(u, u_ref)
    l2_rho, corr_rho, linf_rho = accumulate(rho, rho_ref)
    profile_ok = (l2_p <= 0.20 and l2_u <= 0.20 and l2_rho <= 0.20
                  and corr_p >= 0.92 and corr_u >= 0.92 and corr_rho >= 0.92)
    return dict(pass_=profile_ok and plateau_ok, x_shock=x_shock, dip=dip, hump=hump,
                plateau_l2=plateau_l2, l2_p=l2_p, l2_u=l2_u, l2_rho=l2_rho,
                corr_p=corr_p, corr_u=corr_u, corr_rho=corr_rho)


def dump(cid, env_overlay=None):
    r = subprocess.run([DUMP, cid], capture_output=True, text=True, env=base_env(env_overlay),
                        cwd=ROOT)
    reader = csv.DictReader(io.StringIO(r.stdout))
    cols = {k: [] for k in ("x", "alpha", "p", "u", "rho", "p_ref", "u_ref", "rho_ref")}
    for row in reader:
        for k in cols:
            v = row[k]
            cols[k].append(float("nan") if "nan" in v.lower() else float(v))
    return cols


def validate_all(env_overlay=None):
    r = subprocess.run([VALIDATE], capture_output=True, text=True, env=base_env(env_overlay),
                        cwd=ROOT)
    out = {}
    for line in r.stdout.splitlines():
        if line.startswith('{"case"'):
            d = json.loads(NAN_RE.sub("NaN", line))
            out[d["case"]] = d
    return out


def gatecheck():
    """P1: Python gate must match denner1d_validate's JSON to 6 sig figs on several
    (case,config) pairs, and match the pass verdict."""
    pairs = [
        ("24", {}), ("33", {}), ("34", {}),
        ("24", {"ACID_YADV": "1"}), ("33", {"ACID_YADV": "1"}), ("34", {"ACID_YADV": "1"}),
        ("24", {"ACID_YADV": "1", "ACID_YADV_F3": "1"}),
        ("26", {}), ("27", {}), ("28", {}),
    ]
    ok = True
    for cid, overlay in pairs:
        d = dump(cid, overlay)
        vjson = validate_all(overlay).get(cid, {})
        if cid in ("24", "33", "34"):
            py = case24_spec_pass_py(d["x"], d["p"], d["u"], d["rho"],
                                      d["p_ref"], d["u_ref"], d["rho_ref"])
            fields = [("l2_p", "l2_p"), ("l2_u", "l2_u"), ("l2_rho", "l2_rho"),
                      ("corr_p", "corr_p"), ("corr_u", "corr_u"), ("corr_rho", "corr_rho")]
            match = True
            for pyk, jk in fields:
                jv = vjson.get(jk)
                if jv is None or math.isnan(jv):
                    continue
                pv = py[pyk]
                rel = abs(pv - jv) / max(abs(jv), 1e-300)
                if rel > 1e-5:
                    match = False
                    print(f"  MISMATCH {cid} {overlay}: {pyk} py={pv:.6g} json={jv:.6g} "
                          f"rel={rel:.2e}")
            pass_match = py["pass_"] == vjson.get("pass")
            if not pass_match:
                match = False
                print(f"  PASS-VERDICT MISMATCH {cid} {overlay}: py={py['pass_']} "
                      f"json={vjson.get('pass')}")
            print(f"case {cid} {overlay or 'OFF'}: py_pass={py['pass_']} json_pass={vjson.get('pass')} "
                  f"match={'OK' if match else 'FAIL'}")
            ok = ok and match
    print("P1", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------------------------
# Stage 2 -- reachability
# ---------------------------------------------------------------------------------------------
def gate_stage(results):
    """T1: gate scores of the exact config-B Riemann solution, on an N=800 grid."""
    N = 800
    x = [(i + 0.5) / N for i in range(N)]
    for cid, r in results.items():
        A, star = r["A"], r["star"]
        p = []; u = []; rho = []
        for xi in x:
            if xi < r["x_leftshock"]:
                # L: unshocked closure-A post-shock state (the IC's own left region)
                p.append(A["p_post"]); u.append(A["u_post"]); rho.append(A["rho_post"])
            elif xi < r["x_contact"]:
                # *L: behind the left shock, Y=Y_L
                p.append(star["p"]); u.append(star["u_from_L"]); rho.append(r["rho_star_L"])
            elif xi < r["x_leadshock"]:
                # *R: behind the leading shock, Y=Y_pre (p,u continuous across the contact)
                p.append(star["p"]); u.append(star["u_from_R"]); rho.append(r["rho_star_R"])
            else:
                # R: undisturbed pre-shock state
                p.append(A["p_pre"]); u.append(A["u_pre"]); rho.append(A["rho_pre"])
        p_ref = [A["p_post"] if xi < 0.8 else A["p_pre"] for xi in x]
        u_ref = [A["u_post"] if xi < 0.8 else A["u_pre"] for xi in x]
        rho_ref = [A["rho_post"] if xi < 0.8 else A["rho_pre"] for xi in x]
        py = case24_spec_pass_py(x, p, u, rho, p_ref, u_ref, rho_ref)
        print(f"case {cid} EXACT gate scores: l2_p={py['l2_p']:.4f} l2_u={py['l2_u']:.4f} "
              f"l2_rho={py['l2_rho']:.4f} corr_p={py['corr_p']:.4f} corr_u={py['corr_u']:.4f} "
              f"corr_rho={py['corr_rho']:.4f} dip={py['dip']:.4f} hump={py['hump']:.4f} "
              f"plateau_l2={py['plateau_l2']:.4f} pass={py['pass_']}")


def reachable_scan(results):
    """T2: sweep the entire admissible closure-B Hugoniot locus, report per-metric minimum and
    rho_max^B, vs the gate's implied requirements."""
    a, b = AIR, WATER
    N = 800
    x = [(i + 0.5) / N for i in range(N)]
    for cid, r in results.items():
        A = r["A"]
        p0, T0, Y0 = A["p_pre"], A["T_pre"], A["Y_pre"]
        best_rho = 0.0
        best_p = None
        ps = [p0 * (1.0001 * (1.0e18 / (p0 * 1.0001)) ** (k / 400.0)) for k in range(401)]
        for p1 in ps:
            rB = hugoniot_b(p0, T0, Y0, p1, a, b)
            if rB is None:
                continue
            if rB["rho1"] > best_rho:
                best_rho = rB["rho1"]
                best_p = p1
        rho_A_post, rho_pre = A["rho_post"], A["rho_pre"]
        jump = max(abs(rho_A_post - rho_pre), 1.0)
        dip_bound = (rho_A_post - best_rho) / jump
        l2_bound = math.sqrt(0.8) * (rho_A_post - best_rho) / jump
        gbar = (Y0 * a["gamma"] * a["kv"] + (1.0 - Y0) * b["gamma"] * b["kv"]) / \
               (Y0 * a["kv"] + (1.0 - Y0) * b["kv"])
        asym = (gbar + 1.0) / (gbar - 1.0) if gbar > 1.0 else float("nan")
        print(f"case {cid}: rho_max^B={best_rho:.3f} @p={best_p:.4e}  gbar={gbar:.6f} "
              f"asym_rho={asym:.3f}")
        print(f"  dip_bound(at rho_max)={dip_bound:.4f} (thr 0.02, {dip_bound/0.02:.1f}x) "
              f"l2_bound={l2_bound:.4f} (thr 0.20, {l2_bound/0.20:.2f}x)")
        reachable = dip_bound <= 0.02 or l2_bound <= 0.20
        print(f"  ANY member of the family satisfies dip or l2_rho alone: {reachable}")


# ---------------------------------------------------------------------------------------------
# Stage 4 -- thread (c): case33/24 corr_p sign flip autopsy (existing dumps, no new runs beyond
# what round 25 already produced -- re-dumped here for a self-contained script)
# ---------------------------------------------------------------------------------------------
def autopsy():
    for cid in ("24", "33"):
        dB = dump(cid, {"ACID_YADV": "1"})
        dF3 = dump(cid, {"ACID_YADV": "1", "ACID_YADV_F3": "1"})
        for label, d in (("B", dB), ("B+F3", dF3)):
            finite = all(not math.isnan(v) for v in d["p"])
            if not finite:
                print(f"case {cid} {label}: NOT FINITE, skip")
                continue
            p = d["p"]
            n = len(p)
            lo_mean = sum(p[: n // 4]) / (n // 4)
            hi_mean = sum(p[3 * n // 4:]) / (n - 3 * n // 4)
            l2, corr, _ = accumulate(p, d["p_ref"])
            print(f"case {cid} {label}: p[0:25%] mean={lo_mean:.4e} p[75%:100%] mean={hi_mean:.4e} "
                  f"(increasing={hi_mean>lo_mean}) corr_p={corr:.4f} l2_p={l2:.4f}")


# ---------------------------------------------------------------------------------------------
# Stage 5 -- case15 redirect soundness: reference must track the active config (self-convergent)
# ---------------------------------------------------------------------------------------------
def redirect_check():
    d_off = dump("15", {})
    d_on = dump("15", {"ACID_YADV": "1"})
    diff = max(abs(a - b) for a, b in zip(d_off["p_ref"], d_on["p_ref"]))
    print(f"case15 p_ref OFF-vs-B max|diff|={diff:.6e} "
          f"(nonzero expected -- reference tracks the active config, no closure mismatch)")


# ---------------------------------------------------------------------------------------------
# Stage 3 -- window measurement at ACID_TEND_SCALE=sigma
# ---------------------------------------------------------------------------------------------
def window_measure(sigma):
    a, b = AIR, WATER
    results, _ = stage0(verbose=False)
    for cid, r in results.items():
        A = r["A"]
        t_end = r["t_end"]
        predicted_lead = 0.1 + r["S_R"] * sigma * t_end
        overlays = [
            ("B", {"ACID_YADV": "1"}),
            ("B+F3", {"ACID_YADV": "1", "ACID_YADV_F3": "1"}),
            ("B+RECON+F3", {"ACID_YADV": "1", "ACID_YADV_RECON": "1", "ACID_YADV_F3": "1"}),
            ("C+F3", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1", "ACID_YADV_F3": "1"}),
        ]
        for label, ov in overlays:
            ov = dict(ov, ACID_TEND_SCALE=str(sigma))
            proc = subprocess.run([DUMP, cid], capture_output=True, text=True,
                                   env=base_env(dict(ov, ACID_DBG="1")), cwd=ROOT)
            stderr = proc.stderr
            stalled = "STALLED" in stderr or "DIVERGED" in stderr
            last_t = None
            for line in reversed(stderr.splitlines()):
                m = re.search(r"\bt=([0-9.eE+-]+)", line)
                if m:
                    last_t = float(m.group(1))
                    break
            reader = csv.DictReader(io.StringIO(proc.stdout))
            xs, ps, us, rhos = [], [], [], []
            for row in reader:
                xs.append(float(row["x"]))
                v = row["p"]
                ps.append(float("nan") if "nan" in v.lower() else float(v))
                us.append(float(row["u"]) if "nan" not in row["u"].lower() else float("nan"))
                rhos.append(float(row["rho"]) if "nan" not in row["rho"].lower() else float("nan"))
            finite = all(not math.isnan(v) for v in ps)
            target_t = sigma * t_end
            null_guard = (last_t is not None and last_t >= 0.99 * target_t) if finite else False
            if not finite or stalled or not null_guard:
                print(f"case {cid} {label} sigma={sigma}: REJECTED "
                      f"(finite={finite} stalled={stalled} last_t={last_t} target={target_t:.4e})")
                continue
            wlo, whi = (0.45, 0.70) if cid != "34" else (0.42, 0.63)
            win = [(xs[i], ps[i], us[i], rhos[i]) for i in range(len(xs)) if wlo < xs[i] < whi]
            if not win:
                print(f"case {cid} {label} sigma={sigma}: window empty, skip")
                continue
            pw = [v[1] for v in win]
            spread = (max(pw) - min(pw)) / max(abs(sum(pw) / len(pw)), 1.0)
            p_mean = sum(v[1] for v in win) / len(win)
            u_mean = sum(v[2] for v in win) / len(win)
            rho_mean = sum(v[3] for v in win) / len(win)
            gp = abs(p_mean - r["star"]["p"]) / r["star"]["p"]
            gu = abs(u_mean - r["star"]["u_from_R"]) / max(abs(r["star"]["u_from_R"]), 1e-30)
            grho = abs(rho_mean - r["rho_star_R"]) / r["rho_star_R"]
            print(f"case {cid} {label} sigma={sigma}: window_spread={spread:.4f} "
                  f"p={p_mean:.4e}(gap {gp:.2%}) u={u_mean:.2f}(gap {gu:.2%}) "
                  f"rho={rho_mean:.3f}(gap {grho:.2%}) predicted_lead_x={predicted_lead:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", action="store_true")
    ap.add_argument("--gatecheck", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--reachable", action="store_true")
    ap.add_argument("--autopsy", action="store_true")
    ap.add_argument("--redirect", action="store_true")
    ap.add_argument("--window", type=float, default=None)
    args = ap.parse_args()

    if args.stage0:
        stage0()
    if args.gatecheck:
        gatecheck()
    if args.gate or args.reachable:
        results, ok = stage0(verbose=False)
        if args.gate:
            gate_stage(results)
        if args.reachable:
            reachable_scan(results)
    if args.autopsy:
        autopsy()
    if args.redirect:
        redirect_check()
    if args.window is not None:
        window_measure(args.window)


if __name__ == "__main__":
    main()
