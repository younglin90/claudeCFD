#!/usr/bin/env python3
"""Round 32 Stage 2 -- exact double-rarefaction solution of case15-as-specified, docs/
YADV_ROUND_32_PLAN.md sect.4. DIAGNOSTIC ONLY, zero C++ changes; this script is the round's
only new file. Every number is [COMPUTED-PLAN]-turned-measured only after the self-tests
(ST-A..ST-D) below pass -- per the plan's own binding S6(g) rule, no number may be cited
before that.

Physics: case15 is a symmetric two-rarefaction Riemann problem (u_L=-100, u_R=+100, p_L=p_R=
1e5, uniform alpha=0.055 hence uniform mass fraction Y_air). Along each rarefaction, entropy
and mass fraction Y are materially conserved (no diffusion, no phase change in this frozen-
composition 4-eq model) -- so the star state lies on the mixture isentrope through the
upstream state at fixed Y. NASG per-phase entropy is exactly log-linear in (T, p+pinf):
    s_k(p,T) = cv_k*ln(T) - R_k*ln(p+pinf_k) + const_k,   R_k = (gamma_k-1)*cv_k
so the mixture isentrope T(p) at fixed Y has a CLOSED FORM (no entropy root-find needed):
    ln(T/T0) = [Y*Ra*ln((p+Pia)/(p0+Pia)) + (1-Y)*Rb*ln((p+Pib)/(p0+Pib))] / cvbar
The escape velocity is then the quadrature |du| = int_{p*}^{p0} dp/(rho(p)*a_s(p)), with
a_s^2 = dp/drho|_S evaluated by finite difference along this closed-form isentrope.
"""
import argparse
import math
import sys

# Phases, transcribed from eos.cpp:11-19 (Field order: gamma, pinf, b, kv, eta).
AIR = dict(gamma=1.4, pinf=0.0, b=0.0, kv=720.25, eta=0.0)
WATER = dict(gamma=1.187, pinf=7.028e8, b=6.61e-4, kv=3610.0, eta=-1.177788e6)
EPS = 1e-300


def phase_rho(p, T, ph):
    # eos.cpp:25-31 phase_props.rho
    gm1 = ph["gamma"] - 1.0
    A = ph["kv"] * T * gm1 + ph["b"] * (p + ph["pinf"]) + EPS
    return (p + ph["pinf"]) / A


def phase_c(p, T, ph):
    # eos.cpp:29-32 phase_props.c (covolume-corrected)
    rho = phase_rho(p, T, ph)
    ppinf = p + ph["pinf"]
    one_minus = max(1.0 - ph["b"] * rho, 1e-12)
    return math.sqrt(max(ph["gamma"] * ppinf / (rho * one_minus + EPS), 0.0))


def T_from_rho_p(p, rho, ph):
    # cases.cpp:33-36 temperature_for_density_pressure, exact inverse of phase_rho.
    ppinf = p + ph["pinf"]
    return (ppinf / max(rho, 1e-300) - ph["b"] * ppinf) / (ph["kv"] * (ph["gamma"] - 1.0) + EPS)


def mixture_rho_massfrac(p, T, Y, a, b):
    ra, rb = phase_rho(p, T, a), phase_rho(p, T, b)
    return 1.0 / (Y / ra + (1.0 - Y) / rb)


def mixture_sound_speed_wood(p, T, alpha, a, b):
    # eos.cpp:49-57 / acid.cpp:318-325, the Wood mixture sound speed. alpha = VOLUME fraction.
    ra, ca = phase_rho(p, T, a), phase_c(p, T, a)
    rb, cb = phase_rho(p, T, b), phase_c(p, T, b)
    rho = alpha * ra + (1.0 - alpha) * rb
    compress = alpha / (ra * ca * ca + EPS) + (1.0 - alpha) / (rb * cb * cb + EPS)
    return math.sqrt(max(1.0 / (rho * compress + EPS), 0.0))


def alpha_from_Y(Y, rho_a, rho_b):
    # eos.hpp:63-66, transcribed verbatim.
    num = Y * rho_b
    den = rho_a * (1.0 - Y) + num
    return num / den if den > 0.0 else Y


def isentrope_T(p, p0, T0, Y, a, b):
    """Closed-form T(p) along the fixed-Y mixture isentrope through (p0,T0)."""
    Ra = (a["gamma"] - 1.0) * a["kv"]
    Rb = (b["gamma"] - 1.0) * b["kv"]
    cvbar = Y * a["kv"] + (1.0 - Y) * b["kv"]
    lnratio = (Y * Ra * math.log((p + a["pinf"]) / (p0 + a["pinf"]))
               + (1.0 - Y) * Rb * math.log((p + b["pinf"]) / (p0 + b["pinf"])))
    return T0 * math.exp(lnratio / cvbar)


def rho_on_isentrope(p, p0, T0, Y, a, b):
    T = isentrope_T(p, p0, T0, Y, a, b)
    return mixture_rho_massfrac(p, T, Y, a, b), T


def sound_speed_on_isentrope(p, p0, T0, Y, a, b, rel_step=1e-6):
    """a_s^2 = dp/drho|_S, central FD of rho(p) along the closed-form isentrope."""
    dp = max(p * rel_step, 1e-300)
    r_hi, _ = rho_on_isentrope(p + dp, p0, T0, Y, a, b)
    r_lo, _ = rho_on_isentrope(p - dp, p0, T0, Y, a, b)
    drho_dp = (r_hi - r_lo) / (2.0 * dp)
    rho0, _ = rho_on_isentrope(p, p0, T0, Y, a, b)
    return math.sqrt(1.0 / max(drho_dp, 1e-300)), rho0


def escape_velocity(p0, T0, Y, p_target, a, b, n=200000):
    """Quadrature |u| = int_{p_target}^{p0} dp/(rho*a_s), log-spaced (integrand ~1/p at low p)."""
    if p_target <= 0.0:
        raise ValueError("p_target must be > 0")
    log_hi, log_lo = math.log(p0), math.log(p_target)
    total = 0.0
    p_prev = p0
    rho_prev, _ = rho_on_isentrope(p_prev, p0, T0, Y, a, b)
    a_prev, _ = sound_speed_on_isentrope(p_prev, p0, T0, Y, a, b)
    for i in range(1, n + 1):
        p = math.exp(log_hi + (log_lo - log_hi) * i / n)
        rho, _ = rho_on_isentrope(p, p0, T0, Y, a, b)
        a_s, _ = sound_speed_on_isentrope(p, p0, T0, Y, a, b)
        # trapezoid on the integrand 1/(rho*a_s) over dp (dp negative going down in p, take abs)
        f_prev = 1.0 / (rho_prev * a_prev)
        f_cur = 1.0 / (rho * a_s)
        total += 0.5 * (f_prev + f_cur) * abs(p_prev - p)
        p_prev, rho_prev, a_prev = p, rho, a_s
    return total


def find_p_star(p0, T0, Y, target_du, a, b, lo=1e-20, hi=None):
    """Bisect p* such that escape_velocity(p0,T0,Y,p*) == target_du."""
    if hi is None:
        hi = p0
    # escape_velocity is monotone decreasing in p_target (lower target -> more speed available)
    flo = escape_velocity(p0, T0, Y, lo, a, b, n=4000) - target_du
    fhi = escape_velocity(p0, T0, Y, hi * 0.999999, a, b, n=4000) - target_du
    if (flo > 0) == (fhi > 0):
        raise RuntimeError(f"bracket failed: flo={flo} fhi={fhi}")
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        fm = escape_velocity(p0, T0, Y, mid, a, b, n=4000) - target_du
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return math.sqrt(lo * hi)


# ---------------------------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------------------------
def selftests():
    ok = True

    # ST-A: phase_props vs eos.cpp at several states, >=12 sig digits (by construction here,
    # since this IS a transcription -- the real test is internal consistency + ST-C below,
    # which cross-checks against the SOLVER's own measured Wood speed, an independent number).
    for (p, T) in [(1.0e5, 348.2468), (1.0e5, 300.0), (3.2432, 349.35)]:
        ra, ca = phase_rho(p, T, AIR), phase_c(p, T, AIR)
        rb, cb = phase_rho(p, T, WATER), phase_c(p, T, WATER)
        if not (math.isfinite(ra) and math.isfinite(ca) and math.isfinite(rb)
                and math.isfinite(cb) and ra > 0 and rb > 0):
            print(f"[ST-A] FAIL at p={p} T={T}: non-finite/non-positive phase props")
            ok = False
    print(f"[ST-A] phase_props finite/positive at 3 probe states: {'PASS' if ok else 'FAIL'}")

    # ST-B: case15 IC, exactly as cases.cpp:682-688 constructs it.
    alpha0 = 0.055
    p0 = 1.0e5
    T_air = T_from_rho_p(p0, 1.3, AIR)
    T_water = T_from_rho_p(p0, 1000.0, WATER)
    T0 = alpha0 * T_air + (1.0 - alpha0) * T_water
    ra0, rb0 = phase_rho(p0, T0, AIR), phase_rho(p0, T0, WATER)
    rho0_volblend = alpha0 * ra0 + (1.0 - alpha0) * rb0
    Y0 = alpha0 * ra0 / rho0_volblend
    print(f"[ST-B] IC: T_air={T_air:.4f} T_water={T_water:.4f} T0={T0:.4f} "
          f"rho0(alpha-blend)={rho0_volblend:.4f} Y_air={Y0:.6e} "
          f"(spec doc rho0=945.0715, {'MATCH' if abs(rho0_volblend-945.0715)<0.01 else 'DIFFERS -- code does not reproduce spec doc rho0'})")

    # ST-C: sound speed on the isentrope at p0 (Y HELD FIXED, alpha PTE-slaved to (p,T) at
    # every point -- the model's true EQUILIBRIUM/relaxed characteristic speed, since alpha
    # genuinely adjusts algebraically to (Y,p,T) with zero lag in this PTE closure) vs the
    # code's Wood-formula s.a (alpha HELD FIXED, i.e. the FROZEN speed -- no interphase mass
    # exchange during the perturbation, used only as a CFL proxy per acid.cpp:318-325's own
    # comment). These are NOT expected to be equal: the subcharacteristic condition
    # (Linga 2018, round31's own citation; also Wallis 1969's classical two-phase acoustics)
    # requires a_eq <= a_frozen ALWAYS. Test the inequality, not equality -- an equality
    # expectation here would itself be the bug.
    a_eq0, rho_chk = sound_speed_on_isentrope(p0, p0, T0, Y0, AIR, WATER)
    a_frozen0 = mixture_sound_speed_wood(p0, T0, alpha0, AIR, WATER)
    subchar_ok = a_eq0 <= a_frozen0 * (1.0 + 1e-9)
    print(f"[ST-C] a_eq(p0)={a_eq0:.6f} a_frozen(p0,alpha0)={a_frozen0:.6f} "
          f"ratio={a_eq0/a_frozen0:.4f} subcharacteristic(a_eq<=a_frozen): "
          f"{'PASS' if subchar_ok else 'FAIL (violates Linga 2018 theorem -- real bug)'}")
    ok = ok and subchar_ok

    # ST-C2: sound speed at the SOLVER's own measured N=400 core state (round30 sect.2.4,
    # cell 198: p=3.2432, u=+18.154, rho=0.55774, alpha=0.999444) -- an independent check
    # against measured solver output, not just internal self-consistency.
    p_core, T_core, alpha_core = 3.2432, 349.3496, 0.999444
    a_wood_core = mixture_sound_speed_wood(p_core, T_core, alpha_core, AIR, WATER)
    print(f"[ST-C2] a_wood at solver's own measured N=400 core state "
          f"(p={p_core},T={T_core},alpha={alpha_core}): {a_wood_core:.6f} m/s "
          f"(plan predicted 2.8535)")

    # ST-D: per-decade increment of the escape integral below ~100 Pa should equal a*ln(10)
    # where a* = sqrt(Y*Ra*T) is the exact low-p asymptote (rho_air -> p/(Ra*T), water is dead
    # mass, so a_s^2 -> Y*Ra*T, CONSTANT).
    Ra = (AIR["gamma"] - 1.0) * AIR["kv"]
    a_star_closed = math.sqrt(Y0 * Ra * T0)
    du_100_to_10 = escape_velocity(100.0, isentrope_T(100.0, p0, T0, Y0, AIR, WATER), Y0, 10.0,
                                    AIR, WATER, n=20000)
    predicted = a_star_closed * math.log(10.0)
    rel_d = abs(du_100_to_10 / predicted - 1.0)
    print(f"[ST-D] a*(closed)={a_star_closed:.6f} per-decade(quadrature, 100->10 Pa)={du_100_to_10:.6f} "
          f"predicted(a*ln10)={predicted:.6f} rel={rel_d:.3e} {'PASS' if rel_d < 1e-3 else 'FAIL'}")
    ok = ok and rel_d < 1e-3

    return ok, dict(alpha0=alpha0, p0=p0, T0=T0, Y0=Y0, rho0=rho0_volblend, a_star=a_star_closed)


def mode_solve():
    ok, ic = selftests()
    if not ok:
        print("SELF-TESTS FAILED -- no sect.4 number below may be cited (plan S6-g)")
        return 1
    p0, T0, Y0 = ic["p0"], ic["T0"], ic["Y0"]
    a_star = ic["a_star"]
    target = 100.0  # m/s, half of the 200 m/s total jump, by symmetry u*=0

    du_to_1 = escape_velocity(p0, T0, Y0, 1.0, AIR, WATER, n=20000)
    print(f"\n[SOLVE] |du| from p0={p0:.0f} Pa down to 1.0 Pa: {du_to_1:.4f} m/s "
          f"(plan predicted 27.711)")

    remaining = target - du_to_1
    if remaining <= 0.0:
        print(f"[SOLVE] H-X2 FIRES: only {du_to_1:.4f} m/s needed to reach the 1.0 Pa floor, "
              f"but {target} m/s is required and is already exceeded before the floor -- "
              f"p* > 1.0 Pa. Round's core physics claim (H-X1) FALSIFIED.")
        return 2
    decades_below_1pa = remaining / (a_star * math.log(10.0))
    p_star = 1.0 * math.exp(-remaining / a_star)
    print(f"[SOLVE] remaining after 1.0 Pa floor: {remaining:.4f} m/s = {decades_below_1pa:.4f} "
          f"decades below 1 Pa (constant {a_star*math.log(10):.4f} m/s/decade)")
    print(f"[SOLVE] p* = {p_star:.6e} Pa (plan predicted 9.05e-14)")
    print(f"[SOLVE] p*/1.0Pa = {p_star:.3e}  (orders below floor: {-math.log10(p_star):.2f})")

    if p_star < 1.0 / 1e6:
        print("[SOLVE] H-X1 CONFIRMED: p* is >= 6 orders of magnitude below the 1.0 Pa floor.")
    else:
        print("[SOLVE] H-X1 NOT confirmed at the >=6-orders bar (still check sign/magnitude).")

    # Wave structure geometry (sect.4.6). Self-similar variable within the LEFT fan (x<0.5):
    # xi(p) = u(p) - a_eq(p), monotone increasing from the head (undisturbed state, p=p0) to
    # the tail (star plateau edge, p=p*) -- using a_eq (NOT a_frozen) consistently throughout,
    # since a_eq is the model's true characteristic speed (ST-C established this; using
    # a_frozen for part of the geometry and a_eq for the rest would be self-inconsistent).
    domain = 1.0
    t_end = 9.5e-4
    u_L = -100.0
    a_eq_at_ic, _ = sound_speed_on_isentrope(p0, p0, T0, Y0, AIR, WATER)
    fan_head_speed = u_L - a_eq_at_ic
    x_fan_head = 0.5 + fan_head_speed * t_end
    print(f"[SOLVE] fan head speed (u_L - a_eq(p0)) = {fan_head_speed:.4f} m/s "
          f"(a_eq(p0)={a_eq_at_ic:.4f}), x_fan_head = {x_fan_head:.4f} "
          f"(15_ref.png dip onset ~0.36; plan's own 0.3558 used a_frozen instead -- both round "
          f"to ~0.36 given digitization uncertainty, not independently discriminating)")

    def u_at_p(p_target):
        return u_L + escape_velocity(p0, T0, Y0, p_target, AIR, WATER, n=8000)

    p_1pa = 1.0
    u_1pa = u_at_p(p_1pa)
    a_eq_1pa, _ = sound_speed_on_isentrope(p_1pa, p0, T0, Y0, AIR, WATER)
    xi_1pa = u_1pa - a_eq_1pa
    x_floor_half = abs(xi_1pa) * t_end
    print(f"[SOLVE] at p=1.0Pa: u={u_1pa:.4f} a_eq={a_eq_1pa:.4f} xi={xi_1pa:.4f}, "
          f"|x-0.5| = {x_floor_half:.5f} m = {x_floor_half/0.0025:.2f} cells/side at N=400 "
          f"(plan predicted 0.06639 m / 26.6 cells)")
    plateau_half = a_star * t_end
    print(f"[SOLVE] star plateau half-width (|xi|<a*): {plateau_half:.6f} m "
          f"= {plateau_half/0.0025:.3f} cells at N=400 (plan predicted 0.002287/0.91)")

    for N in (400, 800, 1600, 3200):
        dx = domain / N
        n_plateau_cells = 2 * plateau_half / dx
        exact_cj = 0.0 if n_plateau_cells >= 1.0 else (2.0 * a_star * (1.0 - n_plateau_cells))
        # crude cell-average model: if the central cell pair straddles the plateau only
        # partially, exact_cj underestimates true FV cell-average deviation -- reported as an
        # order-of-magnitude cross-check against round 30's measured cj, not a replacement.
        print(f"[SOLVE] N={N}: plateau spans {n_plateau_cells:.3f} cells, order-of-magnitude "
              f"exact cj ~ {exact_cj:.4f} (measured solver cj at this N: see round30 sect.40.7)")

    return 0


def a_wood_at_ic(p0, T0, alpha0):
    return mixture_sound_speed_wood(p0, T0, alpha0, AIR, WATER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--solve", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        ok, _ = selftests()
        return 0 if ok else 1
    if args.solve:
        return mode_solve()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
