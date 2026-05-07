"""
Validation 06-B: Mixture Sound Speed (Wood Formula)
Denner et al. 2018 JCP 367:192-234 §7.3.1 Fig. 11

Setup (paper): constant α (ψ) across entire domain → sinusoidal inlet
acoustic wave → measure wavelength λ → c_meas = λ·f → compare Wood.

Cases:
  (a) air-helium    f = 5000 Hz
  (b) air-water     f = 6000 Hz
  (c) water-copper  f = 10000 Hz

Common:
  Domain [0, 1] m, Δx = 2×10⁻³ m (N=500)
  u₀ = 1.0 m/s, p₀ = 1e5 Pa, T₀ = 300 K (uniform)
  Δu = 0.01·u₀ = 0.01 m/s
  u_in(t) = u₀ + Δu·sin(2π f t)
  bc_l='inlet', bc_r='transmissive'

Wood formula (Kapila 5-eq closure):
  1/(ρ_mix c_mix²) = α₁/(ρ₁ c₁²) + α₂/(ρ₂ c₂²)
  ρ_mix = α₁ ρ₁ + α₂ ρ₂

For each case, α ∈ {0.0, 0.25, 0.50, 0.75, 1.0} (5 data points).

Usage:
  python run_06B_validation.py
  python run_06B_validation.py --dissipation none
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, SGEOS


L = 1.0
N = 500
dx = L / N
x_grid = np.linspace(dx / 2, L - dx / 2, N)

p0 = 1.0e5
u0 = 1.0
T0 = 300.0
du = 0.01 * u0
eps_pure = 1e-8


# SG EOS properties (Denner 2018 Table 1)
MATERIALS = {
    'air':    dict(gamma=1.400, pinf=0.0,     rho0=1.157, a0=347.8),
    'helium': dict(gamma=1.667, pinf=0.0,     rho0=0.164, a0=1008.2),
    'water':  dict(gamma=4.100, pinf=4.4e8,   rho0=998.0, a0=1344.6),
    'copper': dict(gamma=4.220, pinf=3.24e10, rho0=8960.0, a0=3906.4),
}

CASES = {
    'a': dict(left='air',   right='helium', f=5000.0),
    'b': dict(left='air',   right='water',  f=6000.0),
    'c': dict(left='water', right='copper', f=10000.0),
}

ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]


def _make_eos(name):
    m = MATERIALS[name]
    if m['pinf'] > 0.0:
        return SGEOS(gamma=m['gamma'], pinf=m['pinf'], kv=_kv_uniform_T(m, T0))
    return IdealEOS(gamma=m['gamma'], kv=_kv_uniform_T(m, T0))


def _kv_uniform_T(m, T_target):
    """cv such that T(p0, rho0) = T_target (keeps T uniform)."""
    return (p0 + m['pinf']) / ((m['gamma'] - 1.0) * m['rho0'] * T_target)


def wood_c(alpha, left, right):
    m1, m2 = MATERIALS[left], MATERIALS[right]
    rho1, a1_s = m1['rho0'], m1['a0']
    rho2, a2_s = m2['rho0'], m2['a0']
    rho_mix = alpha * rho1 + (1.0 - alpha) * rho2
    inv_c2 = alpha / (rho1 * a1_s**2) + (1.0 - alpha) / (rho2 * a2_s**2)
    c_mix = np.sqrt(1.0 / (rho_mix * inv_c2))
    return c_mix, rho_mix


def _estimate_wavelength(x, signal, threshold=None):
    """Zero-crossing spacing ×2 → wavelength."""
    s = signal
    if threshold is not None:
        mask = np.abs(s) > threshold
        if mask.sum() < 10:
            return np.nan
        s = s[mask]; xs = x[mask]
    else:
        xs = x
    # zero crossings
    zc = np.where(np.sign(s[:-1]) != np.sign(s[1:]))[0]
    if len(zc) < 3:
        return np.nan
    return 2.0 * float(np.mean(np.diff(xs[zc])))


def run_case(case_key, alpha_val, dissipation='hybrid'):
    c = CASES[case_key]
    left, right = c['left'], c['right']
    m1, m2 = MATERIALS[left], MATERIALS[right]
    # Dynamic f: target wavelength ~20 cells (= 40·Δx = 0.04 m).
    # Default f from case dict as upper bound (pure phases).
    # For mixture cells with low c_wood, adjust f so λ stays resolved.
    c_wood_preview, _ = wood_c(alpha_val, left, right)
    lambda_target = 20.0 * dx   # 0.04 m at Δx=2e-3
    f_default = c['f']
    f_resolved = min(f_default, c_wood_preview / lambda_target)
    f = max(f_resolved, 100.0)  # lower floor to avoid unreasonably slow waves
    T_per = 1.0 / f

    ph1 = dict(gamma=m1['gamma'], pinf=m1['pinf'], kv=_kv_uniform_T(m1, T0))
    ph2 = dict(gamma=m2['gamma'], pinf=m2['pinf'], kv=_kv_uniform_T(m2, T0))
    eos1 = _make_eos(left); eos2 = _make_eos(right)

    # Wood target
    c_wood, rho_mix = wood_c(alpha_val, left, right)
    cfl = 0.44
    # t_end: 6 periods (enough to form 3+ wavelengths + BC transient decay)
    t_end = 6.0 * T_per
    # Estimated wave-occupied region at t_end:
    #   x ∈ [0, c_wood·t_end] (from inlet)
    wave_reach = c_wood_preview * t_end
    # Interior measurement band: avoid inlet BC (≥ 1λ) and wave front (≤ 0.9·reach)
    lam_expected = c_wood_preview / f
    band_lo = min(1.0 * lam_expected, 0.05)
    band_hi = min(0.9 * wave_reach, L - 0.05)
    if band_hi - band_lo < 2.0 * lam_expected:
        # Expand if too narrow
        band_lo = max(0.02, band_hi - 3.0 * lam_expected)

    # Uniform fill
    x = x_grid
    a1_init = np.full(N, np.clip(alpha_val, eps_pure, 1.0 - eps_pure))
    rho1 = np.full(N, m1['rho0'])
    rho2 = np.full(N, m2['rho0'])
    u_init = np.full(N, u0)
    p_init = np.full(N, p0)

    a1r1 = a1_init * rho1
    a2r2 = (1.0 - a1_init) * rho2
    rho_mix_arr = a1r1 + a2r2
    ru = rho_mix_arr * u_init
    e1 = eos1.energy(rho1, p_init)
    e2 = eos2.energy(rho2, p_init)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho_mix_arr * u_init ** 2

    def u_inlet_func(t_):
        return u0 + du * np.sin(2.0 * np.pi * f * t_)

    t_out, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2,
        a1r1, a2r2, ru, rE, a1_init,
        dx=dx, t_end=t_end, cfl=cfl,
        bc_l='inlet', bc_r='transmissive',
        u_inlet_func=u_inlet_func,
        max_steps=20000, print_interval=999999,
        alpha_scheme='tvd', use_mmacm_ex=False,
        primitive_recon='tvd',
        use_acid_face=True,
        acid_interface=True,
        dissipation=dissipation,
    )

    p_out, u_out, T_out, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    dp = p_out - p0

    # Measurement band (wave-occupied, BC-free)
    interior = (x > band_lo) & (x < band_hi)
    if interior.sum() < 10:
        interior = (x > 0.02) & (x < 0.5)
    x_int = x[interior]; dp_int = dp[interior]

    # Amplitude (peak-to-peak / 2)
    dp_amp = 0.5 * (float(np.max(dp_int)) - float(np.min(dp_int)))

    lam = _estimate_wavelength(x_int, dp_int,
                               threshold=0.05 * max(dp_amp, 1e-6))
    c_meas = lam * f if (not np.isnan(lam)) else np.nan
    err_pct = 100.0 * abs(c_meas - c_wood) / c_wood if not np.isnan(c_meas) else np.nan

    return {
        'case_key': case_key, 'left': left, 'right': right,
        'alpha': alpha_val, 'f': f, 'f_default': c['f'],
        'c_wood': c_wood, 'rho_mix': rho_mix,
        'c_meas': c_meas, 'lambda': lam,
        'err_pct': err_pct,
        'dp_amp': dp_amp,
        't_end': t_out,
        'x': x, 'p': p_out, 'u': u_out, 'T': T_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dissipation', choices=['hybrid', 'none', 'project'],
                        default='none')
    args = parser.parse_args()

    outdir = '/home/younglin90/work/claude_code/claudeCFD/results/06B'
    os.makedirs(outdir, exist_ok=True)

    results = {}
    for ck in ['a', 'b', 'c']:
        results[ck] = {}
        cfg = CASES[ck]
        print(f"\n=== 06-B Case ({ck}) {cfg['left']}-{cfg['right']}  f={cfg['f']:.0f} Hz ===")
        for a_val in ALPHAS:
            r = run_case(ck, a_val, dissipation=args.dissipation)
            results[ck][a_val] = r
            c_meas_s = f"{r['c_meas']:.2f}" if not np.isnan(r['c_meas']) else "NaN"
            err_s = f"{r['err_pct']:.2f}%" if not np.isnan(r['err_pct']) else "NaN"
            print(f"  α={a_val:.2f}: c_Wood={r['c_wood']:8.2f} m/s, "
                  f"c_meas={c_meas_s:>9} m/s,  err={err_s:>8},  "
                  f"dp_amp={r['dp_amp']:.3f} Pa")

    suffix = '' if args.dissipation == 'hybrid' else f'_{args.dissipation}'
    pkl_path = os.path.join(outdir, f'summary{suffix}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved summary: {pkl_path}")

    print("\n" + "=" * 78)
    print(f"06-B SUMMARY  [dissipation='{args.dissipation}']")
    print("=" * 78)
    for ck in ['a', 'b', 'c']:
        cfg = CASES[ck]
        print(f"\n({ck}) {cfg['left']}-{cfg['right']}  f={cfg['f']:.0f} Hz")
        print(f"  {'α':>5}  {'c_Wood':>10}  {'c_meas':>10}  {'err':>8}")
        for a_val in ALPHAS:
            r = results[ck][a_val]
            c_meas_s = f"{r['c_meas']:.2f}" if not np.isnan(r['c_meas']) else "NaN"
            err_s = f"{r['err_pct']:.2f}%" if not np.isnan(r['err_pct']) else "NaN"
            print(f"  {a_val:>5.2f}  {r['c_wood']:>10.2f}  {c_meas_s:>10}  {err_s:>8}")
    return results


if __name__ == '__main__':
    main()
