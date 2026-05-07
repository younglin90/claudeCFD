"""
Validation 09-B: Acoustic Impedance Matching — Non-Reflecting Gas-Gas Interface
(Denner et al. 2018 JCP 367:192-234 §7.3.3 Fig. 15)

케이스:
1. Case A: Z = 423.588 Pa·s/m (continuous sinusoidal, f=2000 Hz)
2. Case B: Z = 500    Pa·s/m (single sinusoidal wave, f=5000 Hz)

공통:
  u₀ = 0.30886 m/s (uniform)
  p₀ = 1.0×10⁵ Pa  (uniform)
  Domain [0, 1] m, 계면 x=0.5 m, N=500 (Δx=2e-3)
  BC: bc_l='inlet', bc_r='transmissive'

Case 별:
  Case A: Left ρ=1.2650 γ=1.40 a=334.8522, Right ρ=1.7537 γ=1.01 a=241.5396
          u_in = u₀ + Δu·sin(2πft),  Δu=0.01·u₀,  t_end=3.3e-3 s
  Case B: Left ρ=0.25   γ=9.872 a=2000, Right ρ=1.00 γ=2.468 a=500
          u_in = u₀ + Δu·sin(2πft) for t<1/f else u₀, Δu=0.02·u₀, t_end=0.9e-3 s

이론 해:
  Z_L = Z_R → Δp_L^refl = 0, Δp_R^trans = Δp_L^incid = ρ_L·a_L·Δu
  λ_R/λ_L = a_R/a_L
  Case A λ_ratio = 0.7213, Δp_incid = 1.3082 Pa
  Case B λ_ratio = 0.2500, Δp_incid = 3.0886 Pa

사용법:
  python run_09B_validation.py                       # default dissipation='hybrid'
  python run_09B_validation.py --dissipation none
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS


L = 1.0
N = 500
dx = L / N
x_grid = np.linspace(dx / 2, L - dx / 2, N)

p0 = 1.0e5
u0 = 0.30886
eps_pure = 1e-6


# ── Case별 파라미터 (Denner 2018 JCP 367 §7.3.3 직접 인용) ────────
CASES = {
    'A': dict(
        name='Case A  Z=423.588 sinusoidal',
        rho_L=1.2650, rho_R=1.7537,
        gamma_L=1.40,  gamma_R=1.01,
        a_L=334.8522,  a_R=241.5396,
        f=2000.0, du_rel=0.01,
        t_end=3.3e-3,
        inlet_mode='continuous',
        x_intf=0.5,
    ),
    'B': dict(
        name='Case B  Z=500   single pulse',
        rho_L=0.25,   rho_R=1.00,
        gamma_L=9.872, gamma_R=2.468,
        a_L=2000.0,    a_R=500.0,
        f=5000.0, du_rel=0.02,
        t_end=0.9e-3,
        inlet_mode='single',
        x_intf=0.5,
    ),
}


def _kv_for_uniform_T(gamma, rho, p=p0, T_target=300.0):
    """Compute cv so that T = p/((γ-1)·ρ·cv) equals T_target.
    Needed because γ=1.01 (Case A right) and γ=9.872 (Case B left) produce
    extreme T if kv is fixed to 717.5 — creates huge T discontinuity at the
    interface that our Kapila p-closure cannot handle.  Enforcing T_L = T_R
    = 300 K (论문은 uniform initial state 가정) keeps interface cells sane."""
    return p / ((gamma - 1.0) * rho * T_target)


def run_case(case_id, dissipation='hybrid'):
    c = CASES[case_id]
    x_intf = c['x_intf']
    t_end = c['t_end']
    f = c['f']
    du = c['du_rel'] * u0
    T_per = 1.0 / f

    kv_L = _kv_for_uniform_T(c['gamma_L'], c['rho_L'], p0, 300.0)
    kv_R = _kv_for_uniform_T(c['gamma_R'], c['rho_R'], p0, 300.0)
    ph1 = {'gamma': c['gamma_L'], 'pinf': 0.0, 'kv': kv_L}
    ph2 = {'gamma': c['gamma_R'], 'pinf': 0.0, 'kv': kv_R}
    eos1 = IdealEOS(gamma=c['gamma_L'], kv=kv_L)
    eos2 = IdealEOS(gamma=c['gamma_R'], kv=kv_R)

    Z_L = c['rho_L'] * c['a_L']
    Z_R = c['rho_R'] * c['a_R']
    dp_incid_theory = Z_L * du

    # 초기조건 — 균일 u0, p0
    x = x_grid
    mask_L = x < x_intf
    a1_init = np.where(mask_L, 1.0 - eps_pure, eps_pure)
    rho1 = np.full(N, c['rho_L'])
    rho2 = np.full(N, c['rho_R'])

    u_init = np.full(N, u0)
    p_init = np.full(N, p0)

    a1r1 = a1_init * rho1
    a2r2 = (1.0 - a1_init) * rho2
    rho_mix = a1r1 + a2r2
    ru = rho_mix * u_init
    e1 = eos1.energy(rho1, p_init)
    e2 = eos2.energy(rho2, p_init)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho_mix * u_init ** 2

    # Inlet velocity function
    if c['inlet_mode'] == 'continuous':
        def u_inlet_func(t_):
            return u0 + du * np.sin(2.0 * np.pi * f * t_)
    else:   # 'single' — single wavelength pulse, then u0
        def u_inlet_func(t_):
            if t_ < T_per:
                return u0 + du * np.sin(2.0 * np.pi * f * t_)
            return u0

    print(f"\n=== 09-B {c['name']} ===")
    print(f"  L={L} m, N={N}, dx={dx:.4e} m")
    print(f"  Interface x_intf={x_intf} m, t_end={t_end*1e3:.3f} ms")
    print(f"  Z_L={Z_L:.3f}, Z_R={Z_R:.3f}  (Δ={abs(Z_R-Z_L):.4e})")
    print(f"  δp_incid_theory = {dp_incid_theory:.4f} Pa,  λ_R/λ_L = {c['a_R']/c['a_L']:.4f}")
    print(f"  f={f:.0f} Hz, Δu={du:.6f} m/s,  dissipation='{dissipation}'")

    t_out, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2,
        a1r1, a2r2, ru, rE, a1_init,
        dx=dx, t_end=t_end, cfl=0.4,
        bc_l='inlet', bc_r='transmissive',
        u_inlet_func=u_inlet_func,
        max_steps=50000, print_interval=999999,
        alpha_scheme='tvd', use_mmacm_ex=False,
        primitive_recon='tvd',
        use_acid_face=True,        # Denner 2018 ACID face density EOS
        acid_interface=True,       # ACID-style IM1 block-tridiag: cell-local single-phase impedance
        use_nscbc=True,            # NSCBC inlet BC (acoustic δp=Z·δu consistency)
        dissipation=dissipation,
    )

    p_out, u_out, T_out, rho1_out, rho2_out, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2
    )

    # ── 측정 ────────────────────────────────────────────────
    # Peak-to-peak amplitude on each side of interface
    delta_p = p_out - p0
    # Exclude near-interface transition (avoid ψ-mixed cells)
    # and near-boundary zones
    left_band  = (x > 0.05) & (x < x_intf - 2 * dx)
    right_band = (x > x_intf + 2 * dx) & (x < L - 0.05)

    if left_band.any():
        dp_L_max = float(np.max(delta_p[left_band]))
        dp_L_min = float(np.min(delta_p[left_band]))
        dp_L_pp  = dp_L_max - dp_L_min
    else:
        dp_L_max = dp_L_min = dp_L_pp = 0.0

    if right_band.any():
        dp_R_max = float(np.max(delta_p[right_band]))
        dp_R_min = float(np.min(delta_p[right_band]))
        dp_R_pp  = dp_R_max - dp_R_min
    else:
        dp_R_max = dp_R_min = dp_R_pp = 0.0

    # Incident peak amplitude (half of peak-to-peak) — left side contains
    # incident+reflected; if reflected = 0, peak amplitude = peak-to-peak / 2
    # matches Δp_incid.
    amp_L = 0.5 * dp_L_pp
    amp_R = 0.5 * dp_R_pp

    ratio_trans = amp_R / dp_incid_theory if dp_incid_theory > 0 else 0.0

    # Estimate reflection as deviation of left-side peak from incident theory
    # (if reflection adds constructive/destructive, it leaves a residual)
    # For Case A continuous: if no reflection, amp_L == δp_incid.  If
    # reflection present, interference creates standing-wave pattern.
    # Simple estimate: reflection_amp ≈ |amp_L − δp_incid_theory|
    refl_amp_est = abs(amp_L - dp_incid_theory)
    refl_ratio_est = refl_amp_est / amp_R if amp_R > 0 else 0.0

    # Wavelength estimate: zero-crossings spacing on each side
    def _estimate_wavelength(mask, x_arr, signal):
        if not mask.any():
            return np.nan
        s = signal[mask]
        xs = x_arr[mask]
        # find zero-crossings
        zc = np.where(np.sign(s[:-1]) != np.sign(s[1:]))[0]
        if len(zc) < 3:
            return np.nan
        # wavelength ≈ 2 · mean spacing between zero-crossings
        return 2.0 * float(np.mean(np.diff(xs[zc])))

    lam_L = _estimate_wavelength(left_band,  x, delta_p)
    lam_R = _estimate_wavelength(right_band, x, delta_p)
    lam_ratio_num = lam_R / lam_L if (lam_L and lam_L > 0 and not np.isnan(lam_L)) else np.nan
    lam_ratio_theory = c['a_R'] / c['a_L']

    u_max = float(np.max(np.abs(u_out)))

    print(f"  Result t={t_out*1e3:.3f} ms:")
    print(f"    amp_L(peak) = {amp_L:.4f} Pa,  amp_R(peak) = {amp_R:.4f} Pa")
    print(f"    dp_incid_theory = {dp_incid_theory:.4f} Pa")
    print(f"    Transmitted ratio = {ratio_trans:.4f} (theory 1.0)")
    print(f"    Reflection |dev|  = {refl_amp_est:.4f} Pa  (~{refl_ratio_est*100:.2f}% of trans)")
    lam_ratio_str = f"{lam_ratio_num:.4f}" if (not np.isnan(lam_ratio_num)) else "NaN"
    print(f"    λ_R/λ_L  num = {lam_ratio_str}  theory = {lam_ratio_theory:.4f}")
    print(f"    u_max - u0 = {u_max-u0:+.6f} m/s")

    return {
        'case_id': case_id,
        'name': c['name'],
        'x': x,
        't': t_out,
        'p': p_out,
        'u': u_out,
        'T': T_out,
        'a1': a1_f,
        'x_intf': x_intf,
        'Z_L': Z_L, 'Z_R': Z_R,
        'dp_incid_theory': dp_incid_theory,
        'amp_L': amp_L, 'amp_R': amp_R,
        'ratio_trans': ratio_trans,
        'refl_amp_est': refl_amp_est,
        'refl_ratio_est': refl_ratio_est,
        'lam_L': lam_L, 'lam_R': lam_R,
        'lam_ratio_num': lam_ratio_num,
        'lam_ratio_theory': lam_ratio_theory,
        'u_max': u_max,
        'dissipation': dissipation,
        'f': c['f'], 'du': du, 'inlet_mode': c['inlet_mode'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dissipation', choices=['hybrid', 'none', 'project'],
                        default='hybrid')
    parser.add_argument('--case', choices=['A', 'B', 'both'], default='both')
    args = parser.parse_args()

    outdir = '/home/younglin90/work/claude_code/claudeCFD/results/09B'
    os.makedirs(outdir, exist_ok=True)

    case_ids = ['A', 'B'] if args.case == 'both' else [args.case]
    results = {}
    for cid in case_ids:
        r = run_case(cid, dissipation=args.dissipation)
        results[cid] = r
        suffix = '' if args.dissipation == 'hybrid' else f'_{args.dissipation}'
        pkl_path = os.path.join(outdir, f'case{cid}{suffix}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(r, f)
        print(f"  Saved: {pkl_path}")

    print("\n" + "=" * 70)
    print(f"MEASUREMENT SUMMARY  [dissipation='{args.dissipation}']")
    print("=" * 70)
    for cid, r in results.items():
        print(f"\n09-B {r['name']}")
        print(f"  Trans ratio  = {r['ratio_trans']:.4f}  (theory 1.0)")
        print(f"  Refl   ratio = {r['refl_ratio_est']*100:.3f}%  (theory 0%)")
        print(f"  λ_R/λ_L num  = {r['lam_ratio_num']:.4f}  theory = {r['lam_ratio_theory']:.4f}")
    return results


if __name__ == '__main__':
    main()
