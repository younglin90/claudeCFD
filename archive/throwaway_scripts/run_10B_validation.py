"""
Validation 10-B: Acoustic Reflection/Transmission at Fluid Interfaces (Round 5 재작성)

케이스 (Round 10 reference Fig. 11-13 매칭):
1. Air-Water   (좌 air,    우 water, x_intf=0.5, t_end=1.63 ms, 투과 peak x≈1.15 m)
2. Helium-Air  (좌 helium, 우 air,   x_intf=1.0, t_end=1.61 ms, 투과 peak x≈1.25 m)
3. Argon-Air   (좌 argon,  우 air,   x_intf=0.5, t_end=2.02 ms, 투과 peak x≈0.75 m)

초기조건 (명세서 준수):
  u(x, 0) = u_peak · exp(-(x - x_src)^2 / (2·σ^2))   for x < x_intf   (좌측 phase 만)
  u(x, 0) = 0                                         for x >= x_intf
  p(x, 0) = p0 + Z_L · u(x, 0)                        (right-moving acoustic impedance matching)

경계:
  좌측: reflective (u_ghost = -u[0], p_ghost = p[0])
  우측: transmissive

사용법:
  python run_10B_validation.py                       # dissipation='hybrid'
  python run_10B_validation.py --dissipation none
  python run_10B_validation.py --dissipation project
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, SGEOS

# ── 공통 설정 ─────────────────────────────────────────────
eos_params = {
    'air':    {'gamma': 1.4,   'pinf': 0.0,    'kv': 717.5},
    'helium': {'gamma': 1.667, 'pinf': 0.0,    'kv': 2077.0},
    'argon':  {'gamma': 1.66,  'pinf': 0.0,    'kv': 208.0},
    'water':  {'gamma': 4.1,   'pinf': 4.4e8,  'kv': 474.2},
}

L = 1.5
N = 400          # Round 11: 2× grid refinement (from N=200)
dx = L / N
x_grid = np.linspace(dx / 2, L - dx / 2, N)

x_src = 0.1
u_peak = 0.02
p0 = 1.0e5
eps_pure = 1e-6

# ── Case별 설정 ────────────────────────────────────────────
# Round 12: case-별 σ 개별 설정.
# 목표 투과파 full width ≈ 6·σ_R = 6·σ_L·(c_R/c_L):
#   Case 1 (target 0.25 m): σ_L = 0.25/(6·3.867)  ≈ 0.011 m
#   Case 2 (target 0.80 m): σ_L = 0.80/(6·0.345)  ≈ 0.386 m   (매우 큼 — Helium 느린 변환비)
#   Case 3 (target 0.20 m): σ_L = 0.20/(6·1.128)  ≈ 0.030 m
CASES = {
    1: dict(name='Air-Water', left='air', right='water',
            rho_L=1.157, rho_R=998.0, c_L=347.8, c_R=1344.6,
            x_intf=0.5, t_end=1.63e-3, sigma=0.014, x_src=0.1),    # incid 0.084 / trans 0.325 m
    2: dict(name='Helium-Air', left='helium', right='air',
            rho_L=0.164, rho_R=1.157, c_L=1008.2, c_R=347.8,
            x_intf=1.0, t_end=1.513e-3, sigma=0.049, x_src=0.2),   # incid/refl 0.292 / trans 0.101 m (x_src 0.2 for wall safety)
    3: dict(name='Argon-Air', left='argon', right='air',
            rho_L=1.748, rho_R=1.157, c_L=308.2, c_R=347.8,
            x_intf=0.5, t_end=2.02e-3, sigma=0.038, x_src=0.1),    # incid 0.228 / trans 0.257 m
}


def build_eos(name):
    p = eos_params[name]
    if p['pinf'] > 0.0:
        return SGEOS(gamma=p['gamma'], pinf=p['pinf'], kv=p['kv'])
    return IdealEOS(gamma=p['gamma'], kv=p['kv'])


def run_case(case_id, dissipation='hybrid'):
    c = CASES[case_id]
    x_intf = c['x_intf']
    t_end = c['t_end']
    ph1 = eos_params[c['left']]
    ph2 = eos_params[c['right']]
    eos1 = build_eos(c['left'])
    eos2 = build_eos(c['right'])

    # Impedance (Z_L = ρ_L · c_L)
    Z_L = c['rho_L'] * c['c_L']
    Z_R = c['rho_R'] * c['c_R']

    # 초기조건 — Gaussian u pulse (좌측 phase 만) + impedance-matched p
    sigma_L = c['sigma']             # case-specific initial pulse width
    x_src_c = c.get('x_src', x_src)  # case-specific source location
    x = x_grid
    mask_L = x < x_intf
    u_pulse = u_peak * np.exp(-((x - x_src_c) ** 2) / (2 * sigma_L ** 2))
    u_init = np.where(mask_L, u_pulse, 0.0)
    p_init = p0 + Z_L * u_init

    # α volume fraction (interface 셀에서 eps_pure)
    a1_init = np.where(mask_L, 1.0 - eps_pure, eps_pure)
    # phase 1 = left material (whole domain), phase 2 = right material (whole domain).
    # Kapila 5-eq: a1·ρ1 + a2·ρ2 = ρ_mix. Here a1→0 in right domain → ρ_mix = ρ_R.
    rho1 = np.full(N, c['rho_L'])
    rho2 = np.full(N, c['rho_R'])

    # 보존변수
    a1r1 = a1_init * rho1
    a2r2 = (1.0 - a1_init) * rho2
    rho_mix = a1r1 + a2r2
    ru = rho_mix * u_init
    e1 = eos1.energy(rho1, p_init)
    e2 = eos2.energy(rho2, p_init)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho_mix * u_init ** 2

    print(f"\n=== Case {case_id}: {c['name']} ===")
    print(f"  L={L} m, N={N}, dx={dx:.4e} m")
    print(f"  Interface x_intf={x_intf} m, t_end={t_end*1e3:.2f} ms")
    print(f"  Z_L={Z_L:.2f}, Z_R={Z_R:.2e},  dissipation='{dissipation}'")

    t_out, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2,
        a1r1, a2r2, ru, rE, a1_init,
        dx=dx, t_end=t_end, cfl=0.1,
        bc_l='reflective', bc_r='transmissive',
        max_steps=20000, print_interval=999999,
        alpha_scheme='tvd', use_mmacm_ex=True,
        primitive_recon='tvd', dissipation=dissipation,
    )

    p_out, u_out, T_out, rho1_out, rho2_out, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2
    )

    # 반사/투과 분리 측정
    delta_p = p_out - p0
    mask_left = x < x_intf
    mask_right = x >= x_intf

    if mask_left.any():
        dpL = np.where(mask_left, np.abs(delta_p), -np.inf)
        i_refl = int(np.argmax(dpL))
        dp_refl = float(delta_p[i_refl])
        x_refl = float(x[i_refl])
    else:
        dp_refl, x_refl = 0.0, -1.0

    # Exclude outlet tail band (x > L - 5·dx) to avoid BC-induced trailing-edge
    # amplification when transmitted peak would lie outside the domain.
    outlet_band = L - 5.0 * dx
    mask_right_interior = mask_right & (x <= outlet_band)
    if mask_right_interior.any():
        dpR = np.where(mask_right_interior, np.abs(delta_p), -np.inf)
        i_trans = int(np.argmax(dpR))
        dp_trans = float(delta_p[i_trans])
        x_trans = float(x[i_trans])
    elif mask_right.any():
        dpR = np.where(mask_right, np.abs(delta_p), -np.inf)
        i_trans = int(np.argmax(dpR))
        dp_trans = float(delta_p[i_trans])
        x_trans = float(x[i_trans])
    else:
        dp_trans, x_trans = 0.0, -1.0

    u_max = float(np.abs(u_out).max())
    print(f"  Result t={t_out*1e3:.3f} ms:")
    print(f"    Reflected  |Δp|={abs(dp_refl):.3f} Pa  (sign={'+' if dp_refl>=0 else '-'})  @ x={x_refl:.4f} m")
    print(f"    Transmitted|Δp|={abs(dp_trans):.3f} Pa  (sign={'+' if dp_trans>=0 else '-'})  @ x={x_trans:.4f} m")
    print(f"    u_max={u_max:.5f} m/s")

    return {
        'case_id': case_id,
        'name': c['name'],
        'x': x,
        't': t_out,
        'p': p_out,
        'u': u_out,
        'T': T_out,
        'a1': a1_f,
        'rho1': rho1_out,
        'rho2': rho2_out,
        'rho_mix': a1r1_f + a2r2_f,
        'x_intf': x_intf,
        'sigma': sigma_L,
        'x_src': x_src_c,
        'Z_L': Z_L,
        'Z_R': Z_R,
        'dp_incid_theory': Z_L * u_peak,
        'R_coef': (Z_R - Z_L) / (Z_R + Z_L),
        'T_coef': 2.0 * Z_R / (Z_R + Z_L),
        'dp_refl_meas': dp_refl,
        'x_refl_meas': x_refl,
        'dp_trans_meas': dp_trans,
        'x_trans_meas': x_trans,
        'u_max': u_max,
        'dissipation': dissipation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dissipation', choices=['hybrid', 'none', 'project'],
                        default='hybrid')
    args = parser.parse_args()

    outdir = '/home/younglin90/work/claude_code/claudeCFD/results/10B'
    os.makedirs(outdir, exist_ok=True)

    results = {}
    for case_id in [1, 2, 3]:
        r = run_case(case_id, dissipation=args.dissipation)
        results[case_id] = r
        suffix = '' if args.dissipation == 'hybrid' else f'_{args.dissipation}'
        pkl_path = os.path.join(outdir, f'case{case_id}{suffix}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(r, f)
        print(f"  Saved: {pkl_path}")

    print("\n" + "=" * 70)
    print(f"MEASUREMENT SUMMARY  [dissipation='{args.dissipation}']")
    print("=" * 70)
    for cid, r in results.items():
        dpi = r['dp_incid_theory']
        T_p = r['T_coef']
        R_p = r['R_coef']
        refl_theory = abs(R_p) * dpi
        trans_theory = abs(T_p) * dpi
        print(f"\n10-B-{cid}: {r['name']}   (Z_L={r['Z_L']:.1f}, R={R_p:+.4f}, T={T_p:+.4f})")
        print(f"  dp_incid_theory = {dpi:.3f} Pa")
        print(f"  Reflected  meas = {abs(r['dp_refl_meas']):7.3f} Pa @ x={r['x_refl_meas']:.4f}   "
              f"(sign={'+' if r['dp_refl_meas']>=0 else '-'})  "
              f"theory={refl_theory:.3f} Pa")
        print(f"  Transmitted meas= {abs(r['dp_trans_meas']):7.3f} Pa @ x={r['x_trans_meas']:.4f}   "
              f"(sign={'+' if r['dp_trans_meas']>=0 else '-'})  "
              f"theory={trans_theory:.3f} Pa")
    return results


if __name__ == '__main__':
    main()
