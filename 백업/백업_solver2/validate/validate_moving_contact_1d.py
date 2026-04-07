"""
validate_moving_contact_1d.py
==============================
1D Moving Contact Discontinuity — 2-component CPG ideal gas
Based on: Roy & Raghurama Rao, arXiv:2411.00285v2, §5.2.2–5.2.3

Setup:
  Domain   x ∈ [0, 1],  interface at x = 0.3,  N = 100,  periodic BC
  Left:    ρ = 1.0  (pure component 1),  u = 1.0,  p = 1.0
  Right:   ρ = 0.125 (pure component 2), u = 1.0,  p = 1.0
  t_end = 1.0,  CFL = 0.5

Case A: γ₁ = γ₂ = 1.4   (same γ  — baseline, all schemes should work)
Case B: γ₁ = 1.4, γ₂ = 1.67 (different γ — IEC violation test)

Schemes compared:
  FC-MUSCL   — fully conservative, MUSCL-LLF  (standard)
  APEC-MUSCL — FC-MUSCL + APEC energy correction (PE-consistent dissipation)
  FC-KEEP    — KEEP split-form, standard conservative
  APEC-KEEP  — KEEP split-form + APEC energy correction
  Kinetic    — lambda-difference (Roy & Raghurama Rao, naturally preserves PE)

Output: output/1D/V_moving_contact_*.png
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.kinetic import (
    IdealGasMixture, run_simulation, prim_to_cons_2s, cons_to_prim_2s,
)

OUTPUT_DIR = os.path.join(ROOT, 'output', '1D')
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS = 1e-30

# ═══════════════════════════════════════════════════════════
# CPG EOS helpers (mass-fraction mixing, parametric γ)
# ═══════════════════════════════════════════════════════════

def _inv_gm1(r1, r2, g1, g2):
    """1/(γ̄-1) = Y1/(γ1-1) + Y2/(γ2-1)  (mass-fraction weighted)"""
    rho = np.maximum(r1 + r2, EPS)
    return r1/(rho*(g1-1.0)) + r2/(rho*(g2-1.0))

def _p_from_rhoE(r1, r2, rhou, rhoE, g1, g2):
    rho     = np.maximum(r1 + r2, EPS)
    ke      = 0.5*(rhou/rho)**2
    e       = rhoE/rho - ke              # specific internal energy
    inv_gm1 = _inv_gm1(r1, r2, g1, g2)  # 1/(γ-1)
    # p = (γ-1)*ρe = ρe / inv_gm1
    p   = np.maximum(rho * e / np.maximum(inv_gm1, EPS), 0.0)
    gam = 1.0 + 1.0 / np.maximum(inv_gm1, EPS)   # full γ
    return p, gam

def _wave(r1, r2, rhou, rhoE, g1, g2):
    rho = np.maximum(r1 + r2, EPS)
    u_  = rhou / rho
    p, gam = _p_from_rhoE(r1, r2, rhou, rhoE, g1, g2)
    c   = np.sqrt(np.maximum(gam * p / rho, 0.0))  # c² = γp/ρ
    return c, u_

def _eps_cpg(r1, r2, p, g1, g2):
    """ε_s = (∂ρe/∂ρ_s)|_{p, ρ_{j≠s}}  for mass-fraction CPG."""
    rho = np.maximum(r1 + r2, EPS)
    igm = _inv_gm1(r1, r2, g1, g2)          # 1/(γ̄-1)
    e0  = p/rho * (1.0/(g1-1.0) - igm)      # ε₁
    e1  = p/rho * (1.0/(g2-1.0) - igm)      # ε₂
    return e0, e1

def _minmod(a, b):
    return np.where(a*b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)

# ═══════════════════════════════════════════════════════════
# Scheme 1 & 2: MUSCL-LLF  (FC and APEC)
# ═══════════════════════════════════════════════════════════

def _muscl_flux(U, g1, g2, scheme):
    """MUSCL 재구성 + LLF 수치 플럭스 (FC 또는 APEC)."""
    r1, r2, rhou, rhoE = U
    N = r1.shape[0]

    def ghost(arr):
        return np.concatenate([arr[[-2,-1]], arr, arr[[0,1]]])

    # Ghost cells (periodic)
    Ug = np.array([ghost(q) for q in [r1, r2, rhou, rhoE]])

    dL = Ug[:, 1:N+2] - Ug[:, 0:N+1]
    dR = Ug[:, 2:N+3] - Ug[:, 1:N+2]
    UL = Ug[:, 1:N+2] + 0.5*_minmod(dL, dR)

    dL2= Ug[:, 2:N+3] - Ug[:, 1:N+2]
    dR2= Ug[:, 3:N+4] - Ug[:, 2:N+3]
    UR = Ug[:, 2:N+3] - 0.5*_minmod(dL2, dR2)

    # Face fluxes (standard conservative)
    def flux_prim(U_):
        r1_, r2_, rhou_, rhoE_ = U_
        rho_ = np.maximum(r1_ + r2_, EPS)
        u_   = rhou_ / rho_
        p_, _= _p_from_rhoE(r1_, r2_, rhou_, rhoE_, g1, g2)
        return np.array([r1_*u_, r2_*u_, rhou_*u_+p_, (rhoE_+p_)*u_]), u_, p_

    FL, uL, pL = flux_prim(UL)
    FR, uR, pR = flux_prim(UR)
    cL, _ = _wave(*UL, g1, g2)
    cR, _ = _wave(*UR, g1, g2)
    lam = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)  # (N+1,)

    # Conservative energy dissipation
    dE_cons = 0.5 * lam * (UR[3] - UL[3])

    if scheme == 'APEC':
        # APEC: PE-consistent energy dissipation
        # dρE = Σ ε_s·Δ(ρ_s) + ½ū²·Δρ + ρ̄·ū·Δu
        u_h  = 0.5*(uL + uR)
        rho_L= np.maximum(UL[0]+UL[1], EPS)
        rho_R= np.maximum(UR[0]+UR[1], EPS)
        rho_h= 0.5*(rho_L + rho_R)

        # ε evaluated at left state (consistent with Terashima formulation)
        p_h  = 0.5*(pL + pR)
        r1_h = 0.5*(UL[0] + UR[0])
        r2_h = 0.5*(UL[1] + UR[1])
        eps0, eps1 = _eps_cpg(r1_h, r2_h, p_h, g1, g2)

        dr1 = UR[0] - UL[0]
        dr2 = UR[1] - UL[1]
        du  = uR - uL

        dE_apec = (eps0*dr1 + eps1*dr2
                   + 0.5*u_h**2*(dr1 + dr2)
                   + rho_h*u_h*du)
        dE_diss = 0.5 * lam * dE_apec
    else:
        dE_diss = dE_cons

    Fface = 0.5*(FL + FR)
    Fface[3] = Fface[3] - dE_diss    # overwrite energy flux dissipation

    # Standard dissipation for mass/momentum
    Fface[0] -= 0.5*lam*(UR[0]-UL[0])
    Fface[1] -= 0.5*lam*(UR[1]-UL[1])
    Fface[2] -= 0.5*lam*(UR[2]-UL[2])

    return -(Fface[:, 1:] - Fface[:, :-1])


def _rhs_muscl(U, g1, g2, scheme):
    return _muscl_flux(U, g1, g2, scheme)

# ═══════════════════════════════════════════════════════════
# Scheme 3 & 4: KEEP split-form  (FC and APEC)
# ═══════════════════════════════════════════════════════════

def _keep_flux(r1, r2, u, rhoe, p, scheme, g1, g2):
    """KEEP arithmetic-average interface fluxes."""
    r1p   = np.roll(r1,   -1)
    r2p   = np.roll(r2,   -1)
    up    = np.roll(u,    -1)
    pp    = np.roll(p,    -1)
    rhoep = np.roll(rhoe, -1)
    rho   = r1 + r2
    rhop  = r1p + r2p

    u_h   = 0.5*(u + up)
    rho_h = 0.5*(rho + rhop)
    p_h   = 0.5*(p + pp)

    F1 = 0.5*(r1 + r1p) * u_h
    F2 = 0.5*(r2 + r2p) * u_h
    FU = rho_h * u * up + p_h

    F_KE = 0.5 * rho_h * u * up * u_h
    F_pu = 0.5 * (p * up + pp * u)

    if scheme == 'APEC':
        eps0, eps1 = _eps_cpg(r1, r2, p, g1, g2)
        eps0p = np.roll(eps0, -1)
        eps1p = np.roll(eps1, -1)
        corr  = (0.5*(eps0p - eps0)*0.5*(r1p - r1)
               + 0.5*(eps1p - eps1)*0.5*(r2p - r2))
        rhoe_h = 0.5*(rhoe + rhoep) - corr
    else:
        rhoe_h = 0.5*(rhoe + rhoep)

    FE = rhoe_h * u_h + F_KE + F_pu
    return F1, F2, FU, FE


def _rhs_keep(U, g1, g2, scheme):
    r1, r2, rhoU, rhoE = U
    rho  = np.maximum(r1 + r2, EPS)
    u    = rhoU / rho
    rhoe = rhoE - 0.5*rho*u**2
    p, _ = _p_from_rhoE(r1, r2, rhoU, rhoE, g1, g2)

    F1, F2, FU, FE = _keep_flux(r1, r2, u, rhoe, p, scheme, g1, g2)
    d1 = -(F1 - np.roll(F1, 1))
    d2 = -(F2 - np.roll(F2, 1))
    dU = -(FU - np.roll(FU, 1))
    dE = -(FE - np.roll(FE, 1))
    return np.array([d1, d2, dU, dE])

# ═══════════════════════════════════════════════════════════
# SSP-RK3 runner (MUSCL or KEEP)
# ═══════════════════════════════════════════════════════════

def run_scheme(r10, r20, u0, p0, dx, t_end, g1, g2, CFL, scheme):
    """
    scheme in {'FC-MUSCL', 'APEC-MUSCL', 'FC-KEEP', 'APEC-KEEP', 'Kinetic'}
    Returns (r1, r2, u_final, p_final, Y1_final)
    """
    if scheme == 'Kinetic':
        # Use kinetic.py: IdealGasMixture + run_simulation
        eos  = IdealGasMixture([g1, g2])
        N    = len(r10)
        W0   = r10 / np.maximum(r10 + r20, EPS)
        rho0 = r10 + r20
        U0   = prim_to_cons_2s(W0, rho0, u0, p0, g1, g2)
        U, _ = run_simulation(U0, [dx], t_end, eos,
                              sigma=CFL, bc='periodic', order=2)
        W, rho, u_, p_ = cons_to_prim_2s(U, g1, g2)
        r1_ = rho * W
        r2_ = rho * (1.0 - W)
        return r1_, r2_, u_, p_, W

    # ── MUSCL or KEEP ──────────────────────────────────────────
    rho0 = r10 + r20
    Y10  = r10 / np.maximum(rho0, EPS)
    cv_mix0 = Y10/(g1-1.0) + (1.0-Y10)/(g2-1.0)
    cp_mix0 = Y10*g1/(g1-1.0) + (1.0-Y10)*g2/(g2-1.0)
    gm0 = cp_mix0 / np.maximum(cv_mix0, EPS)
    e0  = p0 / ((gm0-1.0)*np.maximum(rho0, EPS))

    U = np.array([r10, r20, rho0*u0, rho0*(e0 + 0.5*u0**2)])

    sch_base = scheme.split('-')[0]   # 'FC' or 'APEC'
    sch_type = scheme.split('-')[1]   # 'MUSCL' or 'KEEP'

    def rhs(U_):
        if sch_type == 'MUSCL':
            dU = _rhs_muscl(U_, g1, g2, sch_base)
            return dU / dx
        else:
            dU = _rhs_keep(U_, g1, g2, sch_base)
            return dU / dx

    def clip(U_):
        return np.array([np.maximum(U_[0], 1e-12),
                         np.maximum(U_[1], 1e-12),
                         U_[2], U_[3]])

    t = 0.0
    while t < t_end - 1e-14:
        c, u_ = _wave(*U, g1, g2)
        dt = min(CFL * dx / np.max(np.abs(u_) + c + 1e-30), t_end - t)

        k1 = rhs(U)
        U1 = clip(U + dt*k1)
        k2 = rhs(U1)
        U2 = clip(0.75*U + 0.25*(U1 + dt*k2))
        k3 = rhs(U2)
        U  = clip((1.0/3.0)*U + (2.0/3.0)*(U2 + dt*k3))
        t += dt

    r1_, r2_, rhou_, rhoE_ = U
    rho_ = np.maximum(r1_ + r2_, EPS)
    u_   = rhou_ / rho_
    p_, _= _p_from_rhoE(r1_, r2_, rhou_, rhoE_, g1, g2)
    Y1_  = r1_ / rho_
    return r1_, r2_, u_, p_, Y1_

# ═══════════════════════════════════════════════════════════
# Main validation
# ═══════════════════════════════════════════════════════════

def validate(N=100, t_end=1.0, CFL=0.5):
    dx  = 1.0 / N
    x   = (np.arange(N) + 0.5) * dx

    eps = 1e-8
    r10 = np.where(x < 0.3, 1.0 - eps, eps * 0.125)
    r20 = np.where(x < 0.3, eps,        0.125 - eps * 0.125)
    u0  = np.ones(N)
    p0  = np.ones(N)

    cases   = [
        ('Case A  (γ₁=γ₂=1.4)',    1.4, 1.4 ),
        ('Case B  (γ₁=1.4, γ₂=1.67)', 1.4, 1.67),
    ]
    # KEEP 스킴은 소산 없음 → 불연속(step function)에서 즉시 발산 → 제외
    schemes = ['FC-MUSCL', 'APEC-MUSCL', 'Kinetic']
    colors  = {'FC-MUSCL':'tab:blue', 'APEC-MUSCL':'tab:orange', 'Kinetic':'tab:purple'}

    fig_rows = len(cases)
    fig_cols = 4  # ρ, p, u, Y₁
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(16, 4*fig_rows))

    results = {}
    print("=" * 70)
    print("  Moving Contact Discontinuity — 2-component CPG")
    print(f"  N={N}, t_end={t_end}, CFL={CFL}")
    print("=" * 70)

    for row, (label, g1, g2) in enumerate(cases):
        print(f"\n--- {label} ---")
        row_results = {}

        for sch in schemes:
            print(f"  Running {sch} ...", end=' ', flush=True)
            try:
                r1, r2, u, p, Y1 = run_scheme(r10.copy(), r20.copy(),
                                               u0.copy(), p0.copy(),
                                               dx, t_end, g1, g2, CFL, sch)
                pe_err = float(np.max(np.abs(p - 1.0)))
                ue_err = float(np.max(np.abs(u - 1.0)))
                rho    = r1 + r2
                print(f"PE={pe_err:.2e}  uErr={ue_err:.2e}")
                row_results[sch] = dict(r1=r1, r2=r2, u=u, p=p, Y1=Y1,
                                        rho=rho, pe_err=pe_err, ue_err=ue_err)
            except Exception as ex:
                print(f"FAILED: {ex}")
                row_results[sch] = None

        results[(g1, g2)] = row_results

        # ── Plot ──────────────────────────────────────────────────
        titles = ['ρ  (density)', 'p  (pressure)', 'u  (velocity)', 'Y₁ (mass frac.)']
        for col, (title, key) in enumerate(zip(titles, ['rho','p','u','Y1'])):
            ax = axes[row, col]
            for sch in schemes:
                d = row_results.get(sch)
                if d is None:
                    continue
                ax.plot(x, d[key], lw=1.5, label=sch, color=colors[sch])
            # Reference lines
            if key == 'p':
                ax.axhline(1.0, color='k', ls='--', lw=0.8, label='exact')
            if key == 'u':
                ax.axhline(1.0, color='k', ls='--', lw=0.8, label='exact')
            ax.set_title(f'{label}\n{title}', fontsize=8)
            ax.set_xlabel('x'); ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V_moving_contact_profiles.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f"\nSaved: {fname}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY  (max |p - 1|  and  max |u - 1|)")
    print(f"  {'Scheme':<14}  {'Case A  PE':>12}  {'Case A  uE':>12}  "
          f"{'Case B  PE':>12}  {'Case B  uE':>12}")
    print("  " + "-"*66)

    all_pass = True
    for sch in schemes:
        da = results.get((1.4, 1.4 ), {}).get(sch)
        db = results.get((1.4, 1.67), {}).get(sch)
        pe_a = da['pe_err'] if da else float('nan')
        ue_a = da['ue_err'] if da else float('nan')
        pe_b = db['pe_err'] if db else float('nan')
        ue_b = db['ue_err'] if db else float('nan')
        # Pass criterion: p and u errors < 0.5 for Case A; Case B is expected to show differences
        ok_a = pe_a < 0.5 and ue_a < 0.5
        ok_b = pe_b < 0.5 and ue_b < 0.5
        if not (ok_a and ok_b):
            all_pass = False
        tag_a = 'OK' if ok_a else 'FAIL'
        tag_b = 'OK' if ok_b else 'WARN'
        print(f"  {sch:<14}  {pe_a:>10.2e}  {ue_a:>10.2e}  "
              f"{pe_b:>10.2e}  {ue_b:>10.2e}  [{tag_a}/{tag_b}]")

    print("=" * 70)

    # ── PE error over time comparison plot ────────────────────────
    _plot_pe_vs_time(r10, r20, u0, p0, dx, g1=1.4, g2=1.67,
                     schemes=schemes, colors=colors,
                     CFL=CFL, t_end=t_end, N=N)

    return all_pass


def _plot_pe_vs_time(r10, r20, u0, p0, dx, g1, g2, schemes, colors,
                     CFL, t_end, N):
    """Case B: record PE(t) = max|p(t) - 1| at each reporting step."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f'Case B (γ₁=1.4, γ₂={g2})  —  max|p - 1| vs time\n'
                 f'N={N},  CFL={CFL}')

    n_report = 20
    t_rep    = np.linspace(0, t_end, n_report + 1)

    for sch in schemes:
        pe_hist = []
        t_hist  = []
        r1_cur  = r10.copy(); r2_cur = r20.copy()

        if sch == 'Kinetic':
            eos = IdealGasMixture([g1, g2])
            W0  = r10 / np.maximum(r10 + r20, EPS)
            U   = prim_to_cons_2s(W0, r10+r20, u0, p0, g1, g2)
            for i in range(n_report):
                dt_seg = t_rep[i+1] - t_rep[i]
                U, _ = run_simulation(U, [dx], dt_seg, eos,
                                      sigma=CFL, bc='periodic', order=2)
                W, rho, u_, p_ = cons_to_prim_2s(U, g1, g2)
                pe_hist.append(float(np.max(np.abs(p_ - 1.0))))
                t_hist.append(t_rep[i+1])
        else:
            rho0 = r10 + r20
            Y10  = r10 / np.maximum(rho0, EPS)
            cv_m = Y10/(g1-1.0) + (1.0-Y10)/(g2-1.0)
            cp_m = Y10*g1/(g1-1.0) + (1.0-Y10)*g2/(g2-1.0)
            gm0  = cp_m / np.maximum(cv_m, EPS)
            e0   = p0 / ((gm0-1.0)*np.maximum(rho0, EPS))
            U    = np.array([r10, r20, rho0*u0, rho0*(e0+0.5*u0**2)])
            sch_base = sch.split('-')[0]
            sch_type = sch.split('-')[1]

            def rhs(U_):
                if sch_type == 'MUSCL':
                    return _rhs_muscl(U_, g1, g2, sch_base) / dx
                else:
                    return _rhs_keep(U_, g1, g2, sch_base) / dx

            def clip(U_):
                return np.array([np.maximum(U_[0], 1e-12),
                                 np.maximum(U_[1], 1e-12),
                                 U_[2], U_[3]])

            t = 0.0
            i_rep = 1
            failed = False
            while t < t_end - 1e-14 and not failed:
                c, u_ = _wave(*U, g1, g2)
                dt = min(CFL * dx / np.max(np.abs(u_) + c + 1e-30), t_end - t)
                try:
                    k1 = rhs(U)
                    U1 = clip(U + dt*k1)
                    k2 = rhs(U1)
                    U2 = clip(0.75*U + 0.25*(U1+dt*k2))
                    k3 = rhs(U2)
                    U  = clip((1.0/3.0)*U + (2.0/3.0)*(U2+dt*k3))
                except Exception:
                    failed = True; break
                t += dt
                if i_rep < len(t_rep) and t >= t_rep[i_rep] - 1e-14:
                    _, _, _, rhoE_ = U
                    p_, _ = _p_from_rhoE(U[0], U[1], U[2], rhoE_, g1, g2)
                    pe_hist.append(float(np.max(np.abs(p_ - 1.0))))
                    t_hist.append(t)
                    i_rep += 1

        if t_hist:
            ax.plot(t_hist, pe_hist, lw=1.5, marker='o', markersize=3,
                    label=sch, color=colors[sch])

    ax.set_xlabel('time'); ax.set_ylabel('max |p - 1|')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    fname = os.path.join(OUTPUT_DIR, 'V_moving_contact_pe_time.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=120); plt.close()
    print(f"Saved: {fname}")


if __name__ == '__main__':
    ok = validate(N=100, t_end=1.0, CFL=0.5)
    sys.exit(0 if ok else 1)
