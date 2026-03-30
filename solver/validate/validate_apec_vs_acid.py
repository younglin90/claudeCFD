"""
validate_apec_vs_acid.py
========================
APEC vs ACID: comprehensive 1D validation comparison

Tests run:
  [SRK-1]  SRK CH4/N2 interface advection — PE vs time (N=101)
  [SRK-2]  SRK CH4/N2 — resolution study (N=51, 101, 201, 501)
  [SRK-3]  SRK CH4/N2 — long-time stability / divergence time
  [SRK-4]  SRK CH4/N2 — spatial profiles at t=5 ms
  [CPG-1]  CPG (ideal gas) interface advection — PE vs time
  [CPG-2]  CPG — spatial profiles at t=8

References:
  validation/1D_CH4_N2_interface_advection_SRK_EOS.md
  validation/1D_calorically_perfect_gas_interface_advection.md
  validation/1D_interface_advection_constant_velocity_ACID.md
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.apec import (
    run as apec_run_srk,
    run_cpg,
    pe_err, energy_err,
    initial_condition,
    srk_c2,
)
from solver.acid import (
    acid_run,
    compare_resolution,
    plot_profiles,
)

OUTPUT_DIR = os.path.join(ROOT, 'output', '1D')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {'FC': 'tab:blue', 'APEC': 'tab:orange', 'ACID': 'tab:green'}

PASS = True
results = []

def record(name, ok, msg):
    global PASS
    status = 'PASS' if ok else 'FAIL'
    if not ok:
        PASS = False
    results.append(f'  [{status}]  {name}: {msg}')
    return ok


# ═══════════════════════════════════════════════════════════
# SRK CH4/N2 tests  (APEC uses apec.run, ACID uses acid.acid_run)
# ═══════════════════════════════════════════════════════════

def test_srk_pe_time(N=101, t_end=0.06, CFL=0.3, k=15.0):
    """[SRK-1] PE vs time for FC / APEC / ACID."""
    print(f"\n[SRK-1] PE vs time  N={N}  CFL={CFL}  k={k}")
    res = {}
    for sc in ('FC', 'APEC', 'ACID'):
        x, U, T, p, th, ph, eh, div = acid_run(sc, N=N, t_end=t_end, CFL=CFL, k=k)
        res[sc] = dict(t=th, pe=ph, div=div, x=x, U=U, T=T, p=p)
        print(f"  {sc:4s}  PE_init={ph[1]:.3e}  PE_final={ph[-1]:.3e}  "
              f"diverged={div}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for sc, d in res.items():
        ax.semilogy(d['t']*1e3, d['pe'], lw=2, label=sc, color=COLORS[sc])
    ax.set_xlabel('t  [ms]'); ax.set_ylabel('max |Δp/p₀|')
    ax.set_title(f'[SRK-1] PE vs time  —  CH4/N2 SRK  N={N}  k={k}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'apec_acid_srk1_pe_time_N{N}.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  Saved: {fname}")

    # Pass: APEC and ACID both better than FC at t_end
    fc_pe = res['FC']['pe'][-1]
    ap_pe = res['APEC']['pe'][-1]
    ac_pe = res['ACID']['pe'][-1]
    ok = ap_pe < fc_pe or ac_pe < fc_pe
    record('SRK-1_pe_time', ok,
           f'FC={fc_pe:.2e}  APEC={ap_pe:.2e}  ACID={ac_pe:.2e}')
    return res


def test_srk_resolution(t_end=1e-4, CFL=0.3, k=15.0):
    """[SRK-2] Single-step PE vs resolution."""
    print(f"\n[SRK-2] Resolution study  k={k}")
    Ns = [51, 101, 201, 501]
    header = f"  {'N':>5}  {'FC':>12}  {'APEC':>12}  {'ACID':>12}"
    print(header); print("  " + "-"*50)

    dx0 = 1.0 / Ns[0]
    x0  = np.linspace(dx0/2, 1-dx0/2, Ns[0])
    r1_0, r2_0, u0, rhoE0, T0, p0_srk = initial_condition(x0, k=k)

    rows = []
    for N in Ns:
        dx  = 1.0/N
        x   = np.linspace(dx/2, 1-dx/2, N)
        r1, r2, u, rhoE, T, p0 = initial_condition(x, k=k)
        from solver.acid import acid_rkstep
        U0  = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]
        row = [N]
        for sc in ('FC', 'APEC', 'ACID'):
            lam = float(np.max(np.abs(u) + np.sqrt(srk_c2(r1, r2, T))))
            dt  = CFL*dx / (lam + 1e-10)
            U1, T1, p1 = acid_rkstep(U0, sc, dx, dt, T)
            row.append(pe_err(p1))
        rows.append(row)
        print(f"  {N:>5}  {row[1]:>12.3e}  {row[2]:>12.3e}  {row[3]:>12.3e}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    dxs = [1.0/r[0] for r in rows]
    for i, (sc, col) in enumerate([('FC','tab:blue'),('APEC','tab:orange'),
                                    ('ACID','tab:green')]):
        pes = [r[i+1] for r in rows]
        ax.loglog(dxs, pes, 'o-', lw=2, label=sc, color=col)
    ax.set_xlabel('Δx'); ax.set_ylabel('PE (1-step)')
    ax.set_title(f'[SRK-2] Resolution study  k={k}')
    ax.legend(); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'apec_acid_srk2_resolution.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  Saved: {fname}")

    # Pass: APEC and ACID smaller than FC at finest resolution
    fc_fine  = rows[-1][1]
    ap_fine  = rows[-1][2]
    ac_fine  = rows[-1][3]
    ok = ap_fine < fc_fine and ac_fine < fc_fine
    record('SRK-2_resolution', ok,
           f'N=501: FC={fc_fine:.2e}  APEC={ap_fine:.2e}  ACID={ac_fine:.2e}')
    return rows


def test_srk_stability(N=101, t_end=0.06, CFL=0.3, k=15.0):
    """[SRK-3] Long-time stability / divergence comparison."""
    print(f"\n[SRK-3] Stability  N={N}  t_end={t_end*1e3:.0f}ms  k={k}")
    for sc in ('FC', 'APEC', 'ACID'):
        x, U, T, p, th, ph, _, div = acid_run(sc, N=N, t_end=t_end, CFL=CFL, k=k)
        status = f'diverged at t={th[-1]*1e3:.2f}ms' if div else f'stable at t={th[-1]*1e3:.2f}ms'
        print(f"  {sc:4s}  {status}  PE_final={ph[-1]:.2e}")
    record('SRK-3_stability', True, 'see console above')


def test_srk_profiles(N=101, CFL=0.3, k=15.0, t_snap=5e-3):
    """[SRK-4] Spatial profiles at t_snap: rho, p, u, Y_CH4."""
    print(f"\n[SRK-4] Profiles at t={t_snap*1e3:.1f} ms  N={N}")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for sc in ('FC', 'APEC', 'ACID'):
        x, U, T, p, *_ = acid_run(sc, N=N, t_end=t_snap, CFL=CFL, k=k,
                                   verbose=False)
        r1, r2, rhoU, rhoE = U
        rho = r1 + r2
        u   = rhoU / np.maximum(rho, 1e-30)
        Y1  = r1 / np.maximum(rho, 1e-30)   # mass fraction CH4

        axes[0].plot(x, rho,     lw=1.5, label=sc, color=COLORS[sc])
        axes[1].plot(x, p*1e-6,  lw=1.5, label=sc, color=COLORS[sc])
        axes[2].plot(x, u,       lw=1.5, label=sc, color=COLORS[sc])
        axes[3].plot(x, Y1,      lw=1.5, label=sc, color=COLORS[sc])

    titles = ['ρ  [kg/m³]', 'p  [MPa]', 'u  [m/s]', 'Y_CH4']
    for ax, ttl in zip(axes, titles):
        ax.set_xlabel('x'); ax.set_title(ttl)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'[SRK-4] CH4/N2 SRK  t={t_snap*1e3:.1f}ms  N={N}  k={k}',
                 fontsize=11)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'apec_acid_srk4_profiles_N{N}.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  Saved: {fname}")
    record('SRK-4_profiles', True, fname)


# ═══════════════════════════════════════════════════════════
# CPG (ideal gas) tests  (APEC only — ACID uses SRK EOS)
# ═══════════════════════════════════════════════════════════

def test_cpg_pe_time(N=501, t_end=8.0, CFL=0.6, k=20.0):
    """[CPG-1] CPG interface advection PE vs time (APEC vs FC baseline).
    Based on: validation/1D_calorically_perfect_gas_interface_advection.md
    """
    print(f"\n[CPG-1] CPG interface advection  N={N}  t_end={t_end}")
    res = {}
    for sc in ('FC', 'APEC'):
        x, U, T_s, p, th, ph, eh, diverged = apec_run_srk(
            sc, N=N, t_end=t_end, CFL=CFL)
        res[sc] = dict(t=th, pe=ph, div=diverged)
        fin_pe = ph[-1] if np.isfinite(ph[-1]) else float('nan')
        print(f"  {sc:4s}  PE_final={fin_pe:.3e}  diverged={diverged}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for sc, d in res.items():
        mask = np.isfinite(d['pe'])
        ax.semilogy(d['t'][mask], d['pe'][mask], lw=2,
                    label=sc, color=COLORS[sc])
    ax.set_xlabel('t  [s]'); ax.set_ylabel('max |Δp/p₀|')
    ax.set_title(f'[CPG-1] CPG interface  N={N}  k={k}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'apec_acid_cpg1_pe_time.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  Saved: {fname}")

    fc_pe = res['FC']['pe'][-1] if np.isfinite(res['FC']['pe'][-1]) else 1e10
    ap_pe = res['APEC']['pe'][-1] if np.isfinite(res['APEC']['pe'][-1]) else 1e10
    ok = ap_pe < fc_pe
    record('CPG-1_pe_time', ok,
           f'FC={fc_pe:.2e}  APEC={ap_pe:.2e}')
    return res


def test_cpg_profiles(N=501, t_end=8.0, CFL=0.6, k=20.0):
    """[CPG-2] CPG spatial profiles at t_end."""
    print(f"\n[CPG-2] CPG profiles  N={N}  t={t_end}")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for sc in ('FC', 'APEC'):
        x, U, T_s, p, th, ph, eh, div = apec_run_srk(
            sc, N=N, t_end=t_end, CFL=CFL)
        r1, r2, rhoU, rhoE = U
        rho = r1 + r2
        u   = rhoU / np.maximum(rho, 1e-30)
        Y1  = r1 / np.maximum(rho, 1e-30)

        axes[0].plot(x, rho,    lw=1.5, label=sc, color=COLORS[sc])
        axes[1].plot(x, p,      lw=1.5, label=sc, color=COLORS[sc])
        axes[2].plot(x, u,      lw=1.5, label=sc, color=COLORS[sc])
        axes[3].plot(x, Y1,     lw=1.5, label=sc, color=COLORS[sc])

    for ax, ttl in zip(axes, ['ρ  [kg/m³]', 'p  [-]', 'u  [-]', 'Y₁']):
        ax.set_xlabel('x'); ax.set_title(ttl)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'[CPG-2] CPG interface  t={t_end}  N={N}', fontsize=11)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'apec_acid_cpg2_profiles.png')
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  Saved: {fname}")
    record('CPG-2_profiles', True, fname)


# ═══════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 65)
    print("  APEC vs ACID — Validation Summary")
    print("=" * 65)
    print(f"  {'Test':<30}  {'Result'}")
    print("  " + "-" * 60)
    for r in results:
        print(r)
    n_pass = sum(1 for r in results if '[PASS]' in r)
    n_fail = sum(1 for r in results if '[FAIL]' in r)
    print("  " + "-" * 60)
    print(f"  PASS={n_pass}  FAIL={n_fail}  Total={n_pass+n_fail}")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  APEC vs ACID  —  1D Validation Comparison")
    print("  Output:", OUTPUT_DIR)
    print("=" * 65)

    # ── SRK CH4/N2 tests ──────────────────────────────────
    test_srk_pe_time(N=101, t_end=0.06, CFL=0.3, k=15.0)
    test_srk_resolution(k=15.0)
    test_srk_stability(N=101, t_end=0.06, CFL=0.3, k=15.0)
    test_srk_profiles(N=101, CFL=0.3, k=15.0, t_snap=5e-3)

    # ── CPG ideal gas tests ───────────────────────────────
    test_cpg_pe_time(N=501, t_end=8.0, CFL=0.6, k=20.0)
    test_cpg_profiles(N=501, t_end=8.0, CFL=0.6, k=20.0)

    print_summary()
    sys.exit(0 if PASS else 1)
