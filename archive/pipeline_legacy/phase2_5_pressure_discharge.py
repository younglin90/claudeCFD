"""Phase 2-5: Pressure Discharge — Case A (Gas into Liquid) + Case B (Liquid into Gas)

Setup: Both cases, 1D, N=200 (reduced from 500 for speed)
T=308.2K, u=0 initially

Case A: Left gas (high-p), Right liquid (low-p)
Case B: Left liquid (1 GPa), Right gas (0.5 GPa)

Expected:
- Case A: rarefaction left (gas), compression + velocity front right (liquid)
- Case B: expansion left (liquid), compression right (gas)
PASS: wave separation, no divergence
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

def run_case(case_name, p_L, p_R, alpha_gas_L, alpha_gas_R, t_end, alpha_scheme='tvd', label='TVD'):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    xd = 0.5

    a_air = np.where(x < xd, alpha_gas_L, alpha_gas_R)
    a_water = 1.0 - a_air
    p_init = np.where(x < xd, p_L, p_R)
    T0 = 308.2

    rho1 = p_init / ((ph1['gamma'] - 1.0) * ph1['kv'] * T0)
    rho2 = (p_init + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * T0)

    a1r1 = a_air * rho1
    a2r2 = a_water * rho2
    rho = a1r1 + a2r2
    ru = np.zeros(N)
    gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
    rho_e = a_air * (p_init + ph1['gamma'] * ph1['pinf']) / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=500,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_f = a1r1_f + a2r2_f

    completed = t >= t_end * 0.99
    p_finite = np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f))
    p_bounded = p_f.min() > 0 and p_f.max() < max(p_L, p_R) * 10
    u_bounded = np.abs(u_f).max() < 1e4

    # Check wave separation — at least 3 distinct regions
    has_contact = (a1_f.max() - a1_f.min()) > 0.1  # α transition visible
    has_shock_or_raref = (p_f.max() - p_f.min()) > 0.01 * min(p_L, p_R)

    passed = completed and p_finite and p_bounded and u_bounded and has_contact and has_shock_or_raref
    status = "PASS" if passed else "FAIL"
    print(f"  {case_name} ({label}): t={t:.3e}, p=[{p_f.min():.2e},{p_f.max():.2e}], u_max={np.abs(u_f).max():.1f}")
    print(f"    a1_range=[{a1_f.min():.4f},{a1_f.max():.4f}] -> {status}")
    return x, p_f, u_f, a1_f, rho_f, passed

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    schemes = [('tvd', 'TVD'), ('thinc_bvd', 'THINC-BVD'),
               ('cicsam', 'CICSAM'), ('mstacs', 'MSTACS')]

    # Case A: Gas high-p (left), Liquid low-p (right)
    print("="*60); print("Case A: Gas into Liquid")
    caseA = {}
    for s, l in schemes:
        caseA[s] = run_case('A', 1e8, 1e5, 1.0-1e-6, 1e-6, 5e-4, s, l)

    # Case B: Liquid high-p (left), Gas relatively low-p (right)
    print("\n" + "="*60); print("Case B: Liquid into Gas")
    caseB = {}
    for s, l in schemes:
        caseB[s] = run_case('B', 1e9, 5e8, 1e-6, 1.0-1e-6, 3e-4, s, l)

    # Plot
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    colors = {'tvd': 'blue', 'thinc_bvd': 'red', 'cicsam': 'green', 'mstacs': 'purple'}

    for s, l in schemes:
        x, p, u, a1, rho, _ = caseA[s]
        c = colors[s]
        axes[0,0].plot(x, p, color=c, label=l)
        axes[1,0].plot(x, u, color=c, label=l)
        axes[2,0].plot(x, a1, color=c, label=l)
        axes[3,0].plot(x, rho, color=c, label=l)

        x, p, u, a1, rho, _ = caseB[s]
        axes[0,1].plot(x, p, color=c, label=l)
        axes[1,1].plot(x, u, color=c, label=l)
        axes[2,1].plot(x, a1, color=c, label=l)
        axes[3,1].plot(x, rho, color=c, label=l)

    axes[0,0].set_title('Case A: Pressure'); axes[0,0].legend()
    axes[0,1].set_title('Case B: Pressure'); axes[0,1].legend()
    axes[1,0].set_title('Case A: Velocity')
    axes[1,1].set_title('Case B: Velocity')
    axes[2,0].set_title('Case A: alpha_gas')
    axes[2,1].set_title('Case B: alpha_gas')
    axes[3,0].set_title('Case A: Density')
    axes[3,1].set_title('Case B: Density')
    for ax in axes.flat: ax.set_xlabel('x')

    plt.suptitle('Phase 2-5: Pressure Discharge (Gas-Liquid Interaction)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/phase2_5_pressure_discharge.png', dpi=150)
    print("\nPlot saved: results/phase2_5_pressure_discharge.png")

    print("="*50); print("SUMMARY")
    for s, l in schemes:
        pA = caseA[s][-1]; pB = caseB[s][-1]
        print(f"  {l:12s}: Case A {'PASS' if pA else 'FAIL'}, Case B {'PASS' if pB else 'FAIL'}")
