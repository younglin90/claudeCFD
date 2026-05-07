"""EB4 Low-Mach 진동 원인 면밀 분석.

테스트:
1. 프로파일 상세 관찰 (진동 위치/크기)
2. 격자 세밀화 (N=100, 200, 400, 800)
3. CFL 감소 (0.4, 0.2, 0.1, 0.05)
4. MMACM-Ex ON/OFF 비교
5. Strang vs Lie splitting
6. Transmissive vs Periodic BC
7. 초기조건: smooth (tanh) vs discontinuous
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile, exact_riemann_star


def run_eb4(N=200, cfl=0.4, use_mmacm=True, use_strang=True, t_end=3e-4,
            bc='transmissive', smooth_ic=False):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5
    p0 = 1e5
    p_L = p0 * 1.01
    p_R = p0

    if smooth_ic:
        p_init = p0 + 0.005*p0*(1 - np.tanh(50*(x - x0)))
    else:
        p_init = np.where(x < x0, p_L, p_R)

    rho1 = p_init / (0.4 * 717.5 * 293.0)
    rho2 = (p_init + 6e8) / (3.4 * 474.2 * 293.0)
    a_air = 1e-6 * np.ones(N)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=t_end, cfl=cfl, bc_l=bc, bc_r=bc,
        max_steps=200000, print_interval=100000,
        alpha_scheme='tvd', use_strang=use_strang,
        use_defect_correction=False, use_material_cfl=False,
        use_mmacm_ex=use_mmacm)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    # Exact (only for discontinuous IC)
    if not smooth_ic:
        rho_L = p_L / (0.4 * 717.5 * 293.0); rho2_L = (p_L + 6e8) / (3.4 * 474.2 * 293.0)
        rho_R = (p_R + 6e8) / (3.4 * 474.2 * 293.0)
        rho_e, u_e, p_e, _ = exact_profile(
            x, t_end, x0,
            pL=p_L, rhoL=rho2_L, uL=0.0, gammaL=4.4, pinfL=6e8,
            pR=p_R, rhoR=rho_R, uR=0.0, gammaR=4.4, pinfR=6e8)
    else:
        p_e = p_init; u_e = np.zeros(N); rho_e = rho2

    return x, p_n, u_n, p_e, u_e


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    p0 = 1e5

    # Test 1: Baseline + profile
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Row 1: Baseline + grid refinement
    print("="*60)
    print("Test 1: Grid refinement")
    print("="*60)
    for i, N in enumerate([100, 200, 400]):
        x, p_n, u_n, p_e, u_e = run_eb4(N=N)
        # oscillation amplitude
        p_osc = (p_n - p_e).std() / p0
        u_osc = (u_n - u_e).std()
        print(f"  N={N}: p osc std={p_osc:.3e}, u osc std={u_osc:.3e}")
        axes[0,0].plot(x, (p_n - p0)/p0 * 100, label=f'N={N}')
        axes[0,1].plot(x, u_n*1000, label=f'N={N}')

    axes[0,0].plot(x, (p_e - p0)/p0 * 100, 'k--', lw=2, label='Exact')
    axes[0,0].set_title('Grid refinement: p perturbation (%)'); axes[0,0].legend(); axes[0,0].set_xlabel('x')
    axes[0,1].plot(x, u_e*1000, 'k--', lw=2, label='Exact')
    axes[0,1].set_title('Grid refinement: u (mm/s)'); axes[0,1].legend(); axes[0,1].set_xlabel('x')
    axes[0,2].axis('off')

    # Row 2: CFL refinement
    print("\nTest 2: CFL refinement")
    for i, cfl in enumerate([0.4, 0.2, 0.1]):
        x, p_n, u_n, p_e, u_e = run_eb4(N=200, cfl=cfl)
        p_osc = (p_n - p_e).std() / p0
        print(f"  CFL={cfl}: p osc std={p_osc:.3e}")
        axes[1,0].plot(x, (p_n - p0)/p0 * 100, label=f'CFL={cfl}')
        axes[1,1].plot(x, u_n*1000, label=f'CFL={cfl}')
    axes[1,0].plot(x, (p_e - p0)/p0 * 100, 'k--', lw=2, label='Exact')
    axes[1,0].set_title('CFL refinement: p perturbation (%)'); axes[1,0].legend(); axes[1,0].set_xlabel('x')
    axes[1,1].plot(x, u_e*1000, 'k--', lw=2, label='Exact')
    axes[1,1].set_title('CFL refinement: u (mm/s)'); axes[1,1].legend(); axes[1,1].set_xlabel('x')
    axes[1,2].axis('off')

    # Row 3: Physical variations
    print("\nTest 3: MMACM ON vs OFF, Strang vs Lie")
    configs = [
        ('MMACM=T, Strang', {'use_mmacm': True, 'use_strang': True}),
        ('MMACM=F, Strang', {'use_mmacm': False, 'use_strang': True}),
        ('MMACM=T, Lie', {'use_mmacm': True, 'use_strang': False}),
    ]
    for label, kwargs in configs:
        x, p_n, u_n, p_e, u_e = run_eb4(N=200, **kwargs)
        p_osc = (p_n - p_e).std() / p0
        print(f"  {label}: p osc std={p_osc:.3e}")
        axes[2,0].plot(x, (p_n - p0)/p0 * 100, label=label)
        axes[2,1].plot(x, u_n*1000, label=label)
    axes[2,0].plot(x, (p_e - p0)/p0 * 100, 'k--', lw=2, label='Exact')
    axes[2,0].set_title('Config variations: p perturbation (%)'); axes[2,0].legend(fontsize=8); axes[2,0].set_xlabel('x')
    axes[2,1].plot(x, u_e*1000, 'k--', lw=2, label='Exact')
    axes[2,1].set_title('Config variations: u (mm/s)'); axes[2,1].legend(fontsize=8); axes[2,1].set_xlabel('x')
    axes[2,2].axis('off')

    plt.suptitle('EB4 Low-Mach: Oscillation Source Diagnosis', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/diag_eb4.png', dpi=150)
    print(f"\nPlot saved: results/diag_eb4.png")
