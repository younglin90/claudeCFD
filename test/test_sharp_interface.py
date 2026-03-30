"""Test FC vs APEC at different interface sharpnesses (k parameter).

Hypothesis: APEC (PE-consistent dissipation) is stable and superior to FC
only when the interface is sharp enough that the centered correction is
significant. With smooth interfaces (small k), the correction is negligible
and PE-consistent dissipation is less stable than FC.
"""
import sys
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from pressure_eq import (initial_condition, rkstep, srk_c2, pe_err, epsilon_v)

N    = 201
dx   = 1.0 / N
x    = np.linspace(dx/2, 1 - dx/2, N)
CFL  = 0.3
N_STEPS = 500
DIV_THRESH = 5.0   # 500% PE error = diverged

print(f"N={N}  N_STEPS={N_STEPS}  CFL={CFL}  div_thresh={DIV_THRESH*100:.0f}%")
print()

for k in [15, 30, 50, 100, 200]:
    r1, r2, u, rhoE, T, p = initial_condition(x, 5e6, k=k)
    rho  = r1 + r2
    c2   = srk_c2(r1, r2, T)
    lam  = float(np.max(np.abs(u) + np.sqrt(c2)))
    dt   = CFL * dx / lam

    # Check correction magnitude at step 0
    eps0 = epsilon_v(r1, r2, T, 0)
    eps1 = epsilon_v(r1, r2, T, 1)
    dr1 = np.roll(r1, -1) - r1
    dr2 = np.roll(r2, -1) - r2
    corr = 0.5*(np.roll(eps0,-1) - eps0)*0.5*dr1 + 0.5*(np.roll(eps1,-1) - eps1)*0.5*dr2
    rhoe = rhoE - 0.5*rho*u**2
    rhoe_h = 0.5*(rhoe + np.roll(rhoe, -1))
    corr_frac = float(np.max(np.abs(corr)) / (np.max(np.abs(rhoe_h)) + 1e-30))

    results = {}
    for scheme in ['FC', 'APEC']:
        U = [r1.copy(), r2.copy(), rho*u, rhoE.copy()]
        T_cur = T.copy()
        pe_list = []
        div_step = None
        for step in range(1, N_STEPS+1):
            try:
                U, T_cur, p_cur = rkstep(U, scheme, dx, dt, T_cur)
                pe = pe_err(p_cur, 5e6)
                pe_list.append(pe)
                if not np.isfinite(pe) or pe > DIV_THRESH:
                    div_step = step
                    break
            except Exception as e:
                div_step = step
                pe_list.append(float('nan'))
                break
        results[scheme] = (pe_list, div_step)

    fc_pe, fc_div = results['FC']
    ap_pe, ap_div = results['APEC']

    fc_step1 = fc_pe[0] if fc_pe else float('nan')
    ap_step1 = ap_pe[0] if ap_pe else float('nan')
    ratio = fc_step1 / ap_step1 if ap_step1 > 0 else float('nan')

    fc_final = f"div@{fc_div}" if fc_div else f"ok({fc_pe[-1]:.3e})"
    ap_final = f"div@{ap_div}" if ap_div else f"ok({ap_pe[-1]:.3e})"

    print(f"k={k:4d}: corr/rhoe={corr_frac:.2e}  "
          f"step1 FC={fc_step1:.3e}  APEC={ap_step1:.3e}  ratio={ratio:.3f}  "
          f"FC={fc_final}  APEC={ap_final}")
