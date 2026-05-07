"""Debug: 1-step DC method finite-value check."""
import sys; sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
import numpy as np
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, NASGEOS

ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0, 'b': 6.61e-4, 'eta': -1.177788e6}
eos1 = IdealEOS(gamma=1.4, kv=717.5)
eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0, b=6.61e-4, eta=-1.177788e6)
N, L = 10, 1.0; dx = L/N
x = np.linspace(dx/2, L-dx/2, N)
p0, u0 = 1.0e5, 1.0
a_water = ((x >= 0.4) & (x <= 0.6)).astype(float)
a1 = (1 - a_water) * (1 - 1e-6) + a_water * 1e-6
rho1 = eos1.density(np.full(N, p0), np.full(N, 300.0))
rho2 = eos2.density(np.full(N, p0), np.full(N, 300.0))
u = np.full(N, u0); p = np.full(N, p0)
a1r1 = a1*rho1; a2r2 = (1-a1)*rho2; rho_m = a1r1+a2r2; ru = rho_m*u
e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
rE = a1r1*e1 + a2r2*e2 + 0.5*rho_m*u**2

t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
    ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
    dx=dx, t_end=0.01, cfl=0.4, use_material_cfl=True,
    bc_l='periodic', bc_r='periodic',
    max_steps=1, print_interval=1,
    alpha_scheme='tvd', use_mmacm_ex=True, use_apec=True,
    primitive_recon='tvd', use_acid_face=True, acid_interface=True,
    acoustic_method='dumbser_casulli')
p_f, u_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

finite_p = bool(np.all(np.isfinite(p_f)))
finite_u = bool(np.all(np.isfinite(u_f)))
positive_p = bool(np.all(p_f > 0))
err_p = float(np.max(np.abs(p_f-p0))/p0)
err_u = float(np.max(np.abs(u_f-u0)))
print(f'DC 1-step: finite_p={finite_p}, finite_u={finite_u}, p>0={positive_p}')
print(f'err_p={err_p:.3e}, err_u={err_u:.3e}')
print(f'p range: [{float(p_f.min()):.3e}, {float(p_f.max()):.3e}]')
