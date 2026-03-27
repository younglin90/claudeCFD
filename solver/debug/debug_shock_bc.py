"""Trace crash: char=True vs char=False for shock-droplet."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from four_eq_1d import ic_shock_droplet, nasg_prim, rhs_iec, rkstep

sp0, sp1 = 'water', 'air'
N = 100
dx = 1.0 / N
x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)

def run_test(use_char, label):
    r0, r1, m_ic, rhoE = ic_shock_droplet(x, sp0, sp1)
    U = [r0.copy(), r1.copy(), m_ic.copy(), rhoE.copy()]

    def rhs_fn(U, dx):
        return rhs_iec(U, dx, sp0, sp1, iec=True, use_char=use_char,
                       weno_order=5, bc='transmissive')

    t, step = 0.0, 0
    t_end = 3e-4

    while t < t_end - 1e-14:
        r0_, r1_, m_, E_ = U
        rho_ = r0_ + r1_
        Y0_  = r0_ / np.maximum(rho_, 1e-30)
        u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
        lam  = float(np.max(np.abs(u_) + np.sqrt(c2_)))
        dt   = min(0.3 * dx / (lam + 1e-10), t_end - t)
        U, _ = rkstep(U, rhs_fn, dx, dt)
        t += dt; step += 1
        if not np.all(np.isfinite(U[3])):
            bad = np.where(~np.isfinite(U[3]))[0]
            print(f"  [{label}] NaN at step {step} t={t:.4e}  bad cells {bad[:5]}")
            return
    r0f, r1f, mf, Ef = U
    rhof = r0f + r1f
    Y0f  = r0f / np.maximum(rhof, 1e-30)
    uf, ef, Pf, Tf, c2f = nasg_prim(Y0f, rhof, mf, Ef, sp0, sp1)
    print(f"  [{label}] OK t={t:.4e} steps={step} Pmax={Pf.max():.3e} rhomax={rhof.max():.2f}")

print("N=100:")
run_test(True,  'char=True ')
run_test(False, 'char=False')
