"""Check solution structure at early and late times."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from four_eq_1d import ic_riemann_gl, nasg_prim, rhs_iec, rkstep

N = 200
dx = 1.0 / N
x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)
sp0, sp1 = 'water_nd', 'air_nd'

r0, r1, m_ic, rhoE = ic_riemann_gl(x, sp0, sp1)
U = [r0.copy(), r1.copy(), m_ic.copy(), rhoE.copy()]

def rhs_fn(U, dx):
    return rhs_iec(U, dx, sp0, sp1, iec=True, use_char=False,
                   weno_order=5, bc='transmissive')

t, step = 0.0, 0
print_times = [0.005, 0.01, 0.05, 0.14]
ti = 0

while t < 0.14 - 1e-12:
    r0_, r1_, m_, E_ = U
    rho_ = r0_ + r1_; Y0_ = r0_ / np.maximum(rho_, 1e-30)
    u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
    lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
    dt = min(0.3 * dx / (lam + 1e-10), 0.14 - t)
    U, _ = rkstep(U, rhs_fn, dx, dt)
    t += dt; step += 1

    if ti < len(print_times) and t >= print_times[ti] - 1e-12:
        r0f, r1f, mf, Ef = U
        rhof = r0f + r1f
        Y0f  = r0f / np.maximum(rhof, 1e-30)
        uf, ef, Pf, Tf, c2f = nasg_prim(Y0f, rhof, mf, Ef, sp0, sp1)
        imax = int(np.argmax(Pf))
        print(f"t={t:.4f}  Pmin={Pf.min():.4e}  Pmax={Pf.max():.4e} at x={x[imax]:.4f}")
        print(f"  P at x=0 (cell 99): {Pf[99]:.4e}  P at x=0.25 (cell 149): {Pf[149]:.4e}")
        print(f"  u at x=-0.25: {uf[74]:.4f}  u at x=0: {uf[99]:.4f}  u at x=0.25: {uf[149]:.4f}")
        print(f"  Y0 at x=-0.25: {Y0f[74]:.6f}  Y0 at x=0.25: {Y0f[149]:.6f}")
        # Find interface location (max dY0/dx)
        dY0 = np.abs(np.diff(Y0f))
        iif = int(np.argmax(dY0))
        print(f"  Interface near cell {iif}: x={x[iif]:.4f}")
        ti += 1
    if not np.all(np.isfinite(U[3])):
        print(f"NaN at step {step}"); break
