"""Test Riemann with N=501 (paper resolution) and N=100."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from four_eq import ic_riemann_gl, nasg_prim, rhs_iec, rkstep

def run(N, t_end=0.14):
    dx = 1.0 / N
    x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)
    sp0, sp1 = 'water_nd', 'air_nd'

    r0, r1, m_ic, rhoE = ic_riemann_gl(x, sp0, sp1)
    U = [r0.copy(), r1.copy(), m_ic.copy(), rhoE.copy()]

    def rhs_fn(U, dx):
        return rhs_iec(U, dx, sp0, sp1, iec=True, use_char=True,
                       weno_order=5, bc='transmissive')

    r0_, r1_, m_, E_ = U
    rho_ = r0_ + r1_; Y0_ = r0_ / np.maximum(rho_, 1e-30)
    u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
    lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
    dt0 = 0.3 * dx / (lam + 1e-10)
    nsteps = int(t_end / dt0) + 1
    print(f"N={N}  dt={dt0:.4e}  nsteps~{nsteps}")

    t, step = 0.0, 0
    while t < t_end - 1e-12:
        r0_, r1_, m_, E_ = U
        rho_ = r0_ + r1_; Y0_ = r0_ / np.maximum(rho_, 1e-30)
        u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
        lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
        dt = min(0.3 * dx / (lam + 1e-10), t_end - t)
        U, _ = rkstep(U, rhs_fn, dx, dt)
        t += dt; step += 1
        if not np.all(np.isfinite(U[3])):
            print(f"  NaN at step {step}  t={t:.4e}")
            return x, U, False
        if step % 1000 == 0:
            print(f"  step {step}  t={t:.4e}  Pmax={P_.max():.3e}  lam={lam:.1f}")

    print(f"  Completed: t={t:.4f}  ({step} steps)")
    return x, U, True

print("=== N=100 ===")
x100, U100, ok100 = run(100)
print(f"\n=== N=501 ===")
x501, U501, ok501 = run(501)
