"""Test if instability is in WENO reconstruction or base HLLC/EOS."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from four_eq import ic_riemann_gl, nasg_prim, rhs_iec, rkstep

def run(N, scheme, weno_order=5, use_char=True, CFL=0.3, t_end=0.005):
    dx = 1.0 / N
    x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)
    sp0, sp1 = 'water_nd', 'air_nd'

    r0, r1, m_ic, rhoE = ic_riemann_gl(x, sp0, sp1)
    U = [r0.copy(), r1.copy(), m_ic.copy(), rhoE.copy()]

    use_iec = (scheme != 'STD')
    def rhs_fn(U, dx):
        return rhs_iec(U, dx, sp0, sp1, iec=use_iec, use_char=use_char,
                       weno_order=weno_order, bc='transmissive')

    r0_, r1_, m_, E_ = U
    rho_ = r0_ + r1_; Y0_ = r0_ / np.maximum(rho_, 1e-30)
    u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
    lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))

    t, step = 0.0, 0
    while t < t_end - 1e-12:
        r0_, r1_, m_, E_ = U
        rho_ = r0_ + r1_; Y0_ = r0_ / np.maximum(rho_, 1e-30)
        u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
        lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
        dt = min(CFL * dx / (lam + 1e-10), t_end - t)
        U, _ = rkstep(U, rhs_fn, dx, dt)
        t += dt; step += 1
        if not np.all(np.isfinite(U[3])):
            print(f"  [{scheme} W{weno_order}{'C' if use_char else ''}] N={N} CFL={CFL}: NaN at step {step} t={t:.4e}")
            return False
    print(f"  [{scheme} W{weno_order}{'C' if use_char else ''}] N={N} CFL={CFL}: OK through t={t:.4e} ({step} steps)")
    return True

# Sweep different configurations
for N in [100]:
    print(f"\nN={N}:")
    run(N, 'IEC', weno_order=5, use_char=True,  CFL=0.3)  # original
    run(N, 'IEC', weno_order=5, use_char=False, CFL=0.3)  # no char decomp
    run(N, 'IEC', weno_order=3, use_char=False, CFL=0.3)  # WENO3
    run(N, 'STD', weno_order=5, use_char=False, CFL=0.3)  # standard recon
    run(N, 'IEC', weno_order=5, use_char=True,  CFL=0.1)  # smaller CFL
    run(N, 'IEC', weno_order=5, use_char=True,  CFL=0.05) # very small CFL
