"""Debug §4.2.1 Riemann problem — trace Pmin cell and Newton convergence."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from four_eq_1d import (
    ic_riemann_gl, nasg_prim, rhs_iec, rkstep, _NASG,
    nasg_p_from_rho_e, nasg_T_from_P_e
)

N = 100
dx = 1.0 / N
x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)
sp0, sp1 = 'water_nd', 'air_nd'

r0, r1, m_ic, rhoE = ic_riemann_gl(x, sp0, sp1)
U = [r0.copy(), r1.copy(), m_ic.copy(), rhoE.copy()]

def rhs_fn(U, dx):
    return rhs_iec(U, dx, sp0, sp1, iec=True, use_char=True,
                   weno_order=5, bc='transmissive')

r0_, r1_, m_, E_ = U
rho_ = r0_ + r1_
Y0_  = r0_ / np.maximum(rho_, 1e-30)
u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
dt = 0.3 * dx / (lam + 1e-10)
print(f"N={N} dt={dt:.4e} lam={lam:.2f}")
print(f"IC P: {P_.min():.4e}..{P_.max():.4e}")

for step in range(100):
    r0_, r1_, m_, E_ = U
    rho_ = r0_ + r1_
    Y0_  = r0_ / np.maximum(rho_, 1e-30)
    u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
    lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
    dt = 0.3 * dx / (lam + 1e-10)

    imin = int(np.argmin(P_))
    imax = int(np.argmax(P_))
    if step % 10 == 0 or P_.min() <= 1e-14:
        print(f"step {step+1:3d}: Pmax={P_.max():.4e} at x={x[imax]:.4f}  "
              f"Pmin={P_.min():.4e} at x={x[imin]:.4f} (cell {imin})  lam={lam:.2f}")
        if P_.min() <= 1e-14:
            print(f"  FLOOR HIT: cell {imin}")
            print(f"    Y0={Y0_[imin]:.6f}  rho={rho_[imin]:.6f}  u={u_[imin]:.6f}")
            print(f"    e={e_[imin]:.6f}  T={T_[imin]:.6f}")
            # Show neighbors
            for ii in range(max(0,imin-3), min(N,imin+4)):
                print(f"    [{ii:3d}] x={x[ii]:.4f}  Y0={Y0_[ii]:.8f}  rho={rho_[ii]:.6f}  "
                      f"P={P_[ii]:.4e}  u={u_[ii]:.6f}")
            print()

    U, _ = rkstep(U, rhs_fn, dx, dt)

    if not np.all(np.isfinite(U[3])):
        bad = np.where(~np.isfinite(U[3]))[0]
        print(f"\nNaN at step {step+1}, cells: {bad}")
        r0b, r1b, mb, Eb = U
        rhob = r0b + r1b
        Y0b  = r0b / np.maximum(rhob, 1e-30)
        for ii in bad[:5]:
            print(f"  cell {ii}: x={x[ii]:.4f}  Y0={Y0b[ii]:.6f}  rho={rhob[ii]:.6f}  "
                  f"m={mb[ii]:.6f}  E={Eb[ii]:.6f}")
        # Show context around first bad cell
        ii0 = int(bad[0])
        print(f"\nContext around cell {ii0}:")
        for ii in range(max(0,ii0-5), min(N,ii0+6)):
            rr = rhob[ii]
            y0 = Y0b[ii]
            ub_ = mb[ii]/(rr+1e-30)
            eb_ = Eb[ii]/(rr+1e-30) - 0.5*ub_**2
            Pb_ = nasg_p_from_rho_e(np.array([y0]), np.array([rr]), np.array([eb_]), sp0, sp1)
            print(f"  [{ii:3d}] x={x[ii]:.4f}  Y0={y0:.8f}  rho={rr:.6f}  "
                  f"e={eb_:.4f}  P={float(Pb_):.4e}")
        break
