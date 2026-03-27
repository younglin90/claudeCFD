"""Debug: trace first RHS call for droplet IC."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
np.seterr(all='raise')   # make all warnings into errors so we get a traceback

from four_eq_1d import (
    ic_droplet, nasg_prim, rhs_iec, rkstep, _NASG
)

N = 20
dx = 1.0 / N
x  = np.linspace(dx/2, 1.0 - dx/2, N)
P0, T0, u0 = 101325.0, 297.0, 5.0

r0, r1, u_ic, rhoE = ic_droplet(x, P0, T0, u0, eps_factor=2.0)
U = [r0.copy(), r1.copy(), (r0+r1)*u_ic, rhoE.copy()]

print("--- IC values ---")
rho = r0 + r1
Y0  = r0 / rho
print(f"  rho: {rho.min():.4f} .. {rho.max():.4f}")
print(f"  Y0:  {Y0.min():.4f} .. {Y0.max():.4f}")
print(f"  u:   {u_ic.min():.4f} .. {u_ic.max():.4f}")
print(f"  rhoE:{rhoE.min():.4e} .. {rhoE.max():.4e}")

u_, e_, P_, T_, c2_ = nasg_prim(Y0, rho, U[2], U[3])
print(f"\n--- nasg_prim (IC) ---")
print(f"  P: {P_.min():.4e} .. {P_.max():.4e}")
print(f"  T: {T_.min():.4f} .. {T_.max():.4f}")
print(f"  c: {np.sqrt(c2_).min():.2f} .. {np.sqrt(c2_).max():.2f}")

lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
dt  = 0.5 * dx / (lam + 1e-10)
print(f"\n  lam={lam:.2f}  dt={dt:.3e}")

def rhs_fn(U, dx):
    return rhs_iec(U, dx, 'water', 'air', iec=True, use_char=True, weno_order=5)

try:
    print("\n--- calling rhs_iec ---")
    k1, _, _ = rhs_fn(U, dx)
    print("  k1[r0]:  ", k1[0].min(), "..", k1[0].max())
    print("  k1[r1]:  ", k1[1].min(), "..", k1[1].max())
    print("  k1[m]:   ", k1[2].min(), "..", k1[2].max())
    print("  k1[E]:   ", k1[3].min(), "..", k1[3].max())
    print("  all finite:", all(np.all(np.isfinite(k1[q])) for q in range(4)))
except FloatingPointError as e:
    import traceback
    traceback.print_exc()
    print("\nERROR:", e)
