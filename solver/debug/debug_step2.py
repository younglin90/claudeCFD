"""Debug: trace first few steps."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings
warnings.filterwarnings('error')  # make warnings into errors

from four_eq_1d import (
    ic_droplet, nasg_prim, rhs_iec, rkstep, _NASG, nasg_p_from_rho_e
)

N = 100
dx = 1.0 / N
x  = np.linspace(dx/2, 1.0 - dx/2, N)
P0, T0, u0 = 101325.0, 297.0, 5.0

r0, r1, u_ic, rhoE = ic_droplet(x, P0, T0, u0, eps_factor=2.0)
U = [r0.copy(), r1.copy(), (r0+r1)*u_ic, rhoE.copy()]

rho = r0 + r1
Y0  = r0 / rho
u_, e_, P_, T_, c2_ = nasg_prim(Y0, rho, U[2], U[3])
lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
dt  = 0.5 * dx / (lam + 1e-10)
print(f"N={N}  lam={lam:.2f}  dt={dt:.3e}")
print(f"IC P: {P_.min():.4e}..{P_.max():.4e}  T: {T_.min():.2f}..{T_.max():.2f}")
print(f"IC c: {np.sqrt(c2_).min():.2f}..{np.sqrt(c2_).max():.2f}")

def rhs_fn(U, dx):
    return rhs_iec(U, dx, 'water', 'air', iec=True, use_char=True, weno_order=5)

# Run 10 steps
for step in range(20):
    r0_, r1_, m_, E_ = U
    rho_   = r0_ + r1_
    Y0_    = r0_ / np.maximum(rho_, 1e-30)
    try:
        u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_)
    except FloatingPointError as e:
        print(f"  nasg_prim error at step {step}: {e}")
        break
    lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
    dt  = 0.5 * dx / (lam + 1e-10)

    print(f"\nStep {step+1}: lam={lam:.2f}  dt={dt:.3e}")
    print(f"  P: {P_.min():.4e}..{P_.max():.4e}")
    print(f"  T: {T_.min():.2f}..{T_.max():.2f}")
    print(f"  Y0: {Y0_.min():.4f}..{Y0_.max():.4f}")

    try:
        U, _ = rkstep(U, rhs_fn, dx, dt)
    except FloatingPointError as e:
        print(f"  rkstep error at step {step+1}: {e}")
        import traceback; traceback.print_exc()
        break

    if not np.all(np.isfinite(U[3])):
        print(f"  NaN/Inf in U after step {step+1}")
        break

print("\nDone!")
