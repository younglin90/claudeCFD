"""Debug: show interface state at step 81 (before blowup at 82)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from four_eq import ic_riemann_gl, nasg_prim, rhs_iec, rkstep

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
rho_ = r0_ + r1_; Y0_ = r0_ / np.maximum(rho_, 1e-30)
u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
dt = 0.3 * dx / (lam + 1e-10)

for step in range(82):
    r0_, r1_, m_, E_ = U
    rho_ = r0_ + r1_; Y0_ = r0_ / np.maximum(rho_, 1e-30)
    u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
    lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
    dt = 0.3 * dx / (lam + 1e-10)

    if step == 80:
        print(f"\n=== State at step {step+1} (before step {step+2}) ===")
        # Interface region
        for i in range(40, 60):
            Y1i = 1.0 - Y0_[i]
            print(f"  [{i:3d}] x={x[i]:.4f}  Y0={Y0_[i]:.8f}  Y1={Y1i:.2e}  "
                  f"rho={rho_[i]:.6f}  P={P_[i]:.4e}  u={u_[i]:.6f}  T={T_[i]:.2f}")

    U, _ = rkstep(U, rhs_fn, dx, dt)

    if not np.all(np.isfinite(U[3])):
        bad = np.where(~np.isfinite(U[3]))[0]
        print(f"\nNaN at step {step+1}, {len(bad)} cells")
        print(f"First bad: cells {bad[:5]}")
        break

# Show state at step 82 (the blowup step)
r0_, r1_, m_, E_ = U
rho_ = r0_ + r1_; Y0_ = r0_ / np.maximum(rho_, 1e-30)
print(f"\n=== State after blowup ===")
for i in range(40, 60):
    Y1i = 1.0 - Y0_[i]
    print(f"  [{i:3d}] x={x[i]:.4f}  Y0={Y0_[i]:.6f}  Y1={Y1i:.2e}  "
          f"rho={rho_[i]:.4e}  m={m_[i]:.4e}  E={E_[i]:.4e}")
