"""Full Riemann run with use_char=False."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from four_eq import ic_riemann_gl, nasg_prim, rhs_iec, rkstep, _NASG

N = 200
dx = 1.0 / N
x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)
sp0, sp1 = 'water_nd', 'air_nd'

r0, r1, m_ic, rhoE = ic_riemann_gl(x, sp0, sp1)
U = [r0.copy(), r1.copy(), m_ic.copy(), rhoE.copy()]

# Use IEC without char decomp — char decomp breaks at non-equilibrium sharp interface
def rhs_fn(U, dx):
    return rhs_iec(U, dx, sp0, sp1, iec=True, use_char=False,
                   weno_order=5, bc='transmissive')

t, step = 0.0, 0
t_end = 0.14
print(f"N={N}  t_end={t_end}")

while t < t_end - 1e-12:
    r0_, r1_, m_, E_ = U
    rho_ = r0_ + r1_; Y0_ = r0_ / np.maximum(rho_, 1e-30)
    u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
    lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
    dt = min(0.3 * dx / (lam + 1e-10), t_end - t)
    U, _ = rkstep(U, rhs_fn, dx, dt)
    t += dt; step += 1
    if not np.all(np.isfinite(U[3])):
        print(f"  NaN at step {step} t={t:.4e}")
        break
    if step % 5000 == 0:
        print(f"  step {step}  t={t:.4e}  lam={lam:.1f}")

print(f"Done: t={t:.4f}  {step} steps")

# Plot
r0f, r1f, mf, Ef = U
rhof = r0f + r1f
Y0f  = r0f / np.maximum(rhof, 1e-30)
uf, ef, Pf, Tf, c2f = nasg_prim(Y0f, rhof, mf, Ef, sp0, sp1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(x, rhof); axes[0,0].set_title('density'); axes[0,0].set_ylabel('ρ')
axes[0,1].plot(x, uf);   axes[0,1].set_title('velocity')
axes[1,0].plot(x, Pf);   axes[1,0].set_title('pressure')
axes[1,1].plot(x, Y0f);  axes[1,1].set_title('Y0 (water fraction)')
for ax in axes.flat: ax.set_xlabel('x')
plt.tight_layout()
plt.savefig('output/riemann_N200_nochar.png', dpi=100)
print("Plot saved: output/riemann_N200_nochar.png")
