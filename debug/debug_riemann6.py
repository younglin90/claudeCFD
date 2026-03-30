"""§4.2.1 Gas-Liquid Riemann Problem — compare N=200 and N=501 at t=0.14."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from four_eq import run_riemann_gl, nasg_prim

results = {}
for N in [200, 501]:
    print(f"\n--- N={N} ---")
    x, U = run_riemann_gl(N=N, t_end=0.14, CFL=0.3)
    r0f, r1f, mf, Ef = U
    rhof = r0f + r1f
    Y0f  = r0f / np.maximum(rhof, 1e-30)
    uf, ef, Pf, Tf, c2f = nasg_prim(Y0f, rhof, mf, Ef, 'water_nd', 'air_nd')
    results[N] = dict(x=x, rho=rhof, u=uf, P=Pf, Y0=Y0f)
    print(f"  Pmin={Pf.min():.4e}  Pmax={Pf.max():.4e}")
    print(f"  umin={uf.min():.4f}  umax={uf.max():.4f}")
    print(f"  rhomin={rhof.min():.4f}  rhomax={rhof.max():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors = {200: 'b', 501: 'r'}
labels = {200: 'IEC W5 N=200', 501: 'IEC W5 N=501 (paper)'}

for N, res in results.items():
    x, rho, u, P, Y0 = res['x'], res['rho'], res['u'], res['P'], res['Y0']
    c = colors[N]; ls = '-' if N == 501 else '--'
    axes[0].plot(x, rho, c+ls, lw=1.5, label=labels[N])
    axes[1].plot(x, u,   c+ls, lw=1.5, label=labels[N])
    axes[2].plot(x, P,   c+ls, lw=1.5, label=labels[N])

axes[0].set_title('Density ρ');  axes[0].set_xlabel('x'); axes[0].legend()
axes[1].set_title('Velocity u'); axes[1].set_xlabel('x'); axes[1].legend()
axes[2].set_title('Pressure P'); axes[2].set_xlabel('x'); axes[2].legend()
plt.suptitle('§4.2.1 Gas-Liquid Riemann Problem  t=0.14\n'
             'IEC WENO5Z (use_char=False)', fontsize=11)
plt.tight_layout()
fname = 'output/riemann_gl_t014.png'
plt.savefig(fname, dpi=120)
print(f"\nSaved: {fname}")
