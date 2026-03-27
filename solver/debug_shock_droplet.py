"""§4.2.3 Shock-Droplet: IEC vs STD comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from four_eq_1d import (ic_shock_droplet, nasg_prim, rhs_iec, rkstep, _NASG)

sp0, sp1 = 'water', 'air'
N = 200
dx = 1.0 / N
x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)

# Show initial state
r0, r1, m_ic, rhoE = ic_shock_droplet(x, sp0, sp1)
rho0 = r0 + r1
Y0_0 = r0 / np.maximum(rho0, 1e-30)
u0, e0, P0, T0, c20 = nasg_prim(Y0_0, rho0, m_ic, rhoE, sp0, sp1)
print("Initial state:")
print(f"  Pmin={P0.min():.4e}  Pmax={P0.max():.4e}")
print(f"  umin={u0.min():.4f}  umax={u0.max():.4f}")
print(f"  Tmin={T0.min():.2f}  Tmax={T0.max():.2f}")
print(f"  rhomax={rho0.max():.2f}")

# Shock parameters
lam0 = float(np.max(np.abs(u0) + np.sqrt(c20)))
print(f"\nMax wave speed: {lam0:.1f} m/s")
print(f"Estimated shock speed: ~1020 m/s")
print(f"Time for shock to reach droplet (0.2 m): ~{0.2/1020:.4f} s")
print(f"Time for shock to traverse droplet (0.2 m): ~{0.2/1020:.4f} s")

t_end = 5e-4  # shock traverses full droplet

results = {}
for scheme in ['IEC', 'STD']:
    r0_, r1_, m_, rhoE_ = ic_shock_droplet(x, sp0, sp1)
    U = [r0_.copy(), r1_.copy(), m_.copy(), rhoE_.copy()]

    use_iec = (scheme == 'IEC')
    def rhs_fn(U, dx, _iec=use_iec):
        return rhs_iec(U, dx, sp0, sp1, iec=_iec, use_char=False, weno_order=5,
                       bc='transmissive')

    t, step = 0.0, 0
    print(f"\n[{scheme}] Running to t={t_end:.2e} ...")
    crashed = False
    while t < t_end - 1e-14:
        r0c, r1c, mc, Ec = U
        rhoc = r0c + r1c
        Y0c  = r0c / np.maximum(rhoc, 1e-30)
        uc, ec, Pc, Tc, c2c = nasg_prim(Y0c, rhoc, mc, Ec, sp0, sp1)
        lam  = float(np.max(np.abs(uc) + np.sqrt(c2c)))
        dt   = min(0.3 * dx / (lam + 1e-10), t_end - t)
        U, _ = rkstep(U, rhs_fn, dx, dt)
        t += dt; step += 1
        if not np.all(np.isfinite(U[3])):
            print(f"  NaN at step {step} t={t:.4e}"); crashed = True; break
        if step % 200 == 0:
            print(f"  step={step} t={t:.4e}")

    r0f, r1f, mf, Ef = U
    rhof = r0f + r1f
    Y0f  = r0f / np.maximum(rhof, 1e-30)
    uf, ef, Pf, Tf, c2f = nasg_prim(Y0f, rhof, mf, Ef, sp0, sp1)
    print(f"  [{scheme}] t={t:.4e}  steps={step}  Pmin={Pf.min():.3e}  Pmax={Pf.max():.3e}  rhomax={rhof.max():.2f}")
    results[scheme] = dict(x=x, rho=rhof, u=uf, P=Pf, Y0=Y0f, t=t, crashed=crashed)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
colors = {'IEC': 'b', 'STD': 'r'}
ls = {'IEC': '-', 'STD': '--'}

for scheme, res in results.items():
    c = colors[scheme]; l = ls[scheme]
    axes[0,0].plot(res['x'], res['rho'], c+l, lw=1.5, label=f'{scheme} WENO5Z')
    axes[0,1].plot(res['x'], res['u'],   c+l, lw=1.5, label=f'{scheme}')
    axes[1,0].plot(res['x'], res['P'],   c+l, lw=1.5, label=f'{scheme}')
    axes[1,1].plot(res['x'], res['Y0'],  c+l, lw=1.5, label=f'{scheme}')

axes[0,0].set_title('Density ρ [kg/m³]')
axes[0,1].set_title('Velocity u [m/s]')
axes[1,0].set_title('Pressure P [Pa]')
axes[1,1].set_title('Water fraction Y₀')
for ax in axes.flat:
    ax.set_xlabel('x [m]'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

t_shown = results['IEC']['t']
plt.suptitle(f'§4.2.3 Shock-Droplet Interaction  N={N}  t={t_shown:.2e} s\n'
             f'IEC vs STD (WENO5Z, use_char=False)', fontsize=11)
plt.tight_layout()
os.makedirs('output', exist_ok=True)
fname = 'output/shock_droplet_comparison.png'
plt.savefig(fname, dpi=120)
print(f"\nSaved: {fname}")
