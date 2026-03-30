"""§4.2.3 PE error evolution during shock-droplet interaction."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from four_eq import (ic_shock_droplet, nasg_prim, rhs_iec, rkstep)

sp0, sp1 = 'water', 'air'
N = 200
dx = 1.0 / N
x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)

def run_and_track_pe(scheme):
    r0_, r1_, m_, rhoE_ = ic_shock_droplet(x, sp0, sp1)
    U = [r0_.copy(), r1_.copy(), m_.copy(), rhoE_.copy()]

    use_iec = (scheme == 'IEC')
    def rhs_fn(U, dx, _iec=use_iec):
        return rhs_iec(U, dx, sp0, sp1, iec=_iec, use_char=False, weno_order=5,
                       bc='transmissive')

    t, step = 0.0, 0
    t_end = 5e-4
    times, pe_errors = [], []

    print(f"[{scheme}] tracking PE error ...")
    while t < t_end - 1e-14:
        r0c, r1c, mc, Ec = U
        rhoc = r0c + r1c
        Y0c  = r0c / np.maximum(rhoc, 1e-30)
        uc, ec, Pc, Tc, c2c = nasg_prim(Y0c, rhoc, mc, Ec, sp0, sp1)
        lam  = float(np.max(np.abs(uc) + np.sqrt(c2c)))
        dt   = min(0.3 * dx / (lam + 1e-10), t_end - t)

        # PE error: at interface cells (mixed Y0)
        # Once shock reaches droplet, the "equilibrium" pressure is ill-defined.
        # Instead, measure pressure uniformity within the droplet (Y0 > 0.5 cells).
        iwater = np.where(Y0c > 0.5)[0]
        if len(iwater) > 0:
            P_ref = np.mean(Pc[iwater])
            if P_ref > 1e3:  # only track after shock reaches droplet
                pe = float(np.sqrt(np.mean((Pc[iwater] / P_ref - 1.0)**2)))
                times.append(t)
                pe_errors.append(pe)

        U, _ = rkstep(U, rhs_fn, dx, dt)
        t += dt; step += 1
        if not np.all(np.isfinite(U[3])):
            print(f"  NaN at step {step} t={t:.4e}"); break

    r0f, r1f, mf, Ef = U
    rhof = r0f + r1f
    Y0f  = r0f / np.maximum(rhof, 1e-30)
    uf, ef, Pf, Tf, c2f = nasg_prim(Y0f, rhof, mf, Ef, sp0, sp1)
    print(f"  Done: t={t:.4e}  Pmax={Pf.max():.3e}  rhomax={rhof.max():.2f}")
    return np.array(times), np.array(pe_errors), x, rhof, uf, Pf, Y0f

results = {}
for scheme in ['IEC', 'STD']:
    times, pe, xf, rho, u, P, Y0 = run_and_track_pe(scheme)
    results[scheme] = dict(times=times, pe=pe, rho=rho, u=u, P=P, Y0=Y0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = {'IEC': 'b', 'STD': 'r'}

# PE error vs time
ax = axes[0]
for scheme, res in results.items():
    if len(res['times']) > 0:
        ax.semilogy(res['times']*1e3, res['pe'], colors[scheme]+'-', lw=1.5, label=scheme)
ax.set_xlabel('t [ms]')
ax.set_ylabel('PE error (RMS within droplet)')
ax.set_title('§4.2.3 Pressure uniformity in droplet')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# Final profiles
ax = axes[1]
for scheme, res in results.items():
    ax.plot(x, res['P']/1e6, colors[scheme]+'-', lw=1.5, label=scheme)
ax.set_xlabel('x [m]')
ax.set_ylabel('P [MPa]')
ax.set_title(f'Pressure at t=5e-4 s')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
for scheme, res in results.items():
    ax.plot(x, res['rho'], colors[scheme]+'-', lw=1.5, label=scheme)
ax.set_xlabel('x [m]')
ax.set_ylabel('ρ [kg/m³]')
ax.set_title('Density at t=5e-4 s')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(f'§4.2.3 Shock-Droplet  N={N}  (P_shock=10 atm)', fontsize=11)
plt.tight_layout()
os.makedirs('output', exist_ok=True)
fname = 'output/shock_droplet_pe.png'
plt.savefig(fname, dpi=120)
print(f"\nSaved: {fname}")

# Print PE error at specific times
print("\nPE error within droplet:")
for scheme, res in results.items():
    if len(res['times']) > 3:
        # Sample at 3 times
        idx = [len(res['times'])//4, len(res['times'])//2, -1]
        print(f"  {scheme}:", end=' ')
        for i in idx:
            print(f"t={res['times'][i]*1e3:.2f}ms: {res['pe'][i]:.3e}", end='  ')
        print()
