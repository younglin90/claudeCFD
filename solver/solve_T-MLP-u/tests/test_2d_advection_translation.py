"""2D linear advection on a periodic structured Cartesian mesh.

Translation by velocity a = (1, 1) on the unit square [0,1]² with
periodic BCs.  Single round-trip → exact solution = initial condition.

Verifies:
  - first_order  vs  TMLPU(superbee, mc, minmod) on a Gaussian
  - mass conservation (drift < 1e-10)
  - shape correlation with the exact (initial) state
  - LMP property: T-MLP-u outputs satisfy
        max(U_f) ≤ max(U₀) + ε,   min(U_f) ≥ min(U₀) − ε
  - PNG saved to tests/output/2d_advection_translation.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _pkgshim import setup_paths
setup_paths()

from mesh import build_structured_2d
from equations import Advection
from reconstruction import TMLPU
from solver import solve


def _gaussian2d(mesh, sigma=0.07, x0=0.3, y0=0.5):
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def main():
    Nx = Ny = 64
    Lx = Ly = 1.0
    mesh = build_structured_2d(Nx, Ny, Lx=Lx, Ly=Ly, periodic=(True, True))
    eq = Advection(velocity=[1.0, 1.0])

    U0_field = _gaussian2d(mesh)
    U0 = U0_field[None, :]
    init_max = float(np.max(U0_field));  init_min = float(np.min(U0_field))

    cases = [
        ('first_order',    'first_order'),
        ('TMLPU(minmod)',  TMLPU(tvd='minmod')),
        ('TMLPU(superbee)', TMLPU(tvd='superbee')),
        ('TMLPU(mc)',      TMLPU(tvd='mc')),
    ]
    print(f"2D translation advection (a=(1,1), {Nx}×{Ny}, t=1.0 → 1 period)")
    print("=" * 78)

    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    axs = axs.ravel()
    fails = []
    panels = []
    for label, recon in cases:
        res = solve(mesh, eq, U0,
                    reconstruction=recon, flux='upwind',
                    integrator='ssp_rk2',
                    cfl=0.4, t_end=1.0, max_steps=500_000)
        U_f = res['U_final'][0]
        m_drift = abs(np.sum(U_f) - np.sum(U0_field)) / np.sum(U0_field)
        corr = float(np.corrcoef(U_f.ravel(), U0_field.ravel())[0, 1])
        over = float(np.max(U_f) - init_max)
        under = float(init_min - np.min(U_f))
        flag = ' '
        if not np.isfinite(U_f).all():
            flag = 'NaN'; fails.append((label, 'NaN'))
        elif m_drift > 1e-10:
            flag = '!mass'; fails.append((label, f'mass drift {m_drift:.2e}'))
        elif label.startswith('TMLPU') and (over > 1e-9 or under > 1e-9):
            flag = '!LMP'
            fails.append((label,
                f'LMP over={over:.3e} under={under:.3e}'))
        print(f"  [{label:18s}] n={res['n_steps']:5d}  drift={m_drift:.1e}  "
              f"corr={corr:+.3f}  over={over:+.2e}  under={under:+.2e}  {flag}")
        panels.append((label, U_f, corr))

    # Plot — initial + 4 final states
    field = U0_field.reshape(Ny, Nx)
    im0 = axs[0].imshow(field, origin='lower', extent=[0, Lx, 0, Ly], vmin=0, vmax=1)
    axs[0].set_title('initial')
    plt.colorbar(im0, ax=axs[0], fraction=0.046)
    for k, (label, U_f, corr) in enumerate(panels):
        ax = axs[k + 1]
        im = ax.imshow(U_f.reshape(Ny, Nx), origin='lower',
                       extent=[0, Lx, 0, Ly], vmin=0, vmax=1)
        ax.set_title(f"{label}\ncorr={corr:+.3f}")
        plt.colorbar(im, ax=ax, fraction=0.046)
    # Hide unused panel
    if len(panels) + 1 < len(axs):
        for j in range(len(panels) + 1, len(axs)):
            axs[j].axis('off')

    fig.suptitle("2D advection translation — 1 period (Gaussian)", fontsize=11)
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '2d_advection_translation.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")

    print("=" * 78)
    if fails:
        print(f"FAIL — {len(fails)} cases:")
        for r, why in fails:
            print(f"  {r}: {why}")
        return 1
    print("PASS — 2D structured T-MLP-u: finite, conservative, LMP-bounded.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
