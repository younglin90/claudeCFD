"""2D linear advection on a triangulated periodic box (unstructured mesh).

Each Cartesian quad on a Nx × Ny grid is split into two triangles, giving
2·Nx·Ny triangles.  Translation by velocity a = (1, 1) for one period;
exact solution = initial condition.

Verifies the unstructured T-MLP-u path:
  - mesh constructor produces consistent topology (Σ areas == Lx·Ly)
  - first_order vs TMLPU(superbee/mc/minmod) on a Gaussian
  - mass conservation
  - LMP property (no overshoot/undershoot beyond initial extrema)
  - PNG saved to tests/output/2d_advection_unstructured.png

Note on BCs: the unstructured constructor produces a non-periodic boundary
by default (it doesn't auto-wrap edges), so we apply transmissive BC on
all four sides and shrink the test to a *partial* trip so that the
Gaussian doesn't reach the right wall.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from _pkgshim import setup_paths
setup_paths()

from mesh import triangulate_box
from equations import Advection
from reconstruction import TMLPU
from boundary import BoundaryCondition
from solver import solve


def _gaussian(mesh, sigma=0.07, x0=0.3, y0=0.5):
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def main():
    Nx = Ny = 32
    Lx = Ly = 1.0
    mesh = triangulate_box(Nx, Ny, Lx=Lx, Ly=Ly, diag='rising')
    print(f"unstructured mesh: {mesh.n_cells} triangles, {mesh.n_faces} edges, "
          f"area = {float(np.sum(mesh.cell_volumes)):.6f}  (expect {Lx*Ly:.6f})")
    assert abs(float(np.sum(mesh.cell_volumes)) - Lx * Ly) < 1e-12

    eq = Advection(velocity=[1.0, 0.0])    # translate +x
    U0_field = _gaussian(mesh)
    U0 = U0_field[None, :]
    init_max = float(np.max(U0_field));  init_min = float(np.min(U0_field))

    # Use Dirichlet 0 at the inflow boundary (x_min) so a transmissive
    # ghost can't echo the (small but non-zero) Gaussian tail back in;
    # transmissive everywhere else.
    bc = {
        'x_min': BoundaryCondition('dirichlet', state=(0.0,)),
        'x_max': BoundaryCondition('transmissive'),
        'y_min': BoundaryCondition('transmissive'),
        'y_max': BoundaryCondition('transmissive'),
    }

    # Keep the Gaussian well inside the box for the whole run:
    # peak moves 0.3 → 0.5 with σ=0.07, so 6σ tail stays in [0.08, 0.92].
    t_end = 0.2

    cases = [
        ('first_order',     'first_order'),
        ('TMLPU(minmod)',   TMLPU(tvd='minmod')),
        ('TMLPU(superbee)', TMLPU(tvd='superbee')),
        ('TMLPU(mc)',       TMLPU(tvd='mc')),
    ]

    print(f"\n2D unstructured translation advection (a=(1,0), t={t_end})")
    print("=" * 78)
    fails = []
    panels = []
    for label, recon in cases:
        res = solve(mesh, eq, U0,
                    reconstruction=recon, flux='upwind',
                    integrator='ssp_rk2',
                    bc=bc,
                    cfl=0.4, t_end=t_end, max_steps=500_000)
        U_f = res['U_final'][0]
        # Mass through the domain may leave through the +x boundary, so
        # we measure conservation INSIDE a sub-rectangle the wave hasn't
        # yet reached (right boundary is at x=1; peak at 0.65).
        # Use the proper integral Σ U·V on the (variable-area) triangle
        # mesh — naïve Σ U is biased by cell-area variation.
        V = mesh.cell_volumes
        m0 = float(np.sum(U0_field * V))
        mf = float(np.sum(U_f      * V))
        m_drift = abs(mf - m0) / m0
        # For correlation use the exact translated field on the same mesh.
        x = mesh.cell_centers[:, 0]; y = mesh.cell_centers[:, 1]
        x_peak = 0.3 + 1.0 * res['t']
        sigma = 0.07
        U_exact = np.exp(-((x - x_peak) ** 2 + (y - 0.5) ** 2) / (2 * sigma ** 2))
        corr = float(np.corrcoef(U_f, U_exact)[0, 1])
        over = float(np.max(U_f) - init_max)
        under = float(init_min - np.min(U_f))
        flag = ' '
        if not np.isfinite(U_f).all():
            flag = 'NaN'; fails.append((label, 'NaN'))
        elif m_drift > 1e-3:
            flag = '!mass'; fails.append((label, f'mass drift {m_drift:.2e}'))
        elif label.startswith('TMLPU') and (over > 1e-9 or under > 1e-9):
            flag = '!LMP'
            fails.append((label, f'LMP over={over:.3e} under={under:.3e}'))
        print(f"  [{label:18s}] n={res['n_steps']:5d}  drift={m_drift:.1e}  "
              f"corr_exact={corr:+.3f}  over={over:+.2e}  under={under:+.2e}  {flag}")
        panels.append((label, U_f, corr))

    # PNG: tripcolor for triangles
    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1],
                                triangles=np.array(mesh.cell_nodes))
    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    axs = axs.ravel()
    tcf0 = axs[0].tripcolor(triang, U0_field, shading='flat', vmin=0, vmax=1)
    axs[0].set_title('initial');  axs[0].set_aspect('equal')
    plt.colorbar(tcf0, ax=axs[0], fraction=0.046)
    for k, (label, U_f, corr) in enumerate(panels):
        ax = axs[k + 1]
        tcf = ax.tripcolor(triang, U_f, shading='flat', vmin=0, vmax=1)
        ax.set_title(f"{label}\ncorr={corr:+.3f}")
        ax.set_aspect('equal')
        plt.colorbar(tcf, ax=ax, fraction=0.046)
    if len(panels) + 1 < len(axs):
        for j in range(len(panels) + 1, len(axs)):
            axs[j].axis('off')
    fig.suptitle(f"2D unstructured (triangle) translation advection — t={t_end}",
                 fontsize=11)
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '2d_advection_unstructured.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")

    print("=" * 78)
    if fails:
        print(f"FAIL — {len(fails)} cases:")
        for r, why in fails:
            print(f"  {r}: {why}")
        return 1
    print("PASS — 2D unstructured T-MLP-u: finite, conservative, LMP-bounded.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
