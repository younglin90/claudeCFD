"""LeVeque (SIAM J. Numer. Anal. 33(2), 1996) solid-body-rotation test.

Domain: [0, 1]² triangulated as a Criss-cross / Union-Jack mesh
        (4 N² triangles, default N = 100 → 40 000 triangles).
Velocity field (rigid rotation about (½, ½), period T = 1):

    u(x, y) = -2π (y − ½)
    v(x, y) =  2π (x − ½)

Initial scalar field is the canonical superposition of three shapes:

  • Slotted (Zalesak) cylinder at (½, ¾)
  • Cone                       at (½, ¼)
  • Smooth (cosine-bell) hump  at (¼, ½)

After exactly one period (t = 1) the analytic solution recovers φ₀.
The script runs *two* solvers on the *same* mesh:

  Case A:  reconstruction = MinmodTVD1D-style symmetric minmod (no T-MLP-u)
           — this is the classical TVD limiter.
  Case B:  reconstruction = TMLPU(tvd='superbee') (T-MLP-u + superbee)
           — wraps superbee's compressive ψ_TVD with the Local Maximum
           Principle bound.

Both runs use upwind flux, SSP-RK2 in time, transmissive walls (the
shapes never touch the boundary).  The script saves a comparison
figure ``output/2d_leveque_rotation.png`` with seven panels:

    initial  |  Case A final  |  Case B final  |  Case A − exact
                                                |  Case B − exact
    diagonal slices through each shape:
        slotted cylinder (x = 0.5)
        cone             (x = 0.5)
        cosine bell      (y = 0.5)

and prints L₁ errors per shape so the difference between the two
schemes can be quantified.

Note (performance): the criss-cross mesh contains 40 000 triangles, the
rotation field has |u|_max = 2π√(½) ≈ 4.44, so with CFL = 0.4 and
h ≈ 0.01 we expect ≈ 1100 SSP-RK2 steps × ~0.5 s = ≈ 10 min wall.
"""
from __future__ import annotations
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from _pkgshim import setup_paths
setup_paths()

from mesh import criss_cross_box
from equations import Advection
from reconstruction import TMLPU
from boundary import BoundaryCondition
from solver import solve


# ─── LeVeque initial field ──────────────────────────────────────────────────
def phi0(x, y):
    r0 = 0.15
    # Slotted cylinder at (0.5, 0.75)
    r1 = np.sqrt((x - 0.5) ** 2 + (y - 0.75) ** 2) / r0
    in_slot = (np.abs(x - 0.5) < 0.025) & (y < 0.85)
    phi_slot = np.where((r1 <= 1.0) & ~in_slot, 1.0, 0.0)
    # Cone at (0.5, 0.25)
    r2 = np.sqrt((x - 0.5) ** 2 + (y - 0.25) ** 2) / r0
    phi_cone = np.where(r2 <= 1.0, 1.0 - r2, 0.0)
    # Cosine-bell hump at (0.25, 0.5)
    r3 = np.sqrt((x - 0.25) ** 2 + (y - 0.5) ** 2) / r0
    phi_hump = np.where(r3 <= 1.0, 0.25 * (1.0 + np.cos(np.pi * r3)), 0.0)
    return phi_slot + phi_cone + phi_hump


# ─── Run helper ────────────────────────────────────────────────────────────
def _run(mesh, recon, *, t_end, cfl=0.4, integrator='ssp_rk2',
         flux='upwind', n_face_quad=1, label='',
         face_velocity_mode='analytic'):
    eq = Advection(velocity=lambda x, y: (-2 * np.pi * (y - 0.5),
                                          2 * np.pi * (x - 0.5)))
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    U0 = phi0(x, y)[None, :]
    bc = {p: BoundaryCondition('dirichlet', state=(0.0,))
          for p in mesh.bc_patches}

    t0 = time.time()
    try:
        res = solve(mesh, eq, U0,
                    reconstruction=recon, flux=flux,
                    integrator=integrator, bc=bc,
                    cfl=cfl, n_face_quad=n_face_quad,
                    face_velocity_mode=face_velocity_mode,
                    t_end=t_end, max_steps=50_000)
        wall = time.time() - t0
        out = res['U_final'][0]
        print(f"  [{label}]  n={res['n_steps']:5d}  t={res['t']:.4f}/{t_end:.4f}  "
              f"wall={wall:.1f} s")
    except FloatingPointError as e:
        wall = time.time() - t0
        print(f"  [{label}]  DIVERGED ({e})  wall={wall:.1f} s")
        out = np.zeros_like(phi0(x, y))
    return out


# ─── Per-shape error diagnostics ──────────────────────────────────────────
def _shape_masks(x, y):
    r0 = 0.15
    r1 = np.sqrt((x - 0.5) ** 2 + (y - 0.75) ** 2) / r0
    r2 = np.sqrt((x - 0.5) ** 2 + (y - 0.25) ** 2) / r0
    r3 = np.sqrt((x - 0.25) ** 2 + (y - 0.5) ** 2) / r0
    margin = 1.5
    return {
        'slot': r1 <= margin,
        'cone': r2 <= margin,
        'hump': r3 <= margin,
    }


def _l1(field_num, field_exact, V, mask=None):
    if mask is None:
        return float(np.sum(np.abs(field_num - field_exact) * V) / np.sum(V))
    return float(np.sum(np.abs(field_num - field_exact) * V * mask) /
                 np.maximum(np.sum(V * mask), 1e-30))


def main():
    # N=100 criss-cross gives 4·N² = 40 000 triangles.  Wall time per
    # full sweep ~25–35 min; carry-over of the N=50 best config is the
    # baseline starting point.
    N = 100
    mesh = criss_cross_box(N, L=1.0)
    print(f"criss-cross mesh: N={N}, n_cells={mesh.n_cells}, "
          f"n_faces={mesh.n_faces}, area={float(np.sum(mesh.cell_volumes)):.6f}")
    assert abs(float(np.sum(mesh.cell_volumes)) - 1.0) < 1e-12

    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    V = mesh.cell_volumes
    U0_field = phi0(x, y)
    init_max = float(np.max(U0_field))
    init_min = float(np.min(U0_field))

    print(f"\n=== LeVeque rotation @ N={N} (one full period, t=1.0) ===")
    print(f"initial: φ ∈ [{init_min:.4f}, {init_max:.4f}],  "
          f"∫φ = {float(np.sum(U0_field * V)):.6f}")

    # Both runs share the same high-order infrastructure:
    #   • LSQ stencil = vertex (cells sharing any vertex with C)
    #   • polynomial order k = 2 (∇W + Hessian via quadratic LSQ)
    #   • time integrator = SSP-RK3
    #   • face quadrature = 2-point Gauss-Legendre on every edge
    # so the only difference is the face-side limiter.
    common = dict(stencil='vertex2', order=3)

    # -- Case A: T-MLP-u wrapping the *downwind* TVD limiter -------------
    # ψ_DW(r)=max(0,min(2r,2)) is the most compressive symmetric Sweby-
    # region limiter — it would oscillate when used standalone.  The
    # T-MLP-u wrapper supplies the LMP bound that suppresses overshoots,
    # turning the aggressive downwind compression into a monotone scheme.
    case_A = _run(mesh,
                  TMLPU(tvd='downwind', mlp_bound=True,
                        extremum_relax=True, tvb_M=64.0,
                        virtual_uu_gradient=True, **common),
                  t_end=1.0, integrator='ssp_rk3', n_face_quad=2,
                  label='A: T-MLP-u + downwind   (vertex, k=2, RK3, 2pt GQ, virt-UU)')

    # -- Case B: plain van Leer TVD (no T-MLP-u wrapper) -----------------
    # mlp_bound=False makes TMLPU compute ψ = ψ_TVD only — no LMP.  Used
    # here with the smooth ψ_VL(r) = (r+|r|)/(1+|r|) limiter.
    case_B = _run(mesh,
                  TMLPU(tvd='van_leer', mlp_bound=False, **common),
                  t_end=1.0, integrator='ssp_rk3', n_face_quad=2,
                  label='B: plain van Leer       (vertex, k=2, RK3, 2-pt GQ)')

    # -- Case C: textbook central scheme --------------------------------
    # 1st-order reconstruction (W_L = W_owner, W_R = W_neighbour) +
    # central flux F = ½(F_L + F_R) — no upwinding, no limiter.  Pure
    # central differencing; well-known to oscillate on advection
    # without artificial dissipation.  Provided as an absolute reference.
    case_C = _run(mesh,
                  'first_order',
                  t_end=1.0, integrator='ssp_rk3',
                  flux='central', n_face_quad=1,
                  label='C: 1st-order recon + central flux  (RK3, 1-pt midpoint)')

    # ── Quantitative metrics ─────────────────────────────────────────────
    masks = _shape_masks(x, y)
    print("\n  ──── L1 error vs analytic (= initial) per shape ────")
    print("  shape       Case A (T-MLP-u+DW)   Case B (van Leer)   "
          "Case C (central)   B/A    C/A")
    rows = []
    for name in ('slot', 'cone', 'hump'):
        eA = _l1(case_A, U0_field, V, masks[name])
        eB = _l1(case_B, U0_field, V, masks[name])
        eC = _l1(case_C, U0_field, V, masks[name])
        rA = eB / max(eA, 1e-30); rC = eC / max(eA, 1e-30)
        rows.append((name, eA, eB, eC, rA, rC))
        print(f"  {name:8s}    {eA:.5f}             {eB:.5f}             "
              f"{eC:.5f}            {rA:.2f}   {rC:.2f}")
    eA_total = _l1(case_A, U0_field, V)
    eB_total = _l1(case_B, U0_field, V)
    eC_total = _l1(case_C, U0_field, V)
    rB_t = eB_total / max(eA_total, 1e-30)
    rC_t = eC_total / max(eA_total, 1e-30)
    print(f"  {'TOTAL':8s}    {eA_total:.5f}             {eB_total:.5f}             "
          f"{eC_total:.5f}            {rB_t:.2f}   {rC_t:.2f}")
    rows.append(('TOTAL', eA_total, eB_total, eC_total, rB_t, rC_t))

    over_A = float(np.max(case_A) - init_max);  under_A = float(init_min - np.min(case_A))
    over_B = float(np.max(case_B) - init_max);  under_B = float(init_min - np.min(case_B))
    over_C = float(np.max(case_C) - init_max);  under_C = float(init_min - np.min(case_C))
    drift_A = abs(np.sum(case_A * V) - np.sum(U0_field * V)) / np.sum(U0_field * V)
    drift_B = abs(np.sum(case_B * V) - np.sum(U0_field * V)) / np.sum(U0_field * V)
    drift_C = abs(np.sum(case_C * V) - np.sum(U0_field * V)) / np.sum(U0_field * V)
    print()
    print(f"  range  : Case A φ ∈ [{np.min(case_A):.4f}, {np.max(case_A):.4f}]  "
          f"(over={over_A:+.3e}, under={under_A:+.3e})")
    print(f"  range  : Case B φ ∈ [{np.min(case_B):.4f}, {np.max(case_B):.4f}]  "
          f"(over={over_B:+.3e}, under={under_B:+.3e})")
    print(f"  range  : Case C φ ∈ [{np.min(case_C):.4f}, {np.max(case_C):.4f}]  "
          f"(over={over_C:+.3e}, under={under_C:+.3e})")
    print(f"  ∫φ drift A = {drift_A:.2e},  B = {drift_B:.2e},  C = {drift_C:.2e}")

    # ── Plots ───────────────────────────────────────────────────────────
    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1],
                                triangles=np.array(mesh.cell_nodes))

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.2, 1.2, 0.9],
                          hspace=0.40, wspace=0.30)

    def panel(ax, field, title, vmin=None, vmax=None, cmap='viridis'):
        tcf = ax.tripcolor(triang, field, shading='flat',
                           vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal');  ax.set_xticks([]);  ax.set_yticks([])
        plt.colorbar(tcf, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, U0_field,    'initial φ₀',
                                          vmin=0, vmax=1)
    ax = fig.add_subplot(gs[0, 1]); panel(ax, case_A,
                                          'Case A: T-MLP-u + downwind\n(vertex2, k=3, RK3, 2-pt GQ)',
                                          vmin=0, vmax=1)
    ax = fig.add_subplot(gs[0, 2]); panel(ax, case_B,
                                          'Case B: plain van Leer\n(vertex2, k=3, RK3, 2-pt GQ)',
                                          vmin=0, vmax=1)
    ax = fig.add_subplot(gs[0, 3]); panel(ax, case_C,
                                          'Case C: 1st-order + central flux\n(no upwind, no limiter)',
                                          vmin=-0.5, vmax=1.5, cmap='RdBu_r')
    ax = fig.add_subplot(gs[0, 4]); panel(ax, case_B - case_A,
                                          'B − A',
                                          vmin=-0.5, vmax=0.5, cmap='RdBu_r')

    # Error fields (vs exact = φ₀)
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, case_A - U0_field, 'A − exact', vmin=-0.5, vmax=0.5, cmap='RdBu_r')
    ax = fig.add_subplot(gs[1, 2])
    panel(ax, case_B - U0_field, 'B − exact', vmin=-0.5, vmax=0.5, cmap='RdBu_r')
    ax = fig.add_subplot(gs[1, 3])
    panel(ax, case_C - U0_field, 'C − exact', vmin=-0.5, vmax=0.5, cmap='RdBu_r')

    # Bar chart of L1 errors per shape
    ax = fig.add_subplot(gs[1, 0])
    shapes = [r[0] for r in rows[:-1]]
    eA_arr = [r[1] for r in rows[:-1]]
    eB_arr = [r[2] for r in rows[:-1]]
    eC_arr = [r[3] for r in rows[:-1]]
    xpos = np.arange(len(shapes))
    ax.bar(xpos - 0.27, eA_arr, 0.27, label='A: T-MLP-u+DW',  color='tab:blue')
    ax.bar(xpos + 0.00, eB_arr, 0.27, label='B: van Leer',     color='tab:orange')
    ax.bar(xpos + 0.27, eC_arr, 0.27, label='C: central',      color='tab:red')
    ax.set_xticks(xpos);  ax.set_xticklabels(shapes)
    ax.set_ylabel('L1 error');  ax.set_title('L1 error per shape', fontsize=10)
    ax.legend(fontsize=7);  ax.grid(alpha=0.3)

    # 1D slices through each shape
    # Slot at x = 0.5  → vs y
    # Cone at x = 0.5 → vs y
    # Hump at y = 0.5 → vs x
    # Use a vertical strip for x≈0.5 and horizontal strip for y≈0.5.
    eps = 1.5 / N
    strip_x = (np.abs(x - 0.5) < eps)
    strip_y = (np.abs(y - 0.5) < eps)

    ax_slot = fig.add_subplot(gs[2, 0])
    ax_cone = fig.add_subplot(gs[2, 1])
    ax_hump = fig.add_subplot(gs[2, 2])

    if np.any(strip_x):
        order = np.argsort(y[strip_x])
        ys = y[strip_x][order]
        ax_slot.plot(ys, U0_field[strip_x][order], 'k-',  lw=1.6, label='exact')
        ax_slot.plot(ys, case_A   [strip_x][order], 'b-',  lw=1.0, label='A: T-MLP-u+DW')
        ax_slot.plot(ys, case_B   [strip_x][order], 'r-',  lw=1.0, label='B: van Leer')
        ax_slot.plot(ys, case_C   [strip_x][order], 'g--', lw=0.8, alpha=0.6, label='C: central')
        ax_slot.set_xlim(0.55, 0.95);  ax_slot.set_ylim(-0.1, 1.2)
        ax_slot.set_title('slotted cylinder slice (x ≈ ½)', fontsize=10)
        ax_slot.set_xlabel('y');  ax_slot.legend(fontsize=8)
        ax_slot.grid(alpha=0.3)

        order = np.argsort(y[strip_x])
        ax_cone.plot(ys, U0_field[strip_x][order], 'k-',  lw=1.6, label='exact')
        ax_cone.plot(ys, case_A   [strip_x][order], 'b-',  lw=1.0, label='A: T-MLP-u+DW')
        ax_cone.plot(ys, case_B   [strip_x][order], 'r-',  lw=1.0, label='B: van Leer')
        ax_cone.plot(ys, case_C   [strip_x][order], 'g--', lw=0.8, alpha=0.6, label='C: central')
        ax_cone.set_xlim(0.05, 0.45);  ax_cone.set_ylim(-0.1, 1.2)
        ax_cone.set_title('cone slice (x ≈ ½)', fontsize=10)
        ax_cone.set_xlabel('y');  ax_cone.legend(fontsize=8)
        ax_cone.grid(alpha=0.3)

    if np.any(strip_y):
        order = np.argsort(x[strip_y])
        xs = x[strip_y][order]
        ax_hump.plot(xs, U0_field[strip_y][order], 'k-',  lw=1.6, label='exact')
        ax_hump.plot(xs, case_A   [strip_y][order], 'b-',  lw=1.0, label='A: T-MLP-u+DW')
        ax_hump.plot(xs, case_B   [strip_y][order], 'r-',  lw=1.0, label='B: van Leer')
        ax_hump.plot(xs, case_C   [strip_y][order], 'g--', lw=0.8, alpha=0.6, label='C: central')
        ax_hump.set_xlim(0.05, 0.45);  ax_hump.set_ylim(-0.05, 0.6)
        ax_hump.set_title('cosine-bell slice (y ≈ ½)', fontsize=10)
        ax_hump.set_xlabel('x');  ax_hump.legend(fontsize=8)
        ax_hump.grid(alpha=0.3)

    # Caption / summary
    ax_caption = fig.add_subplot(gs[2, 3:]);  ax_caption.axis('off')
    txt = (f"N = {N}, mesh = {mesh.n_cells} triangles\n"
           f"A,B: SSP-RK3, 2-pt GQ, k=3 cubic, vertex2 stencil\n"
           f"C: SSP-RK3, 1-pt midpoint, 1st-order recon, central flux\n\n"
           f"L1 (slot)  A→{rows[0][1]:.4f}, B→{rows[0][2]:.4f}, C→{rows[0][3]:.4f}\n"
           f"L1 (cone)  A→{rows[1][1]:.4f}, B→{rows[1][2]:.4f}, C→{rows[1][3]:.4f}\n"
           f"L1 (hump)  A→{rows[2][1]:.4f}, B→{rows[2][2]:.4f}, C→{rows[2][3]:.4f}\n"
           f"L1 (total) A→{rows[3][1]:.4f}, B→{rows[3][2]:.4f}, C→{rows[3][3]:.4f}\n\n"
           f"Case A range: [{np.min(case_A):.3f}, {np.max(case_A):.3f}]\n"
           f"Case B range: [{np.min(case_B):.3f}, {np.max(case_B):.3f}]\n"
           f"Case C range: [{np.min(case_C):.3f}, {np.max(case_C):.3f}]\n"
           f"∫φ drift  A {drift_A:.1e}, B {drift_B:.1e}, C {drift_C:.1e}")
    ax_caption.text(0.0, 0.95, txt, fontsize=9, family='monospace', va='top')

    fig.suptitle("LeVeque rigid-rotation benchmark — "
                 "A: T-MLP-u+downwind   B: plain van Leer   C: 1st-order + central flux",
                 fontsize=11)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '2d_leveque_rotation.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
