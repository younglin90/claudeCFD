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
from concurrent.futures import ProcessPoolExecutor, as_completed

from _pkgshim import setup_paths
setup_paths()

from mesh import criss_cross_box
from equations import Advection
from reconstruction import TMLPU
from boundary import BoundaryCondition
from solver import solve


# ─── LeVeque velocity field — module-level so child processes can pickle ───
def _leveque_velocity(x, y):
    """Rigid rotation about (½, ½), period T = 1."""
    return (-2.0 * np.pi * (y - 0.5),
             2.0 * np.pi * (x - 0.5))


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
    eq = Advection(velocity=_leveque_velocity)
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


# ─── Parallel worker — runs a single case in its own process ─────────────
def _run_case_worker(spec):
    """Worker for ProcessPoolExecutor.

    spec: dict with keys
        case_id   — 'A' / 'B' / 'C' / 'D'
        N         — mesh resolution (mesh built locally so each process
                    gets its own pristine cache)
        recon     — either a string ('first_order') or a TMLPU kwargs dict
        flux      — flux name
        integrator— integrator name
        n_face_quad — int
        cfl       — float
        t_end     — float
        face_velocity_mode — 'analytic' or 'central_avg'
    Returns: dict(case_id, U_final, n_steps, wall_s, label)
    """
    setup_paths()
    from mesh import criss_cross_box as _ccb
    from solver import solve as _solve
    from reconstruction import TMLPU as _TMLPU
    from equations import Advection as _Adv
    from boundary import BoundaryCondition as _BC

    mesh = _ccb(spec['N'], L=1.0)
    eq = _Adv(velocity=_leveque_velocity)
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    U0 = phi0(x, y)[None, :]
    bc = {p: _BC('dirichlet', state=(0.0,)) for p in mesh.bc_patches}

    if isinstance(spec['recon'], str):
        recon = spec['recon']
    else:
        recon = _TMLPU(**spec['recon'])

    t0 = time.time()
    try:
        res = _solve(mesh, eq, U0,
                     reconstruction=recon, flux=spec['flux'],
                     integrator=spec['integrator'], bc=bc,
                     cfl=spec.get('cfl', 0.4),
                     n_face_quad=spec['n_face_quad'],
                     face_velocity_mode=spec.get('face_velocity_mode', 'analytic'),
                     t_end=spec['t_end'], max_steps=50_000)
        wall = time.time() - t0
        out = res['U_final'][0]
        n_steps = res['n_steps']
    except FloatingPointError:
        wall = time.time() - t0
        out = np.zeros_like(phi0(x, y))
        n_steps = -1
    return dict(case_id=spec['case_id'], U_final=out,
                n_steps=n_steps, wall_s=wall, label=spec.get('label', ''))


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

    # ── Parallel: 4 cases run in their own processes via ProcessPoolExecutor.
    # Each worker rebuilds the mesh + recon (mesh-keyed caches are
    # process-local so there's no contention).  Wall-time becomes the
    # max of {Case A, Case B, Case C, Case D} instead of their sum.
    # Iter 23: CFL scan (untried at adaptive 3-tier setup).
    # baseline cfl=0.4.  Try 0.3 (more steps, less per-step error)
    # and 0.5 (fewer steps).
    common3 = dict(tvd='cicsam_co38', tvd_smooth='van_leer',
                   tvd_smooth2='minmod', mlp_bound=True,
                   extremum_relax=True, tvb_M=64.0,
                   smoothness_threshold=0.10,
                   smoothness_threshold2=0.05,
                   virtual_uu_gradient=True, **common)
    case_specs = [
        dict(case_id='A', N=N, recon=dict(**common3),
             flux='upwind', integrator='ssp_rk3', n_face_quad=2,
             cfl=0.4, t_end=1.0, face_velocity_mode='analytic',
             label='A: cfl=0.4 (iter 21 best 0.02008)'),
        dict(case_id='B', N=N, recon=dict(**common3),
             flux='upwind', integrator='ssp_rk3', n_face_quad=2,
             cfl=0.3, t_end=1.0, face_velocity_mode='analytic',
             label='B: cfl=0.3 (more steps)'),
        dict(case_id='C', N=N, recon='first_order',
             flux='central', integrator='ssp_rk3', n_face_quad=1,
             cfl=0.4, t_end=1.0, face_velocity_mode='analytic',
             label='C: 1st-order + central flux (reference)'),
        dict(case_id='D', N=N, recon=dict(**common3),
             flux='upwind', integrator='ssp_rk3', n_face_quad=2,
             cfl=0.5, t_end=1.0, face_velocity_mode='analytic',
             label='D: cfl=0.5 (fewer steps)'),
    ]

    n_workers = min(len(case_specs), os.cpu_count() or 4)
    print(f"  launching {len(case_specs)} cases on {n_workers} processes")
    results = {}
    t_par_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        future_to_id = {ex.submit(_run_case_worker, s): s['case_id']
                        for s in case_specs}
        for fut in as_completed(future_to_id):
            r = fut.result()
            results[r['case_id']] = r
            print(f"  [{r['label']}]  n={r['n_steps']:5d}  "
                  f"wall={r['wall_s']:.1f} s")
    print(f"  parallel wall = {time.time() - t_par_start:.1f} s "
          f"(max of 4 cases, vs sum ≈ {sum(r['wall_s'] for r in results.values()):.0f} s)")

    case_A = results['A']['U_final']
    case_B = results['B']['U_final']
    case_C = results['C']['U_final']
    case_D = results['D']['U_final']

    # ── Quantitative metrics ─────────────────────────────────────────────
    masks = _shape_masks(x, y)
    print("\n  ──── L1 error vs analytic (= initial) per shape ────")
    print("  shape       Case A (T-MLP-u+DW)   Case B (van Leer)   "
          "Case C (central)   Case D (CICSAM)    B/A   C/A   D/A")
    rows = []
    for name in ('slot', 'cone', 'hump'):
        eA = _l1(case_A, U0_field, V, masks[name])
        eB = _l1(case_B, U0_field, V, masks[name])
        eC = _l1(case_C, U0_field, V, masks[name])
        eD = _l1(case_D, U0_field, V, masks[name])
        rB = eB / max(eA, 1e-30)
        rC = eC / max(eA, 1e-30)
        rD = eD / max(eA, 1e-30)
        rows.append((name, eA, eB, eC, eD, rB, rC, rD))
        print(f"  {name:8s}    {eA:.5f}             {eB:.5f}             "
              f"{eC:.5f}            {eD:.5f}             "
              f"{rB:.2f}  {rC:.2f}  {rD:.2f}")
    eA_total = _l1(case_A, U0_field, V)
    eB_total = _l1(case_B, U0_field, V)
    eC_total = _l1(case_C, U0_field, V)
    eD_total = _l1(case_D, U0_field, V)
    rB_t = eB_total / max(eA_total, 1e-30)
    rC_t = eC_total / max(eA_total, 1e-30)
    rD_t = eD_total / max(eA_total, 1e-30)
    print(f"  {'TOTAL':8s}    {eA_total:.5f}             {eB_total:.5f}             "
          f"{eC_total:.5f}            {eD_total:.5f}             "
          f"{rB_t:.2f}  {rC_t:.2f}  {rD_t:.2f}")
    rows.append(('TOTAL', eA_total, eB_total, eC_total, eD_total, rB_t, rC_t, rD_t))

    over_A = float(np.max(case_A) - init_max);  under_A = float(init_min - np.min(case_A))
    over_B = float(np.max(case_B) - init_max);  under_B = float(init_min - np.min(case_B))
    over_C = float(np.max(case_C) - init_max);  under_C = float(init_min - np.min(case_C))
    over_D = float(np.max(case_D) - init_max);  under_D = float(init_min - np.min(case_D))
    drift_A = abs(np.sum(case_A * V) - np.sum(U0_field * V)) / np.sum(U0_field * V)
    drift_B = abs(np.sum(case_B * V) - np.sum(U0_field * V)) / np.sum(U0_field * V)
    drift_C = abs(np.sum(case_C * V) - np.sum(U0_field * V)) / np.sum(U0_field * V)
    drift_D = abs(np.sum(case_D * V) - np.sum(U0_field * V)) / np.sum(U0_field * V)
    print()
    print(f"  range  : Case A φ ∈ [{np.min(case_A):.4f}, {np.max(case_A):.4f}]  "
          f"(over={over_A:+.3e}, under={under_A:+.3e})")
    print(f"  range  : Case B φ ∈ [{np.min(case_B):.4f}, {np.max(case_B):.4f}]  "
          f"(over={over_B:+.3e}, under={under_B:+.3e})")
    print(f"  range  : Case C φ ∈ [{np.min(case_C):.4f}, {np.max(case_C):.4f}]  "
          f"(over={over_C:+.3e}, under={under_C:+.3e})")
    print(f"  range  : Case D φ ∈ [{np.min(case_D):.4f}, {np.max(case_D):.4f}]  "
          f"(over={over_D:+.3e}, under={under_D:+.3e})")
    print(f"  ∫φ drift A = {drift_A:.2e},  B = {drift_B:.2e},  "
          f"C = {drift_C:.2e},  D = {drift_D:.2e}")

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
    ax = fig.add_subplot(gs[0, 4]); panel(ax, case_D,
                                          'Case D: T-MLP-u + CICSAM\n(vertex2, k=3, RK3, 2-pt GQ)',
                                          vmin=0, vmax=1)

    # Error fields (vs exact = φ₀)
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, case_A - U0_field, 'A − exact', vmin=-0.5, vmax=0.5, cmap='RdBu_r')
    ax = fig.add_subplot(gs[1, 2])
    panel(ax, case_B - U0_field, 'B − exact', vmin=-0.5, vmax=0.5, cmap='RdBu_r')
    ax = fig.add_subplot(gs[1, 3])
    panel(ax, case_C - U0_field, 'C − exact', vmin=-0.5, vmax=0.5, cmap='RdBu_r')
    ax = fig.add_subplot(gs[1, 4])
    panel(ax, case_D - U0_field, 'D − exact', vmin=-0.5, vmax=0.5, cmap='RdBu_r')

    # Bar chart of L1 errors per shape
    ax = fig.add_subplot(gs[1, 0])
    shapes = [r[0] for r in rows[:-1]]
    eA_arr = [r[1] for r in rows[:-1]]
    eB_arr = [r[2] for r in rows[:-1]]
    eC_arr = [r[3] for r in rows[:-1]]
    eD_arr = [r[4] for r in rows[:-1]]
    xpos = np.arange(len(shapes))
    ax.bar(xpos - 0.30, eA_arr, 0.20, label='A: T-MLP-u+DW',  color='tab:blue')
    ax.bar(xpos - 0.10, eB_arr, 0.20, label='B: van Leer',     color='tab:orange')
    ax.bar(xpos + 0.10, eC_arr, 0.20, label='C: central',      color='tab:red')
    ax.bar(xpos + 0.30, eD_arr, 0.20, label='D: CICSAM',       color='tab:green')
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
        ax_slot.plot(ys, case_D   [strip_x][order], 'm:',  lw=1.0, label='D: CICSAM')
        ax_slot.set_xlim(0.55, 0.95);  ax_slot.set_ylim(-0.1, 1.2)
        ax_slot.set_title('slotted cylinder slice (x ≈ ½)', fontsize=10)
        ax_slot.set_xlabel('y');  ax_slot.legend(fontsize=8)
        ax_slot.grid(alpha=0.3)

        order = np.argsort(y[strip_x])
        ax_cone.plot(ys, U0_field[strip_x][order], 'k-',  lw=1.6, label='exact')
        ax_cone.plot(ys, case_A   [strip_x][order], 'b-',  lw=1.0, label='A: T-MLP-u+DW')
        ax_cone.plot(ys, case_B   [strip_x][order], 'r-',  lw=1.0, label='B: van Leer')
        ax_cone.plot(ys, case_C   [strip_x][order], 'g--', lw=0.8, alpha=0.6, label='C: central')
        ax_cone.plot(ys, case_D   [strip_x][order], 'm:',  lw=1.0, label='D: CICSAM')
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
        ax_hump.plot(xs, case_D   [strip_y][order], 'm:',  lw=1.0, label='D: CICSAM')
        ax_hump.set_xlim(0.05, 0.45);  ax_hump.set_ylim(-0.05, 0.6)
        ax_hump.set_title('cosine-bell slice (y ≈ ½)', fontsize=10)
        ax_hump.set_xlabel('x');  ax_hump.legend(fontsize=8)
        ax_hump.grid(alpha=0.3)

    # Caption / summary
    ax_caption = fig.add_subplot(gs[2, 3:]);  ax_caption.axis('off')
    txt = (f"N = {N}, mesh = {mesh.n_cells} triangles\n"
           f"A,B,D: SSP-RK3, 2-pt GQ, k=3 cubic, vertex2 stencil, virt-UU\n"
           f"C: SSP-RK3, 1-pt midpoint, 1st-order recon, central flux\n\n"
           f"L1 (total) A→{rows[3][1]:.4f}, B→{rows[3][2]:.4f}, "
           f"C→{rows[3][3]:.4f}, D→{rows[3][4]:.4f}\n\n"
           f"Case A range: [{np.min(case_A):.3f}, {np.max(case_A):.3f}]\n"
           f"Case B range: [{np.min(case_B):.3f}, {np.max(case_B):.3f}]\n"
           f"Case C range: [{np.min(case_C):.3f}, {np.max(case_C):.3f}]\n"
           f"Case D range: [{np.min(case_D):.3f}, {np.max(case_D):.3f}]\n"
           f"∫φ drift  A {drift_A:.1e}, B {drift_B:.1e}, "
           f"C {drift_C:.1e}, D {drift_D:.1e}")
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
