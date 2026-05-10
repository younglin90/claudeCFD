"""2D paper-evidence harness for T-MLP-u.

The script writes JSON/TSV/PNG artifacts under ``results/T-MLP-u`` and
prints a final JSON metrics line for codex-autoresearch.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from _pkgshim import setup_paths
setup_paths()

from mesh import criss_cross_box, build_unstructured_2d
from equations import Advection, Euler2D
from reconstruction import TMLPU, BarthJespersen, Venkatakrishnan, MLPU1, MLPU2
from boundary import BoundaryCondition
from solver import solve


def _repo_root():
    here = Path(__file__).resolve()
    return here.parents[3]


def _out_dir():
    out = _repo_root() / 'results' / 'T-MLP-u'
    out.mkdir(parents=True, exist_ok=True)
    return out


DOUBLE_MACH_QUICK_GRID = (80, 20)
DOUBLE_MACH_PAPER_GRID = (480, 120)
DOUBLE_MACH_FINE_GRID = (960, 240)
MACH3_STEP_QUICK_GRID = (90, 30)
MACH3_STEP_PAPER_GRID = (240, 80)
MACH3_STEP_FINE_GRID = (480, 160)
LEVEQUE_QUICK_N = 18
LEVEQUE_PAPER_N = 100


def _box_triangle_count(nx, ny):
    return 2 * nx * ny


def _mach3_step_triangle_count(nx, ny, Lx=3.0, Ly=1.0,
                               step_x=0.6, step_h=0.2):
    dx = Lx / nx
    dy = Ly / ny
    kept = 0
    for j in range(ny):
        for i in range(nx):
            cx = (i + 0.5) * dx
            cy = (j + 0.5) * dy
            if not (cx >= step_x and cy < step_h):
                kept += 1
    return 2 * kept


def _triangles(mesh):
    tris = []
    owners = []
    for ci, cell in enumerate(mesh.cell_nodes):
        if len(cell) == 3:
            tris.append(cell)
            owners.append(ci)
        elif len(cell) == 4:
            tris.append((cell[0], cell[1], cell[2]))
            owners.append(ci)
            tris.append((cell[0], cell[2], cell[3]))
            owners.append(ci)
    return np.asarray(tris, dtype=int), np.asarray(owners, dtype=int)


def _plot_field(mesh, field, path, title, vmin=None, vmax=None):
    tris, tri_owner = _triangles(mesh)
    tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1],
                             triangles=tris)
    colors = field
    if len(field) == mesh.n_cells and len(tri_owner) != mesh.n_cells:
        colors = np.asarray(field)[tri_owner]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.0))
    tcf = ax.tripcolor(tri, colors, shading='flat', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    plt.colorbar(tcf, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _node_values_from_cells(mesh, cell_values):
    vals = np.asarray(cell_values, dtype=float)
    sums = np.zeros(mesh.nodes.shape[0], dtype=float)
    counts = np.zeros(mesh.nodes.shape[0], dtype=float)
    for c, nodes in enumerate(mesh.cell_nodes):
        np.add.at(sums, nodes, vals[c])
        np.add.at(counts, nodes, 1.0)
    fallback = float(np.nanmean(vals)) if vals.size else 0.0
    return np.where(counts > 0.0, sums / np.maximum(counts, 1.0), fallback)


def _format_metric(row, key):
    val = row.get(key)
    if isinstance(val, (int, float)) and np.isfinite(val):
        return f"{key}={val:.3g}"
    return None


def _short_error(row):
    err = row.get('error') or ''
    if not err:
        return ''
    err = err.replace('FloatingPointError', 'FPE')
    return err[:58] + ('...' if len(err) > 58 else '')


def _plot_scheme_contours(mesh, fields, rows, path, title, *,
                          vmin=None, vmax=None, metric_keys=()):
    tris, _ = _triangles(mesh)
    tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1],
                             triangles=tris)
    valid_fields = [np.asarray(f, dtype=float) for f in fields.values()
                    if np.all(np.isfinite(f))]
    if vmin is None or vmax is None:
        all_vals = np.concatenate(valid_fields) if valid_fields else np.array([0.0, 1.0])
        if vmin is None:
            vmin = float(np.min(all_vals))
        if vmax is None:
            vmax = float(np.max(all_vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    levels = np.linspace(vmin, vmax, 32)
    line_levels = np.linspace(vmin, vmax, 9)

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.1), constrained_layout=True)
    axes = axes.ravel()
    mappable = None
    for ax, row in zip(axes, rows):
        method = row['method']
        field = fields.get(method)
        metric_text = ', '.join(
            x for x in (_format_metric(row, k) for k in metric_keys) if x)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if field is None:
            ax.set_title(method, fontsize=9)
            ax.text(0.5, 0.56, 'DIVERGED', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, weight='bold')
            ax.text(0.5, 0.42, _short_error(row), transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, wrap=True)
            ax.set_facecolor('#f2f2f2')
            continue
        node_vals = _node_values_from_cells(mesh, field)
        mappable = ax.tricontourf(tri, node_vals, levels=levels,
                                  cmap='viridis', extend='both')
        ax.tricontour(tri, node_vals, levels=line_levels,
                      colors='k', linewidths=0.18, alpha=0.35)
        ax.set_title(f"{method}\n{metric_text}", fontsize=8)
    for ax in axes[len(rows):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes.tolist(), shrink=0.82,
                     pad=0.01, fraction=0.025)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _metric(row, key):
    val = row.get(key)
    if isinstance(val, (int, float)) and np.isfinite(val):
        return float(val)
    return np.nan


def _plot_summary(rows, path):
    methods = [name for name, _ in _comparison_suite('leveque')]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    axes = axes.ravel()
    specs = [
        ('leveque', 'l1', 'lower is better', 'LeVeque L1'),
        ('leveque', 'sharpness', 'higher is sharper', 'LeVeque sharpness'),
        ('double_mach', 'vortex_proxy', 'higher is sharper', 'Double Mach vortices'),
        ('double_mach', 'checker', 'lower is smoother', 'Double Mach checker'),
        ('mach3_step', 'flag_proxy', 'higher is stronger', 'Mach 3 flag-waving'),
        ('mach3_step', 'carbuncle', 'lower is cleaner', 'Mach 3 carbuncle'),
    ]
    colors = ['#707070', '#4c78a8', '#59a14f', '#f28e2b',
              '#b07aa1', '#e15759', '#111111']
    row_map = {(r['case'], r['method']): r for r in rows}
    x = np.arange(len(methods))
    labels = ['1st', 'BJ', 'Venkat', 'MLP-u1', 'MLP-u2', 'T off', 'T on']
    for ax, (case, key, ylabel, title) in zip(axes, specs):
        vals = [_metric(row_map.get((case, method), {}), key)
                for method in methods]
        bars = ax.bar(x, np.nan_to_num(vals, nan=0.0), color=colors,
                      width=0.78)
        for bar, val in zip(bars, vals):
            if not np.isfinite(val):
                bar.set_hatch('//')
                bar.set_alpha(0.35)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.generic):
        return _json_ready(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _quad_mesh(nx, ny, Lx, Ly, *, origin=(0.0, 0.0), keep=None,
               classifier=None, patches=('x_min', 'x_max', 'y_min', 'y_max')):
    x0, y0 = origin
    dx = Lx / nx
    dy = Ly / ny
    nodes = []
    node_id = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            node_id[(i, j)] = len(nodes)
            nodes.append((x0 + i * dx, y0 + j * dy))
    elems = []
    for j in range(ny):
        for i in range(nx):
            cx = x0 + (i + 0.5) * dx
            cy = y0 + (j + 0.5) * dy
            if keep is not None and not keep(cx, cy):
                continue
            elems.append((node_id[(i, j)], node_id[(i + 1, j)],
                          node_id[(i + 1, j + 1)], node_id[(i, j + 1)]))

    def default_classifier(center, normal):
        cx, cy = float(center[0]), float(center[1])
        if cx <= x0 + 1e-9 * Lx:
            return 1
        if cx >= x0 + Lx - 1e-9 * Lx:
            return 2
        if cy <= y0 + 1e-9 * Ly:
            return 3
        if cy >= y0 + Ly - 1e-9 * Ly:
            return 4
        return len(patches)

    return build_unstructured_2d(
        np.asarray(nodes, dtype=float), elems,
        boundary_classifier=classifier or default_classifier,
        bc_patches=patches)


def _tri_mesh(nx, ny, Lx, Ly, *, origin=(0.0, 0.0), keep=None,
              classifier=None, patches=('x_min', 'x_max', 'y_min', 'y_max')):
    """Alternating-diagonal triangular mesh on a logical Cartesian layout."""
    x0, y0 = origin
    dx = Lx / nx
    dy = Ly / ny
    nodes = []
    node_id = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            node_id[(i, j)] = len(nodes)
            nodes.append((x0 + i * dx, y0 + j * dy))
    elems = []
    for j in range(ny):
        for i in range(nx):
            cx = x0 + (i + 0.5) * dx
            cy = y0 + (j + 0.5) * dy
            if keep is not None and not keep(cx, cy):
                continue
            n00 = node_id[(i, j)]
            n10 = node_id[(i + 1, j)]
            n11 = node_id[(i + 1, j + 1)]
            n01 = node_id[(i, j + 1)]
            if (i + j) % 2 == 0:
                elems.append((n00, n10, n11))
                elems.append((n00, n11, n01))
            else:
                elems.append((n00, n10, n01))
                elems.append((n10, n11, n01))

    def default_classifier(center, normal):
        cx, cy = float(center[0]), float(center[1])
        if cx <= x0 + 1e-9 * Lx:
            return 1
        if cx >= x0 + Lx - 1e-9 * Lx:
            return 2
        if cy <= y0 + 1e-9 * Ly:
            return 3
        if cy >= y0 + Ly - 1e-9 * Ly:
            return 4
        return len(patches)

    return build_unstructured_2d(
        np.asarray(nodes, dtype=float), elems,
        boundary_classifier=classifier or default_classifier,
        bc_patches=patches)


def _tmlpu_leveque():
    return TMLPU(tvd='pure_downwind', mlp_bound=True,
                 extremum_relax=False, tvb_M=0.0,
                 virtual_uu_gradient=True, stencil='vertex',
                 order=1, idw_p=6, vertex_mlp=True,
                 vertex_mlp_cap=2.0)


def _tmlpu_off_leveque():
    return TMLPU(tvd='pure_downwind', mlp_bound=False,
                 extremum_relax=False, tvb_M=0.0,
                 virtual_uu_gradient=True, stencil='vertex2',
                 order=3, idw_p=6)


def _tmlpu_euler():
    return TMLPU(tvd='superbee', mlp_bound=True, extremum_relax=False,
                 tvb_M=0.0, vertex_mlp=True, vertex_mlp_cap=2.0,
                 virtual_uu_gradient=True, stencil='vertex', order=1)


def _tmlpu_off_euler():
    return TMLPU(tvd='superbee', mlp_bound=False, extremum_relax=False,
                 tvb_M=0.0, vertex_mlp=False,
                 virtual_uu_gradient=True, stencil='face', order=1)


def _comparison_specs(kind):
    if kind == 'leveque':
        t_on = 'tmlpu_leveque_on'
        t_off = 'tmlpu_leveque_off'
    else:
        t_on = 'tmlpu_euler_on'
        t_off = 'tmlpu_euler_off'
    return [
        ('first_order', 'first_order'),
        ('Barth-Jespersen', 'barth_jespersen'),
        ('Venkatakrishnan', 'venkatakrishnan'),
        ('MLP-u1', 'mlp_u1'),
        ('MLP-u2', 'mlp_u2'),
        ('T-MLP-u OFF', t_off),
        ('T-MLP-u ON', t_on),
    ]


def _reconstruction_from_key(key):
    if key == 'first_order':
        return 'first_order'
    if key == 'barth_jespersen':
        return BarthJespersen(stencil='face')
    if key == 'venkatakrishnan':
        return Venkatakrishnan(stencil='face')
    if key == 'mlp_u1':
        return MLPU1()
    if key == 'mlp_u2':
        return MLPU2()
    if key == 'tmlpu_leveque_on':
        return _tmlpu_leveque()
    if key == 'tmlpu_leveque_off':
        return _tmlpu_off_leveque()
    if key == 'tmlpu_euler_on':
        return _tmlpu_euler()
    if key == 'tmlpu_euler_off':
        return _tmlpu_off_euler()
    raise ValueError(f"unknown reconstruction key {key!r}")


def _comparison_suite(kind):
    return [(name, _reconstruction_from_key(key))
            for name, key in _comparison_specs(kind)]


def _double_mach_states():
    gamma = 1.4
    pre = (1.4, 0.0, 0.0, 1.0)
    post = (8.0,
            8.25 * np.cos(np.pi / 6.0),
            -8.25 * np.sin(np.pi / 6.0),
            116.5)
    return gamma, pre, post


def _double_mach_exact_state(point, time=0.0):
    _, pre, post = _double_mach_states()
    x, y = float(point[0]), float(point[1])
    shock_x = 1.0 / 6.0 + (y + 20.0 * float(time)) / np.sqrt(3.0)
    return post if x < shock_x else pre


def _leveque_phi0(x, y):
    r0 = 0.15
    r1 = np.sqrt((x - 0.5) ** 2 + (y - 0.75) ** 2) / r0
    in_slot = (np.abs(x - 0.5) < 0.025) & (y < 0.85)
    phi_slot = np.where((r1 <= 1.0) & ~in_slot, 1.0, 0.0)
    r2 = np.sqrt((x - 0.5) ** 2 + (y - 0.25) ** 2) / r0
    phi_cone = np.where(r2 <= 1.0, 1.0 - r2, 0.0)
    r3 = np.sqrt((x - 0.25) ** 2 + (y - 0.5) ** 2) / r0
    phi_hump = np.where(r3 <= 1.0, 0.25 * (1.0 + np.cos(np.pi * r3)), 0.0)
    return phi_slot + phi_cone + phi_hump


def _rotation_velocity(x, y):
    return (-2.0 * np.pi * (y - 0.5),
             2.0 * np.pi * (x - 0.5))


def _run_safely(mesh, eq, U0, recon, bc, *, flux, integrator, cfl,
                t_end, n_face_quad=1, face_velocity_mode='analytic'):
    t0 = time.time()
    try:
        res = solve(mesh, eq, U0, reconstruction=recon, flux=flux,
                    integrator=integrator, bc=bc, cfl=cfl, t_end=t_end,
                    max_steps=500_000, n_face_quad=n_face_quad,
                    face_velocity_mode=face_velocity_mode)
        wall = time.time() - t0
        U = res['U_final']
        W = eq.cons_to_prim(U)
        finite = bool(np.all(np.isfinite(W)))
        positive = True
        if getattr(eq, 'prim_names', ()) and 'rho' in eq.prim_names:
            positive = bool(np.min(W[0]) > 0.0 and np.min(W[-1]) > 0.0)
        return dict(ok=finite and positive, U=U, W=W, steps=res['n_steps'],
                    wall=wall, error=None)
    except Exception as exc:
        return dict(ok=False, U=None, W=None, steps=-1,
                    wall=time.time() - t0, error=repr(exc))


def run_leveque(out, quick, workers=1):
    N = LEVEQUE_QUICK_N if quick else LEVEQUE_PAPER_N
    mesh = criss_cross_box(N, L=1.0)
    eq = Advection(velocity=_rotation_velocity)
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    V = mesh.cell_volumes
    exact = _leveque_phi0(x, y)
    U0 = exact[None, :]
    bc = {p: BoundaryCondition('dirichlet', state=(0.0,))
          for p in mesh.bc_patches}
    rows, fields = _run_case_methods('leveque', quick, 'leveque', workers)
    on = next(r for r in rows if r['method'] == 'T-MLP-u ON')
    off = next(r for r in rows if r['method'] == 'T-MLP-u OFF')
    baselines = [r for r in rows if r['method'] not in ('T-MLP-u ON', 'T-MLP-u OFF') and r['ok']]
    best_l1 = min((r['l1'] for r in baselines), default=float('inf'))
    best_sharp = max((r['sharpness'] for r in baselines), default=0.0)

    def _range_violation(row):
        if row['range_min'] is None or row['range_max'] is None:
            return float('inf')
        return max(0.0, -row['range_min']) + max(0.0, row['range_max'] - 1.0)

    on_range_violation = _range_violation(on)
    off_range_violation = _range_violation(off)
    passed = bool(on['ok']
                  and on['sharpness'] >= 1.25 * best_sharp
                  and on['l1'] <= 1.10 * best_l1
                  and on_range_violation <= 1e-8
                  and on_range_violation <= off_range_violation)
    if 'T-MLP-u ON' in fields:
        _plot_field(mesh, fields['T-MLP-u ON'], out / 'leveque_tmlpu_on.png',
                    f'LeVeque T-MLP-u ON N={N}', vmin=0, vmax=1)
    _plot_scheme_contours(
        mesh, fields, rows, out / 'leveque_scheme_contours.png',
        f'LeVeque rotation: all schemes N={N}',
        vmin=0.0, vmax=1.0, metric_keys=('l1', 'sharpness', 'wiggle'))
    return passed, rows


def _euler_state(gamma, rho, u, v, p):
    eq = Euler2D(gamma=gamma)
    W = np.asarray([rho, u, v, p], dtype=float)
    return eq.prim_to_cons(W[:, None])[:, 0], W


def _density_jump_score(mesh, rho, mask=None):
    own = mesh.face_owner
    nei = mesh.face_neighbour
    interior = nei >= 0
    if mask is not None:
        fc = mesh.face_centers
        interior = interior & mask(fc[:, 0], fc[:, 1])
    if not np.any(interior):
        return 0.0
    return float(np.percentile(np.abs(rho[own[interior]] - rho[nei[interior]]), 95))


def _checker_score(mesh, val):
    own = mesh.face_owner
    nei = mesh.face_neighbour
    interior = nei >= 0
    if not np.any(interior):
        return 0.0
    jump = np.abs(val[own[interior]] - val[nei[interior]])
    scale = np.percentile(np.abs(val), 95) + 1e-30
    return float(np.percentile(jump, 95) / scale)


def _cell_gradient_scalar(mesh, values):
    values = np.asarray(values, dtype=float)
    centers = mesh.cell_centers
    grad = np.zeros((mesh.n_cells, 2), dtype=float)
    for ci, neighbours in enumerate(mesh.cell_neighbours):
        nb = np.asarray([n for n in neighbours if n >= 0], dtype=int)
        if nb.size < 2:
            continue
        A = centers[nb] - centers[ci]
        b = values[nb] - values[ci]
        finite = np.isfinite(A).all(axis=1) & np.isfinite(b)
        A = A[finite]
        b = b[finite]
        if A.shape[0] < 2:
            continue
        scale = np.linalg.norm(A, axis=1)
        w = 1.0 / np.maximum(scale, 1e-30)
        Aw = A * w[:, None]
        bw = b * w
        try:
            grad[ci] = np.linalg.lstsq(Aw, bw, rcond=None)[0]
        except np.linalg.LinAlgError:
            grad[ci] = 0.0
    return grad


def _vorticity_metrics(mesh, u, v, mask=None):
    du = _cell_gradient_scalar(mesh, u)
    dv = _cell_gradient_scalar(mesh, v)
    omega = dv[:, 0] - du[:, 1]
    if mask is not None:
        c = mesh.cell_centers
        active = mask(c[:, 0], c[:, 1])
    else:
        active = np.ones(mesh.n_cells, dtype=bool)
    active = active & np.isfinite(omega)
    if not np.any(active):
        return 0.0, 0.0
    vals = np.abs(omega[active])
    return (float(np.percentile(vals, 95)),
            float(np.mean(vals * vals)))


def _run_case_method_worker(payload):
    case = payload['case']
    quick = payload['quick']
    order = payload['order']
    name = payload['name']
    recon = _reconstruction_from_key(payload['recon_key'])
    try:
        if case == 'leveque':
            N = LEVEQUE_QUICK_N if quick else LEVEQUE_PAPER_N
            mesh = criss_cross_box(N, L=1.0)
            eq = Advection(velocity=_rotation_velocity)
            x = mesh.cell_centers[:, 0]
            y = mesh.cell_centers[:, 1]
            V = mesh.cell_volumes
            exact = _leveque_phi0(x, y)
            U0 = exact[None, :]
            bc = {p: BoundaryCondition('dirichlet', state=(0.0,))
                  for p in mesh.bc_patches}
            r = _run_safely(mesh, eq, U0, recon, bc, flux='upwind',
                            integrator='ssp_rk3', cfl=0.4, t_end=1.0,
                            n_face_quad=2,
                            face_velocity_mode='central_avg')
            if r['ok']:
                f = r['W'][0]
                l1 = float(np.sum(np.abs(f - exact) * V) / np.sum(V))
                interior = mesh.face_neighbour >= 0
                jump = np.abs(f[mesh.face_owner[interior]]
                              - f[mesh.face_neighbour[interior]])
                sharp = float(np.percentile(jump, 95))
                rng = [float(np.min(f)), float(np.max(f))]
                wiggle = max(0.0, -rng[0]) + max(0.0, rng[1] - 1.0)
                row_ok = bool(wiggle <= 0.25 and np.isfinite(l1))
                field = f if row_ok else None
            else:
                l1 = None
                sharp = 0.0
                rng = [None, None]
                wiggle = None
                row_ok = False
                field = None
            error = r['error']
            if r['ok'] and not row_ok:
                error = f"range blow-up: min={rng[0]:.3g}, max={rng[1]:.3g}"
            row = dict(case='leveque', method=name, ok=row_ok,
                       mesh='criss_cross_triangles', logical_nx=N,
                       logical_ny=N, mesh_cells=mesh.n_cells,
                       mesh_faces=mesh.n_faces, l1=l1,
                       sharpness=sharp, range_min=rng[0],
                       range_max=rng[1], wiggle=wiggle,
                       steps=r['steps'], wall=r['wall'], error=error)
            return order, name, row, field

        if case == 'double_mach':
            gamma, pre, post_state = _double_mach_states()
            nx, ny = DOUBLE_MACH_QUICK_GRID if quick else DOUBLE_MACH_PAPER_GRID
            Lx, Ly = 4.0, 1.0

            def classify(center, normal):
                cx, cy = float(center[0]), float(center[1])
                if cx <= 1e-9 * Lx:
                    return 1
                if cx >= Lx - 1e-9 * Lx:
                    return 2
                if cy <= 1e-9 * Ly:
                    return 3 if cx <= 1.0 / 6.0 + 1e-12 else 4
                if cy >= Ly - 1e-9 * Ly:
                    return 5
                return 2

            mesh = _tri_mesh(nx, ny, Lx, Ly, classifier=classify,
                             patches=('x_min_postshock', 'x_max_outflow',
                                      'bottom_postshock', 'bottom_reflect',
                                      'top_exact_shock'))
            eq = Euler2D(gamma=gamma)
            rho1, u1, v1, p1 = pre
            rho2, u2, v2, p2 = post_state
            x = mesh.cell_centers[:, 0]
            y = mesh.cell_centers[:, 1]
            shock_x = 1.0 / 6.0 + y / np.sqrt(3.0)
            post = x < shock_x
            W0 = np.vstack([
                np.where(post, rho2, rho1),
                np.where(post, u2, u1),
                np.where(post, v2, v1),
                np.where(post, p2, p1),
            ])
            U0 = eq.prim_to_cons(W0)
            bc = {
                'x_min_postshock': BoundaryCondition('dirichlet', state=post_state),
                'x_max_outflow': BoundaryCondition('transmissive'),
                'bottom_postshock': BoundaryCondition('dirichlet', state=post_state),
                'bottom_reflect': BoundaryCondition('reflective'),
                'top_exact_shock': BoundaryCondition(
                    'dirichlet_func', state=_double_mach_exact_state),
            }
            r = _run_safely(mesh, eq, U0, recon, bc, flux='hllc_adc',
                            integrator='ssp_rk2', cfl=0.35, t_end=0.2)
            if r['ok']:
                W = r['W']
                rho = W[0]
                vortex = _density_jump_score(
                    mesh, rho, mask=lambda xx, yy: (xx > 2.0) & (yy < 0.45))
                vort_p95, enstrophy = _vorticity_metrics(
                    mesh, W[1], W[2],
                    mask=lambda xx, yy: (xx > 2.0) & (yy < 0.45))
                checker = _checker_score(mesh, W[3])
                field = rho
            else:
                vortex = 0.0
                vort_p95 = 0.0
                enstrophy = 0.0
                checker = None
                field = None
            row = dict(case='double_mach', method=name, ok=r['ok'],
                       mesh='tri_alternating', logical_nx=nx,
                       logical_ny=ny, mesh_cells=mesh.n_cells,
                       mesh_faces=mesh.n_faces,
                       vortex_proxy=vortex, vorticity_p95=vort_p95,
                       enstrophy_proxy=enstrophy, checker=checker,
                       steps=r['steps'], wall=r['wall'], error=r['error'])
            return order, name, row, field

        if case == 'mach3_step':
            gamma = 1.4
            nx, ny = MACH3_STEP_QUICK_GRID if quick else MACH3_STEP_PAPER_GRID
            Lx, Ly = 3.0, 1.0
            step_x = 0.6
            step_h = 0.2

            def keep(cx, cy):
                return not (cx >= step_x and cy < step_h)

            def classify(center, normal):
                cx = float(center[0])
                if cx <= 1e-9 * Lx:
                    return 1
                if cx >= Lx - 1e-9 * Lx:
                    return 2
                return 3

            mesh = _tri_mesh(nx, ny, Lx, Ly, keep=keep, classifier=classify,
                             patches=('inflow', 'outflow', 'wall'))
            eq = Euler2D(gamma=gamma)
            rho, p = 1.4, 1.0
            c = np.sqrt(gamma * p / rho)
            u, v = 3.0 * c, 0.0
            W0 = np.vstack([
                np.full(mesh.n_cells, rho),
                np.full(mesh.n_cells, u),
                np.full(mesh.n_cells, v),
                np.full(mesh.n_cells, p),
            ])
            U0 = eq.prim_to_cons(W0)
            bc = {
                'inflow': BoundaryCondition('dirichlet', state=(rho, u, v, p)),
                'outflow': BoundaryCondition('transmissive'),
                'wall': BoundaryCondition('reflective'),
            }
            r = _run_safely(mesh, eq, U0, recon, bc, flux='hllc_adc',
                            integrator='ssp_rk2', cfl=0.35, t_end=4.0)
            if r['ok']:
                W = r['W']
                rho_f = W[0]
                flag = _density_jump_score(
                    mesh, rho_f, mask=lambda xx, yy: (xx > step_x) & (yy < 0.55))
                flag_vort, flag_enstrophy = _vorticity_metrics(
                    mesh, W[1], W[2],
                    mask=lambda xx, yy: (xx > step_x) & (yy < 0.55))
                cells = mesh.cell_centers
                flag_mask = (cells[:, 0] > step_x) & (cells[:, 1] < 0.55)
                transverse = float(np.sqrt(np.mean(W[2, flag_mask] ** 2))
                                   if np.any(flag_mask) else 0.0)
                carbuncle = _checker_score(mesh, W[3])
                field = rho_f
            else:
                flag = 0.0
                flag_vort = 0.0
                flag_enstrophy = 0.0
                transverse = 0.0
                carbuncle = None
                field = None
            row = dict(case='mach3_step', method=name, ok=r['ok'],
                       mesh='tri_alternating', logical_nx=nx,
                       logical_ny=ny, mesh_cells=mesh.n_cells,
                       mesh_faces=mesh.n_faces,
                       flag_proxy=flag, flag_vorticity_p95=flag_vort,
                       flag_enstrophy_proxy=flag_enstrophy,
                       transverse_velocity_rms=transverse,
                       carbuncle=carbuncle,
                       steps=r['steps'], wall=r['wall'], error=r['error'])
            return order, name, row, field

        raise ValueError(f"unknown case {case!r}")
    except Exception as exc:
        row = dict(case=case, method=name, ok=False, steps=-1,
                   wall=0.0, error=repr(exc))
        return order, name, row, None


def _run_case_methods(case, quick, kind, workers):
    specs = _comparison_specs(kind)
    payloads = [
        dict(case=case, quick=quick, order=i, name=name, recon_key=recon_key)
        for i, (name, recon_key) in enumerate(specs)
    ]
    n_workers = max(1, min(int(workers), len(payloads)))
    if n_workers == 1:
        results = [_run_case_method_worker(p) for p in payloads]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_map = {ex.submit(_run_case_method_worker, p): p
                          for p in payloads}
            for fut in as_completed(future_map):
                results.append(fut.result())
    results.sort(key=lambda item: item[0])
    rows = []
    fields = {}
    for _, name, row, field in results:
        rows.append(row)
        if field is not None:
            fields[name] = field
    return rows, fields


def run_double_mach(out, quick, workers=1):
    gamma, pre, post_state = _double_mach_states()
    nx, ny = DOUBLE_MACH_QUICK_GRID if quick else DOUBLE_MACH_PAPER_GRID
    Lx, Ly = 4.0, 1.0

    def classify(center, normal):
        cx, cy = float(center[0]), float(center[1])
        if cx <= 1e-9 * Lx:
            return 1
        if cx >= Lx - 1e-9 * Lx:
            return 2
        if cy <= 1e-9 * Ly:
            return 3 if cx <= 1.0 / 6.0 + 1e-12 else 4
        if cy >= Ly - 1e-9 * Ly:
            return 5
        return 2

    mesh = _tri_mesh(nx, ny, Lx, Ly, classifier=classify,
                     patches=('x_min_postshock', 'x_max_outflow',
                              'bottom_postshock', 'bottom_reflect',
                              'top_exact_shock'))
    eq = Euler2D(gamma=gamma)
    rho1, u1, v1, p1 = pre
    rho2, u2, v2, p2 = post_state
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    shock_x = 1.0 / 6.0 + y / np.sqrt(3.0)
    post = x < shock_x
    W0 = np.vstack([
        np.where(post, rho2, rho1),
        np.where(post, u2, u1),
        np.where(post, v2, v1),
        np.where(post, p2, p1),
    ])
    U0 = eq.prim_to_cons(W0)
    bc = {
        'x_min_postshock': BoundaryCondition('dirichlet', state=post_state),
        'x_max_outflow': BoundaryCondition('transmissive'),
        'bottom_postshock': BoundaryCondition('dirichlet', state=post_state),
        'bottom_reflect': BoundaryCondition('reflective'),
        'top_exact_shock': BoundaryCondition(
            'dirichlet_func', state=_double_mach_exact_state),
    }
    rows, fields = _run_case_methods('double_mach', quick, 'euler', workers)
    on = next(r for r in rows if r['method'] == 'T-MLP-u ON')
    baselines = [r for r in rows if r['method'] not in ('T-MLP-u ON', 'T-MLP-u OFF') and r['ok']]
    best_vortex = max((r['vortex_proxy'] for r in baselines), default=0.0)
    passed = bool(on['ok'] and on['vortex_proxy'] >= 0.85 * best_vortex
                  and on['checker'] < 0.95)
    if 'T-MLP-u ON' in fields:
        _plot_field(mesh, fields['T-MLP-u ON'], out / 'double_mach_tmlpu_on.png',
                    f'Double Mach density T-MLP-u ON tri {nx}x{ny}')
    _plot_scheme_contours(
        mesh, fields, rows, out / 'double_mach_scheme_contours.png',
        f'Double Mach reflection density: all schemes tri {nx}x{ny}',
        metric_keys=('vortex_proxy', 'checker'))
    return passed, rows


def run_mach3_step(out, quick, workers=1):
    gamma = 1.4
    nx, ny = MACH3_STEP_QUICK_GRID if quick else MACH3_STEP_PAPER_GRID
    Lx, Ly = 3.0, 1.0
    step_x = 0.6
    step_h = 0.2

    def keep(cx, cy):
        return not (cx >= step_x and cy < step_h)

    def classify(center, normal):
        cx, cy = float(center[0]), float(center[1])
        if cx <= 1e-9 * Lx:
            return 1
        if cx >= Lx - 1e-9 * Lx:
            return 2
        return 3

    mesh = _tri_mesh(nx, ny, Lx, Ly, keep=keep, classifier=classify,
                     patches=('inflow', 'outflow', 'wall'))
    eq = Euler2D(gamma=gamma)
    rho, p = 1.4, 1.0
    c = np.sqrt(gamma * p / rho)
    u, v = 3.0 * c, 0.0
    W0 = np.vstack([
        np.full(mesh.n_cells, rho),
        np.full(mesh.n_cells, u),
        np.full(mesh.n_cells, v),
        np.full(mesh.n_cells, p),
    ])
    U0 = eq.prim_to_cons(W0)
    bc = {
        'inflow': BoundaryCondition('dirichlet', state=(rho, u, v, p)),
        'outflow': BoundaryCondition('transmissive'),
        'wall': BoundaryCondition('reflective'),
    }
    rows, fields = _run_case_methods('mach3_step', quick, 'euler', workers)
    on = next(r for r in rows if r['method'] == 'T-MLP-u ON')
    baselines = [r for r in rows if r['method'] not in ('T-MLP-u ON', 'T-MLP-u OFF') and r['ok']]
    best_flag = max((r['flag_proxy'] for r in baselines), default=0.0)
    passed = bool(on['ok'] and on['flag_proxy'] >= 0.80 * best_flag
                  and on['carbuncle'] < 0.95)
    if 'T-MLP-u ON' in fields:
        _plot_field(mesh, fields['T-MLP-u ON'], out / 'mach3_step_tmlpu_on.png',
                    f'Mach 3 step density T-MLP-u ON t=4 tri {nx}x{ny}')
    _plot_scheme_contours(
        mesh, fields, rows, out / 'mach3_step_scheme_contours.png',
        f'Mach 3 forward-facing step density: all schemes t=4 tri {nx}x{ny}',
        metric_keys=('flag_proxy', 'carbuncle'))
    return passed, rows


CASE_RUNNERS = [
    ('leveque', run_leveque),
    ('double_mach', run_double_mach),
    ('mach3_step', run_mach3_step),
]


def _selected_cases(case_arg):
    names = [name for name, _ in CASE_RUNNERS]
    if case_arg == 'all':
        return names
    selected = [x.strip() for x in case_arg.split(',') if x.strip()]
    invalid = [x for x in selected if x not in names]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown case(s) {invalid}; choose from {names} or 'all'")
    if not selected:
        raise argparse.ArgumentTypeError("at least one case is required")
    return selected


def _artifact_prefix(selected_names, explicit):
    if explicit:
        return explicit
    all_names = [name for name, _ in CASE_RUNNERS]
    if selected_names == all_names:
        return 'paper_benchmark_summary'
    return 'paper_benchmark_' + '_'.join(selected_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument(
        '--cases', default='all',
        help="comma-separated subset of leveque,double_mach,mach3_step")
    ap.add_argument(
        '--artifact-prefix', default=None,
        help="basename for JSON/TSV/comparison PNG; default depends on cases")
    ap.add_argument(
        '--workers', type=int, default=1,
        help="parallel scheme workers per case; use 1 for serial execution")
    args = ap.parse_args()
    out = _out_dir()
    try:
        selected_names = _selected_cases(args.cases)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))
    runners = dict(CASE_RUNNERS)
    prefix = _artifact_prefix(selected_names, args.artifact_prefix)

    all_rows = []
    case_status = {}
    for name in selected_names:
        ok, rows = runners[name](out, args.quick, workers=args.workers)
        case_status[name] = bool(ok)
        all_rows.extend(rows)

    tsv = out / f'{prefix}.tsv'
    keys = sorted({k for row in all_rows for k in row.keys()})
    with tsv.open('w') as f:
        f.write('\t'.join(keys) + '\n')
        for row in all_rows:
            f.write('\t'.join(str(row.get(k, '')) for k in keys).rstrip('\t') + '\n')

    summary = {
        'comparison_definition': {
            'T-MLP-u OFF': 'SUPERBEE TVD-only reconstruction with mlp_bound=False',
            'T-MLP-u ON': (
                'T-MLP-u wrapper plus SUPERBEE with mlp_bound=True, '
                'vertex_mlp=True, vertex_mlp_cap=2, '
                'virtual_uu_gradient=True, stencil=vertex, order=1, '
                'tvb_M=0, extremum_relax=False'),
            'LeVeque T-MLP-u OFF': 'pure_downwind reconstruction with mlp_bound=False',
            'LeVeque T-MLP-u ON': (
                'T-MLP-u wrapper plus pure_downwind with mlp_bound=True, '
                'vertex_mlp=True, vertex_mlp_cap=2, '
                'virtual_uu_gradient=True, stencil=vertex, order=1, '
                'tvb_M=0, extremum_relax=False'),
            'LeVeque gate': (
                'T-MLP-u ON must keep L1 within 10% of the best non-TMLPU '
                'baseline, exceed 125% of the best interface-jump '
                'sharpness proxy, and remain wiggle-free within [0,1]'),
            'leveque_flux': 'upwind with central-averaged face velocity',
            'leveque_mesh': (
                f'unstructured criss-cross triangles, N={LEVEQUE_PAPER_N} '
                f'({4 * LEVEQUE_PAPER_N * LEVEQUE_PAPER_N} triangles; '
                f'quick N={LEVEQUE_QUICK_N})'),
            'double_mach_flux': 'hllc_adc',
            'mach3_step_flux': 'hllc_adc',
            'euler_reconstruction_space': (
                'primitive variables W=(rho,u,v,p); HLLC-ADC converts '
                'the reconstructed primitives to conservative variables '
                'through the Euler EOS before flux evaluation'),
            'double_mach_vortex_metric': (
                'right-bottom ROI reports both density-jump vortex_proxy '
                'and LSQ curl metrics vorticity_p95/enstrophy_proxy'),
            'mach3_step_flag_metric': (
                'post-step lower-channel ROI reports density-jump flag_proxy, '
                'LSQ vorticity/enstrophy, transverse velocity RMS, and '
                'pressure-checker carbuncle proxy'),
            'double_mach_setup': (
                'Woodward-Colella domain [0,4]x[0,1], Mach 10 shock, '
                'bottom x<=1/6 postshock, bottom x>1/6 reflective, '
                'top exact moving shock'),
            'double_mach_mesh': (
                'unstructured alternating-diagonal triangles from '
                f'{DOUBLE_MACH_PAPER_GRID[0]}x{DOUBLE_MACH_PAPER_GRID[1]} '
                'logical cells '
                f'({_box_triangle_count(*DOUBLE_MACH_PAPER_GRID)} triangles; '
                f'fine reference {DOUBLE_MACH_FINE_GRID[0]}x'
                f'{DOUBLE_MACH_FINE_GRID[1]} = '
                f'{_box_triangle_count(*DOUBLE_MACH_FINE_GRID)} triangles; '
                f'quick={DOUBLE_MACH_QUICK_GRID[0]}x{DOUBLE_MACH_QUICK_GRID[1]})'),
            'mach3_step_mesh': (
                'unstructured alternating-diagonal triangles from '
                f'{MACH3_STEP_PAPER_GRID[0]}x{MACH3_STEP_PAPER_GRID[1]} '
                'logical cells with forward-step cutout '
                f'({_mach3_step_triangle_count(*MACH3_STEP_PAPER_GRID)} '
                'triangles; '
                f'fine reference {MACH3_STEP_FINE_GRID[0]}x'
                f'{MACH3_STEP_FINE_GRID[1]} = '
                f'{_mach3_step_triangle_count(*MACH3_STEP_FINE_GRID)} '
                'triangles; '
                f'quick={MACH3_STEP_QUICK_GRID[0]}x{MACH3_STEP_QUICK_GRID[1]})'),
        },
        'fail_count': int(sum(0 if v else 1 for v in case_status.values())),
        'pass_count': int(sum(1 if v else 0 for v in case_status.values())),
        'total': len(selected_names),
        'selected_cases': selected_names,
        'all_cases_selected': int(selected_names == [name for name, _ in CASE_RUNNERS]),
        'evidence_ready': int(
            selected_names == [name for name, _ in CASE_RUNNERS]
            and all(case_status.values())),
        'selected_ready': int(all(case_status.values())),
        'workers': int(args.workers),
        'leveque_ready': int(case_status.get('leveque', False)),
        'double_mach_ready': int(case_status.get('double_mach', False)),
        'mach3_step_ready': int(case_status.get('mach3_step', False)),
        'quick': bool(args.quick),
        'rows': all_rows,
    }
    comparison_png = (out / 'paper_benchmark_comparison.png'
                      if prefix == 'paper_benchmark_summary'
                      else out / f'{prefix}_comparison.png')
    _plot_summary(all_rows, comparison_png)
    summary['comparison_png'] = str(comparison_png.relative_to(_repo_root()))
    summary = _json_ready(summary)
    json_path = out / f'{prefix}.json'
    json_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + '\n')
    print(f"TSV saved: {tsv}")
    print(f"JSON saved: {json_path}")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary['fail_count'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
