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
from reconstruction import (
    TMLPU, TMLPUBVD, TMLPUSmoothSharpBVD, BarthJespersen, Venkatakrishnan,
    MLPU1, MLPU1TMLPU, MLPU2,
)
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
DOUBLE_MACH_PAPER_GRID = (240, 60)
DOUBLE_MACH_FINE_GRID = (480, 120)
MACH3_STEP_QUICK_GRID = (90, 30)
MACH3_STEP_PAPER_GRID = (240, 80)
MACH3_STEP_FINE_GRID = (480, 160)
LEVEQUE_QUICK_N = 18
LEVEQUE_PAPER_N = 50
STRIP_NY_QUICK = 3
STRIP_NY_PAPER = 4
LAX_QUICK_N = 120
LAX_PAPER_N = 400
BLAST_QUICK_N = 160
BLAST_PAPER_N = 500
SHOCK_TURB_QUICK_N = 180
SHOCK_TURB_PAPER_N = 500


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


def _safe_name(name):
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in name).strip('_')


def _write_vtk_cell_data(mesh, cell_data, path, title):
    """Write legacy ASCII VTK cell data for ParaView."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = [tuple(int(v) for v in cell) for cell in mesh.cell_nodes]
    total_ints = sum(1 + len(cell) for cell in cells)
    cell_type = {3: 5, 4: 9}
    with path.open('w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write(f'{title}\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write(f'POINTS {mesh.nodes.shape[0]} float\n')
        for x, y in mesh.nodes:
            f.write(f'{float(x):.16e} {float(y):.16e} 0.0\n')
        f.write(f'CELLS {len(cells)} {total_ints}\n')
        for cell in cells:
            f.write(str(len(cell)) + ' ' + ' '.join(str(v) for v in cell) + '\n')
        f.write(f'CELL_TYPES {len(cells)}\n')
        for cell in cells:
            f.write(f"{cell_type.get(len(cell), 7)}\n")
        f.write(f'CELL_DATA {mesh.n_cells}\n')

        for key, values in cell_data.items():
            arr = np.asarray(values, dtype=float)
            if arr.shape != (mesh.n_cells,):
                continue
            f.write(f'SCALARS {_safe_name(key)} float 1\n')
            f.write('LOOKUP_TABLE default\n')
            for val in arr:
                f.write(f'{float(val):.16e}\n')

        if 'u' in cell_data and 'v' in cell_data:
            u = np.asarray(cell_data['u'], dtype=float)
            v = np.asarray(cell_data['v'], dtype=float)
            if u.shape == (mesh.n_cells,) and v.shape == (mesh.n_cells,):
                f.write('VECTORS velocity float\n')
                for ux, vy in zip(u, v):
                    f.write(f'{float(ux):.16e} {float(vy):.16e} 0.0\n')


def _write_case_vtks(out, case, mesh, vtk_fields):
    case_dir = out / 'vtk' / case
    if case_dir.exists():
        for old in case_dir.glob('*.vtk'):
            old.unlink()
    written = []
    for method, cell_data in vtk_fields.items():
        path = case_dir / f'{_safe_name(method)}.vtk'
        _write_vtk_cell_data(mesh, cell_data, path, f'{case}: {method}')
        written.append(str(path.relative_to(_repo_root())))
    return written


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


def _missing_field_status(row):
    if row.get('finite') or row.get('run_ok'):
        return 'GATE FAILED'
    err = (row.get('error') or '').lower()
    if 'range' in err or 'boundedness' in err or 'positivity' in err:
        return 'GATE FAILED'
    if err:
        return 'DIVERGED'
    return 'NOT PLOTTED'


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
            status = _missing_field_status(row)
            ax.set_title(method, fontsize=9)
            ax.text(0.5, 0.56, status, transform=ax.transAxes,
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


def _plot_1d_scheme_profiles(mesh, fields, rows, path, title, cfg,
                             metric_keys=()):
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.0))
    nx = cfg['nx']
    x_min = cfg['x_min']
    x_max = cfg['x_max']
    xp = x_min + (np.arange(nx) + 0.5) * (x_max - x_min) / nx
    colors = ['#707070', '#4c78a8', '#59a14f', '#f28e2b',
              '#b07aa1', '#e15759', '#111111']
    for color, row in zip(colors, rows):
        method = row['method']
        field = fields.get(method)
        if field is None:
            continue
        prof = _strip_profile(mesh, field, nx, x_min, x_max)
        metric_text = ', '.join(
            x for x in (_format_metric(row, k) for k in metric_keys) if x)
        label = method if not metric_text else f'{method} ({metric_text})'
        lw = 1.9 if method == 'T-MLP-u ON' else 0.9
        ax.plot(xp, prof, color=color, lw=lw, alpha=0.88, label=label)
    missing = [f"{row['method']} ({_missing_field_status(row).lower()})"
               for row in rows if row['method'] not in fields]
    if missing:
        ax.text(0.01, 0.04, 'Not plotted: ' + ', '.join(missing),
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('density')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
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
        ('lax_1d', 'l1_ref', 'lower is better', 'Lax density L1'),
        ('blast_1d', 'shock_proxy', 'higher is sharper', 'Blast waves'),
        ('shock_turbulence_1d', 'turbulence_proxy', 'higher is richer',
         'Shock/turbulence'),
        ('leveque', 'global_E1', 'lower is better', 'LeVeque 1T shape'),
        ('double_mach', 'vortex_proxy', 'higher is sharper', 'Double Mach vortices'),
        ('mach3_step', 'flag_proxy', 'higher is stronger', 'Mach 3 flag-waving'),
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


def _strip_tri_mesh(nx, ny, x_min, x_max):
    """Thin triangular strip used as a quasi-1D unstructured grid."""
    Lx = x_max - x_min
    Ly = Lx / max(nx, 1)

    def classify(center, normal):
        cx, cy = float(center[0]), float(center[1])
        if cx <= x_min + 1e-9 * Lx:
            return 1
        if cx >= x_max - 1e-9 * Lx:
            return 2
        if cy <= 1e-12 * Ly:
            return 3
        if cy >= Ly - 1e-12 * Ly:
            return 4
        return 4

    return _tri_mesh(nx, ny, Lx, Ly, origin=(x_min, 0.0),
                     classifier=classify,
                     patches=('x_min', 'x_max', 'y_min_wall',
                              'y_max_wall'))


def _tmlpu_leveque():
    return TMLPUSmoothSharpBVD(
        smooth_mode='tmlpu',
        smooth_tvd='bounded_cd',
        smooth_face_increment='lsq',
        sharp_tvd='tmlpu_shape',
        sharp_face_increment='tmlpu',
        stencil='vertex',
        order=1,
        idw_p=0.0,
        vertex_mlp_cap=2.0,
        face_skew_correction=True,
        face_gradient_correction='jasak',
        vertex_mlp_augment=True,
        moment_bvd=False,
    )


def _tmlpu_off_leveque():
    return TMLPU(tvd='pure_downwind', mlp_bound=False,
                 extremum_relax=False, tvb_M=0.0,
                 virtual_uu_gradient=True, stencil='vertex',
                 order=1, idw_p=6.0)


def _tmlpu_euler(idw_p=6.0):
    return TMLPU(tvd='modified_superbee', mlp_bound=True, extremum_relax=False,
                 tvb_M=0.0, vertex_mlp=True, vertex_mlp_cap=2.0,
                 virtual_uu_gradient=True, stencil='vertex', order=1,
                 idw_p=idw_p)


def _tmlpu_double_mach():
    return _tmlpu_euler(idw_p=1.0)


def _tmlpu_mach3_step():
    return _tmlpu_euler(idw_p=0.0)


def _tmlpu_off_euler():
    return TMLPU(tvd='modified_superbee', mlp_bound=False, extremum_relax=False,
                 tvb_M=0.0, vertex_mlp=False,
                 virtual_uu_gradient=True, stencil='face', order=1)


def _comparison_specs(kind):
    if kind == 'leveque':
        t_on = 'tmlpu_leveque_on'
        t_off = 'tmlpu_leveque_off'
    elif kind == 'double_mach':
        t_on = 'tmlpu_double_mach_on'
        t_off = 'tmlpu_euler_off'
    elif kind == 'mach3_step':
        t_on = 'tmlpu_mach3_step_on'
        t_off = 'tmlpu_euler_off'
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
    if key == 'mlp_u1_tmlpu':
        return MLPU1TMLPU()
    if key == 'mlp_u2':
        return MLPU2()
    if key == 'tmlpu_leveque_on':
        return _tmlpu_leveque()
    if key == 'tmlpu_leveque_off':
        return _tmlpu_off_leveque()
    if key == 'tmlpu_double_mach_on':
        return _tmlpu_double_mach()
    if key == 'tmlpu_mach3_step_on':
        return _tmlpu_mach3_step()
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


def _leveque_body_masks(mesh, buffer_cells=3.0):
    """Disjoint local masks for the three LeVeque bodies.

    The exact 1T comparison uses the stored cell initial values.  These masks
    only partition diagnostics by body; cells in overlapping buffered regions
    are assigned to the nearest body center.
    """
    xy = mesh.cell_centers
    h = 2.0 * np.sqrt(float(np.mean(mesh.cell_volumes)))
    r0 = 0.15
    buffer = float(buffer_cells) * h
    body_centers = {
        'smooth_hump': np.asarray([0.25, 0.5], dtype=float),
        'cone': np.asarray([0.5, 0.25], dtype=float),
        'slotted_cylinder': np.asarray([0.5, 0.75], dtype=float),
    }
    names = list(body_centers)
    centers = np.vstack([body_centers[name] for name in names])
    dist = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=2)
    nearest = np.argmin(dist, axis=1)
    active = np.min(dist, axis=1) <= r0 + buffer
    return {name: active & (nearest == i) for i, name in enumerate(names)}


def _weighted_centroid_and_covariance(xy, weights):
    mass = float(np.sum(weights))
    if mass <= 1.0e-30:
        return None, None
    centroid = np.sum(weights[:, None] * xy, axis=0) / mass
    d = xy - centroid
    cov = (d * weights[:, None]).T @ d / mass
    return centroid, cov


def _leveque_body_metrics(mesh, phi0, phiT, mask, *, body_name):
    xy = mesh.cell_centers[mask]
    A = mesh.cell_volumes[mask]
    p0 = np.asarray(phi0, dtype=float)[mask]
    pT = np.asarray(phiT, dtype=float)[mask]
    diff = pT - p0
    den1 = float(np.sum(np.abs(p0) * A))
    den2 = float(np.sum(p0 * p0 * A))
    M0 = float(np.sum(p0 * A))
    MT = float(np.sum(pT * A))
    c0, C0 = _weighted_centroid_and_covariance(xy, p0 * A)
    cT, CT = _weighted_centroid_and_covariance(xy, pT * A)
    centroid_error = None
    moment_error = None
    if c0 is not None and cT is not None:
        centroid_error = float(np.linalg.norm(cT - c0) / 0.15)
    if C0 is not None and CT is not None:
        moment_error = float(
            np.linalg.norm(CT - C0) / max(np.linalg.norm(C0), 1.0e-30))
    out = {
        f'{body_name}_E1': float(np.sum(np.abs(diff) * A) / max(den1, 1.0e-30)),
        f'{body_name}_E2': float(np.sqrt(
            np.sum(diff * diff * A) / max(den2, 1.0e-30))),
        f'{body_name}_Einf': float(np.max(np.abs(diff))) if diff.size else None,
        f'{body_name}_mass_initial': M0,
        f'{body_name}_mass_final': MT,
        f'{body_name}_mass_error': float(abs(MT - M0) / max(abs(M0), 1.0e-30)),
        f'{body_name}_centroid_error': centroid_error,
        f'{body_name}_moment_error': moment_error,
        f'{body_name}_phi_max_initial': float(np.max(p0)) if p0.size else None,
        f'{body_name}_phi_max_final': float(np.max(pT)) if pT.size else None,
        f'{body_name}_phi_min_initial': float(np.min(p0)) if p0.size else None,
        f'{body_name}_phi_min_final': float(np.min(pT)) if pT.size else None,
    }
    max0 = out[f'{body_name}_phi_max_initial']
    maxT = out[f'{body_name}_phi_max_final']
    out[f'{body_name}_phi_max_error'] = (
        float(abs(maxT - max0) / max(abs(max0), 1.0e-30))
        if max0 is not None and maxT is not None else None)
    return out


def _leveque_shape_metrics(mesh, phi0, phiT):
    """1-period LeVeque diagnostics against the stored initial cell field."""
    A = mesh.cell_volumes
    p0 = np.asarray(phi0, dtype=float)
    pT = np.asarray(phiT, dtype=float)
    diff = pT - p0
    M0 = float(np.sum(p0 * A))
    MT = float(np.sum(pT * A))
    metrics = {
        'global_E1': float(
            np.sum(np.abs(diff) * A) / max(np.sum(np.abs(p0) * A), 1.0e-30)),
        'global_E2': float(np.sqrt(
            np.sum(diff * diff * A) / max(np.sum(p0 * p0 * A), 1.0e-30))),
        'global_Einf': float(np.max(np.abs(diff))),
        'global_mass_initial': M0,
        'global_mass_final': MT,
        'global_mass_error': float(abs(MT - M0) / max(abs(M0), 1.0e-30)),
        'undershoot': float(max(0.0, -float(np.min(pT)))),
        'overshoot': float(max(0.0, float(np.max(pT)) - 1.0)),
    }
    masks = _leveque_body_masks(mesh)
    for body_name, mask in masks.items():
        metrics.update(_leveque_body_metrics(
            mesh, p0, pT, mask, body_name=body_name))

    body_names = tuple(masks)
    for key in ('E1', 'E2', 'Einf', 'mass_error',
                'centroid_error', 'moment_error'):
        vals = [
            metrics.get(f'{name}_{key}') for name in body_names
            if metrics.get(f'{name}_{key}') is not None
        ]
        metrics[f'max_body_{key}'] = float(max(vals)) if vals else None

    slot_mask = masks['slotted_cylinder']
    xy = mesh.cell_centers
    slot_probe = (
        (np.abs(xy[:, 0] - 0.5) < 0.025)
        & (xy[:, 1] < 0.85)
        & (np.sqrt((xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.75) ** 2) <= 0.15)
    )
    p0s = p0[slot_mask]
    pTs = pT[slot_mask]
    As = A[slot_mask]
    chi0 = p0s > 0.5
    chiT = pTs > 0.5
    area0 = float(np.sum(As[chi0]))
    areaT = float(np.sum(As[chiT]))
    inter = float(np.sum(As[chi0 & chiT]))
    union = float(np.sum(As[chi0 | chiT]))
    p0c = np.clip(p0s, 0.0, 1.0)
    pTc = np.clip(pTs, 0.0, 1.0)
    fuzzy_inter = float(np.sum(np.minimum(p0c, pTc) * As))
    fuzzy_union = float(np.sum(np.maximum(p0c, pTc) * As))
    metrics.update({
        'slotted_cylinder_area_initial': area0,
        'slotted_cylinder_area_final': areaT,
        'slotted_cylinder_area_error': float(
            abs(areaT - area0) / max(area0, 1.0e-30)),
        'slotted_cylinder_iou': float(inter / max(union, 1.0e-30)),
        'slotted_cylinder_dice': float(
            2.0 * inter / max(area0 + areaT, 1.0e-30)),
        'slotted_cylinder_fuzzy_iou': float(
            fuzzy_inter / max(fuzzy_union, 1.0e-30)),
        'slotted_cylinder_slot_max_phi': (
            float(np.max(pT[slot_probe])) if np.any(slot_probe) else None),
    })
    return metrics


def _rotation_velocity(x, y):
    return (-2.0 * np.pi * (y - 0.5),
             2.0 * np.pi * (x - 0.5))


def _exact_riemann_1d(x, t, left, right, gamma=1.4, x0=0.5):
    """Exact Euler Riemann sample for 1D shock-tube references."""
    rhoL, uL, pL = left
    rhoR, uR, pR = right
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    def f_side(p, pK, rhoK, cK):
        if p > pK:
            A = 2.0 / ((gamma + 1.0) * rhoK)
            B = (gamma - 1.0) / (gamma + 1.0) * pK
            return (p - pK) * np.sqrt(A / (p + B))
        expo = (gamma - 1.0) / (2.0 * gamma)
        return 2.0 * cK / (gamma - 1.0) * ((p / pK) ** expo - 1.0)

    def f_total(p):
        return f_side(p, pL, rhoL, cL) + f_side(p, pR, rhoR, cR) + uR - uL

    p_star = max(1e-12, 0.5 * (pL + pR))
    for _ in range(80):
        f0 = f_total(p_star)
        dp = max(1e-8 * p_star, 1e-10)
        df = (f_total(p_star + dp) - f0) / dp
        p_new = p_star - f0 / max(df, 1e-30)
        if p_new <= 0.0:
            p_new = 0.5 * p_star
        if abs(p_new - p_star) < 1e-10 * max(abs(p_new), 1.0):
            p_star = p_new
            break
        p_star = p_new
    u_star = 0.5 * (uL + uR) + 0.5 * (
        f_side(p_star, pR, rhoR, cR) - f_side(p_star, pL, rhoL, cL))

    def star_density(pK, rhoK):
        if p_star > pK:
            gp = (gamma + 1.0) / (gamma - 1.0)
            return rhoK * ((p_star / pK + 1.0 / gp)
                           / (1.0 + (p_star / pK) / gp))
        return rhoK * (p_star / pK) ** (1.0 / gamma)

    rho_starL = star_density(pL, rhoL)
    rho_starR = star_density(pR, rhoR)
    s = (x - x0) / max(t, 1e-30)
    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)

    left_shock = p_star > pL
    right_shock = p_star > pR
    if left_shock:
        S_L = uL - cL * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * p_star / pL
            + (gamma - 1.0) / (2.0 * gamma))
        SHL = STL = S_L
        c_starL = cL
    else:
        SHL = uL - cL
        c_starL = cL * (p_star / pL) ** ((gamma - 1.0)
                                         / (2.0 * gamma))
        STL = u_star - c_starL
    if right_shock:
        S_R = uR + cR * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * p_star / pR
            + (gamma - 1.0) / (2.0 * gamma))
        SHR = STR = S_R
        c_starR = cR
    else:
        SHR = uR + cR
        c_starR = cR * (p_star / pR) ** ((gamma - 1.0)
                                         / (2.0 * gamma))
        STR = u_star + c_starR

    for i, si in enumerate(s):
        if si < SHL:
            rho[i], u[i], p[i] = rhoL, uL, pL
        elif si < STL:
            u_fan = 2.0 / (gamma + 1.0) * (
                cL + 0.5 * (gamma - 1.0) * uL + si)
            c_fan = 2.0 / (gamma + 1.0) * (
                cL + 0.5 * (gamma - 1.0) * (uL - si))
            p_fan = pL * (c_fan / cL) ** (2.0 * gamma / (gamma - 1.0))
            rho_fan = gamma * p_fan / (c_fan * c_fan)
            rho[i], u[i], p[i] = rho_fan, u_fan, p_fan
        elif si < u_star:
            rho[i], u[i], p[i] = rho_starL, u_star, p_star
        elif si < STR:
            rho[i], u[i], p[i] = rho_starR, u_star, p_star
        elif si < SHR:
            u_fan = 2.0 / (gamma + 1.0) * (
                -cR + 0.5 * (gamma - 1.0) * uR + si)
            c_fan = 2.0 / (gamma + 1.0) * (
                cR - 0.5 * (gamma - 1.0) * (uR - si))
            p_fan = pR * (c_fan / cR) ** (2.0 * gamma / (gamma - 1.0))
            rho_fan = gamma * p_fan / (c_fan * c_fan)
            rho[i], u[i], p[i] = rho_fan, u_fan, p_fan
        else:
            rho[i], u[i], p[i] = rhoR, uR, pR
    return np.vstack([rho, u, p])


def _strip_profile(mesh, values, nx, x_min, x_max):
    x = mesh.cell_centers[:, 0]
    dx = (x_max - x_min) / nx
    bins = np.floor((x - x_min) / dx).astype(int)
    bins = np.clip(bins, 0, nx - 1)
    sums = np.zeros(nx, dtype=float)
    counts = np.zeros(nx, dtype=float)
    np.add.at(sums, bins, np.asarray(values, dtype=float))
    np.add.at(counts, bins, 1.0)
    return sums / np.maximum(counts, 1.0)


def _strip_transverse_checker(mesh, values, nx, x_min, x_max):
    x = mesh.cell_centers[:, 0]
    dx = (x_max - x_min) / nx
    bins = np.floor((x - x_min) / dx).astype(int)
    bins = np.clip(bins, 0, nx - 1)
    vals = np.asarray(values, dtype=float)
    means = np.zeros(nx, dtype=float)
    counts = np.zeros(nx, dtype=float)
    np.add.at(means, bins, vals)
    np.add.at(counts, bins, 1.0)
    means = means / np.maximum(counts, 1.0)
    fluct = vals - means[bins]
    scale = np.percentile(np.abs(vals), 95) + 1e-30
    return float(np.sqrt(np.mean(fluct * fluct)) / scale)


def _strip_case_config(case, quick):
    ny = STRIP_NY_QUICK if quick else STRIP_NY_PAPER
    if case == 'lax_1d':
        nx = LAX_QUICK_N if quick else LAX_PAPER_N
        left = (0.445, 0.698, 3.528)
        right = (0.5, 0.0, 0.571)
        return dict(
            title='Lax shock tube',
            nx=nx, ny=ny, x_min=0.0, x_max=1.0, t_end=0.16, cfl=0.35,
            left=left, right=right, x0=0.5,
            bc_x='transmissive',
            init=lambda x: np.vstack([
                np.where(x < 0.5, left[0], right[0]),
                np.where(x < 0.5, left[1], right[1]),
                np.zeros_like(x),
                np.where(x < 0.5, left[2], right[2]),
            ]))
    if case == 'blast_1d':
        nx = BLAST_QUICK_N if quick else BLAST_PAPER_N
        return dict(
            title='Woodward-Colella blast wave',
            nx=nx, ny=ny, x_min=0.0, x_max=1.0, t_end=0.038, cfl=0.25,
            bc_x='reflective',
            init=lambda x: np.vstack([
                np.ones_like(x),
                np.zeros_like(x),
                np.zeros_like(x),
                np.where(x < 0.1, 1000.0, np.where(x > 0.9, 100.0, 0.01)),
            ]))
    if case == 'shock_turbulence_1d':
        nx = SHOCK_TURB_QUICK_N if quick else SHOCK_TURB_PAPER_N
        return dict(
            title='Shu-Osher shock-density interaction',
            nx=nx, ny=ny, x_min=-5.0, x_max=5.0, t_end=1.8, cfl=0.35,
            bc_x='transmissive',
            init=lambda x: np.vstack([
                np.where(x < -4.0, 3.857143, 1.0 + 0.2 * np.sin(5.0 * x)),
                np.where(x < -4.0, 2.629369, 0.0),
                np.zeros_like(x),
                np.where(x < -4.0, 10.333333, 1.0),
            ]))
    raise ValueError(f'unknown 1D strip case {case!r}')


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


def run_1d_strip_case(case, out, quick, workers=1):
    cfg = _strip_case_config(case, quick)
    nx = cfg['nx']
    ny = cfg['ny']
    mesh = _strip_tri_mesh(nx, ny, cfg['x_min'], cfg['x_max'])
    rows, fields, vtk_fields = _run_case_methods(case, quick, 'euler_1d',
                                                 workers)
    vtk_written = _write_case_vtks(out, case, mesh, vtk_fields)
    on = next(r for r in rows if r['method'] == 'T-MLP-u ON')
    baselines = [
        r for r in rows
        if r['method'] not in ('T-MLP-u ON', 'T-MLP-u OFF') and r['ok']
    ]
    best_shock = max((r['shock_proxy'] for r in baselines), default=0.0)
    best_turb = max((r['turbulence_proxy'] for r in baselines), default=0.0)
    min_checker = min((r['checker'] for r in baselines
                       if r.get('checker') is not None), default=0.0)
    passed = bool(on['ok']
                  and on['checker'] <= max(0.08, 2.0 * min_checker))
    if case == 'lax_1d':
        best_l1 = min((r['l1_ref'] for r in baselines
                       if r.get('l1_ref') is not None), default=float('inf'))
        passed = bool(passed
                      and on['shock_proxy'] >= 0.80 * best_shock
                      and on['l1_ref'] <= 1.30 * best_l1)
    if case == 'shock_turbulence_1d':
        passed = bool(passed and on['turbulence_proxy'] >= 0.95 * best_turb)
    if case == 'blast_1d':
        passed = bool(passed
                      and on['shock_proxy'] >= 0.80 * best_shock
                      and on['turbulence_proxy'] >= 0.80 * best_turb)

    metric_keys = ('l1_ref', 'shock_proxy', 'turbulence_proxy', 'checker')
    if 'T-MLP-u ON' in fields:
        _plot_1d_scheme_profiles(
            mesh, {'T-MLP-u ON': fields['T-MLP-u ON']},
            [on], out / f'{case}_tmlpu_on.png',
            f"{cfg['title']} density T-MLP-u ON tri-strip {nx}x{ny}",
            cfg, metric_keys=metric_keys)
    _plot_1d_scheme_profiles(
        mesh, fields, rows, out / f'{case}_scheme_profiles.png',
        f"{cfg['title']} density: all schemes tri-strip {nx}x{ny}",
        cfg, metric_keys=metric_keys)
    for row in rows:
        row['vtk_written'] = int(
            f"results/T-MLP-u/vtk/{case}/{_safe_name(row['method'])}.vtk"
            in vtk_written)
    return passed, rows


def run_lax_1d(out, quick, workers=1):
    return run_1d_strip_case('lax_1d', out, quick, workers)


def run_blast_1d(out, quick, workers=1):
    return run_1d_strip_case('blast_1d', out, quick, workers)


def run_shock_turbulence_1d(out, quick, workers=1):
    return run_1d_strip_case('shock_turbulence_1d', out, quick, workers)


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
    rows, fields, vtk_fields = _run_case_methods(
        'leveque', quick, 'leveque', workers)
    vtk_written = _write_case_vtks(out, 'leveque', mesh, vtk_fields)
    on = next(r for r in rows if r['method'] == 'T-MLP-u ON')
    off = next(r for r in rows if r['method'] == 'T-MLP-u OFF')
    mlp_u1 = next(
        (r for r in rows if r['method'] == 'MLP-u1' and r['ok']), None)

    def _finite_value(row, key):
        if row is None:
            return None
        val = row.get(key)
        if isinstance(val, (int, float)) and np.isfinite(val):
            return float(val)
        return None

    def _metric_tol(ref, *, abs_tol=1.0e-12, rel_tol=1.0e-12):
        return max(abs_tol, rel_tol * abs(ref))

    def _lt_vs_mlp_u1(key, *, abs_tol=1.0e-12, rel_tol=1.0e-12):
        a = _finite_value(on, key)
        b = _finite_value(mlp_u1, key)
        if a is None or b is None:
            return False
        return bool(a < b - _metric_tol(b, abs_tol=abs_tol,
                                        rel_tol=rel_tol))

    def _le_vs_mlp_u1(key, *, abs_tol=1.0e-12, rel_tol=1.0e-12):
        a = _finite_value(on, key)
        b = _finite_value(mlp_u1, key)
        if a is None or b is None:
            return False
        return bool(a <= b + _metric_tol(b, abs_tol=abs_tol,
                                         rel_tol=rel_tol))

    def _gt_vs_mlp_u1(key, *, abs_tol=1.0e-12, rel_tol=1.0e-12):
        a = _finite_value(on, key)
        b = _finite_value(mlp_u1, key)
        if a is None or b is None:
            return False
        return bool(a > b + _metric_tol(b, abs_tol=abs_tol,
                                        rel_tol=rel_tol))

    def _range_violation(row):
        if row['range_min'] is None or row['range_max'] is None:
            return float('inf')
        return max(0.0, -row['range_min']) + max(0.0, row['range_max'] - 1.0)

    on_range_violation = _range_violation(on)
    off_range_violation = _range_violation(off)
    strict_lower_better = [
        'global_E1', 'global_E2', 'global_Einf',
        'max_body_E1', 'max_body_E2', 'max_body_Einf',
        'max_body_mass_error', 'max_body_centroid_error',
        'max_body_moment_error', 'slotted_cylinder_area_error',
        'slotted_cylinder_slot_max_phi',
    ]
    bounded_nonworse = ['overshoot', 'undershoot', 'wiggle']
    for body_name in ('smooth_hump', 'cone', 'slotted_cylinder'):
        strict_lower_better.extend([
            f'{body_name}_E1',
            f'{body_name}_E2',
            f'{body_name}_Einf',
            f'{body_name}_mass_error',
            f'{body_name}_centroid_error',
            f'{body_name}_moment_error',
            f'{body_name}_phi_max_error',
        ])
    higher_better = [
        'slotted_cylinder_iou',
        'slotted_cylinder_dice',
        'slotted_cylinder_fuzzy_iou',
    ]
    lower_better = strict_lower_better + bounded_nonworse
    required_metrics = tuple(lower_better + higher_better)
    diagnostics_finite = all(_finite_value(on, key) is not None
                             and _finite_value(mlp_u1, key) is not None
                             for key in required_metrics)
    strict_lower_failures = [
        key for key in strict_lower_better if not _lt_vs_mlp_u1(key)]
    bounded_nonworse_failures = [
        key for key in bounded_nonworse if not _le_vs_mlp_u1(key)]
    lower_failures = strict_lower_failures + bounded_nonworse_failures
    higher_failures = [key for key in higher_better
                       if not _gt_vs_mlp_u1(key)]
    bounded_ok = bool(
        _finite_value(on, 'undershoot') is not None
        and _finite_value(on, 'overshoot') is not None
        and on['undershoot'] <= 1.0e-8
        and on['overshoot'] <= 1.0e-8
        and on_range_violation <= 1.0e-8)
    mass_keys = [k for k in lower_better if k.endswith('mass_error')]
    global_mass_ok = bool(
        _finite_value(on, 'global_mass_error') is not None
        and _finite_value(on, 'global_mass_error') <= 1.0e-6)
    error_keys = [k for k in lower_better if (
        k.endswith('_E1') or k.endswith('_E2') or k.endswith('_Einf'))]
    body_shape_keys = [k for k in lower_better if (
        k.endswith('centroid_error') or k.endswith('moment_error')
        or k.endswith('phi_max_error'))]
    slot_keys = [
        'slotted_cylinder_area_error',
        'slotted_cylinder_slot_max_phi',
        'slotted_cylinder_iou',
        'slotted_cylinder_dice',
        'slotted_cylinder_fuzzy_iou',
    ]
    mass_ok = bool(global_mass_ok
                   and all(k not in lower_failures for k in mass_keys))
    error_ok = bool(all(k not in lower_failures for k in error_keys))
    body_shape_ok = bool(all(k not in lower_failures for k in body_shape_keys))
    slot_ok = bool(
        all(k not in lower_failures for k in slot_keys)
        and all(k not in higher_failures for k in slot_keys))
    mlp_u1_dominance_ok = bool(
        mlp_u1 is not None and diagnostics_finite
        and not lower_failures and not higher_failures)
    off_unstable_or_worse = bool(
        not off['ok'] or on_range_violation <= off_range_violation)
    passed = bool(on['ok']
                  and diagnostics_finite
                  and bounded_ok
                  and mlp_u1_dominance_ok
                  and off_unstable_or_worse)
    on.update({
        'leveque_gate_diagnostics_finite': diagnostics_finite,
        'leveque_gate_bounded': bounded_ok,
        'leveque_gate_mass': mass_ok,
        'leveque_gate_global_mass_abs': global_mass_ok,
        'leveque_gate_error': error_ok,
        'leveque_gate_slot': slot_ok,
        'leveque_gate_body_shape': body_shape_ok,
        'leveque_gate_off_unstable_or_worse': off_unstable_or_worse,
        'leveque_gate_mlp_u1_dominance': mlp_u1_dominance_ok,
        'leveque_gate_mlp_u1_lower_failures': lower_failures,
        'leveque_gate_mlp_u1_higher_failures': higher_failures,
        'leveque_gate_mlp_u1_strict_lower_failures': strict_lower_failures,
        'leveque_gate_mlp_u1_bounded_nonworse_failures': (
            bounded_nonworse_failures),
        'leveque_gate_mlp_u1_lower_failure_count': len(lower_failures),
        'leveque_gate_mlp_u1_higher_failure_count': len(higher_failures),
        'leveque_gate_mlp_u1_strict_lower_total': len(strict_lower_better),
        'leveque_gate_mlp_u1_higher_total': len(higher_better),
        'leveque_gate_mlp_u1_bounded_nonworse_total': len(bounded_nonworse),
        'leveque_gate_pass': passed,
    })
    if 'T-MLP-u ON' in fields:
        _plot_field(mesh, fields['T-MLP-u ON'], out / 'leveque_tmlpu_on.png',
                    f'LeVeque T-MLP-u ON N={N}', vmin=0, vmax=1)
    _plot_scheme_contours(
        mesh, fields, rows, out / 'leveque_scheme_contours.png',
        f'LeVeque rotation: all schemes N={N}',
        vmin=0.0, vmax=1.0,
        metric_keys=('global_E1', 'slotted_cylinder_fuzzy_iou', 'wiggle'))
    for row in rows:
        row['vtk_written'] = int(
            f"results/T-MLP-u/vtk/leveque/{_safe_name(row['method'])}.vtk"
            in vtk_written)
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


STRIP_CASES = ('lax_1d', 'blast_1d', 'shock_turbulence_1d')


def _run_case_method_worker(payload):
    case = payload['case']
    quick = payload['quick']
    order = payload['order']
    name = payload['name']
    recon = _reconstruction_from_key(payload['recon_key'])
    try:
        if case in STRIP_CASES:
            cfg = _strip_case_config(case, quick)
            nx = cfg['nx']
            ny = cfg['ny']
            x_min = cfg['x_min']
            x_max = cfg['x_max']
            mesh = _strip_tri_mesh(nx, ny, x_min, x_max)
            eq = Euler2D(gamma=1.4)
            x = mesh.cell_centers[:, 0]
            W0 = cfg['init'](x)
            U0 = eq.prim_to_cons(W0)
            x_bc = BoundaryCondition(cfg['bc_x'])
            bc = {
                'x_min': x_bc,
                'x_max': x_bc,
                'y_min_wall': BoundaryCondition('reflective'),
                'y_max_wall': BoundaryCondition('reflective'),
            }
            r = _run_safely(mesh, eq, U0, recon, bc, flux='hllc_adc',
                            integrator='ssp_rk2', cfl=cfg['cfl'],
                            t_end=cfg['t_end'])
            if r['ok']:
                W = r['W']
                rho = W[0]
                p = W[3]
                rho_prof = _strip_profile(mesh, rho, nx, x_min, x_max)
                xp = x_min + (np.arange(nx) + 0.5) * (x_max - x_min) / nx
                drho = np.abs(np.diff(rho_prof))
                shock_proxy = float(np.percentile(drho, 98)) if drho.size else 0.0
                checker = max(_strip_transverse_checker(mesh, rho, nx, x_min, x_max),
                              _strip_transverse_checker(mesh, p, nx, x_min, x_max))
                l1_ref = None
                turbulence = 0.0
                if case == 'lax_1d':
                    exact = _exact_riemann_1d(
                        xp, cfg['t_end'], cfg['left'], cfg['right'],
                        x0=cfg['x0'])
                    l1_ref = float(np.mean(np.abs(rho_prof - exact[0])))
                elif case == 'shock_turbulence_1d':
                    active = (xp > -2.0) & (xp < 4.0)
                    if np.any(active):
                        osc = rho_prof[active] - np.mean(rho_prof[active])
                        turbulence = float(np.sqrt(np.mean(osc * osc)))
                else:
                    active = (xp > 0.12) & (xp < 0.88)
                    if np.any(active):
                        turbulence = float(np.std(rho_prof[active]))
                rho_min = float(np.min(rho))
                rho_max = float(np.max(rho))
                p_min = float(np.min(p))
                p_max = float(np.max(p))
                row_ok = bool(rho_min > 0.0 and p_min > 0.0
                              and checker < 0.20)
                finite = bool(np.all(np.isfinite(rho)) and np.all(np.isfinite(p)))
                field = rho if finite else None
                vtk = {
                    'rho': W[0], 'u': W[1], 'v': W[2], 'p': W[3],
                    'status_ok': np.ones(mesh.n_cells) if row_ok
                    else np.zeros(mesh.n_cells),
                    'gate_failed': np.zeros(mesh.n_cells) if row_ok
                    else np.ones(mesh.n_cells),
                    'diverged': np.zeros(mesh.n_cells),
                }
            else:
                shock_proxy = 0.0
                turbulence = 0.0
                checker = None
                l1_ref = None
                rho_min = None
                rho_max = None
                p_min = None
                p_max = None
                row_ok = False
                field = None
                finite = False
                vtk = {
                    'initial_rho': W0[0],
                    'initial_u': W0[1],
                    'initial_v': W0[2],
                    'initial_p': W0[3],
                    'status_ok': np.zeros(mesh.n_cells),
                    'diverged': np.ones(mesh.n_cells),
                }
            row = dict(case=case, method=name, ok=row_ok,
                       run_ok=bool(r['ok']), finite=finite,
                       mesh='thin_triangular_strip', logical_nx=nx,
                       logical_ny=ny, mesh_cells=mesh.n_cells,
                       mesh_faces=mesh.n_faces, l1_ref=l1_ref,
                       shock_proxy=shock_proxy,
                       turbulence_proxy=turbulence,
                       checker=checker, range_min=rho_min,
                       range_max=rho_max, pressure_min=p_min,
                       pressure_max=p_max, steps=r['steps'],
                       wall=r['wall'], error=r['error'])
            return order, name, row, field, vtk

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
                            n_face_quad=3,
                            face_velocity_mode='central_avg')
            if r['ok']:
                f = r['W'][0]
                finite = bool(np.all(np.isfinite(f)))
                l1 = float(np.sum(np.abs(f - exact) * V) / np.sum(V))
                shape_metrics = _leveque_shape_metrics(mesh, exact, f)
                interior = mesh.face_neighbour >= 0
                jump = np.abs(f[mesh.face_owner[interior]]
                              - f[mesh.face_neighbour[interior]])
                sharp = float(np.percentile(jump, 95))
                rng = [float(np.min(f)), float(np.max(f))]
                wiggle = max(0.0, -rng[0]) + max(0.0, rng[1] - 1.0)
                row_ok = bool(wiggle <= 0.25 and np.isfinite(l1))
                field = f if finite else None
                vtk = {
                    'phi': f,
                    'initial_phi': exact,
                    'phi_error': f - exact,
                    'status_ok': np.ones(mesh.n_cells) if row_ok
                    else np.zeros(mesh.n_cells),
                    'gate_failed': np.zeros(mesh.n_cells) if row_ok
                    else np.ones(mesh.n_cells),
                    'diverged': np.zeros(mesh.n_cells),
                }
            else:
                l1 = None
                shape_metrics = {}
                sharp = 0.0
                rng = [None, None]
                wiggle = None
                row_ok = False
                field = None
                finite = False
                vtk = {
                    'initial_phi': exact,
                    'status_ok': np.zeros(mesh.n_cells),
                    'gate_failed': np.ones(mesh.n_cells),
                    'diverged': np.ones(mesh.n_cells),
                }
            error = r['error']
            if r['ok'] and not row_ok:
                error = f"boundedness gate failed: min={rng[0]:.3g}, max={rng[1]:.3g}"
            row = dict(case='leveque', method=name, ok=row_ok,
                       run_ok=bool(r['ok']), finite=finite,
                       mesh='criss_cross_triangles', logical_nx=N,
                       logical_ny=N, mesh_cells=mesh.n_cells,
                       mesh_faces=mesh.n_faces, l1=l1,
                       sharpness=sharp, range_min=rng[0],
                       range_max=rng[1], wiggle=wiggle,
                       steps=r['steps'], wall=r['wall'], error=error)
            row.update(shape_metrics)
            return order, name, row, field, vtk

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
                vtk = {
                    'rho': W[0], 'u': W[1], 'v': W[2], 'p': W[3],
                    'status_ok': np.ones(mesh.n_cells),
                    'diverged': np.zeros(mesh.n_cells),
                }
            else:
                vortex = 0.0
                vort_p95 = 0.0
                enstrophy = 0.0
                checker = None
                field = None
                vtk = {
                    'initial_rho': W0[0],
                    'initial_u': W0[1],
                    'initial_v': W0[2],
                    'initial_p': W0[3],
                    'status_ok': np.zeros(mesh.n_cells),
                    'diverged': np.ones(mesh.n_cells),
                }
            row = dict(case='double_mach', method=name, ok=r['ok'],
                       mesh='tri_alternating', logical_nx=nx,
                       logical_ny=ny, mesh_cells=mesh.n_cells,
                       mesh_faces=mesh.n_faces,
                       vortex_proxy=vortex, vorticity_p95=vort_p95,
                       enstrophy_proxy=enstrophy, checker=checker,
                       steps=r['steps'], wall=r['wall'], error=r['error'])
            return order, name, row, field, vtk

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
                vtk = {
                    'rho': W[0], 'u': W[1], 'v': W[2], 'p': W[3],
                    'status_ok': np.ones(mesh.n_cells),
                    'diverged': np.zeros(mesh.n_cells),
                }
            else:
                flag = 0.0
                flag_vort = 0.0
                flag_enstrophy = 0.0
                transverse = 0.0
                carbuncle = None
                field = None
                vtk = {
                    'initial_rho': W0[0],
                    'initial_u': W0[1],
                    'initial_v': W0[2],
                    'initial_p': W0[3],
                    'status_ok': np.zeros(mesh.n_cells),
                    'diverged': np.ones(mesh.n_cells),
                }
            row = dict(case='mach3_step', method=name, ok=r['ok'],
                       mesh='tri_alternating', logical_nx=nx,
                       logical_ny=ny, mesh_cells=mesh.n_cells,
                       mesh_faces=mesh.n_faces,
                       flag_proxy=flag, flag_vorticity_p95=flag_vort,
                       flag_enstrophy_proxy=flag_enstrophy,
                       transverse_velocity_rms=transverse,
                       carbuncle=carbuncle,
                       steps=r['steps'], wall=r['wall'], error=r['error'])
            return order, name, row, field, vtk

        raise ValueError(f"unknown case {case!r}")
    except Exception as exc:
        row = dict(case=case, method=name, ok=False, steps=-1,
                   wall=0.0, error=repr(exc))
        return order, name, row, None, None


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
    vtk_fields = {}
    for _, name, row, field, vtk in results:
        rows.append(row)
        if field is not None:
            fields[name] = field
        if vtk is not None:
            vtk_fields[name] = vtk
    return rows, fields, vtk_fields


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
    rows, fields, vtk_fields = _run_case_methods(
        'double_mach', quick, 'double_mach', workers)
    vtk_written = _write_case_vtks(out, 'double_mach', mesh, vtk_fields)
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
    for row in rows:
        row['vtk_written'] = int(
            f"results/T-MLP-u/vtk/double_mach/{_safe_name(row['method'])}.vtk"
            in vtk_written)
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
    rows, fields, vtk_fields = _run_case_methods(
        'mach3_step', quick, 'mach3_step', workers)
    vtk_written = _write_case_vtks(out, 'mach3_step', mesh, vtk_fields)
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
    for row in rows:
        row['vtk_written'] = int(
            f"results/T-MLP-u/vtk/mach3_step/{_safe_name(row['method'])}.vtk"
            in vtk_written)
    return passed, rows


CASE_RUNNERS = [
    ('lax_1d', run_lax_1d),
    ('blast_1d', run_blast_1d),
    ('shock_turbulence_1d', run_shock_turbulence_1d),
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
        help="comma-separated subset of lax_1d,blast_1d,"
             "shock_turbulence_1d,leveque,double_mach,mach3_step")
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
            'T-MLP-u OFF': (
                'modified SUPERBEE TVD-only reconstruction with '
                'mlp_bound=False'),
            'T-MLP-u ON': (
                'T-MLP-u wrapper plus modified SUPERBEE with mlp_bound=True, '
                'vertex_mlp=True, vertex_mlp_cap=2, '
                'virtual_uu_gradient=True, stencil=vertex, order=1, '
                'tvb_M=0, extremum_relax=False'),
            'Double Mach T-MLP-u ON idw_p': 1.0,
            'Mach 3 step T-MLP-u ON idw_p': 0.0,
            '1D strip setup': (
                'Lax, Woodward-Colella blast wave, and Shu-Osher '
                'shock-density interaction are solved as quasi-1D Euler2D '
                'problems on thin alternating-diagonal triangular strips'),
            '1D strip flux': 'hllc_adc on primitive reconstructed Euler2D states',
            '1D strip grids': (
                f'Lax {LAX_PAPER_N}x{STRIP_NY_PAPER}, '
                f'blast {BLAST_PAPER_N}x{STRIP_NY_PAPER}, '
                f'shock/turbulence {SHOCK_TURB_PAPER_N}x{STRIP_NY_PAPER}'),
            'LeVeque T-MLP-u OFF': (
                'pure downwind reconstruction with psi=2, alpha_f=0.5, '
                'and mlp_bound=False'),
            'LeVeque T-MLP-u ON': (
                'BVD selection between two all-TMLP-u candidates with no '
                'MLP-u1 branch: smooth branch = TMLP-u + '
                'bounded CD with LSQ face increment, sharp branch = '
                'TMLP-u + smooth-extremum-preserving TMLP-u shape.  Both '
                'branches use mlp_bound=True, vertex_mlp=True, '
                'vertex_mlp_cap=2, vertex_mlp_augment=True, '
                'face_skew_correction=True, face_gradient_correction=jasak, '
                'virtual_uu_gradient=True, stencil=vertex, order=1, idw_p=0, '
                'tvb_M=0, extremum_relax=False.  BVD is a no-threshold cell '
                'total-boundary-variation choice.'),
            'LeVeque gate': (
                'T-MLP-u ON is judged after one full rotation against the '
                'stored initial cell field phi^0, not a re-sampled analytic '
                'field.  The gate records global E1/E2/Einf, global and '
                'per-body mass drift, per-body centroid/r0 and covariance '
                'errors, peak/min preservation, boundedness, and slotted '
                'cylinder IoU/Dice/fuzzy-IoU/slot-max diagnostics.  PASS '
                'requires finite diagnostics, 0<=phi<=1 within 1e-8, and '
                'T-MLP-u ON must strictly dominate MLP-u1 metric-by-metric: '
                'lower global/body E1/E2/Einf, global mass drift below '
                '1e-6, lower body mass error, lower centroid/r0 and '
                'covariance errors, lower peak-loss '
                'error, lower slot area error and slot max(phi), and higher '
                'slot IoU/Dice/fuzzy-IoU.  Only boundedness metrics whose '
                'physical optimum is exactly zero, namely overshoot, '
                'undershoot, and wiggle, are allowed to tie MLP-u1 at zero. '
                'T-MLP-u OFF must also be unstable or no better in '
                'boundedness.  Any failed MLP-u1 comparison is written in '
                'leveque_gate_mlp_u1_*_failures.'),
            'leveque_flux': (
                'upwind with central-averaged face velocity and 3-point '
                'Gauss face quadrature'),
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
        'lax_1d_ready': int(case_status.get('lax_1d', False)),
        'blast_1d_ready': int(case_status.get('blast_1d', False)),
        'shock_turbulence_1d_ready': int(
            case_status.get('shock_turbulence_1d', False)),
        'leveque_ready': int(case_status.get('leveque', False)),
        'double_mach_ready': int(case_status.get('double_mach', False)),
        'mach3_step_ready': int(case_status.get('mach3_step', False)),
        'quick': bool(args.quick),
        'vtk_dir': 'results/T-MLP-u/vtk',
        'vtk_note': (
            'Legacy ASCII VTK UNSTRUCTURED_GRID files are written for every '
            'scheme. Successful schemes contain final solution fields; '
            'divergent schemes contain diagnostic status fields and the '
            'initial state because no finite final solution exists.'),
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
