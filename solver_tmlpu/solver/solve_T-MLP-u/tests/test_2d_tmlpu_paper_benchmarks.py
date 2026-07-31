"""2D paper-evidence harness for T-MLP-u.

The script writes JSON/TSV/PNG artifacts under ``results/T-MLP-u`` and
prints a final JSON metrics line for codex-autoresearch.
"""
from __future__ import annotations

import argparse
import traceback
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
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba is optional.
    njit = None
    prange = range
    _NUMBA_AVAILABLE = False

from _pkgshim import setup_paths
setup_paths()

from mesh import (criss_cross_box, build_unstructured_2d,
                  triangulate_box_roi_graded)
from equations import Advection, Euler2D
from reconstruction import (
    TMLPU, TMLPUBVD, TMLPUSmoothSharpBVD, BarthJespersen, Venkatakrishnan,
    MLPU1, MLPU1TMLPU, MLPU1TMLPUContact, MLPU2, _build_vertex_neighbours,
)
from boundary import BoundaryCondition
from solver import solve


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _bvd_use_sharp_kernel(n_cells, o_idx, n_idx, jump_s, jump_h,
                              d_owner, d_nei, use_sharp_out):
        # Legacy single-variable kernel retained for backward compatibility
        # with any callers that still drive one variable at a time.  The
        # fused u,v selector below is preferred for the BVD reconstruction.
        tbv_s = np.zeros(n_cells, dtype=np.float64)
        tbv_h = np.zeros(n_cells, dtype=np.float64)
        cp_s = np.zeros(n_cells, dtype=np.float64)
        cp_h = np.zeros(n_cells, dtype=np.float64)
        for f in range(o_idx.shape[0]):
            o = o_idx[f]
            n = n_idx[f]
            js = jump_s[f]
            jh = jump_h[f]
            tbv_s[o] += js
            tbv_s[n] += js
            tbv_h[o] += jh
            tbv_h[n] += jh
            cp_s[o] += js * d_owner[f]
            cp_s[n] += js * d_nei[f]
            cp_h[o] += jh * d_owner[f]
            cp_h[n] += jh * d_nei[f]
        for c in range(n_cells):
            score_s = (tbv_s[c] * tbv_s[c] + cp_s[c] * cp_s[c]) ** 0.5
            score_h = (tbv_h[c] * tbv_h[c] + cp_h[c] * cp_h[c]) ** 0.5
            use_sharp_out[c] = score_h < score_s

    @njit(cache=True)
    def _bvd_select_uv_apply_kernel(
        interior, o_idx, n_idx, d_owner, d_nei,
        smooth_L, smooth_R, sharp_L, sharp_R,
        tbv_s_u, tbv_h_u, cp_s_u, cp_h_u,
        tbv_s_v, tbv_h_v, cp_s_v, cp_h_v,
        use_sharp_u, use_sharp_v,
        W_L, W_R,
    ):
        n_cells = use_sharp_u.shape[0]
        n_int = interior.shape[0]
        for c in range(n_cells):
            tbv_s_u[c] = 0.0
            tbv_h_u[c] = 0.0
            cp_s_u[c] = 0.0
            cp_h_u[c] = 0.0
            tbv_s_v[c] = 0.0
            tbv_h_v[c] = 0.0
            cp_s_v[c] = 0.0
            cp_h_v[c] = 0.0
        for k in range(n_int):
            f = interior[k]
            o = o_idx[k]
            n = n_idx[k]
            do = d_owner[k]
            dn = d_nei[k]
            sLu = smooth_L[1, f]; sRu = smooth_R[1, f]
            hLu = sharp_L[1, f];  hRu = sharp_R[1, f]
            sLv = smooth_L[2, f]; sRv = smooth_R[2, f]
            hLv = sharp_L[2, f];  hRv = sharp_R[2, f]
            ju_s = sLu - sRu
            if ju_s < 0.0:
                ju_s = -ju_s
            ju_h = hLu - hRu
            if ju_h < 0.0:
                ju_h = -ju_h
            jv_s = sLv - sRv
            if jv_s < 0.0:
                jv_s = -jv_s
            jv_h = hLv - hRv
            if jv_h < 0.0:
                jv_h = -jv_h
            tbv_s_u[o] += ju_s; tbv_s_u[n] += ju_s
            tbv_h_u[o] += ju_h; tbv_h_u[n] += ju_h
            cp_s_u[o]  += ju_s * do; cp_s_u[n]  += ju_s * dn
            cp_h_u[o]  += ju_h * do; cp_h_u[n]  += ju_h * dn
            tbv_s_v[o] += jv_s; tbv_s_v[n] += jv_s
            tbv_h_v[o] += jv_h; tbv_h_v[n] += jv_h
            cp_s_v[o]  += jv_s * do; cp_s_v[n]  += jv_s * dn
            cp_h_v[o]  += jv_h * do; cp_h_v[n]  += jv_h * dn
        for c in range(n_cells):
            ss = (tbv_s_u[c] * tbv_s_u[c] + cp_s_u[c] * cp_s_u[c]) ** 0.5
            sh = (tbv_h_u[c] * tbv_h_u[c] + cp_h_u[c] * cp_h_u[c]) ** 0.5
            use_sharp_u[c] = sh < ss
            ss = (tbv_s_v[c] * tbv_s_v[c] + cp_s_v[c] * cp_s_v[c]) ** 0.5
            sh = (tbv_h_v[c] * tbv_h_v[c] + cp_h_v[c] * cp_h_v[c]) ** 0.5
            use_sharp_v[c] = sh < ss
        for k in range(n_int):
            f = interior[k]
            o = o_idx[k]
            n = n_idx[k]
            if use_sharp_u[o] and use_sharp_u[n]:
                W_L[1, f] = sharp_L[1, f]
                W_R[1, f] = sharp_R[1, f]
            else:
                W_L[1, f] = smooth_L[1, f]
                W_R[1, f] = smooth_R[1, f]
            if use_sharp_v[o] and use_sharp_v[n]:
                W_L[2, f] = sharp_L[2, f]
                W_R[2, f] = sharp_R[2, f]
            else:
                W_L[2, f] = smooth_L[2, f]
                W_R[2, f] = smooth_R[2, f]

    @njit(cache=True, parallel=True)
    def _fast_mlpu1_scalar_kernel(
        phi, owner, nei, face_centers, cell_centers,
        nb_safe, valid_nb, d_nb, ata_inv,
        cell_vertex_ids, vertex_offsets, v2c_safe, v2c_valid,
        W_L, W_R,
    ):
        n_cells = phi.shape[0]
        n_faces = owner.shape[0]
        max_nb = nb_safe.shape[1]
        max_v = cell_vertex_ids.shape[1]
        n_nodes = v2c_safe.shape[0]
        max_v2c = v2c_safe.shape[1]
        eps = 1.0e-30

        grad_x = np.zeros(n_cells, dtype=np.float64)
        grad_y = np.zeros(n_cells, dtype=np.float64)
        for c in prange(n_cells):
            rhs0 = 0.0
            rhs1 = 0.0
            pc = phi[c]
            for k in range(max_nb):
                if valid_nb[c, k]:
                    nb = nb_safe[c, k]
                    dw = phi[nb] - pc
                    rhs0 += d_nb[c, k, 0] * dw
                    rhs1 += d_nb[c, k, 1] * dw
            grad_x[c] = ata_inv[c, 0, 0] * rhs0 + ata_inv[c, 0, 1] * rhs1
            grad_y[c] = ata_inv[c, 1, 0] * rhs0 + ata_inv[c, 1, 1] * rhs1

        vmin = np.empty(n_nodes, dtype=np.float64)
        vmax = np.empty(n_nodes, dtype=np.float64)
        for v in prange(n_nodes):
            lo = 0.0
            hi = 0.0
            first = True
            for k in range(max_v2c):
                if v2c_valid[v, k]:
                    val = phi[v2c_safe[v, k]]
                    if first:
                        lo = val
                        hi = val
                        first = False
                    else:
                        if val < lo:
                            lo = val
                        if val > hi:
                            hi = val
            if first:
                lo = 0.0
                hi = 0.0
            vmin[v] = lo
            vmax[v] = hi

        phi_cell = np.ones(n_cells, dtype=np.float64)
        for c in prange(n_cells):
            center = phi[c]
            psi_min = 1.0
            scale = abs(center)
            if scale < 1.0:
                scale = 1.0
            tol = 1.0e-12 * scale
            for j in range(max_v):
                node = cell_vertex_ids[c, j]
                if node < 0:
                    continue
                delta = (grad_x[c] * vertex_offsets[c, j, 0]
                         + grad_y[c] * vertex_offsets[c, j, 1])
                psi = 1.0
                if delta > tol:
                    allowed = vmax[node] - center
                    if allowed < 0.0:
                        allowed = 0.0
                    psi = allowed / delta
                elif delta < -tol:
                    allowed = center - vmin[node]
                    if allowed < 0.0:
                        allowed = 0.0
                    psi = allowed / (-delta)
                if psi < 0.0:
                    psi = 0.0
                elif psi > 1.0:
                    psi = 1.0
                if psi < psi_min:
                    psi_min = psi
            phi_cell[c] = psi_min

        for f in prange(n_faces):
            o = owner[f]
            n = nei[f]
            W_L[f] = phi[o]
            if n >= 0:
                W_R[f] = phi[n]
            else:
                W_R[f] = phi[o]
        for f in prange(n_faces):
            n = nei[f]
            if n < 0:
                continue
            o = owner[f]
            dx = face_centers[f, 0] - cell_centers[o, 0]
            dy = face_centers[f, 1] - cell_centers[o, 1]
            W_L[f] = phi[o] + phi_cell[o] * (
                grad_x[o] * dx + grad_y[o] * dy)
            dx = face_centers[f, 0] - cell_centers[n, 0]
            dy = face_centers[f, 1] - cell_centers[n, 1]
            W_R[f] = phi[n] + phi_cell[n] * (
                grad_x[n] * dx + grad_y[n] * dy)


def _repo_root():
    here = Path(__file__).resolve()
    return here.parents[3]


def _out_dir():
    out = _repo_root() / 'results' / 'T-MLP-u'
    out.mkdir(parents=True, exist_ok=True)
    return out


def _env_grid(name, default):
    s = os.environ.get(name, '')
    if 'x' in s:
        try:
            a, b = s.lower().split('x')
            return (int(a), int(b))
        except Exception:
            return default
    return default


DOUBLE_MACH_QUICK_GRID = _env_grid('TMLPU_DM_QUICK_GRID', (480, 120))
DOUBLE_MACH_PAPER_GRID = (960, 240)
DOUBLE_MACH_FINE_GRID = (480, 120)
MACH3_STEP_QUICK_GRID = (200, 80)
MACH3_STEP_PAPER_GRID = (480, 160)
MACH3_STEP_FINE_GRID = (480, 160)
LEVEQUE_QUICK_N = 100
LEVEQUE_PAPER_N = 100
LEVEQUE_WIGGLE_PASS_LIMIT = float(os.environ.get(
    'TMLPU_LEVEQUE_WIGGLE_PASS_LIMIT', '2.0e-3'))
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


def _plot_field(mesh, field, path, title, vmin=None, vmax=None,
                style='filled'):
    tris, tri_owner = _triangles(mesh)
    tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1],
                             triangles=tris)
    values = np.asarray(field)
    colors = values
    if len(values) == mesh.n_cells:
        colors = values[tri_owner]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.0))
    if style == 'contour':
        node_vals = _node_values_from_cells(mesh, values)
        if (vmin is None or vmax is None or not np.isfinite(vmin)
                or not np.isfinite(vmax) or vmin == vmax):
            vmin = float(np.min(node_vals)) if node_vals.size else 0.0
            vmax = float(np.max(node_vals)) if node_vals.size else 1.0
        if vmin == vmax:
            vmin, vmax = 0.0, 1.0
        # Use a denser contour stack so the density-line structure reads
        # like a physical roll-up band rather than a coarse blob.
        levels = np.linspace(vmin, vmax, 60)
        ax.set_facecolor('white')
        ax.tricontour(tri, node_vals, levels=levels, colors='k',
                      linewidths=0.25, alpha=0.90)
        ax.set_xlim(float(np.min(mesh.nodes[:, 0])),
                    float(np.max(mesh.nodes[:, 0])))
        ax.set_ylim(float(np.min(mesh.nodes[:, 1])),
                    float(np.max(mesh.nodes[:, 1])))
        tcf = None
        title = ''
    else:
        if len(colors) == len(tris):
            tcf = ax.tripcolor(tri, facecolors=colors, shading='flat',
                               vmin=vmin, vmax=vmax)
        else:
            tcf = ax.tripcolor(tri, colors, shading='flat',
                               vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=10)
    if tcf is not None:
        plt.colorbar(tcf, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
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


class TMLPUUnifiedReconstruction:
    """One benchmark key with scalar and Euler admissibility branches."""

    name = 't_mlp_u_unified'

    def __init__(self, scalar_recon, euler_recon):
        self.scalar_recon = scalar_recon
        self.euler_recon = euler_recon

    def set_timestep_context(self, dt, *, total_dt=None, quad_weight=None,
                             quad_points=None, quad_weights=None):
        for recon in (self.scalar_recon, self.euler_recon):
            if hasattr(recon, 'set_timestep_context'):
                recon.set_timestep_context(
                    dt, total_dt=total_dt, quad_weight=quad_weight,
                    quad_points=quad_points, quad_weights=quad_weights)

    def apply_quadrature_update_bound(self, mesh, W_cell, eq, W_L_quad,
                                      W_R_quad, eval_points_quad,
                                      quad_weights):
        recon = self.scalar_recon if getattr(eq, 'nvar', 0) == 1 else self.euler_recon
        if hasattr(recon, 'apply_quadrature_update_bound'):
            return recon.apply_quadrature_update_bound(
                mesh, W_cell, eq, W_L_quad, W_R_quad, eval_points_quad,
                quad_weights)
        return W_L_quad, W_R_quad

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        recon = self.scalar_recon if getattr(eq, 'nvar', 0) == 1 else self.euler_recon
        return recon.reconstruct(mesh, W_cell, eq, eval_points=eval_points)


class EulerPrimitiveBlendReconstruction:
    """Blend primitive Euler face states from two reconstruction candidates."""

    name = 'euler_primitive_blend'

    def __init__(self, density_velocity_recon, pressure_recon):
        self.density_velocity_recon = density_velocity_recon
        self.pressure_recon = pressure_recon

    def set_timestep_context(self, dt, *, total_dt=None, quad_weight=None,
                             quad_points=None, quad_weights=None):
        for recon in (self.density_velocity_recon, self.pressure_recon):
            if hasattr(recon, 'set_timestep_context'):
                recon.set_timestep_context(
                    dt, total_dt=total_dt, quad_weight=quad_weight,
                    quad_points=quad_points, quad_weights=quad_weights)

    def apply_quadrature_update_bound(self, mesh, W_cell, eq, W_L_quad,
                                      W_R_quad, eval_points_quad,
                                      quad_weights):
        if hasattr(self.pressure_recon, 'apply_quadrature_update_bound'):
            return self.pressure_recon.apply_quadrature_update_bound(
                mesh, W_cell, eq, W_L_quad, W_R_quad, eval_points_quad,
                quad_weights)
        return W_L_quad, W_R_quad

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L_p, W_R_p = self.pressure_recon.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if getattr(eq, 'nvar', 0) < 4:
            return W_L_p, W_R_p
        W_L_dv, W_R_dv = self.density_velocity_recon.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        W_L = W_L_p.copy()
        W_R = W_R_p.copy()
        W_L[0:3] = W_L_dv[0:3]
        W_R[0:3] = W_R_dv[0:3]
        return W_L, W_R


class EulerPrimitiveSlotBlendReconstruction:
    """Use selected Euler primitive slots from an auxiliary reconstruction."""

    name = 'euler_primitive_slot_blend'

    def __init__(self, base_recon, slot_recon, slots):
        self.base_recon = base_recon
        self.slot_recon = slot_recon
        self.slots = tuple(int(slot) for slot in slots)

    def set_timestep_context(self, dt, *, total_dt=None, quad_weight=None,
                             quad_points=None, quad_weights=None):
        for recon in (self.base_recon, self.slot_recon):
            if hasattr(recon, 'set_timestep_context'):
                recon.set_timestep_context(
                    dt, total_dt=total_dt, quad_weight=quad_weight,
                    quad_points=quad_points, quad_weights=quad_weights)

    def apply_quadrature_update_bound(self, mesh, W_cell, eq, W_L_quad,
                                      W_R_quad, eval_points_quad,
                                      quad_weights):
        if hasattr(self.base_recon, 'apply_quadrature_update_bound'):
            return self.base_recon.apply_quadrature_update_bound(
                mesh, W_cell, eq, W_L_quad, W_R_quad, eval_points_quad,
                quad_weights)
        return W_L_quad, W_R_quad

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L_base, W_R_base = self.base_recon.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if getattr(eq, 'nvar', 0) < 4 or not self.slots:
            return W_L_base, W_R_base
        W_L_slot, W_R_slot = self.slot_recon.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        W_L = W_L_base.copy()
        W_R = W_R_base.copy()
        for slot in self.slots:
            W_L[slot] = W_L_slot[slot]
            W_R[slot] = W_R_slot[slot]
        return W_L, W_R


def _tmlpu_unified_scalar():
    variant = os.environ.get('TMLPU_UNIFIED_CANDIDATE', 'u0').strip().lower()
    order_default = '2' if variant == 'u1' else '1'
    zero_delta_default = '0.084' if variant == 'u1' else '2.0'
    interface_tvd = os.environ.get('TMLPU_UNIFIED_INTERFACE_TVD', 'none')
    if interface_tvd.strip().lower() in ('', 'none', 'off', 'false', '0'):
        interface_tvd = None
    return TMLPUSmoothSharpBVD(
        smooth_mode='tmlpu',
        smooth_tvd=os.environ.get('TMLPU_UNIFIED_SMOOTH_TVD', 'bounded_cd'),
        smooth_face_increment=os.environ.get(
            'TMLPU_UNIFIED_SMOOTH_FACE_INCREMENT', 'lsq'),
        sharp_tvd=os.environ.get('TMLPU_UNIFIED_SHARP_TVD', 'tmlpu_shape'),
        sharp_face_increment=os.environ.get(
            'TMLPU_UNIFIED_SHARP_FACE_INCREMENT', 'tmlpu'),
        interface_tvd=interface_tvd,
        stencil=os.environ.get('TMLPU_UNIFIED_STENCIL', 'vertex'),
        order=int(os.environ.get('TMLPU_UNIFIED_ORDER', order_default)),
        idw_p=float(os.environ.get('TMLPU_UNIFIED_IDW_P', '0.0')),
        vertex_mlp_cap=float(os.environ.get(
            'TMLPU_UNIFIED_VERTEX_MLP_CAP', '2.0')),
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_FACE_GRADIENT_CORRECTION', 'jasak'),
        vertex_mlp_augment=True,
        r_form=os.environ.get('TMLPU_R_FORM', 'far_upwind'),
        clamp_tstar=True,
        zero_delta_psi=float(os.environ.get(
            'TMLPU_ZERO_DELTA_PSI', zero_delta_default)),
        moment_bvd=True,
        moment_bvd_mode=os.environ.get(
            'TMLPU_UNIFIED_MOMENT_BVD_MODE', 'product'),
        sharp_bvd_factor=float(os.environ.get(
            'TMLPU_UNIFIED_SHARP_BVD_FACTOR', '0.72')),
        sharp_bvd_factor_mode=os.environ.get(
            'TMLPU_UNIFIED_SHARP_BVD_FACTOR_MODE', 'linear_residual'),
        unit_interval_face_bound=True,
        update_bound_mode=os.environ.get(
            'TMLPU_UNIFIED_UPDATE_BOUND_MODE', 'zalesak_actual'),
    )


def _tmlpu_unified_euler():
    return TMLPU(
        tvd=os.environ.get('TMLPU_UNIFIED_EULER_TVD', 'superbee'),
        hancock_courant=0.0,
        mlp_bound=True,
        extremum_relax=False,
        extremum_relax_curved_otsu=False,
        tvb_M=0.0,
        vertex_mlp=True,
        vertex_mlp_cap=float(os.environ.get(
            'TMLPU_UNIFIED_VERTEX_MLP_CAP', '2.0')),
        euler_shock_flatten=False,
        euler_density_acoustic_flatten=False,
        euler_density_tvd=None,
        euler_density_lsq_increment=False,
        euler_density_no_hancock=False,
        euler_density_entropy_split=False,
        euler_density_entropy_variable=False,
        euler_density_shear_contact=False,
        euler_density_contact_wave_hancock=False,
        euler_density_pressure_entropy=False,
        euler_density_contact_bvd=False,
        euler_density_contact_cell_bvd=False,
        euler_density_first_order=False,
        euler_pressure_first_order=False,
        euler_velocity_no_hancock=False,
        euler_velocity_shock_flatten=False,
        euler_velocity_lsq_increment=False,
        euler_velocity_tvd=None,
        euler_tangential_velocity_tvd=None,
        euler_tangential_velocity_no_hancock=False,
        euler_tangential_contact_wave_hancock=False,
        euler_tangential_velocity_lsq_increment=False,
        euler_local_hancock=False,
        euler_log_positive=False,
        virtual_uu_gradient=True,
        vertex_mlp_augment=True,
        face_skew_correction=True,
        phi_LL_unclipped=False,
        zero_delta_psi=float(os.environ.get('TMLPU_ZERO_DELTA_PSI', '0.084')),
        face_gradient_correction=os.environ.get(
            'TMLPU_FACE_GRADIENT_CORRECTION', 'jasak'),
        face_increment=os.environ.get('TMLPU_UNIFIED_EULER_FACE_INCREMENT',
                                      'tmlpu'),
        r_form=os.environ.get('TMLPU_R_FORM', 'far_upwind'),
        stencil=os.environ.get('TMLPU_UNIFIED_STENCIL', 'vertex'),
        order=int(os.environ.get('TMLPU_UNIFIED_EULER_ORDER', '2')),
        idw_p=float(os.environ.get('TMLPU_UNIFIED_EULER_IDW_P', '0.0')),
    )


def _tmlpu_unified():
    policy = os.environ.get('TMLPU_UNIFIED_FLOW_POLICY', 'auto').lower()
    if policy == 'scalar_bvd':
        return _tmlpu_unified_scalar()
    if policy == 'euler_clean':
        return _tmlpu_unified_euler()
    return TMLPUUnifiedReconstruction(
        _tmlpu_unified_scalar(), _tmlpu_unified_euler())


def _tmlpu_v3_unified_scalar():
    return TMLPU(
        tvd='pure_downwind',
        mlp_bound=True,
        extremum_relax=False,
        tvb_M=float(os.environ.get('TMLPU_V220_TVB_M', '0.0')),
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=1.0,
        vertex_mlp_augment=True,
        face_skew_correction=True,
        face_gradient_correction='jasak',
        face_increment='tmlpu',
        r_form='far_upwind',
        clamp_tstar=True,
        stencil='face',
        order=2,
        idw_p=6.0,
    )


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


class _ScalarJumpDampedReconstruction:
    def __init__(self, inner, jump_min=0.18, jump_full=0.55,
                 min_theta=0.35):
        self.inner = inner
        self.jump_min = float(jump_min)
        self.jump_full = float(jump_full)
        self.min_theta = float(np.clip(min_theta, 0.0, 1.0))

    @staticmethod
    def _smoothstep(edge0, edge1, x):
        width = max(float(edge1) - float(edge0), np.finfo(float).eps)
        t = np.clip((x - float(edge0)) / width, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L, W_R = self.inner.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if W_cell.shape[0] != 1:
            return W_L, W_R
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R
        o_idx = owner[interior]
        n_idx = nei[interior]
        phi_o = W_cell[0, o_idx]
        phi_n = W_cell[0, n_idx]
        jump = np.abs(phi_o - phi_n)
        theta = self.min_theta + (1.0 - self.min_theta) * self._smoothstep(
            self.jump_min, self.jump_full, jump)
        W_L = W_L.copy()
        W_R = W_R.copy()
        W_L[0, interior] = phi_o + theta * (W_L[0, interior] - phi_o)
        W_R[0, interior] = phi_n + theta * (W_R[0, interior] - phi_n)
        return W_L, W_R


class _ScalarJumpDampedBinaryGuardReconstruction(_ScalarJumpDampedReconstruction):
    def __init__(self, inner, jump_min=0.18, jump_full=0.55,
                 min_theta=0.90, low_plateau=0.03, high_plateau=0.92):
        super().__init__(
            inner, jump_min=jump_min, jump_full=jump_full,
            min_theta=min_theta)
        self.low_plateau = float(low_plateau)
        self.high_plateau = float(high_plateau)

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L, W_R = self.inner.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if W_cell.shape[0] != 1:
            return W_L, W_R
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R
        o_idx = owner[interior]
        n_idx = nei[interior]
        phi_o = W_cell[0, o_idx]
        phi_n = W_cell[0, n_idx]
        jump = np.abs(phi_o - phi_n)
        theta = self.min_theta + (1.0 - self.min_theta) * self._smoothstep(
            self.jump_min, self.jump_full, jump)
        plateau = (
            ((phi_o <= self.low_plateau) & (phi_n <= self.low_plateau))
            | ((phi_o >= self.high_plateau) & (phi_n >= self.high_plateau))
        )
        theta = np.where(plateau, 1.0, theta)
        W_L = W_L.copy()
        W_R = W_R.copy()
        W_L[0, interior] = phi_o + theta * (W_L[0, interior] - phi_o)
        W_R[0, interior] = phi_n + theta * (W_R[0, interior] - phi_n)
        return W_L, W_R


class _ScalarSmoothBodyGuardReconstruction(_ScalarJumpDampedBinaryGuardReconstruction):
    def __init__(self, inner, jump_min=0.18, jump_full=0.55,
                 min_theta=0.90, low_plateau=0.03, high_plateau=0.92,
                 smooth_theta=0.68, smooth_value_low=0.06,
                 smooth_value_high=0.90, smooth_jump_min=0.02,
                 smooth_jump_full=0.16, smooth_range_min=None,
                 smooth_range_full=None):
        super().__init__(
            inner, jump_min=jump_min, jump_full=jump_full,
            min_theta=min_theta, low_plateau=low_plateau,
            high_plateau=high_plateau)
        self.smooth_theta = float(np.clip(smooth_theta, 0.0, 1.0))
        self.smooth_value_low = float(smooth_value_low)
        self.smooth_value_high = float(smooth_value_high)
        self.smooth_jump_min = float(smooth_jump_min)
        self.smooth_jump_full = float(smooth_jump_full)
        self.smooth_range_min = (
            None if smooth_range_min is None else float(smooth_range_min))
        self.smooth_range_full = (
            None if smooth_range_full is None else float(smooth_range_full))

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L, W_R = self.inner.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if W_cell.shape[0] != 1:
            return W_L, W_R
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R
        o_idx = owner[interior]
        n_idx = nei[interior]
        phi_o = W_cell[0, o_idx]
        phi_n = W_cell[0, n_idx]
        jump = np.abs(phi_o - phi_n)
        theta = self.min_theta + (1.0 - self.min_theta) * self._smoothstep(
            self.jump_min, self.jump_full, jump)
        plateau = (
            ((phi_o <= self.low_plateau) & (phi_n <= self.low_plateau))
            | ((phi_o >= self.high_plateau) & (phi_n >= self.high_plateau))
        )
        mid_phi = 0.5 * (phi_o + phi_n)
        value_on = self._smoothstep(
            self.smooth_value_low, self.smooth_value_high, mid_phi)
        value_off = 1.0 - self._smoothstep(
            self.smooth_value_high, 1.0, mid_phi)
        smooth_value = np.clip(value_on * value_off, 0.0, 1.0)
        smooth_jump = 1.0 - self._smoothstep(
            self.smooth_jump_min, self.smooth_jump_full, jump)
        smooth_weight = np.where(plateau, 0.0, smooth_value * smooth_jump)
        if self.smooth_range_min is not None and self.smooth_range_full is not None:
            phi = W_cell[0]
            local_min = np.array(phi, copy=True)
            local_max = np.array(phi, copy=True)
            np.minimum.at(local_min, o_idx, phi_n)
            np.minimum.at(local_min, n_idx, phi_o)
            np.maximum.at(local_max, o_idx, phi_n)
            np.maximum.at(local_max, n_idx, phi_o)
            local_range = local_max - local_min
            face_range = np.maximum(local_range[o_idx], local_range[n_idx])
            range_weight = 1.0 - self._smoothstep(
                self.smooth_range_min, self.smooth_range_full, face_range)
            smooth_weight *= np.clip(range_weight, 0.0, 1.0)
        theta = theta + smooth_weight * (self.smooth_theta - theta)
        theta = np.where(plateau, 1.0, theta)
        W_L = W_L.copy()
        W_R = W_R.copy()
        W_L[0, interior] = phi_o + theta * (W_L[0, interior] - phi_o)
        W_R[0, interior] = phi_n + theta * (W_R[0, interior] - phi_n)
        return W_L, W_R


class _ScalarSelfBVDRangeSmoothGuardReconstruction(_ScalarSmoothBodyGuardReconstruction):
    def __init__(self, inner, bvd_mode='product', **kwargs):
        super().__init__(inner, **kwargs)
        self.bvd_mode = str(bvd_mode).lower()

    def _damped_from_raw(self, mesh, W_cell, W_L_raw, W_R_raw):
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L_raw, W_R_raw, interior
        o_idx = owner[interior]
        n_idx = nei[interior]
        phi_o = W_cell[0, o_idx]
        phi_n = W_cell[0, n_idx]
        jump = np.abs(phi_o - phi_n)
        theta = self.min_theta + (1.0 - self.min_theta) * self._smoothstep(
            self.jump_min, self.jump_full, jump)
        plateau = (
            ((phi_o <= self.low_plateau) & (phi_n <= self.low_plateau))
            | ((phi_o >= self.high_plateau) & (phi_n >= self.high_plateau))
        )
        mid_phi = 0.5 * (phi_o + phi_n)
        value_on = self._smoothstep(
            self.smooth_value_low, self.smooth_value_high, mid_phi)
        value_off = 1.0 - self._smoothstep(
            self.smooth_value_high, 1.0, mid_phi)
        smooth_value = np.clip(value_on * value_off, 0.0, 1.0)
        smooth_jump = 1.0 - self._smoothstep(
            self.smooth_jump_min, self.smooth_jump_full, jump)
        smooth_weight = np.where(plateau, 0.0, smooth_value * smooth_jump)
        if self.smooth_range_min is not None and self.smooth_range_full is not None:
            phi = W_cell[0]
            local_min = np.array(phi, copy=True)
            local_max = np.array(phi, copy=True)
            np.minimum.at(local_min, o_idx, phi_n)
            np.minimum.at(local_min, n_idx, phi_o)
            np.maximum.at(local_max, o_idx, phi_n)
            np.maximum.at(local_max, n_idx, phi_o)
            local_range = local_max - local_min
            face_range = np.maximum(local_range[o_idx], local_range[n_idx])
            range_weight = 1.0 - self._smoothstep(
                self.smooth_range_min, self.smooth_range_full, face_range)
            smooth_weight *= np.clip(range_weight, 0.0, 1.0)
        theta = theta + smooth_weight * (self.smooth_theta - theta)
        theta = np.where(plateau, 1.0, theta)
        W_L = W_L_raw.copy()
        W_R = W_R_raw.copy()
        W_L[0, interior] = phi_o + theta * (W_L_raw[0, interior] - phi_o)
        W_R[0, interior] = phi_n + theta * (W_R_raw[0, interior] - phi_n)
        return W_L, W_R, interior

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L_raw, W_R_raw = self.inner.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if W_cell.shape[0] != 1:
            return W_L_raw, W_R_raw
        W_L_damp, W_R_damp, interior = self._damped_from_raw(
            mesh, W_cell, W_L_raw, W_R_raw)
        if interior.size == 0:
            return W_L_raw, W_R_raw
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        o_idx = owner[interior]
        n_idx = nei[interior]
        n_cells = W_cell.shape[1]
        score_points = (mesh.face_centers[interior] if eval_points is None
                        else eval_points[interior])
        try:
            wave = eq.max_wave_speed(
                W_cell[:, o_idx], mesh.face_normals[interior],
                points=score_points)
        except TypeError:
            wave = eq.max_wave_speed(
                W_cell[:, o_idx], mesh.face_normals[interior])
        bvd_weight = mesh.face_areas[interior] * np.asarray(wave, dtype=float)

        def _scores(left, right):
            j = np.abs(left[0, interior] - right[0, interior]) * bvd_weight
            tbv = (
                np.bincount(o_idx, weights=j, minlength=n_cells)
                + np.bincount(n_idx, weights=j, minlength=n_cells))
            d_o = score_points - mesh.cell_centers[o_idx]
            d_n = score_points - mesh.cell_centers[n_idx]
            mx = (
                np.bincount(o_idx, weights=j * d_o[:, 0], minlength=n_cells)
                + np.bincount(n_idx, weights=j * d_n[:, 0], minlength=n_cells))
            my = (
                np.bincount(o_idx, weights=j * d_o[:, 1], minlength=n_cells)
                + np.bincount(n_idx, weights=j * d_n[:, 1], minlength=n_cells))
            return tbv, np.sqrt(mx * mx + my * my)

        tbv_raw, mbv_raw = _scores(W_L_raw, W_R_raw)
        tbv_damp, mbv_damp = _scores(W_L_damp, W_R_damp)
        if self.bvd_mode == 'and':
            use_damp = (tbv_damp < tbv_raw) & (mbv_damp < mbv_raw)
        elif self.bvd_mode == 'moment':
            use_damp = mbv_damp < mbv_raw
        else:
            use_damp = (tbv_damp * mbv_damp) < (tbv_raw * mbv_raw)
        W_L = W_L_raw.copy()
        W_R = W_R_raw.copy()
        mask_L = use_damp[o_idx]
        if np.any(mask_L):
            W_L[0, interior[mask_L]] = W_L_damp[0, interior[mask_L]]
        mask_R = use_damp[n_idx]
        if np.any(mask_R):
            W_R[0, interior[mask_R]] = W_R_damp[0, interior[mask_R]]
        return W_L, W_R


class _ScalarCellBalancedReconstruction:
    def __init__(self, inner, strength=0.5, clip_bounds=True):
        self.inner = inner
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.clip_bounds = bool(clip_bounds)

    def set_timestep_context(self, dt, *, total_dt=None, quad_weight=None,
                             quad_points=None, quad_weights=None):
        if hasattr(self.inner, 'set_timestep_context'):
            self.inner.set_timestep_context(
                dt, total_dt=total_dt, quad_weight=quad_weight,
                quad_points=quad_points, quad_weights=quad_weights)

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L, W_R = self.inner.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if W_cell.shape[0] != 1 or self.strength <= 0.0:
            return W_L, W_R
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        area = np.asarray(mesh.face_areas, dtype=float)
        n_cells = W_cell.shape[1]
        phi = W_cell[0]

        W_L = W_L.copy()
        W_R = W_R.copy()
        delta_L = W_L[0] - phi[owner]
        sum_o = np.bincount(owner, weights=delta_L * area, minlength=n_cells)
        w_o = np.bincount(owner, weights=area, minlength=n_cells)
        mean_o = np.divide(
            sum_o, np.maximum(w_o, np.finfo(float).tiny),
            out=np.zeros_like(sum_o), where=w_o > 0.0)
        W_L[0] -= self.strength * mean_o[owner]

        interior_mask = nei >= 0
        if np.any(interior_mask):
            n_idx = nei[interior_mask]
            delta_R = W_R[0, interior_mask] - phi[n_idx]
            area_i = area[interior_mask]
            sum_n = np.bincount(
                n_idx, weights=delta_R * area_i, minlength=n_cells)
            w_n = np.bincount(n_idx, weights=area_i, minlength=n_cells)
            mean_n = np.divide(
                sum_n, np.maximum(w_n, np.finfo(float).tiny),
                out=np.zeros_like(sum_n), where=w_n > 0.0)
            W_R[0, interior_mask] -= self.strength * mean_n[n_idx]

        if self.clip_bounds:
            local_min = np.array(phi, copy=True)
            local_max = np.array(phi, copy=True)
            if np.any(interior_mask):
                o_i = owner[interior_mask]
                n_i = nei[interior_mask]
                np.minimum.at(local_min, o_i, phi[n_i])
                np.minimum.at(local_min, n_i, phi[o_i])
                np.maximum.at(local_max, o_i, phi[n_i])
                np.maximum.at(local_max, n_i, phi[o_i])
            W_L[0] = np.clip(W_L[0], local_min[owner], local_max[owner])
            if np.any(interior_mask):
                W_R[0, interior_mask] = np.clip(
                    W_R[0, interior_mask],
                    local_min[nei[interior_mask]],
                    local_max[nei[interior_mask]])
        return W_L, W_R


class _ScalarSelectiveSmoothCellBalancedReconstruction:
    def __init__(self, inner, strength=0.32, value_low=0.035,
                 value_high=0.86, range_on_min=0.006,
                 range_on_full=0.045, range_off_min=0.24,
                 range_off_full=0.42, clip_bounds=True):
        self.inner = inner
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.value_low = float(value_low)
        self.value_high = float(value_high)
        self.range_on_min = float(range_on_min)
        self.range_on_full = float(range_on_full)
        self.range_off_min = float(range_off_min)
        self.range_off_full = float(range_off_full)
        self.clip_bounds = bool(clip_bounds)

    @staticmethod
    def _smoothstep(edge0, edge1, x):
        width = max(float(edge1) - float(edge0), np.finfo(float).eps)
        t = np.clip((x - float(edge0)) / width, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def set_timestep_context(self, dt, *, total_dt=None, quad_weight=None,
                             quad_points=None, quad_weights=None):
        if hasattr(self.inner, 'set_timestep_context'):
            self.inner.set_timestep_context(
                dt, total_dt=total_dt, quad_weight=quad_weight,
                quad_points=quad_points, quad_weights=quad_weights)

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L, W_R = self.inner.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if W_cell.shape[0] != 1 or self.strength <= 0.0:
            return W_L, W_R
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R

        n_cells = W_cell.shape[1]
        phi = W_cell[0]
        o_i = owner[interior]
        n_i = nei[interior]
        phi_o = phi[o_i]
        phi_n = phi[n_i]

        local_min = np.array(phi, copy=True)
        local_max = np.array(phi, copy=True)
        np.minimum.at(local_min, o_i, phi_n)
        np.minimum.at(local_min, n_i, phi_o)
        np.maximum.at(local_max, o_i, phi_n)
        np.maximum.at(local_max, n_i, phi_o)
        local_range = local_max - local_min

        mid_value = np.clip(
            self._smoothstep(self.value_low, self.value_high, phi)
            * (1.0 - self._smoothstep(self.value_high, 1.0, phi)),
            0.0, 1.0)
        range_weight = np.clip(
            self._smoothstep(
                self.range_on_min, self.range_on_full, local_range)
            * (1.0 - self._smoothstep(
                self.range_off_min, self.range_off_full, local_range)),
            0.0, 1.0)
        cell_weight = self.strength * mid_value * range_weight
        if not np.any(cell_weight > 0.0):
            return W_L, W_R

        area = np.asarray(mesh.face_areas, dtype=float)
        W_L = W_L.copy()
        W_R = W_R.copy()

        delta_L = W_L[0] - phi[owner]
        sum_o = np.bincount(owner, weights=delta_L * area, minlength=n_cells)
        w_o = np.bincount(owner, weights=area, minlength=n_cells)
        mean_o = np.divide(
            sum_o, np.maximum(w_o, np.finfo(float).tiny),
            out=np.zeros_like(sum_o), where=w_o > 0.0)
        W_L[0] -= cell_weight[owner] * mean_o[owner]

        delta_R = W_R[0, interior] - phi[n_i]
        area_i = area[interior]
        sum_n = np.bincount(
            n_i, weights=delta_R * area_i, minlength=n_cells)
        w_n = np.bincount(n_i, weights=area_i, minlength=n_cells)
        mean_n = np.divide(
            sum_n, np.maximum(w_n, np.finfo(float).tiny),
            out=np.zeros_like(sum_n), where=w_n > 0.0)
        W_R[0, interior] -= cell_weight[n_i] * mean_n[n_i]

        if self.clip_bounds:
            W_L[0] = np.clip(W_L[0], local_min[owner], local_max[owner])
            W_R[0, interior] = np.clip(
                W_R[0, interior], local_min[n_i], local_max[n_i])
        return W_L, W_R


class _ScalarMLPU1JumpSharpener:
    def __init__(self, edge_min=0.28, edge_full=0.58, alpha=1.0):
        self.base = MLPU1()
        self.edge_min = float(edge_min)
        self.edge_full = float(edge_full)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    @staticmethod
    def _smoothstep(edge0, edge1, x):
        width = max(float(edge1) - float(edge0), np.finfo(float).eps)
        t = np.clip((x - float(edge0)) / width, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L, W_R = self.base.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if W_cell.shape[0] != 1 or self.alpha <= 0.0:
            return W_L, W_R
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R
        o_idx = owner[interior]
        n_idx = nei[interior]
        phi_o = W_cell[0, o_idx]
        phi_n = W_cell[0, n_idx]
        theta = self.alpha * self._smoothstep(
            self.edge_min, self.edge_full, np.abs(phi_o - phi_n))
        if not np.any(theta > 0.0):
            return W_L, W_R
        W_L = W_L.copy()
        W_R = W_R.copy()
        W_L[0, interior] = (
            (1.0 - theta) * W_L[0, interior] + theta * phi_o)
        W_R[0, interior] = (
            (1.0 - theta) * W_R[0, interior] + theta * phi_n)
        return W_L, W_R


class _ScalarMLPU1EdgeAntidiffusive:
    def __init__(self, edge_min=0.06, edge_full=0.28, alpha=0.18,
                 clip_unit=True, local_range_min=None,
                 local_range_full=None):
        self.base = MLPU1()
        self.edge_min = float(edge_min)
        self.edge_full = float(edge_full)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.clip_unit = bool(clip_unit)
        self.local_range_min = (
            None if local_range_min is None else float(local_range_min))
        self.local_range_full = (
            None if local_range_full is None else float(local_range_full))

    @staticmethod
    def _smoothstep(edge0, edge1, x):
        width = max(float(edge1) - float(edge0), np.finfo(float).eps)
        t = np.clip((x - float(edge0)) / width, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L, W_R = self.base.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if W_cell.shape[0] != 1 or self.alpha <= 0.0:
            return W_L, W_R
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R
        o_idx = owner[interior]
        n_idx = nei[interior]
        phi_o = W_cell[0, o_idx]
        phi_n = W_cell[0, n_idx]
        jump = phi_o - phi_n
        theta = self.alpha * self._smoothstep(
            self.edge_min, self.edge_full, np.abs(jump))
        if self.local_range_min is not None and self.local_range_full is not None:
            phi = W_cell[0]
            local_min = np.array(phi, copy=True)
            local_max = np.array(phi, copy=True)
            np.minimum.at(local_min, o_idx, phi_n)
            np.minimum.at(local_min, n_idx, phi_o)
            np.maximum.at(local_max, o_idx, phi_n)
            np.maximum.at(local_max, n_idx, phi_o)
            local_range = local_max - local_min
            face_range = np.maximum(local_range[o_idx], local_range[n_idx])
            theta *= self._smoothstep(
                self.local_range_min, self.local_range_full, face_range)
        if not np.any(theta > 0.0):
            return W_L, W_R
        W_L = W_L.copy()
        W_R = W_R.copy()
        W_L[0, interior] = W_L[0, interior] + theta * jump
        W_R[0, interior] = W_R[0, interior] - theta * jump
        if self.clip_unit:
            W_L[0, interior] = np.clip(W_L[0, interior], 0.0, 1.0)
            W_R[0, interior] = np.clip(W_R[0, interior], 0.0, 1.0)
        return W_L, W_R


class _ScalarMLPU1BoundedIncrementBoost:
    def __init__(self, smooth_min=0.006, smooth_full=0.055,
                 edge_min=0.18, edge_full=0.55,
                 smooth_alpha=0.16, edge_alpha=0.90,
                 clip_unit=True):
        self.base = MLPU1()
        self.smooth_min = float(smooth_min)
        self.smooth_full = float(smooth_full)
        self.edge_min = float(edge_min)
        self.edge_full = float(edge_full)
        self.smooth_alpha = float(max(0.0, smooth_alpha))
        self.edge_alpha = float(max(0.0, edge_alpha))
        self.clip_unit = bool(clip_unit)

    @staticmethod
    def _smoothstep(edge0, edge1, x):
        width = max(float(edge1) - float(edge0), np.finfo(float).eps)
        t = np.clip((x - float(edge0)) / width, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        W_L, W_R = self.base.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        if W_cell.shape[0] != 1:
            return W_L, W_R
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R
        o_idx = owner[interior]
        n_idx = nei[interior]
        phi = W_cell[0]
        phi_o = phi[o_idx]
        phi_n = phi[n_idx]

        local_min = np.array(phi, copy=True)
        local_max = np.array(phi, copy=True)
        np.minimum.at(local_min, o_idx, phi_n)
        np.minimum.at(local_min, n_idx, phi_o)
        np.maximum.at(local_max, o_idx, phi_n)
        np.maximum.at(local_max, n_idx, phi_o)
        local_range = local_max - local_min
        face_range = np.maximum(local_range[o_idx], local_range[n_idx])

        smooth_w = self._smoothstep(
            self.smooth_min, self.smooth_full, face_range)
        edge_w = self._smoothstep(self.edge_min, self.edge_full, face_range)
        boost = 1.0 + self.smooth_alpha * smooth_w + self.edge_alpha * edge_w
        if not np.any(boost > 1.0):
            return W_L, W_R

        W_L = W_L.copy()
        W_R = W_R.copy()
        dL = W_L[0, interior] - phi_o
        dR = W_R[0, interior] - phi_n
        W_L[0, interior] = phi_o + boost * dL
        W_R[0, interior] = phi_n + boost * dR
        lo_o = local_min[o_idx]
        hi_o = local_max[o_idx]
        lo_n = local_min[n_idx]
        hi_n = local_max[n_idx]
        if self.clip_unit:
            lo_o = np.maximum(lo_o, 0.0)
            hi_o = np.minimum(hi_o, 1.0)
            lo_n = np.maximum(lo_n, 0.0)
            hi_n = np.minimum(hi_n, 1.0)
        W_L[0, interior] = np.clip(W_L[0, interior], lo_o, hi_o)
        W_R[0, interior] = np.clip(W_R[0, interior], lo_n, hi_n)
        return W_L, W_R


def _fast_mlpu1_scalar_context(mesh):
    ctx = getattr(mesh, '_tmlpu_fast_mlpu1_scalar_ctx', None)
    if ctx is not None:
        return ctx
    n_cells = mesh.cell_centers.shape[0]
    nb_lists = _build_vertex_neighbours(mesh, n_rings=1)
    max_nb = max((len(nbs) for nbs in nb_lists), default=1)
    max_nb = max(max_nb, 1)
    nb = np.full((n_cells, max_nb), -1, dtype=np.int64)
    for c, nbs in enumerate(nb_lists):
        valid = [int(k) for k in nbs if int(k) >= 0]
        nb[c, :len(valid)] = valid
    valid_nb = nb >= 0
    nb_safe = np.where(valid_nb, nb, 0).astype(np.int64, copy=False)
    cc = np.asarray(mesh.cell_centers, dtype=float)
    d_nb = (cc[nb_safe] - cc[:, None, :]) * valid_nb[:, :, None]
    ata = np.einsum('cki,ckj->cij', d_nb, d_nb)
    ata_inv = np.zeros_like(ata)
    det = ata[:, 0, 0] * ata[:, 1, 1] - ata[:, 0, 1] * ata[:, 1, 0]
    ok = np.abs(det) > 1.0e-30
    det_safe = np.where(ok, det, 1.0)
    ata_inv[:, 0, 0] = ata[:, 1, 1] / det_safe
    ata_inv[:, 1, 1] = ata[:, 0, 0] / det_safe
    ata_inv[:, 0, 1] = -ata[:, 0, 1] / det_safe
    ata_inv[:, 1, 0] = -ata[:, 1, 0] / det_safe
    ata_inv = np.where(ok[:, None, None], ata_inv, 0.0)

    if not getattr(mesh, 'cell_nodes', None):
        return None
    max_v = max(len(vs) for vs in mesh.cell_nodes)
    cell_vertex_ids = np.full((n_cells, max_v), -1, dtype=np.int64)
    vertex_offsets = np.zeros((n_cells, max_v, 2), dtype=float)
    for c, vs in enumerate(mesh.cell_nodes):
        cell_vertex_ids[c, :len(vs)] = vs
        vertex_offsets[c, :len(vs)] = mesh.nodes[vs] - cc[c]
    n_nodes = mesh.nodes.shape[0]
    v2c = [[] for _ in range(n_nodes)]
    for c, vs in enumerate(mesh.cell_nodes):
        for node in vs:
            v2c[int(node)].append(c)
    max_v2c = max((len(xs) for xs in v2c), default=1)
    v2c_arr = np.full((n_nodes, max_v2c), -1, dtype=np.int64)
    for node, cells in enumerate(v2c):
        v2c_arr[node, :len(cells)] = cells
    v2c_valid = v2c_arr >= 0
    v2c_safe = np.where(v2c_valid, v2c_arr, 0).astype(np.int64, copy=False)
    ctx = (
        nb_safe.astype(np.int64, copy=False),
        valid_nb.astype(np.bool_, copy=False),
        np.asarray(d_nb, dtype=float),
        np.asarray(ata_inv, dtype=float),
        cell_vertex_ids,
        np.asarray(vertex_offsets, dtype=float),
        v2c_safe,
        v2c_valid.astype(np.bool_, copy=False),
    )
    setattr(mesh, '_tmlpu_fast_mlpu1_scalar_ctx', ctx)
    return ctx


class _ScalarFastMLPU1:
    def __init__(self):
        self.fallback = MLPU1()

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        if (not _NUMBA_AVAILABLE or W_cell.shape[0] != 1
                or mesh.dim != 2
                or mesh.kind not in ('structured_2d', 'unstructured_2d')
                or eval_points is not None):
            return self.fallback.reconstruct(
                mesh, W_cell, eq, eval_points=eval_points)
        ctx = _fast_mlpu1_scalar_context(mesh)
        if ctx is None:
            return self.fallback.reconstruct(
                mesh, W_cell, eq, eval_points=eval_points)
        nb_safe, valid_nb, d_nb, ata_inv, cell_vertex_ids, vertex_offsets, v2c_safe, v2c_valid = ctx
        W_L = np.empty(mesh.n_faces, dtype=float)
        W_R = np.empty(mesh.n_faces, dtype=float)
        _fast_mlpu1_scalar_kernel(
            np.asarray(W_cell[0], dtype=np.float64),
            np.asarray(mesh.face_owner, dtype=np.int64),
            np.asarray(mesh.face_neighbour, dtype=np.int64),
            np.asarray(mesh.face_centers, dtype=np.float64),
            np.asarray(mesh.cell_centers, dtype=np.float64),
            nb_safe, valid_nb, d_nb, ata_inv,
            cell_vertex_ids, vertex_offsets, v2c_safe, v2c_valid,
            W_L, W_R,
        )
        return W_L[None, :], W_R[None, :]


class _ScalarFastTMLPUBVD:
    def __init__(self, tvd='pure_downwind', stencil='face', order=2,
                 idw_p=6.0, vertex_mlp_cap=1.0,
                 vertex_mlp_face_local_branch=False,
                 face_gradient_correction='jasak',
                 r_form='far_upwind', moment_bvd=True,
                 moment_bvd_mode='and', interface_force_tmlpu=True,
                 interface_force_range=0.35, interface_force_only=False):
        self.base = _ScalarFastMLPU1()
        self.sharp = TMLPU(
            tvd=tvd,
            mlp_bound=True,
            extremum_relax=False,
            tvb_M=0.0,
            virtual_uu_gradient=True,
            stencil=stencil,
            order=order,
            idw_p=idw_p,
            hancock_courant=0.0,
            vertex_mlp=True,
            vertex_mlp_cap=vertex_mlp_cap,
            vertex_mlp_face_local=vertex_mlp_face_local_branch,
            vertex_mlp_augment=True,
            face_skew_correction=True,
            face_gradient_correction=face_gradient_correction,
            r_form=r_form,
            clamp_tstar=True,
        )
        self.moment_bvd = bool(moment_bvd)
        self.moment_bvd_mode = str(moment_bvd_mode).lower()
        self.interface_force_tmlpu = bool(interface_force_tmlpu)
        self.interface_force_range = float(interface_force_range)
        self.interface_force_only = bool(interface_force_only)

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        if W_cell.shape[0] != 1:
            return self.sharp.reconstruct(
                mesh, W_cell, eq, eval_points=eval_points)
        A_L, A_R = self.base.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        B_L, B_R = self.sharp.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return A_L, A_R
        n_cells = W_cell.shape[1]
        o_idx = owner[interior]
        n_idx = nei[interior]
        score_points = (mesh.face_centers[interior] if eval_points is None
                        else eval_points[interior])
        try:
            wave = eq.max_wave_speed(
                W_cell[:, o_idx], mesh.face_normals[interior],
                points=score_points)
        except TypeError:
            wave = eq.max_wave_speed(
                W_cell[:, o_idx], mesh.face_normals[interior])
        bvd_weight = mesh.face_areas[interior] * np.asarray(wave, dtype=float)

        interface_cells = None
        if self.interface_force_tmlpu:
            phi = W_cell[0]
            local_min = np.array(phi, copy=True)
            local_max = np.array(phi, copy=True)
            np.minimum.at(local_min, o_idx, phi[n_idx])
            np.minimum.at(local_min, n_idx, phi[o_idx])
            np.maximum.at(local_max, o_idx, phi[n_idx])
            np.maximum.at(local_max, n_idx, phi[o_idx])
            interface_cells = (
                local_max - local_min
                >= max(0.0, float(self.interface_force_range)))
        if self.interface_force_only and interface_cells is not None:
            use_B = interface_cells
        else:
            def _scores(left, right):
                j = np.abs(left[0, interior] - right[0, interior]) * bvd_weight
                tbv = (
                    np.bincount(o_idx, weights=j, minlength=n_cells)
                    + np.bincount(n_idx, weights=j, minlength=n_cells))
                if not self.moment_bvd:
                    return tbv, None
                d_o = score_points - mesh.cell_centers[o_idx]
                d_n = score_points - mesh.cell_centers[n_idx]
                mx = (
                    np.bincount(
                        o_idx, weights=j * d_o[:, 0], minlength=n_cells)
                    + np.bincount(
                        n_idx, weights=j * d_n[:, 0], minlength=n_cells))
                my = (
                    np.bincount(
                        o_idx, weights=j * d_o[:, 1], minlength=n_cells)
                    + np.bincount(
                        n_idx, weights=j * d_n[:, 1], minlength=n_cells))
                return tbv, np.sqrt(mx * mx + my * my)

            tbv_A, mbv_A = _scores(A_L, A_R)
            tbv_B, mbv_B = _scores(B_L, B_R)
            if self.moment_bvd:
                if self.moment_bvd_mode == 'moment':
                    use_B = mbv_B < mbv_A
                elif self.moment_bvd_mode == 'product':
                    use_B = (tbv_B * mbv_B) < (tbv_A * mbv_A)
                elif self.moment_bvd_mode == 'combined':
                    use_B = (
                        tbv_B * tbv_B + mbv_B * mbv_B
                        < tbv_A * tbv_A + mbv_A * mbv_A)
                else:
                    use_B = (tbv_B < tbv_A) & (mbv_B < mbv_A)
            else:
                use_B = tbv_B < tbv_A
        if interface_cells is not None and not self.interface_force_only:
            use_B = use_B | interface_cells
        W_L = A_L.copy()
        W_R = A_R.copy()
        mask_L = use_B[o_idx]
        if np.any(mask_L):
            W_L[0, interior[mask_L]] = B_L[0, interior[mask_L]]
        mask_R = use_B[n_idx]
        if np.any(mask_R):
            W_R[0, interior[mask_R]] = B_R[0, interior[mask_R]]
        return W_L, W_R


def _tmlpu_v3_unified_euler():
    return TMLPU(
        tvd=os.environ.get('TMLPU_V3_EULER_TVD', 'mc'),
        hancock_courant=float(os.environ.get('TMLPU_V3_HANCOCK_COURANT',
                                             '0.4')),
        mlp_bound=True,
        extremum_relax=False,
        extremum_relax_curved_otsu=False,
        tvb_M=0.0,
        vertex_mlp=True,
        vertex_mlp_cap=float(os.environ.get('TMLPU_V3_VERTEX_MLP_CAP',
                                            '1.0')),
        euler_shock_flatten=_env_bool('TMLPU_V3_SHOCK_FLATTEN', True),
        euler_density_acoustic_flatten=False,
        euler_density_tvd=os.environ.get('TMLPU_V3_DENSITY_TVD',
                                         'downwind'),
        euler_density_lsq_increment=_env_bool(
            'TMLPU_V3_DENSITY_LSQ_INCREMENT', True),
        euler_density_full_lsq_increment=False,
        euler_density_no_hancock=False,
        euler_density_entropy_split=False,
        euler_density_entropy_variable=False,
        euler_density_shear_contact=_env_bool(
            'TMLPU_V3_DENSITY_SHEAR_CONTACT', True),
        euler_density_contact_wave_hancock=_env_bool(
            'TMLPU_V3_DENSITY_CONTACT_WAVE_HANCOCK', True),
        euler_density_pressure_entropy=_env_bool(
            'TMLPU_V3_DENSITY_PRESSURE_ENTROPY', True),
        euler_density_contact_bvd=_env_bool('TMLPU_V3_DENSITY_CONTACT_BVD',
                                           False),
        euler_density_contact_cell_bvd=_env_bool(
            'TMLPU_V3_DENSITY_CONTACT_CELL_BVD', False),
        euler_density_first_order=False,
        euler_pressure_first_order=False,
        euler_pressure_shear_lsq_increment=False,
        euler_pressure_nonshock_lsq_increment=False,
        euler_velocity_no_hancock=_env_bool('TMLPU_V3_VELOCITY_NO_HANCOCK',
                                           True),
        euler_velocity_shock_flatten=_env_bool(
            'TMLPU_V3_VELOCITY_SHOCK_FLATTEN', True),
        euler_velocity_lsq_increment=_env_bool(
            'TMLPU_V3_VELOCITY_LSQ_INCREMENT', True),
        euler_density_extrema_lmp=False,
        euler_velocity_extrema_lmp=_env_bool('TMLPU_V3_VELOCITY_EXTREMA_LMP',
                                            True),
        euler_velocity_tvd=os.environ.get('TMLPU_V3_VELOCITY_TVD',
                                         'superbee'),
        euler_velocity_flatten_sensor=os.environ.get(
            'TMLPU_V3_VELOCITY_FLATTEN_SENSOR', 'ducros_normality'),
        euler_tangential_velocity_tvd=os.environ.get(
            'TMLPU_V3_TANGENTIAL_VELOCITY_TVD', 'contact_superbee_shock'),
        euler_tangential_ducros=_env_bool('TMLPU_V3_TANGENTIAL_DUCROS',
                                         True),
        euler_tangential_flatten_mode=os.environ.get(
            'TMLPU_V3_TANGENTIAL_FLATTEN_MODE', 'normality'),
        euler_tangential_velocity_no_hancock=_env_bool(
            'TMLPU_V3_TANGENTIAL_VELOCITY_NO_HANCOCK', False),
        euler_tangential_contact_wave_hancock=_env_bool(
            'TMLPU_V3_TANGENTIAL_CONTACT_WAVE_HANCOCK', True),
        euler_tangential_velocity_lsq_increment=_env_bool(
            'TMLPU_V3_TANGENTIAL_VELOCITY_LSQ_INCREMENT', True),
        euler_local_hancock=False,
        euler_log_positive=False,
        euler_log_pressure_only=False,
        virtual_uu_gradient=True,
        vertex_mlp_augment=False,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V3_FACE_GRADIENT_CORRECTION', 'beta'),
        face_increment=os.environ.get('TMLPU_V3_FACE_INCREMENT', 'tmlpu'),
        r_form=os.environ.get('TMLPU_V3_R_FORM', 'far_upwind'),
        phi_LL_unclipped=False,
        zero_delta_psi=float(os.environ.get('TMLPU_V3_ZERO_DELTA_PSI',
                                            '2.0')),
        stencil=os.environ.get('TMLPU_V3_STENCIL', 'vertex'),
        order=int(os.environ.get('TMLPU_V3_ORDER', '2')),
        idw_p=float(os.environ.get('TMLPU_V3_IDW_P', '0.0')),
    )


def _tmlpu_v3_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v3_unified_euler())


def _tmlpu_v4_unified_euler():
    return TMLPU(
        tvd=os.environ.get('TMLPU_V4_EULER_TVD', 'mc'),
        hancock_courant=float(os.environ.get('TMLPU_V4_HANCOCK_COURANT',
                                             '0.4')),
        mlp_bound=True,
        extremum_relax=False,
        extremum_relax_curved_otsu=False,
        tvb_M=0.0,
        vertex_mlp=True,
        vertex_mlp_cap=float(os.environ.get('TMLPU_V4_VERTEX_MLP_CAP',
                                            '1.0')),
        euler_shock_flatten=_env_bool('TMLPU_V4_SHOCK_FLATTEN', True),
        euler_density_acoustic_flatten=_env_bool(
            'TMLPU_V4_DENSITY_ACOUSTIC_FLATTEN', True),
        euler_density_tvd=os.environ.get('TMLPU_V4_DENSITY_TVD',
                                         'van_leer'),
        euler_density_lsq_increment=_env_bool(
            'TMLPU_V4_DENSITY_LSQ_INCREMENT', True),
        euler_density_full_lsq_increment=False,
        euler_density_no_hancock=_env_bool(
            'TMLPU_V4_DENSITY_NO_HANCOCK', False),
        euler_density_entropy_split=False,
        euler_density_entropy_variable=False,
        euler_density_shear_contact=_env_bool(
            'TMLPU_V4_DENSITY_SHEAR_CONTACT', True),
        euler_density_contact_wave_hancock=_env_bool(
            'TMLPU_V4_DENSITY_CONTACT_WAVE_HANCOCK', True),
        euler_density_pressure_entropy=_env_bool(
            'TMLPU_V4_DENSITY_PRESSURE_ENTROPY', True),
        euler_density_contact_bvd=False,
        euler_density_contact_cell_bvd=False,
        euler_density_first_order=False,
        euler_pressure_first_order=False,
        euler_pressure_shear_lsq_increment=False,
        euler_pressure_nonshock_lsq_increment=_env_bool(
            'TMLPU_V4_PRESSURE_NONSHOCK_LSQ_INCREMENT', True),
        euler_velocity_no_hancock=_env_bool(
            'TMLPU_V4_VELOCITY_NO_HANCOCK', True),
        euler_velocity_shock_flatten=_env_bool(
            'TMLPU_V4_VELOCITY_SHOCK_FLATTEN', True),
        euler_velocity_lsq_increment=_env_bool(
            'TMLPU_V4_VELOCITY_LSQ_INCREMENT', True),
        euler_density_extrema_lmp=False,
        euler_velocity_extrema_lmp=_env_bool(
            'TMLPU_V4_VELOCITY_EXTREMA_LMP', True),
        euler_velocity_tvd=os.environ.get('TMLPU_V4_VELOCITY_TVD',
                                         'mc'),
        euler_velocity_flatten_sensor=os.environ.get(
            'TMLPU_V4_VELOCITY_FLATTEN_SENSOR', 'ducros_normality'),
        euler_tangential_velocity_tvd=os.environ.get(
            'TMLPU_V4_TANGENTIAL_VELOCITY_TVD', 'shear_superbee_blend'),
        euler_tangential_ducros=_env_bool('TMLPU_V4_TANGENTIAL_DUCROS',
                                         True),
        euler_tangential_flatten_mode=os.environ.get(
            'TMLPU_V4_TANGENTIAL_FLATTEN_MODE', 'normality'),
        euler_tangential_velocity_no_hancock=_env_bool(
            'TMLPU_V4_TANGENTIAL_VELOCITY_NO_HANCOCK', False),
        euler_tangential_contact_wave_hancock=_env_bool(
            'TMLPU_V4_TANGENTIAL_CONTACT_WAVE_HANCOCK', True),
        euler_tangential_velocity_lsq_increment=_env_bool(
            'TMLPU_V4_TANGENTIAL_VELOCITY_LSQ_INCREMENT', True),
        euler_local_hancock=False,
        euler_log_positive=False,
        euler_log_pressure_only=False,
        euler_face_positivity_limiter=_env_bool(
            'TMLPU_V4_FACE_POSITIVITY_LIMITER', True),
        virtual_uu_gradient=True,
        vertex_mlp_augment=False,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V4_FACE_GRADIENT_CORRECTION', 'beta_shock_shear'),
        face_increment=os.environ.get('TMLPU_V4_FACE_INCREMENT', 'tmlpu'),
        r_form=os.environ.get('TMLPU_V4_R_FORM', 'far_upwind'),
        tmlpu_bound_tvd_separate=_env_bool(
            'TMLPU_V4_BOUND_TVD_SEPARATE', True),
        phi_LL_unclipped=False,
        zero_delta_psi=float(os.environ.get('TMLPU_V4_ZERO_DELTA_PSI',
                                            '1.0')),
        stencil=os.environ.get('TMLPU_V4_STENCIL', 'vertex'),
        order=int(os.environ.get('TMLPU_V4_ORDER', '2')),
        idw_p=float(os.environ.get('TMLPU_V4_IDW_P', '0.0')),
    )


def _tmlpu_v4_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v4_unified_euler())


def _tmlpu_v4_1_unified_euler():
    return TMLPU(
        tvd=os.environ.get('TMLPU_V4_1_EULER_TVD', 'mc'),
        hancock_courant=float(os.environ.get('TMLPU_V4_1_HANCOCK_COURANT',
                                             '0.4')),
        mlp_bound=True,
        extremum_relax=False,
        extremum_relax_curved_otsu=False,
        tvb_M=0.0,
        vertex_mlp=True,
        vertex_mlp_cap=float(os.environ.get('TMLPU_V4_1_VERTEX_MLP_CAP',
                                            '1.0')),
        euler_shock_flatten=_env_bool('TMLPU_V4_1_SHOCK_FLATTEN', True),
        euler_density_acoustic_flatten=_env_bool(
            'TMLPU_V4_1_DENSITY_ACOUSTIC_FLATTEN', True),
        euler_density_tvd=os.environ.get('TMLPU_V4_1_DENSITY_TVD',
                                         'umist'),
        euler_density_lsq_increment=_env_bool(
            'TMLPU_V4_1_DENSITY_LSQ_INCREMENT', True),
        euler_density_full_lsq_increment=False,
        euler_density_no_hancock=_env_bool(
            'TMLPU_V4_1_DENSITY_NO_HANCOCK', False),
        euler_density_entropy_split=False,
        euler_density_entropy_variable=False,
        euler_density_shear_contact=_env_bool(
            'TMLPU_V4_1_DENSITY_SHEAR_CONTACT', True),
        euler_density_contact_wave_hancock=_env_bool(
            'TMLPU_V4_1_DENSITY_CONTACT_WAVE_HANCOCK', True),
        euler_density_pressure_entropy=_env_bool(
            'TMLPU_V4_1_DENSITY_PRESSURE_ENTROPY', True),
        euler_density_contact_bvd=False,
        euler_density_contact_cell_bvd=False,
        euler_density_first_order=False,
        euler_pressure_first_order=False,
        euler_pressure_shear_lsq_increment=False,
        euler_pressure_nonshock_lsq_increment=_env_bool(
            'TMLPU_V4_1_PRESSURE_NONSHOCK_LSQ_INCREMENT', True),
        euler_velocity_no_hancock=_env_bool(
            'TMLPU_V4_1_VELOCITY_NO_HANCOCK', True),
        euler_velocity_shock_flatten=_env_bool(
            'TMLPU_V4_1_VELOCITY_SHOCK_FLATTEN', True),
        euler_velocity_lsq_increment=_env_bool(
            'TMLPU_V4_1_VELOCITY_LSQ_INCREMENT', True),
        euler_density_extrema_lmp=False,
        euler_velocity_extrema_lmp=_env_bool(
            'TMLPU_V4_1_VELOCITY_EXTREMA_LMP', True),
        euler_velocity_tvd=os.environ.get('TMLPU_V4_1_VELOCITY_TVD',
                                         'mc'),
        euler_velocity_flatten_sensor=os.environ.get(
            'TMLPU_V4_1_VELOCITY_FLATTEN_SENSOR', 'ducros_normality'),
        euler_tangential_velocity_tvd=os.environ.get(
            'TMLPU_V4_1_TANGENTIAL_VELOCITY_TVD',
            'shear_superbee_root_blend'),
        euler_tangential_ducros=_env_bool('TMLPU_V4_1_TANGENTIAL_DUCROS',
                                         True),
        euler_tangential_flatten_mode=os.environ.get(
            'TMLPU_V4_1_TANGENTIAL_FLATTEN_MODE', 'normality'),
        euler_tangential_contact_relax_flatten=_env_bool(
            'TMLPU_V4_1_TANGENTIAL_CONTACT_RELAX_FLATTEN', True),
        euler_tangential_velocity_no_hancock=_env_bool(
            'TMLPU_V4_1_TANGENTIAL_VELOCITY_NO_HANCOCK', False),
        euler_tangential_contact_wave_hancock=_env_bool(
            'TMLPU_V4_1_TANGENTIAL_CONTACT_WAVE_HANCOCK', True),
        euler_tangential_velocity_lsq_increment=_env_bool(
            'TMLPU_V4_1_TANGENTIAL_VELOCITY_LSQ_INCREMENT', True),
        euler_local_hancock=False,
        euler_log_positive=False,
        euler_log_pressure_only=False,
        euler_face_positivity_limiter=_env_bool(
            'TMLPU_V4_1_FACE_POSITIVITY_LIMITER', True),
        virtual_uu_gradient=True,
        vertex_mlp_augment=False,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V4_1_FACE_GRADIENT_CORRECTION', 'beta_shock_shear'),
        face_increment=os.environ.get('TMLPU_V4_1_FACE_INCREMENT', 'tmlpu'),
        r_form=os.environ.get('TMLPU_V4_1_R_FORM', 'far_upwind'),
        tmlpu_bound_tvd_separate=_env_bool(
            'TMLPU_V4_1_BOUND_TVD_SEPARATE', True),
        phi_LL_unclipped=False,
        zero_delta_psi=float(os.environ.get('TMLPU_V4_1_ZERO_DELTA_PSI',
                                            '1.0')),
        stencil=os.environ.get('TMLPU_V4_1_STENCIL', 'vertex'),
        order=int(os.environ.get('TMLPU_V4_1_ORDER', '2')),
        idw_p=float(os.environ.get('TMLPU_V4_1_IDW_P', '0.0')),
    )


def _tmlpu_v4_1_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v4_1_unified_euler())


def _tmlpu_v4_2_unified_euler():
    recon = _tmlpu_v4_1_unified_euler()
    recon.euler_density_contact_weak_face_mlp = _env_bool(
        'TMLPU_V4_2_DENSITY_CONTACT_WEAK_FACE_MLP', True)
    return recon


def _refresh_reconstruction_private_state(recon):
    recon.__post_init__()
    return recon


def _tmlpu_v4_2_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v4_2_unified_euler())


def _tmlpu_v4_3_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_mlp_cap = float(os.environ.get(
        'TMLPU_V4_3_DENSITY_CONTACT_WEAK_FACE_MLP_CAP', '0.55'))
    recon.euler_density_contact_weak_face_shock_power = float(os.environ.get(
        'TMLPU_V4_3_DENSITY_CONTACT_WEAK_FACE_SHOCK_POWER', '2.0'))
    recon.face_gradient_shock_damping = os.environ.get(
        'TMLPU_V4_3_FACE_GRADIENT_SHOCK_DAMPING', 'density_strong')
    return recon


def _tmlpu_v4_3_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v4_3_unified_euler())


def _tmlpu_v5_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_bvd = _env_bool(
        'TMLPU_V5_DENSITY_CONTACT_BVD', True)
    recon.euler_density_contact_bvd_cap = float(os.environ.get(
        'TMLPU_V5_DENSITY_CONTACT_BVD_CAP', '0.30'))
    recon.euler_density_contact_cell_bvd = _env_bool(
        'TMLPU_V5_DENSITY_CONTACT_CELL_BVD', False)
    return recon


def _tmlpu_v5_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v5_unified_euler())


def _tmlpu_v6_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_bvd = _env_bool(
        'TMLPU_V6_DENSITY_CONTACT_BVD', False)
    recon.euler_density_contact_cell_bvd = _env_bool(
        'TMLPU_V6_DENSITY_CONTACT_CELL_BVD', False)
    recon.euler_density_contact_hancock_boost = float(os.environ.get(
        'TMLPU_V6_DENSITY_CONTACT_HANCOCK_BOOST', '0.14'))
    recon.euler_density_contact_hancock_boost_cap = float(os.environ.get(
        'TMLPU_V6_DENSITY_CONTACT_HANCOCK_BOOST_CAP', '0.85'))
    return recon


def _tmlpu_v6_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v6_unified_euler())


def _tmlpu_v7_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_bvd = _env_bool(
        'TMLPU_V7_DENSITY_CONTACT_BVD', False)
    recon.euler_density_contact_cell_bvd = _env_bool(
        'TMLPU_V7_DENSITY_CONTACT_CELL_BVD', False)
    recon.euler_density_contact_hancock_boost = float(os.environ.get(
        'TMLPU_V7_DENSITY_CONTACT_HANCOCK_BOOST', '0.0'))
    recon.euler_density_contact_lsq_root_blend = float(os.environ.get(
        'TMLPU_V7_DENSITY_CONTACT_LSQ_ROOT_BLEND', '0.45'))
    recon.euler_density_contact_lsq_root_blend_cap = float(os.environ.get(
        'TMLPU_V7_DENSITY_CONTACT_LSQ_ROOT_BLEND_CAP', '0.80'))
    return recon


def _tmlpu_v7_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v7_unified_euler())


def _tmlpu_v8_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_bvd = _env_bool(
        'TMLPU_V8_DENSITY_CONTACT_BVD', False)
    recon.euler_density_contact_cell_bvd = _env_bool(
        'TMLPU_V8_DENSITY_CONTACT_CELL_BVD', False)
    recon.euler_density_contact_hancock_boost = float(os.environ.get(
        'TMLPU_V8_DENSITY_CONTACT_HANCOCK_BOOST', '0.0'))
    recon.euler_density_contact_lsq_root_blend = float(os.environ.get(
        'TMLPU_V8_DENSITY_CONTACT_LSQ_ROOT_BLEND', '0.0'))
    recon.euler_density_contact_weak_face_mlp_cap = float(os.environ.get(
        'TMLPU_V8_DENSITY_CONTACT_WEAK_FACE_MLP_CAP', '0.75'))
    recon.euler_density_contact_weak_face_root_blend = float(os.environ.get(
        'TMLPU_V8_DENSITY_CONTACT_WEAK_FACE_ROOT_BLEND', '0.18'))
    return recon


def _tmlpu_v8_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v8_unified_euler())


def _tmlpu_v9_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_bvd = _env_bool(
        'TMLPU_V9_DENSITY_CONTACT_BVD', False)
    recon.euler_density_contact_cell_bvd = _env_bool(
        'TMLPU_V9_DENSITY_CONTACT_CELL_BVD', False)
    recon.euler_density_contact_hancock_boost = float(os.environ.get(
        'TMLPU_V9_DENSITY_CONTACT_HANCOCK_BOOST', '0.0'))
    recon.euler_density_contact_lsq_root_blend = float(os.environ.get(
        'TMLPU_V9_DENSITY_CONTACT_LSQ_ROOT_BLEND', '0.0'))
    recon.euler_density_contact_weak_face_root_blend = float(os.environ.get(
        'TMLPU_V9_DENSITY_CONTACT_WEAK_FACE_ROOT_BLEND', '0.0'))
    recon.euler_density_contact_lsq_shear_floor = float(os.environ.get(
        'TMLPU_V9_DENSITY_CONTACT_LSQ_SHEAR_FLOOR', '0.22'))
    recon.euler_density_contact_lsq_shear_floor_cap = float(os.environ.get(
        'TMLPU_V9_DENSITY_CONTACT_LSQ_SHEAR_FLOOR_CAP', '0.35'))
    return recon


def _tmlpu_v9_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v9_unified_euler())


def _tmlpu_v10_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_bvd = _env_bool(
        'TMLPU_V10_DENSITY_CONTACT_BVD', False)
    recon.euler_density_contact_cell_bvd = _env_bool(
        'TMLPU_V10_DENSITY_CONTACT_CELL_BVD', False)
    recon.euler_density_contact_hancock_boost = float(os.environ.get(
        'TMLPU_V10_DENSITY_CONTACT_HANCOCK_BOOST', '0.0'))
    recon.euler_density_contact_lsq_root_blend = float(os.environ.get(
        'TMLPU_V10_DENSITY_CONTACT_LSQ_ROOT_BLEND', '0.0'))
    recon.euler_density_contact_lsq_shear_floor = float(os.environ.get(
        'TMLPU_V10_DENSITY_CONTACT_LSQ_SHEAR_FLOOR', '0.0'))
    recon.euler_density_contact_weak_face_root_blend = float(os.environ.get(
        'TMLPU_V10_DENSITY_CONTACT_WEAK_FACE_ROOT_BLEND', '0.0'))
    recon.euler_density_contact_weak_face_swirl_extra = float(os.environ.get(
        'TMLPU_V10_DENSITY_CONTACT_WEAK_FACE_SWIRL_EXTRA', '0.10'))
    return recon


def _tmlpu_v10_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v10_unified_euler())


def _tmlpu_v11_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_bvd = _env_bool(
        'TMLPU_V11_DENSITY_CONTACT_BVD', False)
    recon.euler_density_contact_cell_bvd = _env_bool(
        'TMLPU_V11_DENSITY_CONTACT_CELL_BVD', False)
    recon.euler_density_contact_hancock_boost = float(os.environ.get(
        'TMLPU_V11_DENSITY_CONTACT_HANCOCK_BOOST', '0.0'))
    recon.euler_density_contact_lsq_root_blend = float(os.environ.get(
        'TMLPU_V11_DENSITY_CONTACT_LSQ_ROOT_BLEND', '0.0'))
    recon.euler_density_contact_lsq_shear_floor = float(os.environ.get(
        'TMLPU_V11_DENSITY_CONTACT_LSQ_SHEAR_FLOOR', '0.0'))
    recon.euler_density_contact_weak_face_root_blend = float(os.environ.get(
        'TMLPU_V11_DENSITY_CONTACT_WEAK_FACE_ROOT_BLEND', '0.0'))
    recon.euler_density_contact_weak_face_swirl_extra = float(os.environ.get(
        'TMLPU_V11_DENSITY_CONTACT_WEAK_FACE_SWIRL_EXTRA', '0.0'))
    recon.euler_tangential_velocity_tvd = os.environ.get(
        'TMLPU_V11_TANGENTIAL_VELOCITY_TVD',
        'shear_superbee_root_micro')
    recon.euler_tangential_shear_micro_blend = float(os.environ.get(
        'TMLPU_V11_TANGENTIAL_SHEAR_MICRO_BLEND', '0.06'))
    recon.euler_tangential_shear_micro_cap = float(os.environ.get(
        'TMLPU_V11_TANGENTIAL_SHEAR_MICRO_CAP', '0.18'))
    return _refresh_reconstruction_private_state(recon)


def _tmlpu_v11_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v11_unified_euler())


def _tmlpu_v12_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_bvd = _env_bool(
        'TMLPU_V12_DENSITY_CONTACT_BVD', False)
    recon.euler_density_contact_cell_bvd = _env_bool(
        'TMLPU_V12_DENSITY_CONTACT_CELL_BVD', False)
    recon.euler_density_contact_hancock_boost = float(os.environ.get(
        'TMLPU_V12_DENSITY_CONTACT_HANCOCK_BOOST', '0.0'))
    recon.euler_density_contact_lsq_root_blend = float(os.environ.get(
        'TMLPU_V12_DENSITY_CONTACT_LSQ_ROOT_BLEND', '0.0'))
    recon.euler_density_contact_lsq_shear_floor = float(os.environ.get(
        'TMLPU_V12_DENSITY_CONTACT_LSQ_SHEAR_FLOOR', '0.0'))
    recon.euler_density_contact_weak_face_root_blend = float(os.environ.get(
        'TMLPU_V12_DENSITY_CONTACT_WEAK_FACE_ROOT_BLEND', '0.0'))
    recon.euler_density_contact_weak_face_swirl_extra = float(os.environ.get(
        'TMLPU_V12_DENSITY_CONTACT_WEAK_FACE_SWIRL_EXTRA', '0.0'))
    recon.euler_tangential_velocity_tvd = os.environ.get(
        'TMLPU_V12_TANGENTIAL_VELOCITY_TVD',
        'shear_superbee_root_mood')
    recon.euler_tangential_shear_micro_blend = float(os.environ.get(
        'TMLPU_V12_TANGENTIAL_MOOD_BLEND', '0.025'))
    recon.euler_tangential_shear_micro_cap = float(os.environ.get(
        'TMLPU_V12_TANGENTIAL_MOOD_CAP', '0.06'))
    recon.euler_tangential_mood_wavespeed_growth_cap = float(os.environ.get(
        'TMLPU_V12_WAVESPEED_GROWTH_CAP', '0.015'))
    recon.euler_tangential_mood_jump_growth_cap = float(os.environ.get(
        'TMLPU_V12_TANGENTIAL_JUMP_GROWTH_CAP', '0.05'))
    return _refresh_reconstruction_private_state(recon)


def _tmlpu_v12_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v12_unified_euler())


def _tmlpu_v13_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_admissibility_damp = _env_bool(
        'TMLPU_V13_DENSITY_WEAK_FACE_ADMISSIBILITY_DAMP', True)
    recon.euler_density_contact_weak_face_rho_floor = float(os.environ.get(
        'TMLPU_V13_DENSITY_WEAK_FACE_RHO_FLOOR', '0.65'))
    recon.euler_density_contact_weak_face_p_floor = float(os.environ.get(
        'TMLPU_V13_DENSITY_WEAK_FACE_P_FLOOR', '0.80'))
    recon.euler_density_contact_weak_face_admissibility_strength = float(
        os.environ.get(
            'TMLPU_V13_DENSITY_WEAK_FACE_ADMISSIBILITY_STRENGTH', '1.0'))
    return recon


def _tmlpu_v13_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v13_unified_euler())


def _tmlpu_v14_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_entropy_accept = _env_bool(
        'TMLPU_V14_DENSITY_WEAK_FACE_ENTROPY_ACCEPT', True)
    recon.euler_density_contact_weak_face_entropy_accept_eps = float(
        os.environ.get(
            'TMLPU_V14_DENSITY_WEAK_FACE_ENTROPY_ACCEPT_EPS', '0.05'))
    recon.euler_density_contact_weak_face_entropy_reject_scale = float(
        os.environ.get(
            'TMLPU_V14_DENSITY_WEAK_FACE_ENTROPY_REJECT_SCALE', '0.35'))
    return recon


def _tmlpu_v14_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v14_unified_euler())


def _tmlpu_v15_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_shock_gate = _env_bool(
        'TMLPU_V15_DENSITY_WEAK_FACE_SHOCK_GATE', True)
    return recon


def _tmlpu_v15_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v15_unified_euler())




def _tmlpu_v16_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_shock_gate = _env_bool(
        'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE', True)
    recon.euler_density_contact_weak_face_shock_gate_mode = os.environ.get(
        'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_MODE', 'core')
    recon.euler_density_contact_weak_face_shock_gate_strength = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_STRENGTH', '0.55'))
    recon.euler_density_contact_weak_face_shock_gate_floor = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_FLOOR', '0.55'))
    recon.euler_density_contact_weak_face_shock_gate_p_threshold = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_P_THRESHOLD', '0.08'))
    recon.euler_density_contact_weak_face_shock_gate_p_width = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_P_WIDTH', '0.22'))
    recon.euler_density_contact_weak_face_shock_gate_compression_threshold = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_COMPRESSION_THRESHOLD',
            '0.025'))
    recon.euler_density_contact_weak_face_shock_gate_compression_width = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_COMPRESSION_WIDTH',
            '0.12'))
    recon.euler_density_contact_weak_face_shock_gate_normality_threshold = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_NORMALITY_THRESHOLD',
            '0.55'))
    recon.euler_density_contact_weak_face_shock_gate_normality_width = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_NORMALITY_WIDTH',
            '0.30'))
    recon.euler_density_contact_weak_face_shock_gate_shear_threshold = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_SHEAR_THRESHOLD',
            '0.55'))
    recon.euler_density_contact_weak_face_shock_gate_shear_width = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_SHEAR_WIDTH',
            '0.25'))
    recon.euler_density_contact_weak_face_shock_gate_contact_threshold = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_CONTACT_THRESHOLD',
            '0.20'))
    recon.euler_density_contact_weak_face_shock_gate_contact_width = float(
        os.environ.get(
            'TMLPU_V16_DENSITY_WEAK_FACE_SHOCK_GATE_CONTACT_WIDTH',
            '0.40'))
    recon.euler_density_contact_weak_face_admissibility_damp = False
    recon.euler_density_contact_weak_face_entropy_accept = False
    return recon


def _tmlpu_v16_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v16_unified_euler())


def _tmlpu_v17_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V17_DENSITY_WEAK_FACE_VALUE_SCALING', True)
    recon.euler_density_contact_weak_face_rho_floor_factor = float(
        os.environ.get(
            'TMLPU_V17_DENSITY_WEAK_FACE_RHO_FLOOR_FACTOR', '0.88'))
    recon.euler_density_contact_weak_face_theta_floor = float(
        os.environ.get(
            'TMLPU_V17_DENSITY_WEAK_FACE_THETA_FLOOR', '0.0'))
    recon.euler_density_contact_weak_face_admissibility_damp = False
    recon.euler_density_contact_weak_face_entropy_accept = False
    recon.euler_density_contact_weak_face_shock_gate = False
    return recon


def _tmlpu_v17_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v17_unified_euler())


def _tmlpu_v18_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_pressure_contact_entropy_blend = _env_bool(
        'TMLPU_V18_PRESSURE_CONTACT_ENTROPY_BLEND', True)
    recon.euler_pressure_contact_entropy_beta = float(
        os.environ.get(
            'TMLPU_V18_PRESSURE_CONTACT_ENTROPY_BETA', '0.18'))
    recon.euler_pressure_contact_entropy_cap = float(
        os.environ.get(
            'TMLPU_V18_PRESSURE_CONTACT_ENTROPY_CAP', '0.18'))
    recon.euler_pressure_contact_entropy_downscale = float(
        os.environ.get(
            'TMLPU_V18_PRESSURE_CONTACT_ENTROPY_DOWNSCALE', '0.25'))
    recon.euler_pressure_contact_entropy_p_jump_threshold = float(
        os.environ.get(
            'TMLPU_V18_PRESSURE_CONTACT_ENTROPY_P_JUMP_THRESHOLD', '0.04'))
    recon.euler_pressure_contact_entropy_p_jump_width = float(
        os.environ.get(
            'TMLPU_V18_PRESSURE_CONTACT_ENTROPY_P_JUMP_WIDTH', '0.08'))
    recon.euler_pressure_contact_entropy_compression_threshold = float(
        os.environ.get(
            'TMLPU_V18_PRESSURE_CONTACT_ENTROPY_COMPRESSION_THRESHOLD',
            '0.01'))
    recon.euler_pressure_contact_entropy_compression_width = float(
        os.environ.get(
            'TMLPU_V18_PRESSURE_CONTACT_ENTROPY_COMPRESSION_WIDTH', '0.07'))
    recon.euler_pressure_contact_entropy_normality_threshold = float(
        os.environ.get(
            'TMLPU_V18_PRESSURE_CONTACT_ENTROPY_NORMALITY_THRESHOLD',
            '0.45'))
    recon.euler_pressure_contact_entropy_normality_width = float(
        os.environ.get(
            'TMLPU_V18_PRESSURE_CONTACT_ENTROPY_NORMALITY_WIDTH', '0.30'))
    # Ensure this candidate is pressure-only blend on v4.2 baseline.
    recon.euler_density_contact_weak_face_admissibility_damp = False
    recon.euler_density_contact_weak_face_entropy_accept = False
    recon.euler_density_contact_weak_face_shock_gate = False
    recon.euler_density_contact_weak_face_value_scaling = False
    return recon


def _tmlpu_v18_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v18_unified_euler())


def _tmlpu_v19_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_MODE',
            'local_pocket_shock'))
    recon.euler_density_contact_weak_face_value_scaling_strength = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_STRENGTH', '1.0'))
    recon.euler_density_contact_weak_face_rho_floor_factor = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_RHO_FLOOR_FACTOR', '0.86'))
    recon.euler_density_contact_weak_face_value_scaling_p_floor_factor = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_P_FLOOR_FACTOR',
            '0.90'))
    recon.euler_density_contact_weak_face_value_scaling_risk_width = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_RISK_WIDTH',
            '0.08'))
    recon.euler_density_contact_weak_face_value_scaling_p_threshold = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_P_THRESHOLD',
            '0.06'))
    recon.euler_density_contact_weak_face_value_scaling_p_width = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_P_WIDTH',
            '0.10'))
    recon.euler_density_contact_weak_face_value_scaling_compression_threshold = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_COMPRESSION_THRESHOLD',
            '0.015'))
    recon.euler_density_contact_weak_face_value_scaling_compression_width = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_COMPRESSION_WIDTH',
            '0.065'))
    recon.euler_density_contact_weak_face_value_scaling_normality_threshold = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_NORMALITY_THRESHOLD',
            '0.35'))
    recon.euler_density_contact_weak_face_value_scaling_normality_width = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_NORMALITY_WIDTH',
            '0.35'))
    recon.euler_density_contact_weak_face_value_scaling_contact_threshold = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_CONTACT_THRESHOLD',
            '0.25'))
    recon.euler_density_contact_weak_face_value_scaling_contact_width = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_CONTACT_WIDTH',
            '0.35'))
    recon.euler_density_contact_weak_face_value_scaling_shear_threshold = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_SHEAR_THRESHOLD',
            '0.60'))
    recon.euler_density_contact_weak_face_value_scaling_shear_width = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_SHEAR_WIDTH',
            '0.25'))
    recon.euler_density_contact_weak_face_value_scaling_pressure_clean_threshold = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_PRESSURE_CLEAN_THRESHOLD',
            '0.04'))
    recon.euler_density_contact_weak_face_value_scaling_pressure_clean_width = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_PRESSURE_CLEAN_WIDTH',
            '0.06'))
    recon.euler_density_contact_weak_face_value_scaling_hard_protect_cutoff = float(
        os.environ.get(
            'TMLPU_V19_DENSITY_WEAK_FACE_VALUE_SCALING_HARD_PROTECT_CUTOFF',
            '0.65'))
    # Keep other faces unchanged and guard candidate path cleanly.
    recon.euler_density_contact_weak_face_admissibility_damp = False
    recon.euler_density_contact_weak_face_entropy_accept = False
    recon.euler_density_contact_weak_face_shock_gate = False
    recon.euler_pressure_contact_entropy_blend = False
    return recon


def _tmlpu_v19_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v19_unified_euler())


def _tmlpu_v20_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_MODE',
            'local_pocket_shock'))
    recon.euler_density_contact_weak_face_value_scaling_strength = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_STRENGTH', '0.35'))
    recon.euler_density_contact_weak_face_rho_floor_factor = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_RHO_FLOOR_FACTOR', '0.86'))
    recon.euler_density_contact_weak_face_value_scaling_p_floor_factor = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_P_FLOOR_FACTOR',
            '0.90'))
    recon.euler_density_contact_weak_face_value_scaling_risk_width = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_RISK_WIDTH',
            '0.08'))
    recon.euler_density_contact_weak_face_value_scaling_p_threshold = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_P_THRESHOLD',
            '0.06'))
    recon.euler_density_contact_weak_face_value_scaling_p_width = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_P_WIDTH',
            '0.10'))
    recon.euler_density_contact_weak_face_value_scaling_compression_threshold = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_COMPRESSION_THRESHOLD',
            '0.015'))
    recon.euler_density_contact_weak_face_value_scaling_compression_width = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_COMPRESSION_WIDTH',
            '0.065'))
    recon.euler_density_contact_weak_face_value_scaling_normality_threshold = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_NORMALITY_THRESHOLD',
            '0.35'))
    recon.euler_density_contact_weak_face_value_scaling_normality_width = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_NORMALITY_WIDTH',
            '0.35'))
    recon.euler_density_contact_weak_face_value_scaling_contact_threshold = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_CONTACT_THRESHOLD',
            '0.25'))
    recon.euler_density_contact_weak_face_value_scaling_contact_width = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_CONTACT_WIDTH',
            '0.35'))
    recon.euler_density_contact_weak_face_value_scaling_shear_threshold = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_SHEAR_THRESHOLD',
            '0.60'))
    recon.euler_density_contact_weak_face_value_scaling_shear_width = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_SHEAR_WIDTH',
            '0.25'))
    recon.euler_density_contact_weak_face_value_scaling_pressure_clean_threshold = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_PRESSURE_CLEAN_THRESHOLD',
            '0.04'))
    recon.euler_density_contact_weak_face_value_scaling_pressure_clean_width = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_PRESSURE_CLEAN_WIDTH',
            '0.06'))
    recon.euler_density_contact_weak_face_value_scaling_hard_protect_cutoff = float(
        os.environ.get(
            'TMLPU_V20_DENSITY_WEAK_FACE_VALUE_SCALING_HARD_PROTECT_CUTOFF',
            '0.65'))
    recon.euler_density_contact_weak_face_admissibility_damp = False
    recon.euler_density_contact_weak_face_entropy_accept = False
    recon.euler_density_contact_weak_face_shock_gate = False
    recon.euler_pressure_contact_entropy_blend = False
    return recon


def _tmlpu_v20_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v20_unified_euler())


def _tmlpu_v21_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_MODE',
            'local_pocket_shock'))
    recon.euler_density_contact_weak_face_value_scaling_strength = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_STRENGTH', '0.25'))
    recon.euler_density_contact_weak_face_hard_rho_floor_factor = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_HARD_RHO_FLOOR_FACTOR',
            '0.82'))
    recon.euler_density_contact_weak_face_hard_p_floor_factor = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_HARD_P_FLOOR_FACTOR',
            '0.84'))
    recon.euler_density_contact_weak_face_rho_floor_factor = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_RHO_FLOOR_FACTOR',
            '0.88'))
    recon.euler_density_contact_weak_face_value_scaling_p_floor_factor = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_P_FLOOR_FACTOR',
            '0.90'))
    recon.euler_density_contact_weak_face_value_scaling_risk_width = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_RISK_WIDTH',
            '0.08'))
    recon.euler_density_contact_weak_face_value_scaling_p_threshold = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_P_THRESHOLD',
            '0.06'))
    recon.euler_density_contact_weak_face_value_scaling_p_width = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_P_WIDTH',
            '0.10'))
    recon.euler_density_contact_weak_face_value_scaling_compression_threshold = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_COMPRESSION_THRESHOLD',
            '0.015'))
    recon.euler_density_contact_weak_face_value_scaling_compression_width = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_COMPRESSION_WIDTH',
            '0.065'))
    recon.euler_density_contact_weak_face_value_scaling_normality_threshold = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_NORMALITY_THRESHOLD',
            '0.35'))
    recon.euler_density_contact_weak_face_value_scaling_normality_width = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_NORMALITY_WIDTH',
            '0.35'))
    recon.euler_density_contact_weak_face_value_scaling_contact_threshold = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_CONTACT_THRESHOLD',
            '0.25'))
    recon.euler_density_contact_weak_face_value_scaling_contact_width = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_CONTACT_WIDTH',
            '0.35'))
    recon.euler_density_contact_weak_face_value_scaling_shear_threshold = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_SHEAR_THRESHOLD',
            '0.60'))
    recon.euler_density_contact_weak_face_value_scaling_shear_width = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_SHEAR_WIDTH',
            '0.25'))
    recon.euler_density_contact_weak_face_value_scaling_pressure_clean_threshold = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_PRESSURE_CLEAN_THRESHOLD',
            '0.04'))
    recon.euler_density_contact_weak_face_value_scaling_pressure_clean_width = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_PRESSURE_CLEAN_WIDTH',
            '0.06'))
    recon.euler_density_contact_weak_face_value_scaling_hard_protect_cutoff = float(
        os.environ.get(
            'TMLPU_V21_DENSITY_WEAK_FACE_VALUE_SCALING_HARD_PROTECT_CUTOFF',
            '0.65'))
    # Keep v19/v20 feature gates off so only value-scaling path is active.
    recon.euler_density_contact_weak_face_admissibility_damp = False
    recon.euler_density_contact_weak_face_entropy_accept = False
    recon.euler_density_contact_weak_face_shock_gate = False
    recon.euler_pressure_contact_entropy_blend = False
    return recon


def _tmlpu_v21_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v21_unified_euler())


def _tmlpu_v22_unified_euler():
    recon = _tmlpu_v19_unified_euler()
    recon.euler_tangential_velocity_tvd = os.environ.get(
        'TMLPU_V22_TANGENTIAL_VELOCITY_TVD',
        'shear_superbee_root_mood')
    recon.euler_tangential_shear_micro_blend = float(
        os.environ.get(
            'TMLPU_V22_TANGENTIAL_MOOD_BLEND', '0.06'))
    recon.euler_tangential_shear_micro_cap = float(
        os.environ.get(
            'TMLPU_V22_TANGENTIAL_MOOD_CAP', '0.16'))
    recon.euler_tangential_mood_wavespeed_growth_cap = float(
        os.environ.get(
            'TMLPU_V22_WAVESPEED_GROWTH_CAP', '0.02'))
    recon.euler_tangential_mood_jump_growth_cap = float(
        os.environ.get(
            'TMLPU_V22_TANGENTIAL_JUMP_GROWTH_CAP', '0.05'))
    _refresh_reconstruction_private_state(recon)
    return recon


def _tmlpu_v22_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v22_unified_euler())


def _tmlpu_v23_unified_euler():
    recon = _tmlpu_v19_unified_euler()
    recon.euler_tangential_velocity_tvd = os.environ.get(
        'TMLPU_V23_TANGENTIAL_VELOCITY_TVD',
        'shear_superbee_root_mood')
    recon.euler_tangential_shear_micro_blend = float(
        os.environ.get(
            'TMLPU_V23_TANGENTIAL_MOOD_BLEND', '0.035'))
    recon.euler_tangential_shear_micro_cap = float(
        os.environ.get(
            'TMLPU_V23_TANGENTIAL_MOOD_CAP', '0.10'))
    recon.euler_tangential_mood_wavespeed_growth_cap = float(
        os.environ.get(
            'TMLPU_V23_WAVESPEED_GROWTH_CAP', '0.012'))
    recon.euler_tangential_mood_jump_growth_cap = float(
        os.environ.get(
            'TMLPU_V23_TANGENTIAL_JUMP_GROWTH_CAP', '0.030'))
    _refresh_reconstruction_private_state(recon)
    return recon


def _tmlpu_v23_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v23_unified_euler())


def _tmlpu_v24_unified_euler():
    recon = _tmlpu_v19_unified_euler()
    recon.euler_tangential_velocity_tvd = os.environ.get(
        'TMLPU_V24_TANGENTIAL_VELOCITY_TVD',
        'shear_superbee_root_mood')
    recon.euler_tangential_shear_micro_blend = float(
        os.environ.get(
            'TMLPU_V24_TANGENTIAL_MOOD_BLEND', '0.06'))
    recon.euler_tangential_shear_micro_cap = float(
        os.environ.get(
            'TMLPU_V24_TANGENTIAL_MOOD_CAP', '0.16'))
    recon.euler_tangential_mood_wavespeed_growth_cap = float(
        os.environ.get(
            'TMLPU_V24_WAVESPEED_GROWTH_CAP', '0.010'))
    recon.euler_tangential_mood_jump_growth_cap = float(
        os.environ.get(
            'TMLPU_V24_TANGENTIAL_JUMP_GROWTH_CAP', '0.025'))
    _refresh_reconstruction_private_state(recon)
    return recon


def _tmlpu_v24_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v24_unified_euler())


def _tmlpu_v25_unified_euler():
    recon = _tmlpu_v13_unified_euler()
    recon.euler_density_contact_weak_face_rho_floor = float(os.environ.get(
        'TMLPU_V25_DENSITY_WEAK_FACE_RHO_FLOOR', '0.655'))
    recon.euler_density_contact_weak_face_p_floor = float(os.environ.get(
        'TMLPU_V25_DENSITY_WEAK_FACE_P_FLOOR', '0.875'))
    recon.euler_density_contact_weak_face_admissibility_strength = float(
        os.environ.get(
            'TMLPU_V25_DENSITY_WEAK_FACE_ADMISSIBILITY_STRENGTH',
            '1.0'))
    return recon


def _tmlpu_v25_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v25_unified_euler())


def _tmlpu_v26_unified_euler():
    recon = _tmlpu_v13_unified_euler()
    recon.euler_density_contact_weak_face_rho_floor = float(os.environ.get(
        'TMLPU_V26_DENSITY_WEAK_FACE_RHO_FLOOR', '0.655'))
    recon.euler_density_contact_weak_face_p_floor = float(os.environ.get(
        'TMLPU_V26_DENSITY_WEAK_FACE_P_FLOOR', '0.875'))
    recon.euler_density_contact_weak_face_admissibility_strength = float(
        os.environ.get(
            'TMLPU_V26_DENSITY_WEAK_FACE_ADMISSIBILITY_STRENGTH',
            '1.0'))
    recon.euler_density_contact_weak_face_admissibility_shear_protect = _env_bool(
        'TMLPU_V26_DENSITY_WEAK_FACE_ADMISSIBILITY_SHEAR_PROTECT',
        True)
    return recon


def _tmlpu_v26_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v26_unified_euler())


def _tmlpu_v27_unified_euler():
    recon = _tmlpu_v13_unified_euler()
    recon.euler_face_positivity_limiter = _env_bool(
        'TMLPU_V27_FACE_POSITIVITY_LIMITER', True)
    recon.euler_face_rho_abs_floor = float(os.environ.get(
        'TMLPU_V27_FACE_RHO_ABS_FLOOR', '0.645'))
    recon.euler_face_p_abs_floor = float(os.environ.get(
        'TMLPU_V27_FACE_P_ABS_FLOOR', '0.860'))
    return recon


def _tmlpu_v27_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v27_unified_euler())


def _tmlpu_v28_unified_euler():
    recon = _tmlpu_v13_unified_euler()
    recon.euler_density_contact_weak_face_swirl_extra = float(
        os.environ.get(
            'TMLPU_V28_DENSITY_CONTACT_WEAK_FACE_SWIRL_EXTRA',
            '0.18'))
    return recon


def _tmlpu_v28_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v28_unified_euler())


def _tmlpu_v29_unified_euler():
    recon = _tmlpu_v13_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V29_DENSITY_WEAK_FACE_VALUE_SCALING', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        os.environ.get(
            'TMLPU_V29_DENSITY_WEAK_FACE_VALUE_SCALING_MODE',
            'shear_floor_blend'))
    recon.euler_density_contact_weak_face_rho_floor_factor = float(
        os.environ.get(
            'TMLPU_V29_DENSITY_WEAK_FACE_RHO_FLOOR_FACTOR', '0.82'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get(
            'TMLPU_V29_DENSITY_WEAK_FACE_SHEAR_BLEND_ALPHA', '0.12'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get(
            'TMLPU_V29_DENSITY_WEAK_FACE_SHEAR_BLEND_BOUND_PAD', '0.02'))
    return recon


def _tmlpu_v29_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v29_unified_euler())


def _tmlpu_v30_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_mlp = _env_bool(
        'TMLPU_V30_DENSITY_WEAK_FACE_MLP', False)
    recon.euler_tangential_contact_relax_flatten = _env_bool(
        'TMLPU_V30_TANGENTIAL_FLATTEN_DEFAULT', True)
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V30_DENSITY_MLP_MICRO_ON', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        'clean_shear_micro_restore')
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get(
            'TMLPU_V30_DENSITY_MLP_MICRO_ALPHA', '0.08'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get(
            'TMLPU_V30_DENSITY_MLP_MICRO_PAD', '0.012'))
    recon.euler_density_contact_weak_face_admissibility_damp = False
    recon.euler_density_contact_weak_face_entropy_accept = False
    recon.euler_density_contact_weak_face_shock_gate = False
    recon.euler_density_contact_weak_face_swirl_extra = 0.0
    recon.euler_pressure_contact_entropy_blend = False
    recon.euler_density_contact_bvd = False
    recon.euler_density_contact_cell_bvd = False
    recon.euler_face_rho_abs_floor = 0.0
    recon.euler_face_p_abs_floor = 0.0
    return recon


def _tmlpu_v30_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v30_unified_euler())


def _tmlpu_v31_unified_euler():
    recon = _tmlpu_v30_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V31_DENSITY_MLP_MICRO_ON', True)
    recon.euler_tangential_contact_relax_flatten = _env_bool(
        'TMLPU_V31_TANGENTIAL_FLATTEN_DEFAULT', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        'coherent_shear_micro_restore')
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V31_DENSITY_MLP_MICRO_ALPHA', '0.045'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V31_DENSITY_MLP_MICRO_PAD', '0.008'))
    recon.euler_density_contact_weak_face_value_scaling_require_coherent_shear = (
        _env_bool('TMLPU_V31_REQUIRE_COHERENT_SHEAR', '1'))
    recon.euler_density_contact_weak_face_value_scaling_artifact_reject = (
        _env_bool('TMLPU_V31_ARTIFACT_REJECT', '1'))
    return recon


def _tmlpu_v31_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v31_unified_euler())


def _tmlpu_v32_unified_euler():
    recon = _tmlpu_v31_unified_euler()
    recon.euler_tangential_pair_restore_on = _env_bool(
        'TMLPU_V32_TANGENTIAL_PAIR_RESTORE_ON', True)
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get(
            'TMLPU_V32_TANGENTIAL_PAIR_ALPHA', '0.045'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get(
            'TMLPU_V32_TANGENTIAL_PAIR_CAP', '0.075'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get(
            'TMLPU_V32_TANGENTIAL_PAIR_WAVE_CAP', '0.010'))
    recon.euler_tangential_contact_relax_flatten = _env_bool(
        'TMLPU_V32_TANGENTIAL_FLATTEN_DEFAULT', True)
    return recon


def _tmlpu_v32_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v32_unified_euler())


def _tmlpu_v33_unified_euler():
    recon = _tmlpu_v32_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V33_DENSITY_MLP_MICRO_ALPHA', '0.070'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V33_DENSITY_MLP_MICRO_PAD', '0.010'))
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V33_TANGENTIAL_PAIR_ALPHA', '0.030'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V33_TANGENTIAL_PAIR_CAP', '0.055'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V33_TANGENTIAL_PAIR_WAVE_CAP', '0.008'))
    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V33_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V33_STREAM_COHERENCE_MIN', '0.20'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V33_STREAM_COHERENCE_FULL', '0.60'))
    recon.euler_density_contact_weak_face_downstream_rho_beta = float(
        os.environ.get('TMLPU_V33_DOWNSTREAM_RHO_BETA', '0.035'))
    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V33_DOWNSTREAM_TANGENTIAL_BETA', '0.020'))
    recon.euler_density_contact_weak_face_downstream_rho_cap = float(
        os.environ.get('TMLPU_V33_DOWNSTREAM_RHO_CAP', '0.006'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V33_DOWNSTREAM_TANGENTIAL_CAP', '0.030'))
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = float(
        os.environ.get('TMLPU_V33_DOWNSTREAM_WAVE_CAP', '0.004'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V33_DOWNSTREAM_WAVE_CAP', '0.004'))
    recon.euler_density_contact_weak_face_stream_coherence_on = _env_bool(
        'TMLPU_V33_STREAM_COHERENCE_ON', True)
    recon.euler_density_contact_weak_face_stream_coherence_min = float(
        os.environ.get('TMLPU_V33_STREAM_COHERENCE_MIN', '0.20'))
    recon.euler_density_contact_weak_face_stream_coherence_full = float(
        os.environ.get('TMLPU_V33_STREAM_COHERENCE_FULL', '0.60'))
    recon.euler_tangential_contact_relax_flatten = _env_bool(
        'TMLPU_V33_TANGENTIAL_FLATTEN_DEFAULT', True)
    return recon


def _tmlpu_v33_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v33_unified_euler())


def _tmlpu_v34_unified_euler():
    recon = _tmlpu_v32_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V34_DENSITY_MLP_MICRO_ON', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        'contour_continuity_micro_restore')
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V34_DENSITY_MICRO_ALPHA', '0.075'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V34_DENSITY_MICRO_PAD', '0.010'))
    recon.euler_density_contact_weak_face_contour_continuity_on = _env_bool(
        'TMLPU_V34_CONTOUR_CONTINUITY_ON', True)
    recon.euler_density_contact_weak_face_contour_continuity_min = float(
        os.environ.get('TMLPU_V34_CONTOUR_CONTINUITY_MIN', '0.55'))
    recon.euler_density_contact_weak_face_contour_continuity_full = float(
        os.environ.get('TMLPU_V34_CONTOUR_CONTINUITY_FULL', '0.85'))
    recon.euler_density_contact_weak_face_density_increment_cap = float(
        os.environ.get('TMLPU_V34_DENSITY_INCREMENT_CAP', '0.008'))
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V34_TANGENTIAL_PAIR_ALPHA', '0.030'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V34_TANGENTIAL_PAIR_CAP', '0.055'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V34_TANGENTIAL_PAIR_WAVE_CAP', '0.008'))
    recon.euler_tangential_contact_relax_flatten = _env_bool(
        'TMLPU_V34_TANGENTIAL_FLATTEN_DEFAULT', True)
    recon.euler_tangential_pair_restore_stream_coherence_on = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = 0.0
    recon.euler_pressure_contact_entropy_blend = False
    return recon


def _tmlpu_v34_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v34_unified_euler())


def _tmlpu_v35_unified_euler():
    recon = _tmlpu_v32_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V35_DENSITY_MLP_MICRO_ON', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        'coherent_shear_micro_restore')
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V35_DENSITY_MICRO_ALPHA', '0.070'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V35_DENSITY_MICRO_PAD', '0.010'))
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V35_TANGENTIAL_PAIR_ALPHA', '0.030'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V35_TANGENTIAL_PAIR_CAP', '0.055'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V35_TANGENTIAL_PAIR_WAVE_CAP', '0.008'))
    recon.euler_tangential_pair_extend_on = _env_bool(
        'TMLPU_V35_PAIR_EXTEND_ON', True)
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V35_PAIR_EXTEND_BETA', '0.018'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V35_PAIR_EXTEND_CAP', '0.025'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V35_PAIR_EXTEND_WAVE_CAP', '0.0035'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V35_PAIR_EXTEND_ALIGNMENT_MIN', '0.65'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V35_PAIR_EXTEND_ALIGNMENT_FULL', '0.90'))
    recon.euler_tangential_contact_relax_flatten = _env_bool(
        'TMLPU_V35_TANGENTIAL_FLATTEN_DEFAULT', True)
    recon.euler_tangential_pair_restore_stream_coherence_on = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = 0.0
    recon.euler_density_contact_weak_face_contour_continuity_on = False
    recon.euler_density_contact_weak_face_density_increment_cap = 0.0
    recon.euler_pressure_contact_entropy_blend = False
    return recon


def _tmlpu_v35_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v35_unified_euler())


def _tmlpu_v36_unified_euler():
    recon = _tmlpu_v32_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V36_DENSITY_MLP_MICRO_ON', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        'coherent_shear_micro_restore')
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V36_DENSITY_MICRO_ALPHA', '0.070'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V36_DENSITY_MICRO_PAD', '0.010'))
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V36_TANGENTIAL_PAIR_ALPHA', '0.030'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V36_TANGENTIAL_PAIR_CAP', '0.055'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V36_TANGENTIAL_PAIR_WAVE_CAP', '0.008'))
    recon.euler_tangential_pair_gate_contact_min = float(
        os.environ.get('TMLPU_V36_PAIR_GATE_CONTACT_MIN', '0.24'))
    recon.euler_tangential_pair_gate_contact_full = float(
        os.environ.get('TMLPU_V36_PAIR_GATE_CONTACT_FULL', '0.58'))
    recon.euler_tangential_pair_gate_shear_min = float(
        os.environ.get('TMLPU_V36_PAIR_GATE_SHEAR_MIN', '0.62'))
    recon.euler_tangential_pair_gate_shear_full = float(
        os.environ.get('TMLPU_V36_PAIR_GATE_SHEAR_FULL', '0.88'))
    recon.euler_tangential_pair_gate_density_support_min = float(
        os.environ.get(
            'TMLPU_V36_PAIR_GATE_DENSITY_SUPPORT_MIN', '0.018'))
    recon.euler_tangential_pair_gate_density_support_full = float(
        os.environ.get(
            'TMLPU_V36_PAIR_GATE_DENSITY_SUPPORT_FULL', '0.075'))
    recon.euler_tangential_pair_gate_shock_reject_keep_v32 = _env_bool(
        'TMLPU_V36_SHOCK_REJECT_KEEP_V32', True)
    recon.euler_tangential_contact_relax_flatten = _env_bool(
        'TMLPU_V36_TANGENTIAL_FLATTEN_DEFAULT', True)
    recon.euler_tangential_pair_extend_on = False
    recon.euler_tangential_pair_extend_beta = 0.0
    recon.euler_tangential_pair_extend_cap = 0.0
    recon.euler_tangential_pair_extend_wave_cap = 0.0
    recon.euler_tangential_pair_restore_stream_coherence_on = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = 0.0
    recon.euler_density_contact_weak_face_contour_continuity_on = False
    recon.euler_density_contact_weak_face_density_increment_cap = 0.0
    recon.euler_pressure_contact_entropy_blend = False
    return recon


def _tmlpu_v36_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v36_unified_euler())


def _tmlpu_v37_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_legacy_order = _env_bool(
        'TMLPU_V37_LEGACY_DENSITY_WEAK_ORDER', True)
    recon.euler_density_contact_weak_face_legacy_relax = _env_bool(
        'TMLPU_V37_LEGACY_DENSITY_RELAX', True)
    recon.euler_density_contact_weak_face_legacy_relax_cap = float(
        os.environ.get('TMLPU_V37_DENSITY_RELAX_CAP', '1.0'))
    recon.euler_density_contact_weak_face_legacy_tvd_after_weak = _env_bool(
        'TMLPU_V37_DENSITY_TVD_AFTER_WEAK', True)
    recon.euler_density_contact_weak_face_admissibility_damp = _env_bool(
        'TMLPU_V37_DENSITY_WEAK_ADMISSIBILITY_DAMP', False)
    recon.euler_density_contact_weak_face_entropy_accept = False
    recon.euler_density_contact_weak_face_shock_gate = False
    recon.euler_density_contact_weak_face_value_scaling = False
    recon.euler_density_contact_weak_face_root_blend = 0.0
    recon.euler_density_contact_weak_face_swirl_extra = 0.0
    recon.euler_density_contact_weak_face_shock_power = 1.0
    recon.euler_density_contact_weak_face_mlp_cap = 1.0
    recon.euler_density_contact_bvd = False
    recon.euler_density_contact_cell_bvd = False
    recon.euler_density_contact_hancock_boost = 0.0
    recon.euler_density_contact_lsq_root_blend = 0.0
    recon.euler_density_contact_lsq_shear_floor = 0.0
    recon.euler_pressure_contact_entropy_blend = False
    recon.euler_tangential_pair_restore_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_tangential_pair_restore_stream_coherence_on = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    return recon


def _tmlpu_v37_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v37_unified_euler())


def _tmlpu_v38_unified_euler():
    recon = _tmlpu_v4_2_unified_euler()
    recon.euler_density_contact_weak_face_head_generic = _env_bool(
        'TMLPU_V38_USE_HEAD_GENERIC_DENSITY_WEAK_FACE', True)
    recon.euler_density_contact_weak_face_disable_specialized_relax = (
        _env_bool('TMLPU_V38_DISABLE_SPECIALIZED_DENSITY_WEAK_RELAX', True))
    recon.euler_density_contact_weak_face_head_generic_blend_cap = float(
        os.environ.get('TMLPU_V38_GENERIC_WEAK_FACE_BLEND_CAP', '1.0'))
    recon.euler_density_contact_weak_face_admissibility_damp = _env_bool(
        'TMLPU_V38_DENSITY_WEAK_ADMISSIBILITY_DAMP', False)
    recon.tmlpu_bound_tvd_separate = _env_bool(
        'TMLPU_V38_KEEP_BOUND_TVD_SEPARATE', True)
    recon.euler_face_positivity_limiter = _env_bool(
        'TMLPU_V38_KEEP_FACE_POSITIVITY', True)
    recon.euler_density_contact_weak_face_legacy_order = False
    recon.euler_density_contact_weak_face_legacy_relax = False
    recon.euler_density_contact_weak_face_legacy_tvd_after_weak = False
    recon.euler_density_contact_weak_face_entropy_accept = False
    recon.euler_density_contact_weak_face_shock_gate = False
    recon.euler_density_contact_weak_face_value_scaling = False
    recon.euler_density_contact_weak_face_root_blend = 0.0
    recon.euler_density_contact_weak_face_swirl_extra = 0.0
    recon.euler_density_contact_weak_face_shock_power = 1.0
    recon.euler_density_contact_weak_face_mlp_cap = 1.0
    recon.euler_pressure_contact_entropy_blend = False
    recon.euler_tangential_pair_restore_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_tangential_pair_restore_stream_coherence_on = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    return recon


def _tmlpu_v38_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v38_unified_euler())


def _tmlpu_v39_unified_euler():
    recon = _tmlpu_v32_unified_euler()
    recon.euler_density_contact_weak_face_mlp = False
    recon.euler_density_contact_weak_face_value_scaling = True
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        'coherent_shear_micro_restore')
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V39_DENSITY_MICRO_ALPHA', '0.070'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V39_DENSITY_MICRO_PAD', '0.010'))
    recon.euler_density_contact_weak_face_value_scaling_require_coherent_shear = True
    recon.euler_density_contact_weak_face_value_scaling_artifact_reject = True
    recon.euler_tangential_pair_restore_on = True
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V39_TANGENTIAL_PAIR_ALPHA', '0.030'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V39_TANGENTIAL_PAIR_CAP', '0.055'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V39_TANGENTIAL_PAIR_WAVE_CAP', '0.008'))
    recon.euler_tangential_contact_relax_flatten = True
    recon.tmlpu_bound_tvd_separate = True
    recon.euler_face_positivity_limiter = True

    recon.euler_contact_characteristic_postpass_on = _env_bool(
        'TMLPU_V39_CONTACT_POSTPASS_ON', True)
    recon.euler_contact_characteristic_entropy_alpha = float(
        os.environ.get('TMLPU_V39_CONTACT_ENTROPY_ALPHA', '0.035'))
    recon.euler_contact_characteristic_tangential_alpha = float(
        os.environ.get('TMLPU_V39_TANGENTIAL_ALPHA', '0.020'))
    recon.euler_contact_characteristic_entropy_cap = float(
        os.environ.get('TMLPU_V39_CONTACT_ENTROPY_CAP', '0.010'))
    recon.euler_contact_characteristic_tangential_cap = float(
        os.environ.get('TMLPU_V39_TANGENTIAL_CAP', '0.025'))
    recon.euler_contact_characteristic_tangential_wave_cap = float(
        os.environ.get('TMLPU_V39_TANGENTIAL_WAVE_CAP', '0.0035'))
    recon.euler_contact_characteristic_pressure_alpha = float(
        os.environ.get('TMLPU_V39_PRESSURE_ALPHA', '0.000'))
    recon.euler_contact_characteristic_normal_alpha = float(
        os.environ.get('TMLPU_V39_NORMAL_ALPHA', '0.000'))
    recon.euler_contact_characteristic_mood_fallback_on = _env_bool(
        'TMLPU_V39_MOOD_FALLBACK_ON', True)

    recon.euler_pressure_contact_entropy_blend = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_tangential_pair_restore_stream_coherence_on = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_head_generic = False
    recon.euler_density_contact_weak_face_disable_specialized_relax = False
    return recon


def _tmlpu_v39_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v39_unified_euler())


def _tmlpu_v40_unified_euler():
    recon = _tmlpu_v32_unified_euler()
    recon.euler_density_contact_weak_face_mlp = False
    recon.euler_density_contact_weak_face_value_scaling = True
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        'coherent_shear_micro_restore')
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V40_DENSITY_MICRO_ALPHA', '0.070'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V40_DENSITY_MICRO_PAD', '0.010'))
    recon.euler_density_contact_weak_face_value_scaling_require_coherent_shear = True
    recon.euler_density_contact_weak_face_value_scaling_artifact_reject = True
    recon.euler_tangential_pair_restore_on = True
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V40_TANGENTIAL_PAIR_ALPHA', '0.030'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V40_TANGENTIAL_PAIR_CAP', '0.055'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V40_TANGENTIAL_PAIR_WAVE_CAP', '0.008'))
    recon.euler_tangential_contact_relax_flatten = True
    recon.tmlpu_bound_tvd_separate = True
    recon.euler_face_positivity_limiter = True

    recon.euler_patch_contact_shear_postpass_on = _env_bool(
        'TMLPU_V40_PATCH_ON', True)
    recon.euler_patch_contact_shear_neighbor_blend = float(
        os.environ.get('TMLPU_V40_PATCH_NEIGHBOR_BLEND', '0.30'))
    recon.euler_patch_contact_shear_entropy_alpha = float(
        os.environ.get('TMLPU_V40_ENTROPY_ALPHA', '0.030'))
    recon.euler_patch_contact_shear_tangential_alpha = float(
        os.environ.get('TMLPU_V40_TANGENTIAL_ALPHA', '0.018'))
    recon.euler_patch_contact_shear_entropy_cap = float(
        os.environ.get('TMLPU_V40_ENTROPY_CAP', '0.008'))
    recon.euler_patch_contact_shear_tangential_cap = float(
        os.environ.get('TMLPU_V40_TANGENTIAL_CAP', '0.020'))
    recon.euler_patch_contact_shear_tangential_wave_cap = float(
        os.environ.get('TMLPU_V40_TANGENTIAL_WAVE_CAP', '0.003'))
    recon.euler_patch_contact_shear_min_valid_neighbours = int(
        os.environ.get('TMLPU_V40_MIN_VALID_NEIGHBOURS', '2'))

    recon.euler_contact_characteristic_postpass_on = False
    recon.euler_pressure_contact_entropy_blend = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_tangential_pair_restore_stream_coherence_on = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_legacy_order = False
    recon.euler_density_contact_weak_face_head_generic = False
    recon.euler_density_contact_weak_face_disable_specialized_relax = False
    return recon


def _tmlpu_v40_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v40_unified_euler())


def _tmlpu_v45_unified_scalar():
    return TMLPUBVD(
        tvd=os.environ.get('TMLPU_V45_SCALAR_TVD', 'pure_downwind'),
        stencil=os.environ.get('TMLPU_V45_SCALAR_STENCIL', 'face'),
        order=int(os.environ.get('TMLPU_V45_SCALAR_ORDER', '2')),
        idw_p=float(os.environ.get('TMLPU_V45_SCALAR_IDW_P', '6.0')),
        vertex_mlp_cap=float(os.environ.get(
            'TMLPU_V45_SCALAR_VERTEX_MLP_CAP', '1.0')),
        vertex_mlp_face_local_branch=_env_bool(
            'TMLPU_V45_SCALAR_FACE_LOCAL_BRANCH', False),
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V45_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        vertex_mlp_augment=True,
        r_form=os.environ.get('TMLPU_V45_SCALAR_R_FORM', 'far_upwind'),
        clamp_tstar=True,
        hancock_courant=0.0,
        moment_bvd=_env_bool('TMLPU_V45_SCALAR_MOMENT_BVD', True),
        moment_bvd_mode=os.environ.get(
            'TMLPU_V45_SCALAR_MOMENT_BVD_MODE', 'and'),
        moment_bvd_normalize_length=_env_bool(
            'TMLPU_V45_SCALAR_MOMENT_BVD_NORMALIZE_LENGTH', False),
        interface_force_tmlpu=_env_bool(
            'TMLPU_V45_SCALAR_INTERFACE_FORCE_TMLPU', True),
        interface_force_range=float(os.environ.get(
            'TMLPU_V45_SCALAR_INTERFACE_FORCE_RANGE', '0.35')),
        interface_force_only=_env_bool(
            'TMLPU_V45_SCALAR_INTERFACE_FORCE_ONLY', False),
    )


def _tmlpu_v45_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v40_unified_euler())


def _tmlpu_v46_unified_euler():
    recon = _tmlpu_v40_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V46_DENSITY_MICRO_ALPHA', '0.080'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V46_DENSITY_MICRO_PAD', '0.010'))
    recon.euler_density_contact_weak_face_density_increment_cap = float(
        os.environ.get('TMLPU_V46_DENSITY_INCREMENT_CAP', '0.006'))
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V46_TANGENTIAL_PAIR_ALPHA', '0.030'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V46_TANGENTIAL_PAIR_CAP', '0.055'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V46_TANGENTIAL_PAIR_WAVE_CAP', '0.008'))
    recon.euler_tangential_pair_extend_on = _env_bool(
        'TMLPU_V46_PAIR_EXTEND_ON', True)
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V46_PAIR_EXTEND_BETA', '0.024'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V46_PAIR_EXTEND_CAP', '0.032'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V46_PAIR_EXTEND_WAVE_CAP', '0.0040'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V46_PAIR_EXTEND_ALIGNMENT_MIN', '0.55'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V46_PAIR_EXTEND_ALIGNMENT_FULL', '0.86'))
    recon.euler_tangential_pair_restore_stream_coherence_on = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = 0.0
    return recon


def _tmlpu_v46_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v46_unified_euler())


def _tmlpu_v47_unified_euler():
    recon = _tmlpu_v46_unified_euler()
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V47_TANGENTIAL_PAIR_ALPHA', '0.045'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V47_TANGENTIAL_PAIR_CAP', '0.075'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V47_TANGENTIAL_PAIR_WAVE_CAP', '0.010'))
    recon.euler_tangential_pair_gate_contact_min = float(
        os.environ.get('TMLPU_V47_PAIR_GATE_CONTACT_MIN', '0.22'))
    recon.euler_tangential_pair_gate_contact_full = float(
        os.environ.get('TMLPU_V47_PAIR_GATE_CONTACT_FULL', '0.50'))
    recon.euler_tangential_pair_gate_shear_min = float(
        os.environ.get('TMLPU_V47_PAIR_GATE_SHEAR_MIN', '0.55'))
    recon.euler_tangential_pair_gate_shear_full = float(
        os.environ.get('TMLPU_V47_PAIR_GATE_SHEAR_FULL', '0.82'))
    recon.euler_tangential_pair_gate_density_support_min = float(
        os.environ.get('TMLPU_V47_PAIR_GATE_DENSITY_SUPPORT_MIN', '0.0'))
    recon.euler_tangential_pair_gate_density_support_full = float(
        os.environ.get('TMLPU_V47_PAIR_GATE_DENSITY_SUPPORT_FULL', '0.035'))
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V47_PAIR_EXTEND_BETA', '0.040'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V47_PAIR_EXTEND_CAP', '0.045'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V47_PAIR_EXTEND_WAVE_CAP', '0.0060'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V47_PAIR_EXTEND_ALIGNMENT_MIN', '0.48'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V47_PAIR_EXTEND_ALIGNMENT_FULL', '0.80'))
    recon.euler_patch_contact_shear_tangential_alpha = float(
        os.environ.get('TMLPU_V47_PATCH_TANGENTIAL_ALPHA', '0.016'))
    recon.euler_patch_contact_shear_tangential_cap = float(
        os.environ.get('TMLPU_V47_PATCH_TANGENTIAL_CAP', '0.018'))
    recon.euler_patch_contact_shear_tangential_wave_cap = float(
        os.environ.get('TMLPU_V47_PATCH_TANGENTIAL_WAVE_CAP', '0.0025'))
    return recon


def _tmlpu_v47_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v47_unified_euler())


def _tmlpu_v48_unified_euler():
    recon = _tmlpu_v17_unified_euler()
    recon.euler_density_contact_weak_face_rho_floor_factor = float(
        os.environ.get('TMLPU_V48_DENSITY_WEAK_FACE_RHO_FLOOR_FACTOR', '0.82'))
    recon.euler_density_contact_weak_face_theta_floor = float(
        os.environ.get('TMLPU_V48_DENSITY_WEAK_FACE_THETA_FLOOR', '0.0'))
    recon.euler_face_positivity_limiter = _env_bool(
        'TMLPU_V48_FACE_POSITIVITY_LIMITER', True)
    recon.euler_tangential_pair_restore_on = _env_bool(
        'TMLPU_V48_TANGENTIAL_PAIR_RESTORE_ON', True)
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V48_TANGENTIAL_PAIR_ALPHA', '0.035'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V48_TANGENTIAL_PAIR_CAP', '0.060'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V48_TANGENTIAL_PAIR_WAVE_CAP', '0.008'))
    recon.euler_tangential_pair_gate_contact_min = float(
        os.environ.get('TMLPU_V48_PAIR_GATE_CONTACT_MIN', '0.22'))
    recon.euler_tangential_pair_gate_contact_full = float(
        os.environ.get('TMLPU_V48_PAIR_GATE_CONTACT_FULL', '0.50'))
    recon.euler_tangential_pair_gate_shear_min = float(
        os.environ.get('TMLPU_V48_PAIR_GATE_SHEAR_MIN', '0.55'))
    recon.euler_tangential_pair_gate_shear_full = float(
        os.environ.get('TMLPU_V48_PAIR_GATE_SHEAR_FULL', '0.82'))
    recon.euler_tangential_pair_gate_density_support_min = float(
        os.environ.get('TMLPU_V48_PAIR_GATE_DENSITY_SUPPORT_MIN', '0.0'))
    recon.euler_tangential_pair_gate_density_support_full = float(
        os.environ.get('TMLPU_V48_PAIR_GATE_DENSITY_SUPPORT_FULL', '0.035'))
    recon.euler_tangential_pair_extend_on = _env_bool(
        'TMLPU_V48_PAIR_EXTEND_ON', True)
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V48_PAIR_EXTEND_BETA', '0.028'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V48_PAIR_EXTEND_CAP', '0.035'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V48_PAIR_EXTEND_WAVE_CAP', '0.0045'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V48_PAIR_EXTEND_ALIGNMENT_MIN', '0.48'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V48_PAIR_EXTEND_ALIGNMENT_FULL', '0.80'))
    return recon


def _tmlpu_v48_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v48_unified_euler())


def _tmlpu_v49_unified_euler():
    recon = _tmlpu_v47_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V49_DENSITY_MICRO_ALPHA', '0.120'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V49_DENSITY_MICRO_PAD', '0.012'))
    recon.euler_density_contact_weak_face_density_increment_cap = float(
        os.environ.get('TMLPU_V49_DENSITY_INCREMENT_CAP', '0.012'))
    recon.euler_density_contact_weak_face_value_scaling_require_coherent_shear = (
        _env_bool('TMLPU_V49_REQUIRE_COHERENT_SHEAR', False))
    recon.euler_density_contact_weak_face_value_scaling_artifact_reject = (
        _env_bool('TMLPU_V49_ARTIFACT_REJECT', True))
    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V49_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V49_STREAM_COHERENCE_MIN', '0.05'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V49_STREAM_COHERENCE_FULL', '0.45'))
    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V49_DOWNSTREAM_TANGENTIAL_BETA', '0.065'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V49_DOWNSTREAM_TANGENTIAL_CAP', '0.075'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V49_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.010'))
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0
    return recon


def _tmlpu_v49_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v49_unified_euler())


def _tmlpu_v50_unified_euler():
    recon = _tmlpu_v49_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V50_DENSITY_MICRO_ALPHA', '0.100'))
    recon.euler_density_contact_weak_face_density_increment_cap = float(
        os.environ.get('TMLPU_V50_DENSITY_INCREMENT_CAP', '0.015'))
    recon.euler_density_contact_weak_face_stream_coherence_on = _env_bool(
        'TMLPU_V50_DENSITY_STREAM_COHERENCE_ON', True)
    recon.euler_density_contact_weak_face_stream_coherence_min = float(
        os.environ.get('TMLPU_V50_DENSITY_STREAM_COHERENCE_MIN', '0.05'))
    recon.euler_density_contact_weak_face_stream_coherence_full = float(
        os.environ.get('TMLPU_V50_DENSITY_STREAM_COHERENCE_FULL', '0.45'))
    recon.euler_density_contact_weak_face_downstream_rho_beta = float(
        os.environ.get('TMLPU_V50_DOWNSTREAM_RHO_BETA', '0.110'))
    recon.euler_density_contact_weak_face_downstream_rho_cap = float(
        os.environ.get('TMLPU_V50_DOWNSTREAM_RHO_CAP', '0.018'))
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = float(
        os.environ.get('TMLPU_V50_DOWNSTREAM_RHO_WAVE_CAP', '0.008'))
    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V50_DOWNSTREAM_TANGENTIAL_BETA', '0.065'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V50_DOWNSTREAM_TANGENTIAL_CAP', '0.075'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V50_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.010'))
    return recon


def _tmlpu_v50_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v50_unified_euler())


def _tmlpu_v51_unified_euler():
    recon = _tmlpu_v49_unified_euler()
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V51_STREAM_COHERENCE_MIN', '0.0'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V51_STREAM_COHERENCE_FULL', '0.28'))
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V51_TANGENTIAL_PAIR_ALPHA', '0.055'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V51_TANGENTIAL_PAIR_CAP', '0.090'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V51_TANGENTIAL_PAIR_WAVE_CAP', '0.012'))
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V51_PAIR_EXTEND_BETA', '0.070'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V51_PAIR_EXTEND_CAP', '0.060'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V51_PAIR_EXTEND_WAVE_CAP', '0.0080'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V51_PAIR_EXTEND_ALIGNMENT_MIN', '0.35'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V51_PAIR_EXTEND_ALIGNMENT_FULL', '0.72'))
    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V51_DOWNSTREAM_TANGENTIAL_BETA', '0.120'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V51_DOWNSTREAM_TANGENTIAL_CAP', '0.110'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V51_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.016'))
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0
    return recon


def _tmlpu_v51_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v51_unified_euler())


def _tmlpu_v52_unified_euler():
    recon = _tmlpu_v49_unified_euler()
    recon.euler_density_contact_weak_face_value_scaling = False
    recon.euler_density_contact_weak_face_mlp = _env_bool(
        'TMLPU_V52_DENSITY_WEAK_FACE_MLP', True)
    recon.euler_density_contact_weak_face_mlp_cap = float(
        os.environ.get('TMLPU_V52_DENSITY_WEAK_FACE_MLP_CAP', '0.62'))
    recon.euler_density_contact_weak_face_shock_power = float(
        os.environ.get('TMLPU_V52_DENSITY_WEAK_FACE_SHOCK_POWER', '2.0'))
    recon.euler_density_contact_weak_face_legacy_order = False
    recon.euler_density_contact_weak_face_legacy_relax = False
    recon.euler_density_contact_weak_face_head_generic = False
    recon.euler_density_contact_weak_face_disable_specialized_relax = False
    recon.euler_density_contact_weak_face_admissibility_damp = False
    recon.euler_density_contact_weak_face_entropy_accept = False
    recon.euler_density_contact_weak_face_shock_gate = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0
    return recon


def _tmlpu_v52_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v52_unified_euler())


def _tmlpu_v53_unified_euler():
    recon = _tmlpu_v52_unified_euler()
    recon.euler_density_contact_weak_face_mlp_cap = float(
        os.environ.get('TMLPU_V53_DENSITY_WEAK_FACE_MLP_CAP', '1.0'))
    recon.euler_density_contact_weak_face_shock_power = float(
        os.environ.get('TMLPU_V53_DENSITY_WEAK_FACE_SHOCK_POWER', '2.2'))
    return recon


def _tmlpu_v53_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v53_unified_euler())


def _tmlpu_v54_unified_euler():
    recon = _tmlpu_v52_unified_euler()
    recon.euler_tangential_pair_restore_on = _env_bool(
        'TMLPU_V54_TANGENTIAL_PAIR_RESTORE_ON', False)
    recon.euler_tangential_pair_extend_on = _env_bool(
        'TMLPU_V54_PAIR_EXTEND_ON', False)
    recon.euler_tangential_pair_restore_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_tangential_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = 0.0
    return recon


def _tmlpu_v54_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v54_unified_euler())


def _tmlpu_v55_unified_euler():
    recon = _tmlpu_v54_unified_euler()
    recon.euler_tangential_pair_restore_on = _env_bool(
        'TMLPU_V55_TANGENTIAL_PAIR_RESTORE_ON', True)
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V55_TANGENTIAL_PAIR_ALPHA', '0.014'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V55_TANGENTIAL_PAIR_CAP', '0.030'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V55_TANGENTIAL_PAIR_WAVE_CAP', '0.004'))
    recon.euler_tangential_pair_gate_contact_min = float(
        os.environ.get('TMLPU_V55_PAIR_GATE_CONTACT_MIN', '0.22'))
    recon.euler_tangential_pair_gate_contact_full = float(
        os.environ.get('TMLPU_V55_PAIR_GATE_CONTACT_FULL', '0.50'))
    recon.euler_tangential_pair_gate_shear_min = float(
        os.environ.get('TMLPU_V55_PAIR_GATE_SHEAR_MIN', '0.55'))
    recon.euler_tangential_pair_gate_shear_full = float(
        os.environ.get('TMLPU_V55_PAIR_GATE_SHEAR_FULL', '0.82'))
    recon.euler_tangential_pair_gate_density_support_min = float(
        os.environ.get('TMLPU_V55_PAIR_GATE_DENSITY_SUPPORT_MIN', '0.0'))
    recon.euler_tangential_pair_gate_density_support_full = float(
        os.environ.get('TMLPU_V55_PAIR_GATE_DENSITY_SUPPORT_FULL', '0.035'))
    return recon


def _tmlpu_v55_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v55_unified_euler())


def _tmlpu_v56_unified_euler():
    recon = _tmlpu_v52_unified_euler()
    recon.euler_tangential_pair_gate_density_support_min = float(
        os.environ.get('TMLPU_V56_PAIR_GATE_DENSITY_SUPPORT_MIN', '0.045'))
    recon.euler_tangential_pair_gate_density_support_full = float(
        os.environ.get('TMLPU_V56_PAIR_GATE_DENSITY_SUPPORT_FULL', '0.120'))
    recon.euler_tangential_pair_extend_on = _env_bool(
        'TMLPU_V56_PAIR_EXTEND_ON', False)
    recon.euler_density_contact_weak_face_downstream_tangential_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = 0.0
    return recon


def _tmlpu_v56_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v56_unified_euler())


def _tmlpu_v57_unified_euler():
    recon = _tmlpu_v54_unified_euler()
    recon.euler_pressure_first_order = _env_bool(
        'TMLPU_V57_PRESSURE_FIRST_ORDER', True)
    return recon


def _tmlpu_v57_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v57_unified_euler())


def _tmlpu_v58_unified_euler():
    recon = _tmlpu_v54_unified_euler()
    recon.euler_velocity_shock_flatten = _env_bool(
        'TMLPU_V58_VELOCITY_SHOCK_FLATTEN', True)
    recon.euler_velocity_flatten_sensor = os.environ.get(
        'TMLPU_V58_VELOCITY_FLATTEN_SENSOR', 'ducros_geomean_core')
    recon.euler_tangential_contact_relax_flatten = _env_bool(
        'TMLPU_V58_TANGENTIAL_RELAX_FLATTEN', True)
    return recon


def _tmlpu_v58_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v58_unified_euler())


def _tmlpu_v59_unified_euler():
    recon = _tmlpu_v54_unified_euler()
    recon.euler_wall_tangential_flatten = _env_bool(
        'TMLPU_V59_WALL_TANGENTIAL_FLATTEN', True)
    return recon


def _tmlpu_v59_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v59_unified_euler())


def _tmlpu_v60_unified_euler():
    recon = _tmlpu_v59_unified_euler()
    recon.euler_density_contact_weak_face_mlp_cap = float(
        os.environ.get('TMLPU_V60_DENSITY_WEAK_FACE_MLP_CAP', '0.78'))
    return recon


def _tmlpu_v60_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v60_unified_euler())


def _tmlpu_v61_unified_euler():
    recon = _tmlpu_v59_unified_euler()
    recon.euler_density_contact_weak_face_stream_coherence_on = _env_bool(
        'TMLPU_V61_DENSITY_STREAM_COHERENCE_ON', True)
    recon.euler_density_contact_weak_face_stream_coherence_min = float(
        os.environ.get('TMLPU_V61_DENSITY_STREAM_COHERENCE_MIN', '0.05'))
    recon.euler_density_contact_weak_face_stream_coherence_full = float(
        os.environ.get('TMLPU_V61_DENSITY_STREAM_COHERENCE_FULL', '0.45'))
    recon.euler_density_contact_weak_face_downstream_rho_beta = float(
        os.environ.get('TMLPU_V61_DOWNSTREAM_RHO_BETA', '0.060'))
    recon.euler_density_contact_weak_face_downstream_rho_cap = float(
        os.environ.get('TMLPU_V61_DOWNSTREAM_RHO_CAP', '0.010'))
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = float(
        os.environ.get('TMLPU_V61_DOWNSTREAM_RHO_WAVE_CAP', '0.004'))
    return recon


def _tmlpu_v61_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v61_unified_euler())


def _tmlpu_v62_unified_euler():
    recon = _tmlpu_v59_unified_euler()
    recon.euler_density_contact_weak_face_swirl_extra = float(
        os.environ.get('TMLPU_V62_DENSITY_WEAK_FACE_SWIRL_EXTRA', '0.10'))
    return recon


def _tmlpu_v62_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v62_unified_euler())


def _tmlpu_v63_unified_euler():
    recon = _tmlpu_v59_unified_euler()
    recon.euler_tangential_pair_extend_on = _env_bool(
        'TMLPU_V63_PAIR_EXTEND_ON', True)
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V63_PAIR_EXTEND_BETA', '0.055'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V63_PAIR_EXTEND_CAP', '0.055'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V63_PAIR_EXTEND_WAVE_CAP', '0.006'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V63_PAIR_EXTEND_ALIGNMENT_MIN', '0.42'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V63_PAIR_EXTEND_ALIGNMENT_FULL', '0.76'))
    return recon


def _tmlpu_v63_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v63_unified_euler())


def _tmlpu_v64_unified_euler():
    recon = _tmlpu_v59_unified_euler()
    recon.euler_tangential_pair_extend_on = _env_bool(
        'TMLPU_V64_PAIR_EXTEND_ON', True)
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V64_PAIR_EXTEND_BETA', '0.20'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V64_PAIR_EXTEND_CAP', '0.12'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V64_PAIR_EXTEND_WAVE_CAP', '0.020'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V64_PAIR_EXTEND_ALIGNMENT_MIN', '0.0'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V64_PAIR_EXTEND_ALIGNMENT_FULL', '0.25'))
    return recon


def _tmlpu_v64_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v64_unified_euler())


def _tmlpu_v65_unified_euler():
    return EulerPrimitiveBlendReconstruction(
        MLPU1(), _tmlpu_v59_unified_euler())


def _tmlpu_v65_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v65_unified_euler())


def _tmlpu_v66_unified_euler():
    recon = _tmlpu_v54_unified_euler()
    recon.euler_wall_tangential_flatten = _env_bool(
        'TMLPU_V66_WALL_TANGENTIAL_FLATTEN', True)
    recon.euler_wall_tangential_flatten_mode = os.environ.get(
        'TMLPU_V66_WALL_TANGENTIAL_FLATTEN_MODE', 'shock_only')
    recon.euler_wall_tangential_flatten_strength = float(os.environ.get(
        'TMLPU_V66_WALL_TANGENTIAL_FLATTEN_STRENGTH', '1.0'))
    return recon


def _tmlpu_v66_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v66_unified_euler())


def _tmlpu_v67_unified_euler():
    recon = _tmlpu_v59_unified_euler()
    recon.euler_tangential_pair_restore_on = _env_bool(
        'TMLPU_V67_TANGENTIAL_PAIR_RESTORE_ON', True)
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V67_TANGENTIAL_PAIR_ALPHA', '0.026'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V67_TANGENTIAL_PAIR_CAP', '0.045'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V67_TANGENTIAL_PAIR_WAVE_CAP', '0.006'))
    recon.euler_tangential_pair_gate_contact_min = float(
        os.environ.get('TMLPU_V67_PAIR_GATE_CONTACT_MIN', '0.18'))
    recon.euler_tangential_pair_gate_contact_full = float(
        os.environ.get('TMLPU_V67_PAIR_GATE_CONTACT_FULL', '0.46'))
    recon.euler_tangential_pair_gate_shear_min = float(
        os.environ.get('TMLPU_V67_PAIR_GATE_SHEAR_MIN', '0.50'))
    recon.euler_tangential_pair_gate_shear_full = float(
        os.environ.get('TMLPU_V67_PAIR_GATE_SHEAR_FULL', '0.78'))
    recon.euler_tangential_pair_gate_density_support_min = float(
        os.environ.get('TMLPU_V67_PAIR_GATE_DENSITY_SUPPORT_MIN', '0.0'))
    recon.euler_tangential_pair_gate_density_support_full = float(
        os.environ.get('TMLPU_V67_PAIR_GATE_DENSITY_SUPPORT_FULL', '0.030'))
    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V67_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V67_STREAM_COHERENCE_MIN', '0.04'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V67_STREAM_COHERENCE_FULL', '0.36'))
    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V67_DOWNSTREAM_TANGENTIAL_BETA', '0.045'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V67_DOWNSTREAM_TANGENTIAL_CAP', '0.045'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V67_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.006'))
    return recon


def _tmlpu_v67_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v67_unified_euler())


def _tmlpu_v68_unified_euler():
    recon = _tmlpu_v67_unified_euler()
    recon.euler_tangential_pair_extend_on = _env_bool(
        'TMLPU_V68_PAIR_EXTEND_ON', True)
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V68_PAIR_EXTEND_BETA', '0.120'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V68_PAIR_EXTEND_CAP', '0.080'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V68_PAIR_EXTEND_WAVE_CAP', '0.012'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V68_PAIR_EXTEND_ALIGNMENT_MIN', '0.10'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V68_PAIR_EXTEND_ALIGNMENT_FULL', '0.40'))
    return recon


def _tmlpu_v68_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v68_unified_euler())


def _tmlpu_v69_unified_euler():
    return EulerPrimitiveSlotBlendReconstruction(
        _tmlpu_v59_unified_euler(), MLPU1(), slots=(0,))


def _tmlpu_v69_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v69_unified_euler())


def _tmlpu_v70_unified_euler():
    return EulerPrimitiveSlotBlendReconstruction(
        _tmlpu_v59_unified_euler(), MLPU1(), slots=(1, 2))


def _tmlpu_v70_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v70_unified_euler())


def _tmlpu_v71_unified_euler():
    recon = _tmlpu_v59_unified_euler()
    recon.euler_wall_tangential_flatten_strength = float(os.environ.get(
        'TMLPU_V71_WALL_TANGENTIAL_FLATTEN_STRENGTH', '0.85'))
    return recon


def _tmlpu_v71_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v71_unified_euler())


def _tmlpu_v72_unified_euler():
    recon = _tmlpu_v67_unified_euler()
    recon.euler_tangential_pair_ignore_normality_gate = _env_bool(
        'TMLPU_V72_IGNORE_PAIR_NORMALITY_GATE', True)
    return recon


def _tmlpu_v72_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v72_unified_euler())


def _tmlpu_v73_unified_euler():
    recon = _tmlpu_v72_unified_euler()
    recon.euler_tangential_pair_gate_density_support_min = float(
        os.environ.get('TMLPU_V73_PAIR_GATE_DENSITY_SUPPORT_MIN', '-0.01'))
    recon.euler_tangential_pair_gate_density_support_full = float(
        os.environ.get('TMLPU_V73_PAIR_GATE_DENSITY_SUPPORT_FULL', '0.0'))
    recon.euler_tangential_pair_extend_on = _env_bool(
        'TMLPU_V73_PAIR_EXTEND_ON', True)
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V73_PAIR_EXTEND_BETA', '0.080'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V73_PAIR_EXTEND_CAP', '0.060'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V73_PAIR_EXTEND_WAVE_CAP', '0.008'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V73_PAIR_EXTEND_ALIGNMENT_MIN', '0.05'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V73_PAIR_EXTEND_ALIGNMENT_FULL', '0.35'))
    return recon


def _tmlpu_v73_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v73_unified_euler())


def _tmlpu_v74_unified_euler():
    recon = _tmlpu_v73_unified_euler()
    recon.euler_density_contact_weak_face_stream_coherence_on = _env_bool(
        'TMLPU_V74_DENSITY_STREAM_COHERENCE_ON', True)
    recon.euler_density_contact_weak_face_stream_coherence_min = float(
        os.environ.get('TMLPU_V74_DENSITY_STREAM_COHERENCE_MIN', '0.03'))
    recon.euler_density_contact_weak_face_stream_coherence_full = float(
        os.environ.get('TMLPU_V74_DENSITY_STREAM_COHERENCE_FULL', '0.32'))
    recon.euler_density_contact_weak_face_downstream_rho_beta = float(
        os.environ.get('TMLPU_V74_DOWNSTREAM_RHO_BETA', '0.045'))
    recon.euler_density_contact_weak_face_downstream_rho_cap = float(
        os.environ.get('TMLPU_V74_DOWNSTREAM_RHO_CAP', '0.008'))
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = float(
        os.environ.get('TMLPU_V74_DOWNSTREAM_RHO_WAVE_CAP', '0.003'))
    return recon


def _tmlpu_v74_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v74_unified_euler())


def _tmlpu_v75_unified_euler():
    recon = _tmlpu_v73_unified_euler()

    # Revert v74 direction: no downstream density stream bridge.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    # Local bounded density support only in coherent clean shear/contact.
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V75_DENSITY_MICRO_RESTORE_ON', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = os.environ.get(
        'TMLPU_V75_DENSITY_MICRO_MODE', 'coherent_shear_micro_restore')
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V75_DENSITY_MICRO_ALPHA', '0.045'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V75_DENSITY_MICRO_PAD', '0.006'))
    recon.euler_density_contact_weak_face_value_scaling_require_coherent_shear = _env_bool(
        'TMLPU_V75_REQUIRE_COHERENT_SHEAR', True)
    recon.euler_density_contact_weak_face_value_scaling_artifact_reject = _env_bool(
        'TMLPU_V75_ARTIFACT_REJECT', True)
    recon.euler_density_contact_weak_face_density_increment_cap = float(
        os.environ.get('TMLPU_V75_DENSITY_INCREMENT_CAP', '0.006'))
    return recon


def _tmlpu_v75_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v75_unified_euler())


def _tmlpu_v76_unified_euler():
    recon = _tmlpu_v73_unified_euler()

    # Keep v74 downstream density bridge disabled.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    # Weaker v75 density support: enough for contour hooks, not enough to erase pairs.
    recon.euler_density_contact_weak_face_value_scaling = _env_bool(
        'TMLPU_V76_DENSITY_MICRO_RESTORE_ON', True)
    recon.euler_density_contact_weak_face_value_scaling_mode = os.environ.get(
        'TMLPU_V76_DENSITY_MICRO_MODE', 'coherent_shear_micro_restore')
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V76_DENSITY_MICRO_ALPHA', '0.022'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V76_DENSITY_MICRO_PAD', '0.004'))
    recon.euler_density_contact_weak_face_value_scaling_require_coherent_shear = _env_bool(
        'TMLPU_V76_REQUIRE_COHERENT_SHEAR', True)
    recon.euler_density_contact_weak_face_value_scaling_artifact_reject = _env_bool(
        'TMLPU_V76_ARTIFACT_REJECT', True)
    recon.euler_density_contact_weak_face_density_increment_cap = float(
        os.environ.get('TMLPU_V76_DENSITY_INCREMENT_CAP', '0.003'))

    # Recover v73 signed-pair evidence with a bounded tangential boost.
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V76_TANGENTIAL_PAIR_ALPHA', '0.034'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V76_TANGENTIAL_PAIR_CAP', '0.055'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V76_TANGENTIAL_PAIR_WAVE_CAP', '0.008'))
    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V76_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V76_STREAM_COHERENCE_MIN', '0.02'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V76_STREAM_COHERENCE_FULL', '0.30'))

    return recon


def _tmlpu_v76_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v76_unified_euler())


def _tmlpu_v77_unified_euler():
    recon = _tmlpu_v76_unified_euler()

    # Keep broad downstream density bridge off; v74 showed this breaks shocks.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    # Preserve v76 hook support, slightly reduce density amplitude risk.
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V77_DENSITY_MICRO_ALPHA', '0.018'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V77_DENSITY_MICRO_PAD', '0.004'))
    recon.euler_density_contact_weak_face_density_increment_cap = float(
        os.environ.get('TMLPU_V77_DENSITY_INCREMENT_CAP', '0.0025'))

    # Main v77 change: extend existing tangential pair support downstream.
    recon.euler_tangential_pair_extend_on = _env_bool(
        'TMLPU_V77_PAIR_EXTEND_ON', True)
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V77_PAIR_EXTEND_BETA', '0.115'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V77_PAIR_EXTEND_CAP', '0.070'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V77_PAIR_EXTEND_WAVE_CAP', '0.010'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V77_PAIR_EXTEND_ALIGNMENT_MIN', '0.00'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V77_PAIR_EXTEND_ALIGNMENT_FULL', '0.24'))

    # Keep pair restore bounded but a touch stronger than v76.
    recon.euler_tangential_pair_restore_alpha = float(
        os.environ.get('TMLPU_V77_TANGENTIAL_PAIR_ALPHA', '0.036'))
    recon.euler_tangential_pair_restore_cap = float(
        os.environ.get('TMLPU_V77_TANGENTIAL_PAIR_CAP', '0.058'))
    recon.euler_tangential_pair_restore_wave_cap = float(
        os.environ.get('TMLPU_V77_TANGENTIAL_PAIR_WAVE_CAP', '0.008'))

    return recon


def _tmlpu_v77_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v77_unified_euler())


def _tmlpu_v78_unified_euler():
    recon = _tmlpu_v77_unified_euler()

    # Keep the v74 broad density bridge disabled.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    # Preserve v77 hook support without increasing density forcing.
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V78_DENSITY_MICRO_ALPHA', '0.016'))
    recon.euler_density_contact_weak_face_density_increment_cap = float(
        os.environ.get('TMLPU_V78_DENSITY_INCREMENT_CAP', '0.0025'))

    # Main v78 change: extend existing signed-pair tails downstream through
    # stream-coherent tangential reconstruction, not density bridging.
    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V78_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V78_STREAM_COHERENCE_MIN', '0.0'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V78_STREAM_COHERENCE_FULL', '0.20'))

    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V78_DOWNSTREAM_TANGENTIAL_BETA', '0.085'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V78_DOWNSTREAM_TANGENTIAL_CAP', '0.080'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V78_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.012'))

    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V78_PAIR_EXTEND_BETA', '0.135'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V78_PAIR_EXTEND_CAP', '0.080'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V78_PAIR_EXTEND_WAVE_CAP', '0.012'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V78_PAIR_EXTEND_ALIGNMENT_MIN', '0.0'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V78_PAIR_EXTEND_ALIGNMENT_FULL', '0.18'))

    return recon


def _tmlpu_v78_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v78_unified_euler())


def _tmlpu_v79_unified_euler():
    recon = _tmlpu_v77_unified_euler()

    # Keep broad downstream density bridge disabled.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    # Revert density support to v77 risk level; v78 reduced hooks too much.
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V79_DENSITY_MICRO_ALPHA', '0.018'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V79_DENSITY_MICRO_PAD', '0.004'))
    recon.euler_density_contact_weak_face_density_increment_cap = float(
        os.environ.get('TMLPU_V79_DENSITY_INCREMENT_CAP', '0.0025'))

    # Mild downstream tangential tail: between v77 and v78.
    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V79_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V79_STREAM_COHERENCE_MIN', '0.0'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V79_STREAM_COHERENCE_FULL', '0.26'))

    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V79_DOWNSTREAM_TANGENTIAL_BETA', '0.052'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V79_DOWNSTREAM_TANGENTIAL_CAP', '0.052'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V79_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.007'))

    # Less aggressive than v78; slightly above v77 to retain some extent gain.
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V79_PAIR_EXTEND_BETA', '0.122'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V79_PAIR_EXTEND_CAP', '0.072'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V79_PAIR_EXTEND_WAVE_CAP', '0.010'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V79_PAIR_EXTEND_ALIGNMENT_MIN', '0.0'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V79_PAIR_EXTEND_ALIGNMENT_FULL', '0.22'))

    return recon


def _tmlpu_v79_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v79_unified_euler())


def _tmlpu_v80_unified_euler():
    recon = _tmlpu_v77_unified_euler()

    # Keep broad downstream density bridge disabled.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    # Keep v77 density micro level; do not add density transport.
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = float(
        os.environ.get('TMLPU_V80_DENSITY_MICRO_ALPHA', '0.018'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = float(
        os.environ.get('TMLPU_V80_DENSITY_MICRO_PAD', '0.004'))
    recon.euler_density_contact_weak_face_density_increment_cap = float(
        os.environ.get('TMLPU_V80_DENSITY_INCREMENT_CAP', '0.0025'))

    # Reset pair extension to v77-safe geometry; v79's wider alignment leaked
    # into global shock-split bands.
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V80_PAIR_EXTEND_BETA', '0.116'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V80_PAIR_EXTEND_CAP', '0.070'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V80_PAIR_EXTEND_WAVE_CAP', '0.010'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V80_PAIR_EXTEND_ALIGNMENT_MIN', '0.0'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V80_PAIR_EXTEND_ALIGNMENT_FULL', '0.18'))

    # Mild downstream tangential tail: smaller than v79, slightly above v77.
    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V80_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V80_STREAM_COHERENCE_MIN', '0.0'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V80_STREAM_COHERENCE_FULL', '0.24'))

    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V80_DOWNSTREAM_TANGENTIAL_BETA', '0.047'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V80_DOWNSTREAM_TANGENTIAL_CAP', '0.046'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V80_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.006'))

    return recon


def _tmlpu_v80_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v80_unified_euler())


def _tmlpu_v81_unified_euler():
    recon = _tmlpu_v80_unified_euler()

    # Keep broad downstream density bridge disabled.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    recon.euler_tangential_downstream_shock_exclude = _env_bool(
        'TMLPU_V81_DOWNSTREAM_SHOCK_EXCLUDE', True)

    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V81_DOWNSTREAM_TANGENTIAL_BETA', '0.062'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V81_DOWNSTREAM_TANGENTIAL_CAP', '0.058'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V81_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.008'))

    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V81_PAIR_EXTEND_BETA', '0.120'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V81_PAIR_EXTEND_CAP', '0.072'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V81_PAIR_EXTEND_WAVE_CAP', '0.010'))

    return recon


def _tmlpu_v81_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v81_unified_euler())


def _tmlpu_v82_unified_euler():
    recon = _tmlpu_v81_unified_euler()

    # Keep broad density transport off.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    recon.euler_tangential_downstream_shock_exclude = _env_bool(
        'TMLPU_V82_DOWNSTREAM_SHOCK_EXCLUDE', True)

    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V82_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V82_STREAM_COHERENCE_MIN', '0.0'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V82_STREAM_COHERENCE_FULL', '0.18'))

    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V82_DOWNSTREAM_TANGENTIAL_BETA', '0.078'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V82_DOWNSTREAM_TANGENTIAL_CAP', '0.068'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V82_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.009'))

    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V82_PAIR_EXTEND_BETA', '0.128'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V82_PAIR_EXTEND_CAP', '0.076'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V82_PAIR_EXTEND_WAVE_CAP', '0.011'))

    return recon


def _tmlpu_v82_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v82_unified_euler())


def _tmlpu_v83_unified_euler():
    recon = _tmlpu_v82_unified_euler()

    # Keep broad density transport off.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    recon.euler_tangential_downstream_shock_exclude = _env_bool(
        'TMLPU_V83_DOWNSTREAM_SHOCK_EXCLUDE', True)
    recon.euler_tangential_downstream_shock_pressure_min = float(
        os.environ.get('TMLPU_V83_DOWNSTREAM_SHOCK_PRESSURE_MIN', '0.015'))
    recon.euler_tangential_downstream_shock_compression_min = float(
        os.environ.get('TMLPU_V83_DOWNSTREAM_SHOCK_COMPRESSION_MIN', '0.003'))
    recon.euler_tangential_downstream_shock_normality_min = float(
        os.environ.get('TMLPU_V83_DOWNSTREAM_SHOCK_NORMALITY_MIN', '0.25'))

    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V83_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V83_STREAM_COHERENCE_MIN', '0.0'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V83_STREAM_COHERENCE_FULL', '0.20'))

    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V83_DOWNSTREAM_TANGENTIAL_BETA', '0.070'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V83_DOWNSTREAM_TANGENTIAL_CAP', '0.063'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V83_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.0085'))

    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V83_PAIR_EXTEND_BETA', '0.122'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V83_PAIR_EXTEND_CAP', '0.072'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V83_PAIR_EXTEND_WAVE_CAP', '0.010'))

    return recon


def _tmlpu_v83_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v83_unified_euler())


def _tmlpu_v84_unified_euler():
    return _tmlpu_v83_unified_euler()


def _tmlpu_v84_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v84_unified_euler())


def _tmlpu_v85_unified_euler():
    recon = _tmlpu_v81_unified_euler()

    # Keep broad density transport off.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    recon.euler_tangential_pair_extend_shock_exclude = _env_bool(
        'TMLPU_V85_PAIR_EXTEND_SHOCK_EXCLUDE', False)
    recon.euler_tangential_downstream_shock_exclude = _env_bool(
        'TMLPU_V85_DOWNSTREAM_SHOCK_EXCLUDE', True)

    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V85_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V85_STREAM_COHERENCE_MIN', '0.0'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V85_STREAM_COHERENCE_FULL', '0.22'))

    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V85_DOWNSTREAM_TANGENTIAL_BETA', '0.066'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V85_DOWNSTREAM_TANGENTIAL_CAP', '0.060'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V85_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.008'))

    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V85_PAIR_EXTEND_BETA', '0.120'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V85_PAIR_EXTEND_CAP', '0.072'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V85_PAIR_EXTEND_WAVE_CAP', '0.010'))

    return recon


def _tmlpu_v85_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v85_unified_euler())


def _tmlpu_v86_unified_euler():
    recon = _tmlpu_v85_unified_euler()

    recon.euler_tangential_pair_extend_shock_exclude = False

    # Keep broad density transport off.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    recon.euler_tangential_downstream_branch_damp_on = _env_bool(
        'TMLPU_V86_BRANCH_DAMP_ON', True)
    recon.euler_tangential_downstream_branch_pressure_min = float(
        os.environ.get('TMLPU_V86_BRANCH_PRESSURE_MIN', '0.010'))
    recon.euler_tangential_downstream_branch_compression_min = float(
        os.environ.get('TMLPU_V86_BRANCH_COMPRESSION_MIN', '0.002'))
    recon.euler_tangential_downstream_branch_normality_min = float(
        os.environ.get('TMLPU_V86_BRANCH_NORMALITY_MIN', '0.18'))
    recon.euler_tangential_downstream_branch_floor = float(
        os.environ.get('TMLPU_V86_BRANCH_FLOOR', '0.78'))

    return recon


def _tmlpu_v86_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v86_unified_euler())


def _tmlpu_v87_unified_euler():
    recon = _tmlpu_v81_unified_euler()

    # Keep broad density transport off.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    recon.euler_tangential_pair_extend_shock_exclude = False
    recon.euler_tangential_downstream_branch_damp_on = False

    recon.euler_tangential_clean_contact_tail_on = _env_bool(
        'TMLPU_V87_CLEAN_CONTACT_TAIL_ON', True)
    recon.euler_tangential_clean_contact_tail_beta = float(
        os.environ.get('TMLPU_V87_CLEAN_CONTACT_TAIL_BETA', '0.055'))
    recon.euler_tangential_clean_contact_tail_cap = float(
        os.environ.get('TMLPU_V87_CLEAN_CONTACT_TAIL_CAP', '0.050'))
    recon.euler_tangential_clean_contact_tail_wave_cap = float(
        os.environ.get('TMLPU_V87_CLEAN_CONTACT_TAIL_WAVE_CAP', '0.006'))
    recon.euler_tangential_clean_contact_tail_stream_full = float(
        os.environ.get('TMLPU_V87_CLEAN_CONTACT_TAIL_STREAM_FULL', '0.18'))
    recon.euler_tangential_clean_contact_tail_pressure_lo = float(
        os.environ.get('TMLPU_V87_CLEAN_CONTACT_TAIL_PRESSURE_LO', '0.006'))
    recon.euler_tangential_clean_contact_tail_pressure_hi = float(
        os.environ.get('TMLPU_V87_CLEAN_CONTACT_TAIL_PRESSURE_HI', '0.020'))
    recon.euler_tangential_clean_contact_tail_compression_lo = float(
        os.environ.get('TMLPU_V87_CLEAN_CONTACT_TAIL_COMPRESSION_LO', '0.001'))
    recon.euler_tangential_clean_contact_tail_compression_hi = float(
        os.environ.get('TMLPU_V87_CLEAN_CONTACT_TAIL_COMPRESSION_HI', '0.006'))
    recon.euler_tangential_clean_contact_tail_normality_lo = float(
        os.environ.get('TMLPU_V87_CLEAN_CONTACT_TAIL_NORMALITY_LO', '0.10'))
    recon.euler_tangential_clean_contact_tail_normality_hi = float(
        os.environ.get('TMLPU_V87_CLEAN_CONTACT_TAIL_NORMALITY_HI', '0.24'))

    return recon


def _tmlpu_v87_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v87_unified_euler())


def _tmlpu_v88_unified_euler():
    recon = _tmlpu_v81_unified_euler()

    # Keep broad density transport off.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    recon.euler_tangential_pair_extend_shock_exclude = False
    recon.euler_tangential_downstream_branch_damp_on = False
    recon.euler_tangential_clean_contact_tail_on = False

    recon.euler_tangential_swirl_tail_on = _env_bool(
        'TMLPU_V88_SWIRL_TAIL_ON', True)
    recon.euler_tangential_swirl_tail_beta = float(
        os.environ.get('TMLPU_V88_SWIRL_TAIL_BETA', '0.050'))
    recon.euler_tangential_swirl_tail_cap = float(
        os.environ.get('TMLPU_V88_SWIRL_TAIL_CAP', '0.045'))
    recon.euler_tangential_swirl_tail_wave_cap = float(
        os.environ.get('TMLPU_V88_SWIRL_TAIL_WAVE_CAP', '0.006'))
    recon.euler_tangential_swirl_tail_q_min = float(
        os.environ.get('TMLPU_V88_SWIRL_TAIL_Q_MIN', '0.015'))
    recon.euler_tangential_swirl_tail_q_full = float(
        os.environ.get('TMLPU_V88_SWIRL_TAIL_Q_FULL', '0.055'))
    recon.euler_tangential_swirl_tail_pressure_hi = float(
        os.environ.get('TMLPU_V88_SWIRL_TAIL_PRESSURE_HI', '0.018'))
    recon.euler_tangential_swirl_tail_compression_hi = float(
        os.environ.get('TMLPU_V88_SWIRL_TAIL_COMPRESSION_HI', '0.004'))
    recon.euler_tangential_swirl_tail_normality_hi = float(
        os.environ.get('TMLPU_V88_SWIRL_TAIL_NORMALITY_HI', '0.20'))

    return recon


def _tmlpu_v88_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v88_unified_euler())


def _tmlpu_v89_unified_euler():
    recon = _tmlpu_v81_unified_euler()

    # Keep broad density transport off.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    # Disable post-v81 experimental paths.
    recon.euler_tangential_pair_extend_shock_exclude = False
    recon.euler_tangential_downstream_branch_damp_on = False
    recon.euler_tangential_clean_contact_tail_on = False
    recon.euler_tangential_swirl_tail_on = False

    recon.euler_tangential_pair_restore_stream_coherence_on = _env_bool(
        'TMLPU_V89_STREAM_COHERENCE_ON', True)
    recon.euler_tangential_pair_restore_stream_coherence_min = float(
        os.environ.get('TMLPU_V89_STREAM_COHERENCE_MIN', '0.0'))
    recon.euler_tangential_pair_restore_stream_coherence_full = float(
        os.environ.get('TMLPU_V89_STREAM_COHERENCE_FULL', '0.235'))

    recon.euler_density_contact_weak_face_downstream_tangential_beta = float(
        os.environ.get('TMLPU_V89_DOWNSTREAM_TANGENTIAL_BETA', '0.063'))
    recon.euler_density_contact_weak_face_downstream_tangential_cap = float(
        os.environ.get('TMLPU_V89_DOWNSTREAM_TANGENTIAL_CAP', '0.0585'))
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = float(
        os.environ.get('TMLPU_V89_DOWNSTREAM_TANGENTIAL_WAVE_CAP', '0.008'))

    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V89_PAIR_EXTEND_BETA', '0.120'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V89_PAIR_EXTEND_CAP', '0.072'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V89_PAIR_EXTEND_WAVE_CAP', '0.010'))

    return recon


def _tmlpu_v89_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v89_unified_euler())


def _tmlpu_v90_unified_euler():
    recon = _tmlpu_v81_unified_euler()

    # Keep broad density transport off.
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0

    # Disable v82+ broad downstream/contact/swirl tails.
    recon.euler_tangential_pair_extend_shock_exclude = False
    recon.euler_tangential_downstream_branch_damp_on = False
    recon.euler_tangential_clean_contact_tail_on = False
    recon.euler_tangential_swirl_tail_on = False

    recon.euler_tangential_signed_pair_tail_on = True
    recon.euler_tangential_signed_pair_tail_beta = 0.040
    recon.euler_tangential_signed_pair_tail_cap = 0.035
    recon.euler_tangential_signed_pair_tail_wave_cap = 0.0045

    return recon


def _tmlpu_v90_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v90_unified_euler())


def _tmlpu_v91_unified_euler():
    recon = _tmlpu_v90_unified_euler()
    recon.euler_tangential_shockline_rollback_on = True
    recon.euler_tangential_shockline_rollback_theta = 0.55
    recon.euler_tangential_shockline_pressure_min = 0.012
    recon.euler_tangential_shockline_compression_min = 0.0025
    recon.euler_tangential_shockline_normality_min = 0.18
    recon.euler_tangential_shockline_shear_max = 0.86
    return recon


def _tmlpu_v91_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v91_unified_euler())


def _tmlpu_v92_unified_euler():
    recon = _tmlpu_v90_unified_euler()
    recon.euler_tangential_shockline_rollback_on = False
    recon.euler_tangential_pair_extend_shock_exclude = False
    recon.euler_tangential_downstream_branch_damp_on = False
    recon.euler_tangential_clean_contact_tail_on = False
    recon.euler_tangential_swirl_tail_on = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0
    recon.euler_tangential_density_curve_pair_tail_on = True
    recon.euler_tangential_density_curve_pair_tail_beta = 0.035
    recon.euler_tangential_density_curve_pair_tail_cap = 0.030
    recon.euler_tangential_density_curve_pair_tail_wave_cap = 0.004
    return recon


def _tmlpu_v92_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v92_unified_euler())


def _tmlpu_v93_diag_unified_euler():
    return _tmlpu_v92_unified_euler()


def _tmlpu_v93_diag_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v93_diag_unified_euler())


def _tmlpu_v95_legacy_pair_target_euler():
    recon = _tmlpu_v92_unified_euler()
    recon.euler_tangential_legacy_pair_target_on = True
    recon.euler_tangential_legacy_pair_target_blend = 1.0
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0
    return recon


def _tmlpu_v95_legacy_pair_target():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v95_legacy_pair_target_euler())


def _tmlpu_v96_legacy_pair_target_half_euler():
    recon = _tmlpu_v95_legacy_pair_target_euler()
    recon.euler_tangential_legacy_pair_target_on = True
    recon.euler_tangential_legacy_pair_target_blend = 0.50
    return recon


def _tmlpu_v96_legacy_pair_target_half():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v96_legacy_pair_target_half_euler())


def _tmlpu_v97_legacy_pair_target_075_euler():
    recon = _tmlpu_v95_legacy_pair_target_euler()
    recon.euler_tangential_legacy_pair_target_on = True
    recon.euler_tangential_legacy_pair_target_blend = float(
        os.environ.get('TMLPU_V97_LEGACY_PAIR_TARGET_BLEND', '0.75'))
    return recon


def _tmlpu_v97_legacy_pair_target_075():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v97_legacy_pair_target_075_euler())


def _tmlpu_v98_legacy_pair_target_0875_euler():
    recon = _tmlpu_v95_legacy_pair_target_euler()
    recon.euler_tangential_legacy_pair_target_on = True
    recon.euler_tangential_legacy_pair_target_blend = float(
        os.environ.get('TMLPU_V98_LEGACY_PAIR_TARGET_BLEND', '0.875'))
    return recon


def _tmlpu_v98_legacy_pair_target_0875():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v98_legacy_pair_target_0875_euler())


def _tmlpu_v99_split_legacy_target_euler():
    recon = _tmlpu_v95_legacy_pair_target_euler()
    recon.euler_tangential_legacy_pair_target_on = True
    recon.euler_tangential_signed_pair_legacy_target_blend = float(
        os.environ.get('TMLPU_V99_SIGNED_LEGACY_BLEND', '0.75'))
    recon.euler_tangential_density_curve_legacy_target_blend = float(
        os.environ.get('TMLPU_V99_CURVE_LEGACY_BLEND', '1.0'))
    return recon


def _tmlpu_v99_split_legacy_target():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v99_split_legacy_target_euler())


def _tmlpu_v100_safe_legacy_target_euler():
    recon = _tmlpu_v95_legacy_pair_target_euler()
    recon.euler_tangential_legacy_pair_target_on = True
    recon.euler_tangential_legacy_pair_target_blend = 1.0
    recon.euler_tangential_safe_legacy_gate_on = True
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.82
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    return recon


def _tmlpu_v100_safe_legacy_target():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v100_safe_legacy_target_euler())


def _tmlpu_v101_safe_legacy_pressure014_euler():
    recon = _tmlpu_v100_safe_legacy_target_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = float(
        os.environ.get('TMLPU_V101_SAFE_LEGACY_PRESSURE_HI', '0.014'))
    return recon


def _tmlpu_v101_safe_legacy_pressure014():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v101_safe_legacy_pressure014_euler())


def _tmlpu_v102_safe_legacy_contact035_euler():
    recon = _tmlpu_v100_safe_legacy_target_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_contact_min = float(
        os.environ.get('TMLPU_V102_SAFE_LEGACY_CONTACT_MIN', '0.35'))
    return recon


def _tmlpu_v102_safe_legacy_contact035():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v102_safe_legacy_contact035_euler())


def _tmlpu_v103_safe_legacy_shear072_euler():
    recon = _tmlpu_v100_safe_legacy_target_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_safe_legacy_shear_min = float(
        os.environ.get('TMLPU_V103_SAFE_LEGACY_SHEAR_MIN', '0.72'))
    return recon


def _tmlpu_v103_safe_legacy_shear072():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v103_safe_legacy_shear072_euler())


def _tmlpu_v104_safe_legacy_shear072_norm020_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_normality_hi = float(
        os.environ.get('TMLPU_V104_SAFE_LEGACY_NORMALITY_HI', '0.20'))
    return recon


def _tmlpu_v104_safe_legacy_shear072_norm020():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v104_safe_legacy_shear072_norm020_euler())


def _tmlpu_v105_safe_legacy_coherence_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_safe_legacy_coherence_on = True
    recon.euler_tangential_safe_legacy_coherence_beta = 0.25
    recon.euler_tangential_safe_legacy_coherence_floor = 0.08
    recon.euler_tangential_safe_legacy_coherence_cap = 0.35
    return recon


def _tmlpu_v105_safe_legacy_coherence():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v105_safe_legacy_coherence_euler())


def _tmlpu_v106_safe_legacy_qcurv_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_safe_legacy_qcurv_on = True
    recon.euler_tangential_safe_legacy_qcurv_beta = 0.18
    recon.euler_tangential_safe_legacy_qcurv_q_min = 0.012
    recon.euler_tangential_safe_legacy_qcurv_q_full = 0.045
    recon.euler_tangential_safe_legacy_qcurv_curve_min = 0.20
    recon.euler_tangential_safe_legacy_qcurv_curve_full = 0.50
    return recon


def _tmlpu_v106_safe_legacy_qcurv():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v106_safe_legacy_qcurv_euler())


def _tmlpu_v107_safe_legacy_capboost_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_signed_pair_tail_cap = float(
        os.environ.get('TMLPU_V107_SIGNED_TAIL_CAP', '0.045'))
    recon.euler_tangential_signed_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V107_SIGNED_TAIL_WAVE_CAP', '0.0060'))
    recon.euler_tangential_density_curve_pair_tail_cap = float(
        os.environ.get('TMLPU_V107_CURVE_TAIL_CAP', '0.040'))
    recon.euler_tangential_density_curve_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V107_CURVE_TAIL_WAVE_CAP', '0.0055'))
    return recon


def _tmlpu_v107_safe_legacy_capboost():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v107_safe_legacy_capboost_euler())


def _tmlpu_v108_signed_density_support018_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_tail_density_support_min = float(
        os.environ.get('TMLPU_V108_TAIL_DENSITY_SUPPORT_MIN', '0.014'))
    recon.euler_tangential_tail_density_support_full = float(
        os.environ.get('TMLPU_V108_TAIL_DENSITY_SUPPORT_FULL', '0.060'))
    return recon


def _tmlpu_v108_signed_density_support018():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v108_signed_density_support018_euler())


def _tmlpu_v109_tail_density_support_mid_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_tail_density_support_min = float(
        os.environ.get('TMLPU_V109_TAIL_DENSITY_SUPPORT_MIN', '0.017'))
    recon.euler_tangential_tail_density_support_full = float(
        os.environ.get('TMLPU_V109_TAIL_DENSITY_SUPPORT_FULL', '0.070'))
    return recon


def _tmlpu_v109_tail_density_support_mid():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v109_tail_density_support_mid_euler())


def _tmlpu_v110_tail_density_shockdamp_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_tail_density_support_min = 0.014
    recon.euler_tangential_tail_density_support_full = 0.060
    recon.euler_tangential_tail_density_shock_damp_on = True
    recon.euler_tangential_tail_density_shock_damp_theta = 0.65
    recon.euler_tangential_tail_density_shock_damp_pressure_min = 0.010
    recon.euler_tangential_tail_density_shock_damp_compression_min = 0.002
    recon.euler_tangential_tail_density_shock_damp_normality_min = 0.16
    return recon


def _tmlpu_v110_tail_density_shockdamp():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v110_tail_density_shockdamp_euler())


def _tmlpu_v111_tail_density_min015_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_tail_density_support_min = float(
        os.environ.get('TMLPU_V111_TAIL_DENSITY_SUPPORT_MIN', '0.015'))
    recon.euler_tangential_tail_density_support_full = float(
        os.environ.get('TMLPU_V111_TAIL_DENSITY_SUPPORT_FULL', '0.060'))
    return recon


def _tmlpu_v111_tail_density_min015():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v111_tail_density_min015_euler())


def _tmlpu_v112_tail_density_min0145_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_tail_density_support_min = float(
        os.environ.get('TMLPU_V112_TAIL_DENSITY_SUPPORT_MIN', '0.0145'))
    recon.euler_tangential_tail_density_support_full = float(
        os.environ.get('TMLPU_V112_TAIL_DENSITY_SUPPORT_FULL', '0.060'))
    return recon


def _tmlpu_v112_tail_density_min0145():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v112_tail_density_min0145_euler())


def _tmlpu_v113_tail_density_min015_full055_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_tail_density_support_min = float(
        os.environ.get('TMLPU_V113_TAIL_DENSITY_SUPPORT_MIN', '0.015'))
    recon.euler_tangential_tail_density_support_full = float(
        os.environ.get('TMLPU_V113_TAIL_DENSITY_SUPPORT_FULL', '0.055'))
    return recon


def _tmlpu_v113_tail_density_min015_full055():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v113_tail_density_min015_full055_euler())


def _tmlpu_v114_tail_density_beta115_euler():
    recon = _tmlpu_v103_safe_legacy_shear072_euler()
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_tail_density_support_min = 0.015
    recon.euler_tangential_tail_density_support_full = 0.060
    base_curve_beta = 0.035
    recon.euler_tangential_density_curve_pair_tail_beta = float(
        os.environ.get(
            'TMLPU_V114_CURVE_TAIL_BETA', str(1.15 * base_curve_beta)))
    return recon


def _tmlpu_v114_tail_density_beta115():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v114_tail_density_beta115_euler())


def _tmlpu_v115_v111_taildiag_euler():
    return _tmlpu_v111_tail_density_min015_euler()


def _tmlpu_v115_v111_taildiag():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v115_v111_taildiag_euler())


def _tmlpu_v116_tail_safe_floor_euler():
    recon = _tmlpu_v111_tail_density_min015_euler()
    recon.euler_tangential_tail_density_support_min = 0.015
    recon.euler_tangential_tail_density_support_full = 0.060
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_tail_safe_floor_on = True
    recon.euler_tangential_tail_safe_floor = float(
        os.environ.get('TMLPU_V116_TAIL_SAFE_FLOOR', '0.18'))
    return recon


def _tmlpu_v116_tail_safe_floor():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v116_tail_safe_floor_euler())


def _tmlpu_v117_v111_featurediag_euler():
    return _tmlpu_v111_tail_density_min015_euler()


def _tmlpu_v117_v111_featurediag():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v117_v111_featurediag_euler())


def _tmlpu_v118_shear_contact_relief_euler():
    recon = _tmlpu_v111_tail_density_min015_euler()
    recon.euler_tangential_tail_density_support_min = 0.015
    recon.euler_tangential_tail_density_support_full = 0.060
    recon.euler_tangential_safe_legacy_pressure_hi = 0.010
    recon.euler_tangential_safe_legacy_compression_hi = 0.002
    recon.euler_tangential_safe_legacy_normality_hi = 0.14
    recon.euler_tangential_safe_legacy_shear_min = 0.72
    recon.euler_tangential_safe_legacy_contact_min = 0.45
    recon.euler_tangential_tail_shear_contact_relief_on = True
    recon.euler_tangential_tail_shear_contact_relief_floor = 0.08
    recon.euler_tangential_tail_shear_contact_shear_min = 0.94
    recon.euler_tangential_tail_shear_contact_normality_max = 0.08
    recon.euler_tangential_tail_shear_contact_pressure_max = 0.008
    recon.euler_tangential_tail_shear_contact_compression_max = 0.0015
    return recon


def _tmlpu_v118_shear_contact_relief():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(), _tmlpu_v118_shear_contact_relief_euler())


def _tmlpu_v119_shear_contact_relief_floor04_euler():
    recon = _tmlpu_v118_shear_contact_relief_euler()
    recon.euler_tangential_tail_shear_contact_relief_floor = float(
        os.environ.get('TMLPU_V119_RELIEF_FLOOR', '0.04'))
    return recon


def _tmlpu_v119_shear_contact_relief_floor04():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v119_shear_contact_relief_floor04_euler())


def _tmlpu_v120_shear_contact_relief_floor02_euler():
    recon = _tmlpu_v118_shear_contact_relief_euler()
    recon.euler_tangential_tail_shear_contact_relief_floor = float(
        os.environ.get('TMLPU_V120_RELIEF_FLOOR', '0.02'))
    return recon


def _tmlpu_v120_shear_contact_relief_floor02():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v120_shear_contact_relief_floor02_euler())


def _tmlpu_v121_shear_contact_relief_p006_euler():
    recon = _tmlpu_v118_shear_contact_relief_euler()
    recon.euler_tangential_tail_shear_contact_relief_floor = 0.04
    recon.euler_tangential_tail_shear_contact_pressure_max = float(
        os.environ.get('TMLPU_V121_RELIEF_PRESSURE_MAX', '0.006'))
    return recon


def _tmlpu_v121_shear_contact_relief_p006():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v121_shear_contact_relief_p006_euler())


def _tmlpu_v122_shear_contact_relief_c0010_euler():
    recon = _tmlpu_v118_shear_contact_relief_euler()
    recon.euler_tangential_tail_shear_contact_relief_floor = 0.04
    recon.euler_tangential_tail_shear_contact_pressure_max = 0.008
    recon.euler_tangential_tail_shear_contact_compression_max = float(
        os.environ.get('TMLPU_V122_RELIEF_COMPRESSION_MAX', '0.0010'))
    return recon


def _tmlpu_v122_shear_contact_relief_c0010():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v122_shear_contact_relief_c0010_euler())


def _tmlpu_v123_shear_contact_relief_floor03_euler():
    recon = _tmlpu_v118_shear_contact_relief_euler()
    recon.euler_tangential_tail_shear_contact_pressure_max = 0.008
    recon.euler_tangential_tail_shear_contact_compression_max = 0.0015
    recon.euler_tangential_tail_shear_contact_shear_min = 0.94
    recon.euler_tangential_tail_shear_contact_normality_max = 0.08
    recon.euler_tangential_tail_shear_contact_relief_floor = float(
        os.environ.get('TMLPU_V123_RELIEF_FLOOR', '0.03'))
    return recon


def _tmlpu_v123_shear_contact_relief_floor03():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v123_shear_contact_relief_floor03_euler())


def _tmlpu_v124_shear_contact_relief_floor035_euler():
    recon = _tmlpu_v118_shear_contact_relief_euler()
    recon.euler_tangential_tail_shear_contact_pressure_max = 0.008
    recon.euler_tangential_tail_shear_contact_compression_max = 0.0015
    recon.euler_tangential_tail_shear_contact_shear_min = 0.94
    recon.euler_tangential_tail_shear_contact_normality_max = 0.08
    recon.euler_tangential_tail_shear_contact_relief_floor = float(
        os.environ.get('TMLPU_V124_RELIEF_FLOOR', '0.035'))
    return recon


def _tmlpu_v124_shear_contact_relief_floor035():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v124_shear_contact_relief_floor035_euler())


def _tmlpu_v125_shear_contact_relief_floor0375_euler():
    recon = _tmlpu_v118_shear_contact_relief_euler()
    recon.euler_tangential_tail_shear_contact_pressure_max = 0.008
    recon.euler_tangential_tail_shear_contact_compression_max = 0.0015
    recon.euler_tangential_tail_shear_contact_shear_min = 0.94
    recon.euler_tangential_tail_shear_contact_normality_max = 0.08
    recon.euler_tangential_tail_shear_contact_relief_floor = float(
        os.environ.get('TMLPU_V125_RELIEF_FLOOR', '0.0375'))
    return recon


def _tmlpu_v125_shear_contact_relief_floor0375():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v125_shear_contact_relief_floor0375_euler())


def _tmlpu_v126_curve_only_relief_floor04_euler():
    recon = _tmlpu_v118_shear_contact_relief_euler()
    recon.euler_tangential_tail_shear_contact_relief_floor = 0.04
    recon.euler_tangential_tail_shear_contact_pressure_max = 0.008
    recon.euler_tangential_tail_shear_contact_compression_max = 0.0015
    recon.euler_tangential_tail_shear_contact_shear_min = 0.94
    recon.euler_tangential_tail_shear_contact_normality_max = 0.08
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = False
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = True
    return recon


def _tmlpu_v126_curve_only_relief_floor04():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v126_curve_only_relief_floor04_euler())


def _tmlpu_v127_signed_only_relief_floor04_euler():
    recon = _tmlpu_v118_shear_contact_relief_euler()
    recon.euler_tangential_tail_shear_contact_relief_floor = 0.04
    recon.euler_tangential_tail_shear_contact_pressure_max = 0.008
    recon.euler_tangential_tail_shear_contact_compression_max = 0.0015
    recon.euler_tangential_tail_shear_contact_shear_min = 0.94
    recon.euler_tangential_tail_shear_contact_normality_max = 0.08
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    return recon


def _tmlpu_v127_signed_only_relief_floor04():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v127_signed_only_relief_floor04_euler())


def _tmlpu_v128_asym_relief_signed04_curve02_euler():
    recon = _tmlpu_v118_shear_contact_relief_euler()
    recon.euler_tangential_tail_shear_contact_pressure_max = 0.008
    recon.euler_tangential_tail_shear_contact_compression_max = 0.0015
    recon.euler_tangential_tail_shear_contact_shear_min = 0.94
    recon.euler_tangential_tail_shear_contact_normality_max = 0.08
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = True
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = 0.02
    return recon


def _tmlpu_v128_asym_relief_signed04_curve02():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v128_asym_relief_signed04_curve02_euler())


def _tmlpu_v129_signed_relief_density014_euler():
    recon = _tmlpu_v127_signed_only_relief_floor04_euler()
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_signed_tail_density_support_min = float(
        os.environ.get('TMLPU_V129_SIGNED_DENSITY_SUPPORT_MIN', '0.014'))
    recon.euler_tangential_signed_tail_density_support_full = float(
        os.environ.get('TMLPU_V129_SIGNED_DENSITY_SUPPORT_FULL', '0.060'))
    return recon


def _tmlpu_v129_signed_relief_density014():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v129_signed_relief_density014_euler())


def _tmlpu_v130_signed_only_relief_floor06_euler():
    recon = _tmlpu_v127_signed_only_relief_floor04_euler()
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.06
    recon.euler_tangential_tail_shear_contact_relief_floor = 0.06
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    return recon


def _tmlpu_v130_signed_only_relief_floor06():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v130_signed_only_relief_floor06_euler())


def _tmlpu_v131_signed_gate_decay_relief_euler():
    recon = _tmlpu_v127_signed_only_relief_floor04_euler()
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    return recon


def _tmlpu_v131_signed_gate_decay_relief():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v131_signed_gate_decay_relief_euler())


def _tmlpu_v132_signed_gate_decay_floor07_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.07
    return recon


def _tmlpu_v132_signed_gate_decay_floor07():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v132_signed_gate_decay_floor07_euler())


def _tmlpu_v133_signed_decay_floor10_capboost_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_signed_pair_tail_cap = 0.060
    return recon


def _tmlpu_v133_signed_decay_floor10_capboost():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v133_signed_decay_floor10_capboost_euler())


def _tmlpu_v134_signed_postrollback_preserve_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_signed_tail_postrollback_preserve_on = True
    recon.euler_tangential_signed_tail_postrollback_theta = 0.35
    return recon


def _tmlpu_v134_signed_postrollback_preserve():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v134_signed_postrollback_preserve_euler())


def _tmlpu_v135_signed_anchored_curve_assist_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = True
    recon.euler_tangential_tail_signed_anchored_curve_floor = 0.04
    return recon


def _tmlpu_v135_signed_anchored_curve_assist():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v135_signed_anchored_curve_assist_euler())


def _tmlpu_v136_signed_aligned_curve_assist_euler():
    recon = _tmlpu_v135_signed_anchored_curve_assist_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = True
    recon.euler_tangential_tail_signed_anchored_curve_floor = 0.04
    recon.euler_tangential_tail_signed_anchored_curve_align_on = True
    return recon


def _tmlpu_v136_signed_aligned_curve_assist():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v136_signed_aligned_curve_assist_euler())


def _tmlpu_v137_signed_anchor_curve_floor06_euler():
    recon = _tmlpu_v135_signed_anchored_curve_assist_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = True
    recon.euler_tangential_tail_signed_anchored_curve_floor = 0.06
    recon.euler_tangential_tail_signed_anchored_curve_align_on = False
    return recon


def _tmlpu_v137_signed_anchor_curve_floor06():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v137_signed_anchor_curve_floor06_euler())


def _tmlpu_v138_signed_anchor_curve_keep_signed_euler():
    recon = _tmlpu_v135_signed_anchored_curve_assist_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = True
    recon.euler_tangential_tail_signed_anchored_curve_floor = 0.04
    recon.euler_tangential_tail_signed_anchored_curve_align_on = False
    recon.euler_tangential_tail_signed_anchored_curve_preserve_signed_on = True
    return recon


def _tmlpu_v138_signed_anchor_curve_keep_signed():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v138_signed_anchor_curve_keep_signed_euler())


def _tmlpu_v139_signed_anchor_density_trace_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_tail_signed_anchored_curve_align_on = False
    recon.euler_tangential_tail_signed_anchored_curve_preserve_signed_on = False
    recon.euler_density_signed_tail_trace_on = True
    recon.euler_density_signed_tail_trace_beta = 0.15
    recon.euler_density_signed_tail_trace_cap = 0.004
    recon.euler_density_signed_tail_trace_wave_cap = 0.0015
    return recon


def _tmlpu_v139_signed_anchor_density_trace():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v139_signed_anchor_density_trace_euler())


def _tmlpu_v140_v131_signed_anchor_curve_gate_diag_euler():
    return _tmlpu_v135_signed_anchored_curve_assist_euler()


def _tmlpu_v140_v131_signed_anchor_curve_gate_diag():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v140_v131_signed_anchor_curve_gate_diag_euler())


def _tmlpu_v141_anchor_curve_diag_epsfix_euler():
    return _tmlpu_v140_v131_signed_anchor_curve_gate_diag_euler()


def _tmlpu_v141_anchor_curve_diag_epsfix():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v141_anchor_curve_diag_epsfix_euler())


def _tmlpu_v142_highsafe_raw_curve_microassist_euler():
    recon = _tmlpu_v141_anchor_curve_diag_epsfix_euler()
    recon.euler_tangential_highsafe_raw_curve_microassist_on = True
    recon.euler_tangential_highsafe_raw_curve_microassist_floor = 0.015
    recon.euler_tangential_highsafe_raw_curve_microassist_cap = 0.020
    recon.euler_tangential_highsafe_raw_curve_microassist_wave_cap = 0.0025
    recon.euler_tangential_highsafe_raw_curve_safe_min = 0.40
    recon.euler_tangential_highsafe_raw_curve_shear_min = 0.94
    recon.euler_tangential_highsafe_raw_curve_normality_max = 0.08
    recon.euler_tangential_highsafe_raw_curve_pressure_max = 0.008
    recon.euler_tangential_highsafe_raw_curve_compression_max = 0.0015
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    return recon


def _tmlpu_v142_highsafe_raw_curve_microassist():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v142_highsafe_raw_curve_microassist_euler())


def _tmlpu_v143_signed_sidecar_decay_euler():
    recon = _tmlpu_v127_signed_only_relief_floor04_euler()
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_signed_tail_safe_decay_relief_on = False
    recon.euler_tangential_signed_tail_sidecar_decay_on = True
    recon.euler_tangential_signed_tail_sidecar_safe_floor = 0.10
    recon.euler_tangential_signed_tail_sidecar_blend = 0.35
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_tangential_pair_extend_beta = 0.0
    recon.euler_tangential_pair_extend_cap = 0.0
    recon.euler_tangential_pair_extend_wave_cap = 0.0
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    return recon


def _tmlpu_v143_signed_sidecar_decay():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v143_signed_sidecar_decay_euler())


def _tmlpu_v144_signed_sidecar_decay_blend015_euler():
    recon = _tmlpu_v143_signed_sidecar_decay_euler()
    recon.euler_tangential_signed_tail_sidecar_decay_on = True
    recon.euler_tangential_signed_tail_sidecar_safe_floor = 0.10
    recon.euler_tangential_signed_tail_sidecar_blend = 0.15
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_tangential_pair_extend_beta = 0.0
    recon.euler_tangential_pair_extend_cap = 0.0
    recon.euler_tangential_pair_extend_wave_cap = 0.0
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    return recon


def _tmlpu_v144_signed_sidecar_decay_blend015():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v144_signed_sidecar_decay_blend015_euler())


def _tmlpu_v145_signed_decay_floor12_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.12
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_signed_tail_sidecar_decay_on = False
    return recon


def _tmlpu_v145_signed_decay_floor12():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v145_signed_decay_floor12_euler())


def _tmlpu_v146_signed_gate_shadow_diag_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_signed_tail_sidecar_decay_on = False
    return recon


def _tmlpu_v146_signed_gate_shadow_diag():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v146_signed_gate_shadow_diag_euler())


def _tmlpu_v147_signed_beta044_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_pair_tail_beta = 0.044
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_signed_tail_sidecar_decay_on = False
    return recon


def _tmlpu_v147_signed_beta044():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v147_signed_beta044_euler())


def _tmlpu_v148_signed_beta038_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_pair_tail_beta = 0.038
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_tail_shear_contact_curve_floor = -1.0
    recon.euler_tangential_signed_tail_density_support_min = -1.0
    recon.euler_tangential_signed_tail_density_support_full = -1.0
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_signed_tail_sidecar_decay_on = False
    return recon


def _tmlpu_v148_signed_beta038():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v148_signed_beta038_euler())


def _tmlpu_v149_v135_downstream_density_micro_euler():
    recon = _tmlpu_v135_signed_anchored_curve_assist_euler()
    recon.euler_density_contact_weak_face_stream_coherence_on = True
    recon.euler_density_contact_weak_face_stream_coherence_min = 0.20
    recon.euler_density_contact_weak_face_stream_coherence_full = 0.60
    recon.euler_density_contact_weak_face_downstream_rho_beta = 0.018
    recon.euler_density_contact_weak_face_downstream_rho_cap = 0.003
    recon.euler_density_contact_weak_face_downstream_rho_wave_cap = 0.0015
    recon.euler_density_contact_weak_face_downstream_tangential_beta = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_cap = 0.0
    recon.euler_density_contact_weak_face_downstream_tangential_wave_cap = 0.0
    return recon


def _tmlpu_v149_v135_downstream_density_micro():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v149_v135_downstream_density_micro_euler())


def _tmlpu_v150_v135_pair_extend_micro_euler():
    recon = _tmlpu_v135_signed_anchored_curve_assist_euler()
    recon.euler_tangential_pair_extend_on = True
    recon.euler_tangential_pair_extend_beta = 0.015
    recon.euler_tangential_pair_extend_cap = 0.015
    recon.euler_tangential_pair_extend_wave_cap = 0.002
    recon.euler_tangential_pair_extend_alignment_min = 0.20
    recon.euler_tangential_pair_extend_alignment_full = 0.60
    recon.euler_tangential_pair_extend_shock_exclude = True
    return recon


def _tmlpu_v150_v135_pair_extend_micro():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v150_v135_pair_extend_micro_euler())


def _tmlpu_v153_v131_pair_extend_micro_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_pair_extend_on = True
    recon.euler_tangential_pair_extend_beta = 0.012
    recon.euler_tangential_pair_extend_cap = 0.010
    recon.euler_tangential_pair_extend_wave_cap = 0.0015
    recon.euler_tangential_pair_extend_alignment_min = 0.25
    recon.euler_tangential_pair_extend_alignment_full = 0.65
    recon.euler_tangential_pair_extend_shock_exclude = True
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    return recon


def _tmlpu_v153_v131_pair_extend_micro():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v153_v131_pair_extend_micro_euler())


def _tmlpu_v155_v131_reduced_signed_tail_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_pair_tail_beta = 0.032
    recon.euler_tangential_signed_pair_tail_cap = 0.026
    recon.euler_tangential_signed_pair_tail_wave_cap = 0.0032
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.04
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_tangential_pair_extend_beta = 0.0
    recon.euler_tangential_pair_extend_cap = 0.0
    recon.euler_tangential_pair_extend_wave_cap = 0.0
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_signed_tail_sidecar_decay_on = False
    return recon


def _tmlpu_v155_v131_reduced_signed_tail():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v155_v131_reduced_signed_tail_euler())


def _tmlpu_v157_v131_antisheet_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_antisheet_on = True
    recon.euler_tangential_signed_tail_antisheet_strength = 0.45
    recon.euler_tangential_signed_tail_antisheet_min_factor = 0.55
    recon.euler_tangential_signed_tail_antisheet_q_hi = 0.070
    recon.euler_tangential_signed_tail_antisheet_contact_min = 0.25
    recon.euler_tangential_signed_tail_antisheet_contact_full = 0.60
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    return recon


def _tmlpu_v157_v131_antisheet():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v157_v131_antisheet_euler())


def _tmlpu_v158_v131_strong_antisheet_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_antisheet_on = True
    recon.euler_tangential_signed_tail_antisheet_strength = 0.90
    recon.euler_tangential_signed_tail_antisheet_min_factor = 0.20
    recon.euler_tangential_signed_tail_antisheet_q_hi = 0.18
    recon.euler_tangential_signed_tail_antisheet_contact_min = 0.10
    recon.euler_tangential_signed_tail_antisheet_contact_full = 0.45
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    return recon


def _tmlpu_v158_v131_strong_antisheet():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v158_v131_strong_antisheet_euler())


def _tmlpu_v159_v131_qcore_gate_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_pair_tail_q_min = 0.050
    recon.euler_tangential_signed_pair_tail_q_full = 0.140
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_signed_tail_antisheet_on = False
    return recon


def _tmlpu_v159_v131_qcore_gate():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v159_v131_qcore_gate_euler())


def _tmlpu_v160_v131_signed_tail_hffilter_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_signed_tail_antisheet_on = False
    recon.euler_tangential_signed_tail_hf_filter_on = True
    recon.euler_tangential_signed_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V160_SIGNED_TAIL_HF_FILTER_STRENGTH', '0.35'))
    recon.euler_tangential_signed_tail_hf_filter_min_weight = float(
        os.environ.get('TMLPU_V160_SIGNED_TAIL_HF_FILTER_MIN_WEIGHT',
                       '1e-10'))
    recon.euler_tangential_signed_tail_hf_filter_shock_exclude = _env_bool(
        'TMLPU_V160_SIGNED_TAIL_HF_FILTER_SHOCK_EXCLUDE', True)
    return recon


def _tmlpu_v160_v131_signed_tail_hffilter():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v160_v131_signed_tail_hffilter_euler())


def _tmlpu_v161_v131_bridge_cut_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_signed_tail_antisheet_on = False
    recon.euler_tangential_signed_tail_bridge_cut_on = True
    recon.euler_tangential_signed_tail_bridge_cut_strength = float(
        os.environ.get('TMLPU_V161_BRIDGE_CUT_STRENGTH', '0.55'))
    recon.euler_tangential_signed_tail_bridge_cut_min_factor = float(
        os.environ.get('TMLPU_V161_BRIDGE_CUT_MIN_FACTOR', '0.25'))
    recon.euler_tangential_signed_tail_bridge_cut_q_min = float(
        os.environ.get('TMLPU_V161_BRIDGE_CUT_Q_MIN', '0.08'))
    recon.euler_tangential_signed_tail_bridge_cut_q_full = float(
        os.environ.get('TMLPU_V161_BRIDGE_CUT_Q_FULL', '0.22'))
    recon.euler_tangential_signed_tail_bridge_cut_omega_lo_pct = float(
        os.environ.get('TMLPU_V161_BRIDGE_CUT_OMEGA_LO_PCT', '70.0'))
    recon.euler_tangential_signed_tail_bridge_cut_omega_hi_pct = float(
        os.environ.get('TMLPU_V161_BRIDGE_CUT_OMEGA_HI_PCT', '92.0'))
    return recon


def _tmlpu_v161_v131_bridge_cut():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v161_v131_bridge_cut_euler())


def _tmlpu_v162_v131_conservative_bridge_cut_euler():
    recon = _tmlpu_v161_v131_bridge_cut_euler()
    recon.euler_tangential_signed_tail_bridge_cut_strength = float(
        os.environ.get('TMLPU_V162_BRIDGE_CUT_STRENGTH', '0.25'))
    recon.euler_tangential_signed_tail_bridge_cut_min_factor = float(
        os.environ.get('TMLPU_V162_BRIDGE_CUT_MIN_FACTOR', '0.55'))
    recon.euler_tangential_signed_tail_bridge_cut_q_min = float(
        os.environ.get('TMLPU_V162_BRIDGE_CUT_Q_MIN', '0.12'))
    recon.euler_tangential_signed_tail_bridge_cut_q_full = float(
        os.environ.get('TMLPU_V162_BRIDGE_CUT_Q_FULL', '0.28'))
    recon.euler_tangential_signed_tail_bridge_cut_omega_lo_pct = float(
        os.environ.get('TMLPU_V162_BRIDGE_CUT_OMEGA_LO_PCT', '75.0'))
    recon.euler_tangential_signed_tail_bridge_cut_omega_hi_pct = float(
        os.environ.get('TMLPU_V162_BRIDGE_CUT_OMEGA_HI_PCT', '96.0'))
    return recon


def _tmlpu_v162_v131_conservative_bridge_cut():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v162_v131_conservative_bridge_cut_euler())


def _tmlpu_v163_v131_shock_ridge_guard_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_tail_safe_decay_relief_on = True
    recon.euler_tangential_signed_tail_safe_floor = 0.10
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_density_signed_tail_trace_on = False
    recon.euler_tangential_signed_tail_postrollback_preserve_on = False
    recon.euler_tangential_signed_tail_antisheet_on = False
    recon.euler_tangential_signed_tail_bridge_cut_on = False
    recon.euler_tangential_signed_tail_hf_filter_on = False
    recon.euler_tangential_signed_tail_shock_ridge_clean_on = True
    recon.euler_tangential_signed_tail_shock_ridge_strength = float(
        os.environ.get('TMLPU_V163_SHOCK_RIDGE_STRENGTH', '0.50'))
    recon.euler_tangential_signed_tail_shock_ridge_min_factor = float(
        os.environ.get('TMLPU_V163_SHOCK_RIDGE_MIN_FACTOR', '0.60'))
    recon.euler_tangential_signed_tail_shock_ridge_density_min = float(
        os.environ.get('TMLPU_V163_SHOCK_RIDGE_DENSITY_MIN', '0.35'))
    recon.euler_tangential_signed_tail_shock_ridge_density_full = float(
        os.environ.get('TMLPU_V163_SHOCK_RIDGE_DENSITY_FULL', '0.85'))
    recon.euler_tangential_signed_tail_shock_ridge_q_keep_min = float(
        os.environ.get('TMLPU_V163_SHOCK_RIDGE_Q_KEEP_MIN', '0.10'))
    recon.euler_tangential_signed_tail_shock_ridge_q_keep_full = float(
        os.environ.get('TMLPU_V163_SHOCK_RIDGE_Q_KEEP_FULL', '0.25'))
    return recon


def _tmlpu_v163_v131_shock_ridge_guard():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v163_v131_shock_ridge_guard_euler())


def _tmlpu_v164_v131_density_support_damp_euler():
    recon = _tmlpu_v163_v131_shock_ridge_guard_euler()
    recon.euler_tangential_signed_tail_shock_ridge_strength = float(
        os.environ.get('TMLPU_V164_DENSITY_SUPPORT_DAMP_STRENGTH', '0.65'))
    recon.euler_tangential_signed_tail_shock_ridge_min_factor = float(
        os.environ.get('TMLPU_V164_DENSITY_SUPPORT_DAMP_MIN_FACTOR', '0.55'))
    recon.euler_tangential_signed_tail_shock_ridge_density_min = float(
        os.environ.get('TMLPU_V164_DENSITY_SUPPORT_DAMP_MIN', '0.20'))
    recon.euler_tangential_signed_tail_shock_ridge_density_full = float(
        os.environ.get('TMLPU_V164_DENSITY_SUPPORT_DAMP_FULL', '0.70'))
    # Set the keep window above attainable qratio values to damp strong
    # density-support ridges directly rather than only low-q sheets.
    recon.euler_tangential_signed_tail_shock_ridge_q_keep_min = float(
        os.environ.get('TMLPU_V164_DENSITY_SUPPORT_DAMP_Q_KEEP_MIN', '2.0'))
    recon.euler_tangential_signed_tail_shock_ridge_q_keep_full = float(
        os.environ.get('TMLPU_V164_DENSITY_SUPPORT_DAMP_Q_KEEP_FULL', '3.0'))
    return recon


def _tmlpu_v164_v131_density_support_damp():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v164_v131_density_support_damp_euler())


def _tmlpu_v165_v131_signed_tail_ablation_euler():
    recon = _tmlpu_v131_signed_gate_decay_relief_euler()
    recon.euler_tangential_signed_pair_tail_on = False
    recon.euler_tangential_signed_pair_tail_beta = 0.0
    recon.euler_tangential_signed_pair_tail_cap = 0.0
    recon.euler_tangential_signed_pair_tail_wave_cap = 0.0
    recon.euler_tangential_density_curve_pair_tail_on = False
    recon.euler_tangential_density_curve_pair_tail_beta = 0.0
    recon.euler_tangential_density_curve_pair_tail_cap = 0.0
    recon.euler_tangential_density_curve_pair_tail_wave_cap = 0.0
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = False
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = False
    recon.euler_tangential_signed_tail_safe_decay_relief_on = False
    recon.euler_tangential_signed_tail_bridge_cut_on = False
    recon.euler_tangential_signed_tail_hf_filter_on = False
    recon.euler_tangential_signed_tail_shock_ridge_clean_on = False
    recon.euler_tangential_tail_signed_anchored_curve_assist_on = False
    recon.euler_tangential_highsafe_raw_curve_microassist_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_density_signed_tail_trace_on = False
    return recon


def _tmlpu_v165_v131_signed_tail_ablation():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v165_v131_signed_tail_ablation_euler())


def _tmlpu_v166_v165_micro_signed_restore_euler():
    recon = _tmlpu_v165_v131_signed_tail_ablation_euler()
    recon.euler_tangential_signed_pair_tail_on = True
    recon.euler_tangential_signed_pair_tail_beta = float(
        os.environ.get('TMLPU_V166_SIGNED_RESTORE_BETA', '0.016'))
    recon.euler_tangential_signed_pair_tail_cap = float(
        os.environ.get('TMLPU_V166_SIGNED_RESTORE_CAP', '0.013'))
    recon.euler_tangential_signed_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V166_SIGNED_RESTORE_WAVE_CAP', '0.0016'))
    recon.euler_tangential_tail_shear_contact_relief_apply_signed = True
    recon.euler_tangential_tail_shear_contact_signed_floor = 0.02
    recon.euler_tangential_signed_tail_safe_decay_relief_on = False
    return recon


def _tmlpu_v166_v165_micro_signed_restore():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v166_v165_micro_signed_restore_euler())


def _tmlpu_v167_v165_curve_restore_euler():
    recon = _tmlpu_v166_v165_micro_signed_restore_euler()
    recon.euler_tangential_signed_pair_tail_beta = float(
        os.environ.get('TMLPU_V167_SIGNED_RESTORE_BETA', '0.020'))
    recon.euler_tangential_signed_pair_tail_cap = float(
        os.environ.get('TMLPU_V167_SIGNED_RESTORE_CAP', '0.016'))
    recon.euler_tangential_signed_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V167_SIGNED_RESTORE_WAVE_CAP', '0.0020'))
    recon.euler_tangential_density_curve_pair_tail_on = True
    recon.euler_tangential_density_curve_pair_tail_beta = float(
        os.environ.get('TMLPU_V167_CURVE_RESTORE_BETA', '0.012'))
    recon.euler_tangential_density_curve_pair_tail_cap = float(
        os.environ.get('TMLPU_V167_CURVE_RESTORE_CAP', '0.010'))
    recon.euler_tangential_density_curve_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V167_CURVE_RESTORE_WAVE_CAP', '0.0012'))
    recon.euler_tangential_tail_shear_contact_relief_apply_curve = True
    recon.euler_tangential_tail_shear_contact_curve_floor = 0.015
    recon.euler_tangential_signed_tail_safe_decay_relief_on = False
    recon.euler_tangential_pair_extend_on = False
    recon.euler_density_signed_tail_trace_on = False
    return recon


def _tmlpu_v167_v165_curve_restore():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v167_v165_curve_restore_euler())


def _tmlpu_v168_v167_curve_hffilter_euler():
    recon = _tmlpu_v167_v165_curve_restore_euler()
    recon.euler_tangential_signed_tail_hf_filter_on = True
    recon.euler_tangential_signed_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V168_SIGNED_HF_FILTER_STRENGTH', '0.20'))
    recon.euler_tangential_signed_tail_hf_filter_min_weight = float(
        os.environ.get('TMLPU_V168_SIGNED_HF_FILTER_MIN_WEIGHT', '1e-10'))
    recon.euler_tangential_signed_tail_hf_filter_shock_exclude = _env_bool(
        'TMLPU_V168_SIGNED_HF_FILTER_SHOCK_EXCLUDE', True)
    recon.euler_tangential_density_curve_tail_hf_filter_on = True
    recon.euler_tangential_density_curve_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V168_CURVE_HF_FILTER_STRENGTH', '0.35'))
    recon.euler_tangential_density_curve_tail_hf_filter_min_weight = float(
        os.environ.get('TMLPU_V168_CURVE_HF_FILTER_MIN_WEIGHT', '1e-10'))
    recon.euler_tangential_density_curve_tail_hf_filter_shock_exclude = (
        _env_bool('TMLPU_V168_CURVE_HF_FILTER_SHOCK_EXCLUDE', True))
    return recon


def _tmlpu_v168_v167_curve_hffilter():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v168_v167_curve_hffilter_euler())


def _tmlpu_v206_v168_fast_weakoff_euler():
    recon = _tmlpu_v168_v167_curve_hffilter_euler()
    recon.euler_density_contact_weak_face_mlp = False
    recon.euler_density_contact_weak_face_value_scaling = True
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        'coherent_shear_micro_restore')
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_head_generic = False
    recon.euler_density_contact_weak_face_disable_specialized_relax = False
    recon.euler_density_contact_weak_face_legacy_order = False
    return recon


def _tmlpu_v206_v168_fast_weakoff():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v206_v168_fast_weakoff_euler())


def _tmlpu_v207_v206_fast_shockridge_euler():
    recon = _tmlpu_v206_v168_fast_weakoff_euler()
    recon.euler_tangential_tail_density_shock_damp_on = _env_bool(
        'TMLPU_V207_TAIL_SHOCK_DAMP_ON', True)
    recon.euler_tangential_tail_density_shock_damp_theta = float(
        os.environ.get('TMLPU_V207_TAIL_SHOCK_DAMP_THETA', '0.42'))
    recon.euler_tangential_tail_density_shock_damp_pressure_min = float(
        os.environ.get('TMLPU_V207_TAIL_SHOCK_PRESSURE_MIN', '0.010'))
    recon.euler_tangential_tail_density_shock_damp_compression_min = float(
        os.environ.get('TMLPU_V207_TAIL_SHOCK_COMPRESSION_MIN', '0.002'))
    recon.euler_tangential_tail_density_shock_damp_normality_min = float(
        os.environ.get('TMLPU_V207_TAIL_SHOCK_NORMALITY_MIN', '0.14'))

    recon.euler_tangential_signed_tail_shock_ridge_clean_on = _env_bool(
        'TMLPU_V207_SIGNED_SHOCK_RIDGE_CLEAN_ON', True)
    recon.euler_tangential_signed_tail_shock_ridge_strength = float(
        os.environ.get('TMLPU_V207_SIGNED_SHOCK_RIDGE_STRENGTH', '0.26'))
    recon.euler_tangential_signed_tail_shock_ridge_min_factor = float(
        os.environ.get('TMLPU_V207_SIGNED_SHOCK_RIDGE_MIN_FACTOR', '0.74'))
    recon.euler_tangential_signed_tail_shock_ridge_density_min = float(
        os.environ.get('TMLPU_V207_SIGNED_SHOCK_RIDGE_DENSITY_MIN', '0.22'))
    recon.euler_tangential_signed_tail_shock_ridge_density_full = float(
        os.environ.get('TMLPU_V207_SIGNED_SHOCK_RIDGE_DENSITY_FULL', '0.64'))
    recon.euler_tangential_signed_tail_shock_ridge_q_keep_min = float(
        os.environ.get('TMLPU_V207_SIGNED_SHOCK_RIDGE_Q_KEEP_MIN', '0.055'))
    recon.euler_tangential_signed_tail_shock_ridge_q_keep_full = float(
        os.environ.get('TMLPU_V207_SIGNED_SHOCK_RIDGE_Q_KEEP_FULL', '0.18'))

    recon.euler_tangential_density_curve_tail_shock_ridge_clean_on = _env_bool(
        'TMLPU_V207_CURVE_SHOCK_RIDGE_CLEAN_ON', True)
    recon.euler_tangential_density_curve_tail_shock_ridge_strength = float(
        os.environ.get('TMLPU_V207_CURVE_SHOCK_RIDGE_STRENGTH', '0.34'))
    recon.euler_tangential_density_curve_tail_shock_ridge_min_factor = float(
        os.environ.get('TMLPU_V207_CURVE_SHOCK_RIDGE_MIN_FACTOR', '0.68'))
    recon.euler_tangential_density_curve_tail_shock_ridge_density_min = float(
        os.environ.get('TMLPU_V207_CURVE_SHOCK_RIDGE_DENSITY_MIN', '0.20'))
    recon.euler_tangential_density_curve_tail_shock_ridge_density_full = float(
        os.environ.get('TMLPU_V207_CURVE_SHOCK_RIDGE_DENSITY_FULL', '0.60'))
    recon.euler_tangential_density_curve_tail_shock_ridge_q_keep_min = float(
        os.environ.get('TMLPU_V207_CURVE_SHOCK_RIDGE_Q_KEEP_MIN', '0.05'))
    recon.euler_tangential_density_curve_tail_shock_ridge_q_keep_full = float(
        os.environ.get('TMLPU_V207_CURVE_SHOCK_RIDGE_Q_KEEP_FULL', '0.17'))
    return recon


def _tmlpu_v207_v206_fast_shockridge():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v207_v206_fast_shockridge_euler())


def _tmlpu_v208_v206_fast_curve_guard_euler():
    recon = _tmlpu_v206_v168_fast_weakoff_euler()
    recon.euler_tangential_density_curve_tail_shock_ridge_clean_on = _env_bool(
        'TMLPU_V208_CURVE_SHOCK_RIDGE_CLEAN_ON', True)
    recon.euler_tangential_density_curve_tail_shock_ridge_strength = float(
        os.environ.get('TMLPU_V208_CURVE_SHOCK_RIDGE_STRENGTH', '0.16'))
    recon.euler_tangential_density_curve_tail_shock_ridge_min_factor = float(
        os.environ.get('TMLPU_V208_CURVE_SHOCK_RIDGE_MIN_FACTOR', '0.84'))
    recon.euler_tangential_density_curve_tail_shock_ridge_density_min = float(
        os.environ.get('TMLPU_V208_CURVE_SHOCK_RIDGE_DENSITY_MIN', '0.24'))
    recon.euler_tangential_density_curve_tail_shock_ridge_density_full = float(
        os.environ.get('TMLPU_V208_CURVE_SHOCK_RIDGE_DENSITY_FULL', '0.68'))
    recon.euler_tangential_density_curve_tail_shock_ridge_q_keep_min = float(
        os.environ.get('TMLPU_V208_CURVE_SHOCK_RIDGE_Q_KEEP_MIN', '0.07'))
    recon.euler_tangential_density_curve_tail_shock_ridge_q_keep_full = float(
        os.environ.get('TMLPU_V208_CURVE_SHOCK_RIDGE_Q_KEEP_FULL', '0.22'))
    return recon


def _tmlpu_v208_v206_fast_curve_guard():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v208_v206_fast_curve_guard_euler())


def _tmlpu_v209_v206_fast_pressure_jump_euler():
    recon = _tmlpu_v206_v168_fast_weakoff_euler()
    recon.euler_pressure_face_jump_limiter_on = _env_bool(
        'TMLPU_V209_PRESSURE_JUMP_LIMIT_ON', True)
    recon.euler_pressure_face_jump_limiter_strength = float(
        os.environ.get('TMLPU_V209_PRESSURE_JUMP_LIMIT_STRENGTH', '0.62'))
    recon.euler_pressure_face_jump_limiter_growth_cap = float(
        os.environ.get('TMLPU_V209_PRESSURE_JUMP_LIMIT_GROWTH_CAP', '0.06'))
    recon.euler_pressure_face_jump_limiter_abs_floor = float(
        os.environ.get('TMLPU_V209_PRESSURE_JUMP_LIMIT_ABS_FLOOR', '1e-10'))
    recon.euler_pressure_face_jump_limiter_p_jump_threshold = float(
        os.environ.get('TMLPU_V209_PRESSURE_JUMP_LIMIT_P_THRESHOLD', '0.020'))
    recon.euler_pressure_face_jump_limiter_p_jump_width = float(
        os.environ.get('TMLPU_V209_PRESSURE_JUMP_LIMIT_P_WIDTH', '0.060'))
    recon.euler_pressure_face_jump_limiter_compression_threshold = float(
        os.environ.get('TMLPU_V209_PRESSURE_JUMP_LIMIT_C_THRESHOLD', '0.006'))
    recon.euler_pressure_face_jump_limiter_compression_width = float(
        os.environ.get('TMLPU_V209_PRESSURE_JUMP_LIMIT_C_WIDTH', '0.055'))
    recon.euler_pressure_face_jump_limiter_normality_threshold = float(
        os.environ.get('TMLPU_V209_PRESSURE_JUMP_LIMIT_N_THRESHOLD', '0.36'))
    recon.euler_pressure_face_jump_limiter_normality_width = float(
        os.environ.get('TMLPU_V209_PRESSURE_JUMP_LIMIT_N_WIDTH', '0.30'))
    return recon


def _tmlpu_v209_v206_fast_pressure_jump():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v209_v206_fast_pressure_jump_euler())


def _tmlpu_v210_v168_fast_weakmlp_no_valuescale_euler():
    recon = _tmlpu_v168_v167_curve_hffilter_euler()
    recon.euler_density_contact_weak_face_mlp = True
    recon.euler_density_contact_weak_face_value_scaling = False
    recon.euler_density_contact_weak_face_stream_coherence_on = False
    recon.euler_density_contact_weak_face_head_generic = False
    recon.euler_density_contact_weak_face_disable_specialized_relax = False
    recon.euler_density_contact_weak_face_legacy_order = False
    recon.euler_density_contact_weak_face_admissibility_damp = False
    recon.euler_density_contact_weak_face_entropy_accept = False
    recon.euler_density_contact_weak_face_shock_gate = False
    recon.euler_density_contact_weak_face_root_blend = 0.0
    recon.euler_density_contact_weak_face_swirl_extra = 0.0
    return recon


def _tmlpu_v210_v168_fast_weakmlp_no_valuescale():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v210_v168_fast_weakmlp_no_valuescale_euler())


def _tmlpu_v211_v210_fast_weakmlp_valuescale_euler():
    recon = _tmlpu_v210_v168_fast_weakmlp_no_valuescale_euler()
    recon.euler_density_contact_weak_face_value_scaling = True
    recon.euler_density_contact_weak_face_value_scaling_mode = (
        os.environ.get(
            'TMLPU_V211_WEAK_FACE_VALUE_SCALING_MODE',
            'coherent_shear_micro_restore'))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha = (
        float(os.environ.get('TMLPU_V211_WEAK_FACE_VALUE_ALPHA', '1.0')))
    recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad = (
        float(os.environ.get('TMLPU_V211_WEAK_FACE_VALUE_BOUND_PAD', '0.0')))
    recon.euler_density_contact_weak_face_value_scaling_require_coherent_shear = (
        _env_bool('TMLPU_V211_WEAK_FACE_VALUE_REQUIRE_COHERENT', True))
    recon.euler_density_contact_weak_face_value_scaling_artifact_reject = (
        _env_bool('TMLPU_V211_WEAK_FACE_VALUE_ARTIFACT_REJECT', True))
    return recon


def _tmlpu_v211_v210_fast_weakmlp_valuescale():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v211_v210_fast_weakmlp_valuescale_euler())


def _tmlpu_v212_v210_fast_wallshock_flatten_euler():
    recon = _tmlpu_v210_v168_fast_weakmlp_no_valuescale_euler()
    recon.euler_wall_tangential_flatten = _env_bool(
        'TMLPU_V212_WALL_TANGENTIAL_FLATTEN', True)
    recon.euler_wall_tangential_flatten_mode = os.environ.get(
        'TMLPU_V212_WALL_TANGENTIAL_FLATTEN_MODE', 'shock')
    recon.euler_wall_tangential_flatten_strength = float(
        os.environ.get('TMLPU_V212_WALL_TANGENTIAL_FLATTEN_STRENGTH', '0.55'))
    return recon


def _tmlpu_v212_v210_fast_wallshock_flatten():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v212_v210_fast_wallshock_flatten_euler())


def _tmlpu_v213_v212_wallshock_flatten035_euler():
    recon = _tmlpu_v212_v210_fast_wallshock_flatten_euler()
    recon.euler_wall_tangential_flatten_strength = float(
        os.environ.get('TMLPU_V213_WALL_TANGENTIAL_FLATTEN_STRENGTH', '0.35'))
    return recon


def _tmlpu_v213_v212_wallshock_flatten035():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v213_v212_wallshock_flatten035_euler())


def _tmlpu_v214_v212_wallpressure_flatten_euler():
    recon = _tmlpu_v212_v210_fast_wallshock_flatten_euler()
    recon.euler_wall_tangential_flatten_mode = os.environ.get(
        'TMLPU_V214_WALL_TANGENTIAL_FLATTEN_MODE', 'pressure')
    recon.euler_wall_tangential_flatten_strength = float(
        os.environ.get('TMLPU_V214_WALL_TANGENTIAL_FLATTEN_STRENGTH', '0.55'))
    return recon


def _tmlpu_v214_v212_wallpressure_flatten():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v214_v212_wallpressure_flatten_euler())


def _tmlpu_v216_v210_fast_shockline_rollback_euler():
    recon = _tmlpu_v210_v168_fast_weakmlp_no_valuescale_euler()
    recon.euler_tangential_shockline_rollback_on = _env_bool(
        'TMLPU_V216_SHOCKLINE_ROLLBACK_ON', True)
    recon.euler_tangential_shockline_rollback_theta = float(
        os.environ.get('TMLPU_V216_SHOCKLINE_ROLLBACK_THETA', '0.22'))
    recon.euler_tangential_shockline_pressure_min = float(
        os.environ.get('TMLPU_V216_SHOCKLINE_PRESSURE_MIN', '0.018'))
    recon.euler_tangential_shockline_compression_min = float(
        os.environ.get('TMLPU_V216_SHOCKLINE_COMPRESSION_MIN', '0.0040'))
    recon.euler_tangential_shockline_normality_min = float(
        os.environ.get('TMLPU_V216_SHOCKLINE_NORMALITY_MIN', '0.30'))
    recon.euler_tangential_shockline_shear_max = float(
        os.environ.get('TMLPU_V216_SHOCKLINE_SHEAR_MAX', '0.78'))
    return recon


def _tmlpu_v216_v210_fast_shockline_rollback():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v216_v210_fast_shockline_rollback_euler())


def _tmlpu_v217_v212_wallshock_tailboost_euler():
    recon = _tmlpu_v212_v210_fast_wallshock_flatten_euler()
    recon.euler_tangential_signed_pair_tail_beta = float(
        os.environ.get('TMLPU_V217_SIGNED_RESTORE_BETA', '0.026'))
    recon.euler_tangential_signed_pair_tail_cap = float(
        os.environ.get('TMLPU_V217_SIGNED_RESTORE_CAP', '0.020'))
    recon.euler_tangential_signed_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V217_SIGNED_RESTORE_WAVE_CAP', '0.0024'))
    recon.euler_tangential_density_curve_pair_tail_beta = float(
        os.environ.get('TMLPU_V217_CURVE_RESTORE_BETA', '0.017'))
    recon.euler_tangential_density_curve_pair_tail_cap = float(
        os.environ.get('TMLPU_V217_CURVE_RESTORE_CAP', '0.014'))
    recon.euler_tangential_density_curve_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V217_CURVE_RESTORE_WAVE_CAP', '0.0017'))
    return recon


def _tmlpu_v217_v212_wallshock_tailboost():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v217_v212_wallshock_tailboost_euler())


def _tmlpu_v174_v168_roi_strength_euler():
    recon = _tmlpu_v168_v167_curve_hffilter_euler()
    recon.euler_tangential_signed_pair_tail_beta = float(
        os.environ.get('TMLPU_V174_SIGNED_RESTORE_BETA', '0.030'))
    recon.euler_tangential_signed_pair_tail_cap = float(
        os.environ.get('TMLPU_V174_SIGNED_RESTORE_CAP', '0.024'))
    recon.euler_tangential_signed_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V174_SIGNED_RESTORE_WAVE_CAP', '0.0030'))
    recon.euler_tangential_density_curve_pair_tail_beta = float(
        os.environ.get('TMLPU_V174_CURVE_RESTORE_BETA', '0.020'))
    recon.euler_tangential_density_curve_pair_tail_cap = float(
        os.environ.get('TMLPU_V174_CURVE_RESTORE_CAP', '0.016'))
    recon.euler_tangential_density_curve_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V174_CURVE_RESTORE_WAVE_CAP', '0.0020'))
    recon.euler_tangential_signed_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V174_SIGNED_HF_FILTER_STRENGTH', '0.16'))
    recon.euler_tangential_density_curve_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V174_CURVE_HF_FILTER_STRENGTH', '0.26'))
    return recon


def _tmlpu_v174_v168_roi_strength():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v174_v168_roi_strength_euler())


def _tmlpu_v175_v174_stronger_filtered_roi_euler():
    recon = _tmlpu_v168_v167_curve_hffilter_euler()
    recon.euler_tangential_signed_pair_tail_beta = float(
        os.environ.get('TMLPU_V175_SIGNED_RESTORE_BETA', '0.045'))
    recon.euler_tangential_signed_pair_tail_cap = float(
        os.environ.get('TMLPU_V175_SIGNED_RESTORE_CAP', '0.034'))
    recon.euler_tangential_signed_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V175_SIGNED_RESTORE_WAVE_CAP', '0.0045'))
    recon.euler_tangential_density_curve_pair_tail_beta = float(
        os.environ.get('TMLPU_V175_CURVE_RESTORE_BETA', '0.030'))
    recon.euler_tangential_density_curve_pair_tail_cap = float(
        os.environ.get('TMLPU_V175_CURVE_RESTORE_CAP', '0.024'))
    recon.euler_tangential_density_curve_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V175_CURVE_RESTORE_WAVE_CAP', '0.0030'))
    recon.euler_tangential_signed_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V175_SIGNED_HF_FILTER_STRENGTH', '0.24'))
    recon.euler_tangential_density_curve_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V175_CURVE_HF_FILTER_STRENGTH', '0.42'))
    return recon


def _tmlpu_v175_v174_stronger_filtered_roi():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v175_v174_stronger_filtered_roi_euler())


def _tmlpu_v176_v174_pair_extend_roi_euler():
    recon = _tmlpu_v174_v168_roi_strength_euler()
    recon.euler_tangential_pair_extend_on = True
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V176_PAIR_EXTEND_BETA', '0.010'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V176_PAIR_EXTEND_CAP', '0.008'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V176_PAIR_EXTEND_WAVE_CAP', '0.0012'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V176_PAIR_EXTEND_ALIGN_MIN', '0.30'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V176_PAIR_EXTEND_ALIGN_FULL', '0.70'))
    recon.euler_tangential_pair_extend_shock_exclude = True
    return recon


def _tmlpu_v176_v174_pair_extend_roi():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v176_v174_pair_extend_roi_euler())


def _tmlpu_v177_v174_mid_strength_roi_euler():
    recon = _tmlpu_v168_v167_curve_hffilter_euler()
    recon.euler_tangential_signed_pair_tail_beta = float(
        os.environ.get('TMLPU_V177_SIGNED_RESTORE_BETA', '0.036'))
    recon.euler_tangential_signed_pair_tail_cap = float(
        os.environ.get('TMLPU_V177_SIGNED_RESTORE_CAP', '0.028'))
    recon.euler_tangential_signed_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V177_SIGNED_RESTORE_WAVE_CAP', '0.0036'))
    recon.euler_tangential_density_curve_pair_tail_beta = float(
        os.environ.get('TMLPU_V177_CURVE_RESTORE_BETA', '0.024'))
    recon.euler_tangential_density_curve_pair_tail_cap = float(
        os.environ.get('TMLPU_V177_CURVE_RESTORE_CAP', '0.019'))
    recon.euler_tangential_density_curve_pair_tail_wave_cap = float(
        os.environ.get('TMLPU_V177_CURVE_RESTORE_WAVE_CAP', '0.0024'))
    recon.euler_tangential_signed_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V177_SIGNED_HF_FILTER_STRENGTH', '0.18'))
    recon.euler_tangential_density_curve_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V177_CURVE_HF_FILTER_STRENGTH', '0.30'))
    return recon


def _tmlpu_v177_v174_mid_strength_roi():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v177_v174_mid_strength_roi_euler())


def _tmlpu_v178_v174_dual_bridge_cut_euler():
    recon = _tmlpu_v174_v168_roi_strength_euler()
    recon.euler_tangential_signed_tail_bridge_cut_on = True
    recon.euler_tangential_signed_tail_bridge_cut_strength = float(
        os.environ.get('TMLPU_V178_SIGNED_BRIDGE_CUT_STRENGTH', '0.14'))
    recon.euler_tangential_signed_tail_bridge_cut_min_factor = float(
        os.environ.get('TMLPU_V178_SIGNED_BRIDGE_CUT_MIN_FACTOR', '0.78'))
    recon.euler_tangential_signed_tail_bridge_cut_q_min = float(
        os.environ.get('TMLPU_V178_SIGNED_BRIDGE_CUT_Q_MIN', '0.14'))
    recon.euler_tangential_signed_tail_bridge_cut_q_full = float(
        os.environ.get('TMLPU_V178_SIGNED_BRIDGE_CUT_Q_FULL', '0.32'))
    recon.euler_tangential_signed_tail_bridge_cut_contact_min = float(
        os.environ.get('TMLPU_V178_SIGNED_BRIDGE_CUT_CONTACT_MIN', '0.30'))
    recon.euler_tangential_signed_tail_bridge_cut_contact_full = float(
        os.environ.get('TMLPU_V178_SIGNED_BRIDGE_CUT_CONTACT_FULL', '0.68'))
    recon.euler_tangential_signed_tail_bridge_cut_omega_lo_pct = float(
        os.environ.get('TMLPU_V178_SIGNED_BRIDGE_CUT_OMEGA_LO_PCT', '76.0'))
    recon.euler_tangential_signed_tail_bridge_cut_omega_hi_pct = float(
        os.environ.get('TMLPU_V178_SIGNED_BRIDGE_CUT_OMEGA_HI_PCT', '96.0'))
    recon.euler_tangential_density_curve_tail_bridge_cut_on = True
    recon.euler_tangential_density_curve_tail_bridge_cut_strength = float(
        os.environ.get('TMLPU_V178_CURVE_BRIDGE_CUT_STRENGTH', '0.16'))
    recon.euler_tangential_density_curve_tail_bridge_cut_min_factor = float(
        os.environ.get('TMLPU_V178_CURVE_BRIDGE_CUT_MIN_FACTOR', '0.76'))
    recon.euler_tangential_density_curve_tail_bridge_cut_q_min = float(
        os.environ.get('TMLPU_V178_CURVE_BRIDGE_CUT_Q_MIN', '0.12'))
    recon.euler_tangential_density_curve_tail_bridge_cut_q_full = float(
        os.environ.get('TMLPU_V178_CURVE_BRIDGE_CUT_Q_FULL', '0.30'))
    recon.euler_tangential_density_curve_tail_bridge_cut_contact_min = float(
        os.environ.get('TMLPU_V178_CURVE_BRIDGE_CUT_CONTACT_MIN', '0.28'))
    recon.euler_tangential_density_curve_tail_bridge_cut_contact_full = float(
        os.environ.get('TMLPU_V178_CURVE_BRIDGE_CUT_CONTACT_FULL', '0.64'))
    recon.euler_tangential_density_curve_tail_bridge_cut_omega_lo_pct = float(
        os.environ.get('TMLPU_V178_CURVE_BRIDGE_CUT_OMEGA_LO_PCT', '74.0'))
    recon.euler_tangential_density_curve_tail_bridge_cut_omega_hi_pct = float(
        os.environ.get('TMLPU_V178_CURVE_BRIDGE_CUT_OMEGA_HI_PCT', '95.0'))
    return recon


def _tmlpu_v178_v174_dual_bridge_cut():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v178_v174_dual_bridge_cut_euler())


def _tmlpu_v179_v174_antisheet_euler():
    recon = _tmlpu_v174_v168_roi_strength_euler()
    recon.euler_tangential_signed_tail_antisheet_on = True
    recon.euler_tangential_signed_tail_antisheet_strength = float(
        os.environ.get('TMLPU_V179_ANTISHEET_STRENGTH', '0.35'))
    recon.euler_tangential_signed_tail_antisheet_min_factor = float(
        os.environ.get('TMLPU_V179_ANTISHEET_MIN_FACTOR', '0.60'))
    recon.euler_tangential_signed_tail_antisheet_q_hi = float(
        os.environ.get('TMLPU_V179_ANTISHEET_Q_HI', '0.055'))
    recon.euler_tangential_signed_tail_antisheet_contact_min = float(
        os.environ.get('TMLPU_V179_ANTISHEET_CONTACT_MIN', '0.28'))
    recon.euler_tangential_signed_tail_antisheet_contact_full = float(
        os.environ.get('TMLPU_V179_ANTISHEET_CONTACT_FULL', '0.66'))
    return recon


def _tmlpu_v179_v174_antisheet():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v179_v174_antisheet_euler())


def _tmlpu_v180_v174_swirl_core_euler():
    recon = _tmlpu_v174_v168_roi_strength_euler()
    recon.euler_tangential_swirl_tail_on = True
    recon.euler_tangential_swirl_tail_beta = float(
        os.environ.get('TMLPU_V180_SWIRL_TAIL_BETA', '0.018'))
    recon.euler_tangential_swirl_tail_cap = float(
        os.environ.get('TMLPU_V180_SWIRL_TAIL_CAP', '0.014'))
    recon.euler_tangential_swirl_tail_wave_cap = float(
        os.environ.get('TMLPU_V180_SWIRL_TAIL_WAVE_CAP', '0.0018'))
    recon.euler_tangential_swirl_tail_q_min = float(
        os.environ.get('TMLPU_V180_SWIRL_TAIL_Q_MIN', '0.010'))
    recon.euler_tangential_swirl_tail_q_full = float(
        os.environ.get('TMLPU_V180_SWIRL_TAIL_Q_FULL', '0.036'))
    recon.euler_tangential_swirl_tail_pressure_hi = float(
        os.environ.get('TMLPU_V180_SWIRL_TAIL_PRESSURE_HI', '0.018'))
    recon.euler_tangential_swirl_tail_compression_hi = float(
        os.environ.get('TMLPU_V180_SWIRL_TAIL_COMPRESSION_HI', '0.004'))
    recon.euler_tangential_swirl_tail_normality_hi = float(
        os.environ.get('TMLPU_V180_SWIRL_TAIL_NORMALITY_HI', '0.18'))
    return recon


def _tmlpu_v180_v174_swirl_core():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v180_v174_swirl_core_euler())


def _tmlpu_v181_v174_qbridge_cut_euler():
    recon = _tmlpu_v174_v168_roi_strength_euler()
    recon.euler_tangential_signed_tail_qbridge_cut_on = True
    recon.euler_tangential_signed_tail_qbridge_cut_strength = float(
        os.environ.get('TMLPU_V181_SIGNED_QBRIDGE_CUT_STRENGTH', '0.42'))
    recon.euler_tangential_signed_tail_qbridge_cut_min_factor = float(
        os.environ.get('TMLPU_V181_SIGNED_QBRIDGE_CUT_MIN_FACTOR', '0.62'))
    recon.euler_tangential_signed_tail_qbridge_cut_q_lo_pct = float(
        os.environ.get('TMLPU_V181_SIGNED_QBRIDGE_Q_LO_PCT', '28.0'))
    recon.euler_tangential_signed_tail_qbridge_cut_q_mid_pct = float(
        os.environ.get('TMLPU_V181_SIGNED_QBRIDGE_Q_MID_PCT', '60.0'))
    recon.euler_tangential_signed_tail_qbridge_cut_q_core_pct = float(
        os.environ.get('TMLPU_V181_SIGNED_QBRIDGE_Q_CORE_PCT', '82.0'))
    recon.euler_tangential_signed_tail_qbridge_cut_q_top_pct = float(
        os.environ.get('TMLPU_V181_SIGNED_QBRIDGE_Q_TOP_PCT', '96.0'))
    recon.euler_tangential_signed_tail_qbridge_cut_contact_min = float(
        os.environ.get('TMLPU_V181_SIGNED_QBRIDGE_CONTACT_MIN', '0.24'))
    recon.euler_tangential_signed_tail_qbridge_cut_contact_full = float(
        os.environ.get('TMLPU_V181_SIGNED_QBRIDGE_CONTACT_FULL', '0.58'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_on = True
    recon.euler_tangential_density_curve_tail_qbridge_cut_strength = float(
        os.environ.get('TMLPU_V181_CURVE_QBRIDGE_CUT_STRENGTH', '0.50'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_min_factor = float(
        os.environ.get('TMLPU_V181_CURVE_QBRIDGE_CUT_MIN_FACTOR', '0.56'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_q_lo_pct = float(
        os.environ.get('TMLPU_V181_CURVE_QBRIDGE_Q_LO_PCT', '24.0'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_q_mid_pct = float(
        os.environ.get('TMLPU_V181_CURVE_QBRIDGE_Q_MID_PCT', '58.0'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_q_core_pct = float(
        os.environ.get('TMLPU_V181_CURVE_QBRIDGE_Q_CORE_PCT', '80.0'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_q_top_pct = float(
        os.environ.get('TMLPU_V181_CURVE_QBRIDGE_Q_TOP_PCT', '95.0'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_contact_min = float(
        os.environ.get('TMLPU_V181_CURVE_QBRIDGE_CONTACT_MIN', '0.22'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_contact_full = float(
        os.environ.get('TMLPU_V181_CURVE_QBRIDGE_CONTACT_FULL', '0.56'))
    return recon


def _tmlpu_v181_v174_qbridge_cut():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v181_v174_qbridge_cut_euler())


def _tmlpu_v182_v174_total_qbridge_damp_euler():
    recon = _tmlpu_v174_v168_roi_strength_euler()
    recon.euler_tangential_total_qbridge_damp_on = True
    recon.euler_tangential_total_qbridge_damp_strength = float(
        os.environ.get('TMLPU_V182_TOTAL_QBRIDGE_DAMP_STRENGTH', '0.30'))
    recon.euler_tangential_total_qbridge_damp_min_factor = float(
        os.environ.get('TMLPU_V182_TOTAL_QBRIDGE_DAMP_MIN_FACTOR', '0.70'))
    recon.euler_tangential_total_qbridge_damp_q_lo_pct = float(
        os.environ.get('TMLPU_V182_TOTAL_QBRIDGE_Q_LO_PCT', '25.0'))
    recon.euler_tangential_total_qbridge_damp_q_mid_pct = float(
        os.environ.get('TMLPU_V182_TOTAL_QBRIDGE_Q_MID_PCT', '58.0'))
    recon.euler_tangential_total_qbridge_damp_q_core_pct = float(
        os.environ.get('TMLPU_V182_TOTAL_QBRIDGE_Q_CORE_PCT', '80.0'))
    recon.euler_tangential_total_qbridge_damp_q_top_pct = float(
        os.environ.get('TMLPU_V182_TOTAL_QBRIDGE_Q_TOP_PCT', '95.0'))
    recon.euler_tangential_total_qbridge_damp_contact_min = float(
        os.environ.get('TMLPU_V182_TOTAL_QBRIDGE_CONTACT_MIN', '0.22'))
    recon.euler_tangential_total_qbridge_damp_contact_full = float(
        os.environ.get('TMLPU_V182_TOTAL_QBRIDGE_CONTACT_FULL', '0.55'))
    return recon


def _tmlpu_v182_v174_total_qbridge_damp():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v182_v174_total_qbridge_damp_euler())


def _tmlpu_v183_v174_micro_pair_extend_euler():
    recon = _tmlpu_v174_v168_roi_strength_euler()
    recon.euler_tangential_pair_extend_on = True
    recon.euler_tangential_pair_extend_beta = float(
        os.environ.get('TMLPU_V183_PAIR_EXTEND_BETA', '0.0035'))
    recon.euler_tangential_pair_extend_cap = float(
        os.environ.get('TMLPU_V183_PAIR_EXTEND_CAP', '0.0028'))
    recon.euler_tangential_pair_extend_wave_cap = float(
        os.environ.get('TMLPU_V183_PAIR_EXTEND_WAVE_CAP', '0.00045'))
    recon.euler_tangential_pair_extend_alignment_min = float(
        os.environ.get('TMLPU_V183_PAIR_EXTEND_ALIGN_MIN', '0.42'))
    recon.euler_tangential_pair_extend_alignment_full = float(
        os.environ.get('TMLPU_V183_PAIR_EXTEND_ALIGN_FULL', '0.82'))
    recon.euler_tangential_pair_extend_shock_exclude = True
    return recon


def _tmlpu_v183_v174_micro_pair_extend():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v183_v174_micro_pair_extend_euler())


def _tmlpu_v184_v174_midq_cell_blend_euler():
    recon = _tmlpu_v174_v168_roi_strength_euler()
    recon.euler_tangential_midq_cell_blend_on = True
    recon.euler_tangential_midq_cell_blend_strength = float(
        os.environ.get('TMLPU_V184_MIDQ_CELL_BLEND_STRENGTH', '0.32'))
    recon.euler_tangential_midq_cell_blend_q_lo_pct = float(
        os.environ.get('TMLPU_V184_MIDQ_CELL_BLEND_Q_LO_PCT', '12.0'))
    recon.euler_tangential_midq_cell_blend_q_mid_pct = float(
        os.environ.get('TMLPU_V184_MIDQ_CELL_BLEND_Q_MID_PCT', '48.0'))
    recon.euler_tangential_midq_cell_blend_q_core_pct = float(
        os.environ.get('TMLPU_V184_MIDQ_CELL_BLEND_Q_CORE_PCT', '78.0'))
    recon.euler_tangential_midq_cell_blend_q_top_pct = float(
        os.environ.get('TMLPU_V184_MIDQ_CELL_BLEND_Q_TOP_PCT', '95.0'))
    recon.euler_tangential_midq_cell_blend_contact_min = float(
        os.environ.get('TMLPU_V184_MIDQ_CELL_BLEND_CONTACT_MIN', '0.18'))
    recon.euler_tangential_midq_cell_blend_contact_full = float(
        os.environ.get('TMLPU_V184_MIDQ_CELL_BLEND_CONTACT_FULL', '0.52'))
    return recon


def _tmlpu_v184_v174_midq_cell_blend():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v184_v174_midq_cell_blend_euler())


def _tmlpu_v185_v174_soft_midq_cell_blend_euler():
    recon = _tmlpu_v174_v168_roi_strength_euler()
    recon.euler_tangential_midq_cell_blend_on = True
    recon.euler_tangential_midq_cell_blend_strength = float(
        os.environ.get('TMLPU_V185_MIDQ_CELL_BLEND_STRENGTH', '0.16'))
    recon.euler_tangential_midq_cell_blend_q_lo_pct = float(
        os.environ.get('TMLPU_V185_MIDQ_CELL_BLEND_Q_LO_PCT', '20.0'))
    recon.euler_tangential_midq_cell_blend_q_mid_pct = float(
        os.environ.get('TMLPU_V185_MIDQ_CELL_BLEND_Q_MID_PCT', '58.0'))
    recon.euler_tangential_midq_cell_blend_q_core_pct = float(
        os.environ.get('TMLPU_V185_MIDQ_CELL_BLEND_Q_CORE_PCT', '86.0'))
    recon.euler_tangential_midq_cell_blend_q_top_pct = float(
        os.environ.get('TMLPU_V185_MIDQ_CELL_BLEND_Q_TOP_PCT', '97.0'))
    recon.euler_tangential_midq_cell_blend_contact_min = float(
        os.environ.get('TMLPU_V185_MIDQ_CELL_BLEND_CONTACT_MIN', '0.22'))
    recon.euler_tangential_midq_cell_blend_contact_full = float(
        os.environ.get('TMLPU_V185_MIDQ_CELL_BLEND_CONTACT_FULL', '0.58'))
    return recon


def _tmlpu_v185_v174_soft_midq_cell_blend():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v185_v174_soft_midq_cell_blend_euler())


def _tmlpu_v186_v184_shockridge_guard_euler():
    recon = _tmlpu_v184_v174_midq_cell_blend_euler()

    # v184 gives strong Mach3 upper-ROI vortex clarity, but it can bend/split
    # the step-top reflected shock.  Keep that vortex-producing branch and
    # damp only shock-ridge-like tail support with global feature gates.
    recon.euler_tangential_tail_density_shock_damp_on = _env_bool(
        'TMLPU_V186_TAIL_SHOCK_DAMP_ON', True)
    recon.euler_tangential_tail_density_shock_damp_theta = float(
        os.environ.get('TMLPU_V186_TAIL_SHOCK_DAMP_THETA', '0.46'))
    recon.euler_tangential_tail_density_shock_damp_pressure_min = float(
        os.environ.get('TMLPU_V186_TAIL_SHOCK_PRESSURE_MIN', '0.010'))
    recon.euler_tangential_tail_density_shock_damp_compression_min = float(
        os.environ.get('TMLPU_V186_TAIL_SHOCK_COMPRESSION_MIN', '0.002'))
    recon.euler_tangential_tail_density_shock_damp_normality_min = float(
        os.environ.get('TMLPU_V186_TAIL_SHOCK_NORMALITY_MIN', '0.14'))

    recon.euler_tangential_signed_tail_shock_ridge_clean_on = _env_bool(
        'TMLPU_V186_SIGNED_SHOCK_RIDGE_CLEAN_ON', True)
    recon.euler_tangential_signed_tail_shock_ridge_strength = float(
        os.environ.get('TMLPU_V186_SIGNED_SHOCK_RIDGE_STRENGTH', '0.34'))
    recon.euler_tangential_signed_tail_shock_ridge_min_factor = float(
        os.environ.get('TMLPU_V186_SIGNED_SHOCK_RIDGE_MIN_FACTOR', '0.70'))
    recon.euler_tangential_signed_tail_shock_ridge_density_min = float(
        os.environ.get('TMLPU_V186_SIGNED_SHOCK_RIDGE_DENSITY_MIN', '0.24'))
    recon.euler_tangential_signed_tail_shock_ridge_density_full = float(
        os.environ.get('TMLPU_V186_SIGNED_SHOCK_RIDGE_DENSITY_FULL', '0.68'))
    recon.euler_tangential_signed_tail_shock_ridge_q_keep_min = float(
        os.environ.get('TMLPU_V186_SIGNED_SHOCK_RIDGE_Q_KEEP_MIN', '0.06'))
    recon.euler_tangential_signed_tail_shock_ridge_q_keep_full = float(
        os.environ.get('TMLPU_V186_SIGNED_SHOCK_RIDGE_Q_KEEP_FULL', '0.20'))

    return recon


def _tmlpu_v186_v184_shockridge_guard():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v186_v184_shockridge_guard_euler())


def _tmlpu_v187_v186_curve_shockridge_guard_euler():
    recon = _tmlpu_v186_v184_shockridge_guard_euler()

    # v186 still leaves a weak step-top shock branch.  Apply the same
    # shock-ridge density support cleanup to density-curve tail corrections,
    # while keeping q-dominant vortex cores open.
    recon.euler_tangential_density_curve_tail_shock_ridge_clean_on = _env_bool(
        'TMLPU_V187_CURVE_SHOCK_RIDGE_CLEAN_ON', True)
    recon.euler_tangential_density_curve_tail_shock_ridge_strength = float(
        os.environ.get('TMLPU_V187_CURVE_SHOCK_RIDGE_STRENGTH', '0.42'))
    recon.euler_tangential_density_curve_tail_shock_ridge_min_factor = float(
        os.environ.get('TMLPU_V187_CURVE_SHOCK_RIDGE_MIN_FACTOR', '0.64'))
    recon.euler_tangential_density_curve_tail_shock_ridge_density_min = float(
        os.environ.get('TMLPU_V187_CURVE_SHOCK_RIDGE_DENSITY_MIN', '0.20'))
    recon.euler_tangential_density_curve_tail_shock_ridge_density_full = float(
        os.environ.get('TMLPU_V187_CURVE_SHOCK_RIDGE_DENSITY_FULL', '0.62'))
    recon.euler_tangential_density_curve_tail_shock_ridge_q_keep_min = float(
        os.environ.get('TMLPU_V187_CURVE_SHOCK_RIDGE_Q_KEEP_MIN', '0.055'))
    recon.euler_tangential_density_curve_tail_shock_ridge_q_keep_full = float(
        os.environ.get('TMLPU_V187_CURVE_SHOCK_RIDGE_Q_KEEP_FULL', '0.18'))
    return recon


def _tmlpu_v187_v186_curve_shockridge_guard():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v187_v186_curve_shockridge_guard_euler())


def _tmlpu_v188_v126_soft_midq_rollup_euler():
    recon = _tmlpu_v126_curve_only_relief_floor04_euler()

    # Start from the only recent Mach3 branch that gives a clean step-top
    # reflected shock, then add a weaker mid-Q cell blend to recover upper-ROI
    # roll-up without reopening the shock branch.
    recon.euler_tangential_midq_cell_blend_on = True
    recon.euler_tangential_midq_cell_blend_strength = float(
        os.environ.get('TMLPU_V188_MIDQ_CELL_BLEND_STRENGTH', '0.20'))
    recon.euler_tangential_midq_cell_blend_q_lo_pct = float(
        os.environ.get('TMLPU_V188_MIDQ_CELL_BLEND_Q_LO_PCT', '14.0'))
    recon.euler_tangential_midq_cell_blend_q_mid_pct = float(
        os.environ.get('TMLPU_V188_MIDQ_CELL_BLEND_Q_MID_PCT', '50.0'))
    recon.euler_tangential_midq_cell_blend_q_core_pct = float(
        os.environ.get('TMLPU_V188_MIDQ_CELL_BLEND_Q_CORE_PCT', '80.0'))
    recon.euler_tangential_midq_cell_blend_q_top_pct = float(
        os.environ.get('TMLPU_V188_MIDQ_CELL_BLEND_Q_TOP_PCT', '95.0'))
    recon.euler_tangential_midq_cell_blend_contact_min = float(
        os.environ.get('TMLPU_V188_MIDQ_CELL_BLEND_CONTACT_MIN', '0.18'))
    recon.euler_tangential_midq_cell_blend_contact_full = float(
        os.environ.get('TMLPU_V188_MIDQ_CELL_BLEND_CONTACT_FULL', '0.52'))
    return recon


def _tmlpu_v188_v126_soft_midq_rollup():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v188_v126_soft_midq_rollup_euler())


def _tmlpu_v189_v188_checker_safe_midq_euler():
    recon = _tmlpu_v188_v126_soft_midq_rollup_euler()

    # v188 passes Mach3 but its Double Mach paper-grid result leaves too much
    # smooth-region checker and a slightly merged primary ROI cluster.  Tighten
    # the mid-Q bridge damping and filter only the density-curve tail
    # correction, while keeping the top-Q core and shocks mostly excluded.
    recon.euler_tangential_midq_cell_blend_strength = float(
        os.environ.get('TMLPU_V189_MIDQ_CELL_BLEND_STRENGTH', '0.32'))
    recon.euler_tangential_midq_cell_blend_q_lo_pct = float(
        os.environ.get('TMLPU_V189_MIDQ_CELL_BLEND_Q_LO_PCT', '12.0'))
    recon.euler_tangential_midq_cell_blend_q_mid_pct = float(
        os.environ.get('TMLPU_V189_MIDQ_CELL_BLEND_Q_MID_PCT', '45.0'))
    recon.euler_tangential_midq_cell_blend_q_core_pct = float(
        os.environ.get('TMLPU_V189_MIDQ_CELL_BLEND_Q_CORE_PCT', '72.0'))
    recon.euler_tangential_midq_cell_blend_q_top_pct = float(
        os.environ.get('TMLPU_V189_MIDQ_CELL_BLEND_Q_TOP_PCT', '93.0'))
    recon.euler_tangential_midq_cell_blend_contact_min = float(
        os.environ.get('TMLPU_V189_MIDQ_CELL_BLEND_CONTACT_MIN', '0.16'))
    recon.euler_tangential_midq_cell_blend_contact_full = float(
        os.environ.get('TMLPU_V189_MIDQ_CELL_BLEND_CONTACT_FULL', '0.48'))
    recon.euler_tangential_density_curve_tail_hf_filter_on = _env_bool(
        'TMLPU_V189_CURVE_HF_FILTER_ON', True)
    recon.euler_tangential_density_curve_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V189_CURVE_HF_FILTER_STRENGTH', '0.22'))
    recon.euler_tangential_density_curve_tail_hf_filter_min_weight = float(
        os.environ.get('TMLPU_V189_CURVE_HF_FILTER_MIN_WEIGHT', '1e-10'))
    recon.euler_tangential_density_curve_tail_hf_filter_shock_exclude = (
        _env_bool('TMLPU_V189_CURVE_HF_FILTER_SHOCK_EXCLUDE', True))
    return recon


def _tmlpu_v189_v188_checker_safe_midq():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v189_v188_checker_safe_midq_euler())


def _tmlpu_v195_v188_curve_hffilter_euler():
    recon = _tmlpu_v188_v126_soft_midq_rollup_euler()

    # Keep the Mach3-passing v188 mid-Q roll-up settings unchanged, and only
    # damp face-to-face high-frequency density-curve tail correction.  v189
    # tightened the mid-Q blend as well and broke the Mach3 top-floor shock.
    recon.euler_tangential_density_curve_tail_hf_filter_on = _env_bool(
        'TMLPU_V195_CURVE_HF_FILTER_ON', True)
    recon.euler_tangential_density_curve_tail_hf_filter_strength = float(
        os.environ.get('TMLPU_V195_CURVE_HF_FILTER_STRENGTH', '0.16'))
    recon.euler_tangential_density_curve_tail_hf_filter_min_weight = float(
        os.environ.get('TMLPU_V195_CURVE_HF_FILTER_MIN_WEIGHT', '1e-10'))
    recon.euler_tangential_density_curve_tail_hf_filter_shock_exclude = (
        _env_bool('TMLPU_V195_CURVE_HF_FILTER_SHOCK_EXCLUDE', True))
    return recon


def _tmlpu_v195_v188_curve_hffilter():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v195_v188_curve_hffilter_euler())


def _tmlpu_v196_v188_curve_qbridge_euler():
    recon = _tmlpu_v188_v126_soft_midq_rollup_euler()

    # Target Double Mach bridge merging without generic smoothing: damp only
    # mid-Q contact/shear density-curve tail support and leave top-Q cores and
    # shocks open.  This is narrower than v195's HF averaging, which broke
    # Mach3 roll-up/top-floor visual gates.
    recon.euler_tangential_density_curve_tail_qbridge_cut_on = _env_bool(
        'TMLPU_V196_CURVE_QBRIDGE_CUT_ON', True)
    recon.euler_tangential_density_curve_tail_qbridge_cut_strength = float(
        os.environ.get('TMLPU_V196_CURVE_QBRIDGE_CUT_STRENGTH', '0.18'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_min_factor = float(
        os.environ.get('TMLPU_V196_CURVE_QBRIDGE_CUT_MIN_FACTOR', '0.84'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_q_lo_pct = float(
        os.environ.get('TMLPU_V196_CURVE_QBRIDGE_CUT_Q_LO_PCT', '32.0'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_q_mid_pct = float(
        os.environ.get('TMLPU_V196_CURVE_QBRIDGE_CUT_Q_MID_PCT', '58.0'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_q_core_pct = float(
        os.environ.get('TMLPU_V196_CURVE_QBRIDGE_CUT_Q_CORE_PCT', '84.0'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_q_top_pct = float(
        os.environ.get('TMLPU_V196_CURVE_QBRIDGE_CUT_Q_TOP_PCT', '96.0'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_contact_min = float(
        os.environ.get('TMLPU_V196_CURVE_QBRIDGE_CUT_CONTACT_MIN', '0.22'))
    recon.euler_tangential_density_curve_tail_qbridge_cut_contact_full = float(
        os.environ.get('TMLPU_V196_CURVE_QBRIDGE_CUT_CONTACT_FULL', '0.54'))
    return recon


def _tmlpu_v196_v188_curve_qbridge():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v196_v188_curve_qbridge_euler())


def _tmlpu_v190_v45scalar_v3fast_euler():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v3_unified_euler())


def _tmlpu_v191_fast_scalar_balanced():
    scalar = TMLPU(
        tvd=os.environ.get('TMLPU_V191_SCALAR_TVD', 'vanleer'),
        mlp_bound=True,
        extremum_relax=_env_bool('TMLPU_V191_SCALAR_EXTREMUM_RELAX', False),
        tvb_M=float(os.environ.get('TMLPU_V191_SCALAR_TVB_M', '0.0')),
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=float(os.environ.get(
            'TMLPU_V191_SCALAR_VERTEX_MLP_CAP', '1.0')),
        vertex_mlp_augment=True,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V191_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        face_increment=os.environ.get(
            'TMLPU_V191_SCALAR_FACE_INCREMENT', 'tmlpu'),
        r_form=os.environ.get('TMLPU_V191_SCALAR_R_FORM', 'far_upwind'),
        clamp_tstar=True,
        stencil=os.environ.get('TMLPU_V191_SCALAR_STENCIL', 'face'),
        order=int(os.environ.get('TMLPU_V191_SCALAR_ORDER', '2')),
        idw_p=float(os.environ.get('TMLPU_V191_SCALAR_IDW_P', '6.0')),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v192_fast_scalar_jump_damped():
    inner = TMLPU(
        tvd=os.environ.get('TMLPU_V192_SCALAR_TVD', 'pure_downwind'),
        mlp_bound=True,
        extremum_relax=_env_bool('TMLPU_V220_EXTREMUM_RELAX', False),
        tvd_smooth=(os.environ.get('TMLPU_V220_TVD_SMOOTH', '') or None),
        tvd_smooth2=(os.environ.get('TMLPU_V220_TVD_SMOOTH2', '') or None),
        smoothness_threshold=float(os.environ.get(
            'TMLPU_V220_SMOOTHNESS_THRESHOLD', '0.1')),
        smoothness_threshold2=float(os.environ.get(
            'TMLPU_V220_SMOOTHNESS_THRESHOLD2', '0.05')),
        tvb_M=float(os.environ.get('TMLPU_V220_TVB_M', '0.0')),
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=1.0,
        vertex_mlp_augment=True,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V192_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        face_increment='tmlpu',
        r_form='far_upwind',
        clamp_tstar=True,
        stencil='face',
        order=2,
        idw_p=6.0,
    )
    scalar = _ScalarJumpDampedReconstruction(
        inner,
        jump_min=float(os.environ.get('TMLPU_V192_JUMP_MIN', '0.18')),
        jump_full=float(os.environ.get('TMLPU_V192_JUMP_FULL', '0.55')),
        min_theta=float(os.environ.get('TMLPU_V192_MIN_THETA', '0.35')),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v193_fast_scalar_binary_guard():
    inner = TMLPU(
        tvd=os.environ.get('TMLPU_V193_SCALAR_TVD', 'pure_downwind'),
        mlp_bound=True,
        extremum_relax=False,
        tvb_M=0.0,
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=1.0,
        vertex_mlp_augment=True,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V193_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        face_increment='tmlpu',
        r_form='far_upwind',
        clamp_tstar=True,
        stencil='face',
        order=2,
        idw_p=6.0,
    )
    scalar = _ScalarJumpDampedBinaryGuardReconstruction(
        inner,
        jump_min=float(os.environ.get('TMLPU_V193_JUMP_MIN', '0.18')),
        jump_full=float(os.environ.get('TMLPU_V193_JUMP_FULL', '0.55')),
        min_theta=float(os.environ.get('TMLPU_V193_MIN_THETA', '0.90')),
        low_plateau=float(os.environ.get('TMLPU_V193_LOW_PLATEAU', '0.03')),
        high_plateau=float(os.environ.get('TMLPU_V193_HIGH_PLATEAU', '0.92')),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v200_fast_scalar_smooth_body_guard():
    inner = TMLPU(
        tvd=os.environ.get('TMLPU_V200_SCALAR_TVD', 'pure_downwind'),
        mlp_bound=True,
        extremum_relax=False,
        tvb_M=0.0,
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=1.0,
        vertex_mlp_augment=True,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V200_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        face_increment='tmlpu',
        r_form='far_upwind',
        clamp_tstar=True,
        stencil='face',
        order=2,
        idw_p=6.0,
    )
    scalar = _ScalarSmoothBodyGuardReconstruction(
        inner,
        jump_min=float(os.environ.get('TMLPU_V200_JUMP_MIN', '0.18')),
        jump_full=float(os.environ.get('TMLPU_V200_JUMP_FULL', '0.55')),
        min_theta=float(os.environ.get('TMLPU_V200_MIN_THETA', '0.90')),
        low_plateau=float(os.environ.get('TMLPU_V200_LOW_PLATEAU', '0.03')),
        high_plateau=float(os.environ.get('TMLPU_V200_HIGH_PLATEAU', '0.92')),
        smooth_theta=float(os.environ.get('TMLPU_V200_SMOOTH_THETA', '0.68')),
        smooth_value_low=float(os.environ.get(
            'TMLPU_V200_SMOOTH_VALUE_LOW', '0.06')),
        smooth_value_high=float(os.environ.get(
            'TMLPU_V200_SMOOTH_VALUE_HIGH', '0.90')),
        smooth_jump_min=float(os.environ.get(
            'TMLPU_V200_SMOOTH_JUMP_MIN', '0.02')),
        smooth_jump_full=float(os.environ.get(
            'TMLPU_V200_SMOOTH_JUMP_FULL', '0.16')),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v201_fast_scalar_range_smooth_guard():
    inner = TMLPU(
        tvd=os.environ.get('TMLPU_V201_SCALAR_TVD', 'pure_downwind'),
        mlp_bound=True,
        extremum_relax=False,
        tvb_M=0.0,
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=1.0,
        vertex_mlp_augment=True,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V201_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        face_increment='tmlpu',
        r_form='far_upwind',
        clamp_tstar=True,
        stencil='face',
        order=2,
        idw_p=6.0,
    )
    scalar = _ScalarSmoothBodyGuardReconstruction(
        inner,
        jump_min=float(os.environ.get('TMLPU_V201_JUMP_MIN', '0.18')),
        jump_full=float(os.environ.get('TMLPU_V201_JUMP_FULL', '0.55')),
        min_theta=float(os.environ.get('TMLPU_V201_MIN_THETA', '0.90')),
        low_plateau=float(os.environ.get('TMLPU_V201_LOW_PLATEAU', '0.03')),
        high_plateau=float(os.environ.get('TMLPU_V201_HIGH_PLATEAU', '0.92')),
        smooth_theta=float(os.environ.get('TMLPU_V201_SMOOTH_THETA', '0.45')),
        smooth_value_low=float(os.environ.get(
            'TMLPU_V201_SMOOTH_VALUE_LOW', '0.06')),
        smooth_value_high=float(os.environ.get(
            'TMLPU_V201_SMOOTH_VALUE_HIGH', '0.82')),
        smooth_jump_min=float(os.environ.get(
            'TMLPU_V201_SMOOTH_JUMP_MIN', '0.02')),
        smooth_jump_full=float(os.environ.get(
            'TMLPU_V201_SMOOTH_JUMP_FULL', '0.16')),
        smooth_range_min=float(os.environ.get(
            'TMLPU_V201_SMOOTH_RANGE_MIN', '0.10')),
        smooth_range_full=float(os.environ.get(
            'TMLPU_V201_SMOOTH_RANGE_FULL', '0.34')),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v202_fast_scalar_self_bvd_smooth_guard():
    inner = TMLPU(
        tvd=os.environ.get('TMLPU_V202_SCALAR_TVD', 'pure_downwind'),
        mlp_bound=True,
        extremum_relax=False,
        tvb_M=0.0,
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=1.0,
        vertex_mlp_augment=True,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V202_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        face_increment='tmlpu',
        r_form='far_upwind',
        clamp_tstar=True,
        stencil='face',
        order=2,
        idw_p=6.0,
    )
    scalar = _ScalarSelfBVDRangeSmoothGuardReconstruction(
        inner,
        bvd_mode=os.environ.get('TMLPU_V202_BVD_MODE', 'product'),
        jump_min=float(os.environ.get('TMLPU_V202_JUMP_MIN', '0.18')),
        jump_full=float(os.environ.get('TMLPU_V202_JUMP_FULL', '0.55')),
        min_theta=float(os.environ.get('TMLPU_V202_MIN_THETA', '0.90')),
        low_plateau=float(os.environ.get('TMLPU_V202_LOW_PLATEAU', '0.03')),
        high_plateau=float(os.environ.get('TMLPU_V202_HIGH_PLATEAU', '0.92')),
        smooth_theta=float(os.environ.get('TMLPU_V202_SMOOTH_THETA', '0.45')),
        smooth_value_low=float(os.environ.get(
            'TMLPU_V202_SMOOTH_VALUE_LOW', '0.06')),
        smooth_value_high=float(os.environ.get(
            'TMLPU_V202_SMOOTH_VALUE_HIGH', '0.82')),
        smooth_jump_min=float(os.environ.get(
            'TMLPU_V202_SMOOTH_JUMP_MIN', '0.02')),
        smooth_jump_full=float(os.environ.get(
            'TMLPU_V202_SMOOTH_JUMP_FULL', '0.16')),
        smooth_range_min=float(os.environ.get(
            'TMLPU_V202_SMOOTH_RANGE_MIN', '0.10')),
        smooth_range_full=float(os.environ.get(
            'TMLPU_V202_SMOOTH_RANGE_FULL', '0.34')),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v218_fast_scalar_selective_balance():
    inner = TMLPU(
        tvd=os.environ.get('TMLPU_V218_SCALAR_TVD', 'pure_downwind'),
        mlp_bound=True,
        extremum_relax=False,
        tvb_M=0.0,
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=1.0,
        vertex_mlp_augment=True,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V218_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        face_increment='tmlpu',
        r_form='far_upwind',
        clamp_tstar=True,
        stencil='face',
        order=2,
        idw_p=6.0,
    )
    damped = _ScalarSelfBVDRangeSmoothGuardReconstruction(
        inner,
        bvd_mode=os.environ.get('TMLPU_V218_BVD_MODE', 'product'),
        jump_min=float(os.environ.get('TMLPU_V218_JUMP_MIN', '0.18')),
        jump_full=float(os.environ.get('TMLPU_V218_JUMP_FULL', '0.55')),
        min_theta=float(os.environ.get('TMLPU_V218_MIN_THETA', '0.90')),
        low_plateau=float(os.environ.get('TMLPU_V218_LOW_PLATEAU', '0.03')),
        high_plateau=float(os.environ.get('TMLPU_V218_HIGH_PLATEAU', '0.92')),
        smooth_theta=float(os.environ.get('TMLPU_V218_SMOOTH_THETA', '0.45')),
        smooth_value_low=float(os.environ.get(
            'TMLPU_V218_SMOOTH_VALUE_LOW', '0.06')),
        smooth_value_high=float(os.environ.get(
            'TMLPU_V218_SMOOTH_VALUE_HIGH', '0.82')),
        smooth_jump_min=float(os.environ.get(
            'TMLPU_V218_SMOOTH_JUMP_MIN', '0.02')),
        smooth_jump_full=float(os.environ.get(
            'TMLPU_V218_SMOOTH_JUMP_FULL', '0.16')),
        smooth_range_min=float(os.environ.get(
            'TMLPU_V218_SMOOTH_RANGE_MIN', '0.10')),
        smooth_range_full=float(os.environ.get(
            'TMLPU_V218_SMOOTH_RANGE_FULL', '0.34')),
    )
    scalar = _ScalarSelectiveSmoothCellBalancedReconstruction(
        damped,
        strength=float(os.environ.get('TMLPU_V218_BALANCE_STRENGTH', '0.32')),
        value_low=float(os.environ.get('TMLPU_V218_BALANCE_VALUE_LOW', '0.035')),
        value_high=float(os.environ.get(
            'TMLPU_V218_BALANCE_VALUE_HIGH', '0.86')),
        range_on_min=float(os.environ.get(
            'TMLPU_V218_BALANCE_RANGE_ON_MIN', '0.006')),
        range_on_full=float(os.environ.get(
            'TMLPU_V218_BALANCE_RANGE_ON_FULL', '0.045')),
        range_off_min=float(os.environ.get(
            'TMLPU_V218_BALANCE_RANGE_OFF_MIN', '0.24')),
        range_off_full=float(os.environ.get(
            'TMLPU_V218_BALANCE_RANGE_OFF_FULL', '0.42')),
        clip_bounds=_env_bool('TMLPU_V218_BALANCE_CLIP_BOUNDS', True),
    )
    return TMLPUUnifiedReconstruction(
        scalar, _tmlpu_v217_v212_wallshock_tailboost_euler())


def _tmlpu_v204_mlpu1_bounded_increment_boost():
    scalar = _ScalarMLPU1BoundedIncrementBoost(
        smooth_min=float(os.environ.get('TMLPU_V204_SMOOTH_MIN', '0.006')),
        smooth_full=float(os.environ.get('TMLPU_V204_SMOOTH_FULL', '0.055')),
        edge_min=float(os.environ.get('TMLPU_V204_EDGE_MIN', '0.18')),
        edge_full=float(os.environ.get('TMLPU_V204_EDGE_FULL', '0.55')),
        smooth_alpha=float(os.environ.get(
            'TMLPU_V204_SMOOTH_ALPHA', '0.16')),
        edge_alpha=float(os.environ.get('TMLPU_V204_EDGE_ALPHA', '0.90')),
        clip_unit=_env_bool('TMLPU_V204_CLIP_UNIT', True),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v220_exact_beta():
    tvd = os.environ.get('TMLPU_V220_TVD', 'pure_downwind')
    common = dict(
        tvd=tvd,
        hancock_courant=0.0,
        mlp_bound=True,
        extremum_relax=_env_bool('TMLPU_V220_EXTREMUM_RELAX', False),
        tvb_M=float(os.environ.get('TMLPU_V220_TVB_M', '0.0')),
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=float(os.environ.get('TMLPU_V220_VERTEX_MLP_CAP',
                                            '1.0')),
        vertex_mlp_augment=_env_bool('TMLPU_V220_VERTEX_MLP_AUGMENT', False),
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V220_FACE_GRADIENT_CORRECTION', 'beta'),
        face_increment=os.environ.get('TMLPU_V220_FACE_INCREMENT', 'tmlpu'),
        r_form='far_upwind',
        clamp_tstar=True,
        stencil=os.environ.get('TMLPU_V220_STENCIL', 'face'),
        order=int(os.environ.get('TMLPU_V220_ORDER', '2')),
        idw_p=float(os.environ.get('TMLPU_V220_IDW_P', '6.0')),
        zero_delta_psi=float(os.environ.get('TMLPU_V220_ZERO_DELTA_PSI',
                                            '2.0')),
        cicsam_full=_env_bool('TMLPU_V220_CICSAM', False),
        cicsam_courant=float(os.environ.get('TMLPU_V220_CICSAM_CO', '0.4')),
        phi_LL_unclipped=_env_bool('TMLPU_V220_PHI_LL_UNCLIPPED', False),
        weak_face_mlp=_env_bool('TMLPU_V220_WEAK_FACE_MLP', False),
        euler_face_positivity_limiter=_env_bool('TMLPU_V220_POSITIVITY', False),
        euler_shock_flatten=_env_bool('TMLPU_V220_SHOCK_FLATTEN', False),
        euler_density_acoustic_flatten=_env_bool(
            'TMLPU_V220_ACOUSTIC_FLATTEN', False),
        euler_pressure_face_jump_limiter_on=_env_bool(
            'TMLPU_V220_PJUMP_LIMITER', False),
        euler_pressure_face_jump_limiter_strength=float(os.environ.get(
            'TMLPU_V220_PJUMP_STRENGTH', '0.0')),
    )
    scalar = TMLPU(**common)
    euler = TMLPU(**common)
    return TMLPUUnifiedReconstruction(scalar, euler)


def _tmlpu_v221_bvd_unified():
    """Unified BVD candidate: smooth bounded-CD <-> sharp TMLP-u-preserve, chosen
    per cell by boundary-variation (data-driven, not ROI). Compression is applied
    only at discontinuities so smooth regions stay polynomial-smooth.  Same single
    reconstruction for scalar (LeVeque) and Euler (Mach3 / Double Mach).  Euler
    robustness via the env positivity floor (TMLPU_EULER_RHO/P_FLOOR_FACTOR)."""
    return TMLPUSmoothSharpBVD(
        smooth_mode='tmlpu',
        smooth_tvd=os.environ.get('TMLPU_V221_SMOOTH_TVD', 'bounded_cd'),
        smooth_face_increment='lsq',
        sharp_tvd=os.environ.get('TMLPU_V221_SHARP_TVD', 'tmlpu_preserve'),
        sharp_face_increment='tmlpu',
        interface_tvd=None,
        smooth_stencil='face',
        sharp_stencil='face',
        order=int(os.environ.get('TMLPU_V221_ORDER', '1')),
        idw_p=float(os.environ.get('TMLPU_V221_IDW_P', '1.0')),
        vertex_mlp_cap=float(os.environ.get('TMLPU_V221_VERTEX_MLP_CAP', '2.0')),
        moment_bvd=False,
        euler_shock_flatten=_env_bool('TMLPU_V221_SHOCK_FLATTEN', False),
        euler_density_acoustic_flatten=_env_bool(
            'TMLPU_V221_ACOUSTIC_FLATTEN', False),
        euler_pressure_face_jump_limiter_on=_env_bool(
            'TMLPU_V221_PJUMP_LIMITER', False),
        euler_pressure_face_jump_limiter_strength=float(os.environ.get(
            'TMLPU_V221_PJUMP_STRENGTH', '0.0')),
        euler_face_positivity_limiter=_env_bool('TMLPU_V221_POSITIVITY', False),
    )


def _tmlpu_v205_fast_scalar_bvd():
    scalar = _ScalarFastTMLPUBVD(
        tvd=os.environ.get('TMLPU_V205_SCALAR_TVD', 'pure_downwind'),
        stencil=os.environ.get('TMLPU_V205_SCALAR_STENCIL', 'face'),
        order=int(os.environ.get('TMLPU_V205_SCALAR_ORDER', '2')),
        idw_p=float(os.environ.get('TMLPU_V205_SCALAR_IDW_P', '6.0')),
        vertex_mlp_cap=float(os.environ.get(
            'TMLPU_V205_SCALAR_VERTEX_MLP_CAP', '1.0')),
        vertex_mlp_face_local_branch=_env_bool(
            'TMLPU_V205_SCALAR_FACE_LOCAL_BRANCH', False),
        face_gradient_correction=os.environ.get(
            'TMLPU_V205_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        r_form=os.environ.get('TMLPU_V205_SCALAR_R_FORM', 'far_upwind'),
        moment_bvd=_env_bool('TMLPU_V205_SCALAR_MOMENT_BVD', True),
        moment_bvd_mode=os.environ.get(
            'TMLPU_V205_SCALAR_MOMENT_BVD_MODE', 'and'),
        interface_force_tmlpu=_env_bool(
            'TMLPU_V205_SCALAR_INTERFACE_FORCE_TMLPU', True),
        interface_force_range=float(os.environ.get(
            'TMLPU_V205_SCALAR_INTERFACE_FORCE_RANGE', '0.35')),
        interface_force_only=_env_bool(
            'TMLPU_V205_SCALAR_INTERFACE_FORCE_ONLY', False),
    )
    return TMLPUUnifiedReconstruction(
        scalar, _tmlpu_v196_v188_curve_qbridge_euler())


def _tmlpu_v194_mlpu1_jump_sharpener():
    scalar = _ScalarMLPU1JumpSharpener(
        edge_min=float(os.environ.get('TMLPU_V194_EDGE_MIN', '0.28')),
        edge_full=float(os.environ.get('TMLPU_V194_EDGE_FULL', '0.58')),
        alpha=float(os.environ.get('TMLPU_V194_ALPHA', '1.0')),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v199_fast_scalar_edge_antidiff():
    scalar = _ScalarMLPU1EdgeAntidiffusive(
        edge_min=float(os.environ.get('TMLPU_V199_EDGE_MIN', '0.06')),
        edge_full=float(os.environ.get('TMLPU_V199_EDGE_FULL', '0.28')),
        alpha=float(os.environ.get('TMLPU_V199_ALPHA', '0.18')),
        clip_unit=_env_bool('TMLPU_V199_CLIP_UNIT', True),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v203_mlpu1_range_edge_antidiff():
    scalar = _ScalarMLPU1EdgeAntidiffusive(
        edge_min=float(os.environ.get('TMLPU_V203_EDGE_MIN', '0.08')),
        edge_full=float(os.environ.get('TMLPU_V203_EDGE_FULL', '0.32')),
        alpha=float(os.environ.get('TMLPU_V203_ALPHA', '0.10')),
        clip_unit=_env_bool('TMLPU_V203_CLIP_UNIT', True),
        local_range_min=float(os.environ.get(
            'TMLPU_V203_LOCAL_RANGE_MIN', '0.42')),
        local_range_full=float(os.environ.get(
            'TMLPU_V203_LOCAL_RANGE_FULL', '0.70')),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v197_fast_scalar_hancock_jump_damped():
    inner = TMLPU(
        tvd=os.environ.get('TMLPU_V197_SCALAR_TVD', 'pure_downwind'),
        mlp_bound=True,
        extremum_relax=False,
        tvb_M=0.0,
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=1.0,
        vertex_mlp_augment=True,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V197_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        face_increment='tmlpu',
        r_form='far_upwind',
        clamp_tstar=True,
        hancock_courant=float(os.environ.get(
            'TMLPU_V197_SCALAR_HANCOCK_COURANT', '0.25')),
        stencil='face',
        order=2,
        idw_p=6.0,
    )
    scalar = _ScalarJumpDampedBinaryGuardReconstruction(
        inner,
        jump_min=float(os.environ.get('TMLPU_V197_JUMP_MIN', '0.18')),
        jump_full=float(os.environ.get('TMLPU_V197_JUMP_FULL', '0.55')),
        min_theta=float(os.environ.get('TMLPU_V197_MIN_THETA', '0.90')),
        low_plateau=float(os.environ.get('TMLPU_V197_LOW_PLATEAU', '0.03')),
        high_plateau=float(os.environ.get('TMLPU_V197_HIGH_PLATEAU', '0.92')),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v198_fast_scalar_cell_balanced():
    inner = TMLPU(
        tvd=os.environ.get('TMLPU_V198_SCALAR_TVD', 'pure_downwind'),
        mlp_bound=True,
        extremum_relax=False,
        tvb_M=0.0,
        virtual_uu_gradient=True,
        vertex_mlp=True,
        vertex_mlp_cap=1.0,
        vertex_mlp_augment=True,
        face_skew_correction=True,
        face_gradient_correction=os.environ.get(
            'TMLPU_V198_SCALAR_FACE_GRADIENT_CORRECTION', 'jasak'),
        face_increment='tmlpu',
        r_form='far_upwind',
        clamp_tstar=True,
        stencil='face',
        order=2,
        idw_p=6.0,
    )
    damped = _ScalarJumpDampedBinaryGuardReconstruction(
        inner,
        jump_min=float(os.environ.get('TMLPU_V198_JUMP_MIN', '0.18')),
        jump_full=float(os.environ.get('TMLPU_V198_JUMP_FULL', '0.55')),
        min_theta=float(os.environ.get('TMLPU_V198_MIN_THETA', '0.90')),
        low_plateau=float(os.environ.get('TMLPU_V198_LOW_PLATEAU', '0.03')),
        high_plateau=float(os.environ.get('TMLPU_V198_HIGH_PLATEAU', '0.92')),
    )
    scalar = _ScalarCellBalancedReconstruction(
        damped,
        strength=float(os.environ.get('TMLPU_V198_BALANCE_STRENGTH', '0.50')),
        clip_bounds=_env_bool('TMLPU_V198_BALANCE_CLIP_BOUNDS', True),
    )
    return TMLPUUnifiedReconstruction(scalar, _tmlpu_v3_unified_euler())


def _tmlpu_v169_v167_qtight_core_euler():
    recon = _tmlpu_v167_v165_curve_restore_euler()
    recon.euler_tangential_signed_pair_tail_q_min = float(
        os.environ.get('TMLPU_V169_SIGNED_Q_MIN', '0.022'))
    recon.euler_tangential_signed_pair_tail_q_full = float(
        os.environ.get('TMLPU_V169_SIGNED_Q_FULL', '0.070'))
    recon.euler_tangential_density_curve_pair_tail_q_min = float(
        os.environ.get('TMLPU_V169_CURVE_Q_MIN', '0.020'))
    recon.euler_tangential_density_curve_pair_tail_q_full = float(
        os.environ.get('TMLPU_V169_CURVE_Q_FULL', '0.065'))
    recon.euler_tangential_signed_tail_hf_filter_on = False
    recon.euler_tangential_density_curve_tail_hf_filter_on = False
    return recon


def _tmlpu_v169_v167_qtight_core():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v169_v167_qtight_core_euler())


def _tmlpu_v170_v167_pressure_entropy_euler():
    recon = _tmlpu_v167_v165_curve_restore_euler()
    recon.euler_pressure_contact_entropy_blend = True
    recon.euler_pressure_contact_entropy_beta = float(
        os.environ.get('TMLPU_V170_PRESSURE_ENTROPY_BETA', '0.12'))
    recon.euler_pressure_contact_entropy_cap = float(
        os.environ.get('TMLPU_V170_PRESSURE_ENTROPY_CAP', '0.08'))
    recon.euler_pressure_contact_entropy_downscale = float(
        os.environ.get('TMLPU_V170_PRESSURE_ENTROPY_DOWNSCALE', '1.0'))
    recon.euler_pressure_contact_entropy_p_jump_threshold = float(
        os.environ.get('TMLPU_V170_PRESSURE_ENTROPY_P_THRESHOLD', '0.025'))
    recon.euler_pressure_contact_entropy_p_jump_width = float(
        os.environ.get('TMLPU_V170_PRESSURE_ENTROPY_P_WIDTH', '0.070'))
    recon.euler_pressure_contact_entropy_compression_threshold = float(
        os.environ.get('TMLPU_V170_PRESSURE_ENTROPY_C_THRESHOLD', '0.008'))
    recon.euler_pressure_contact_entropy_compression_width = float(
        os.environ.get('TMLPU_V170_PRESSURE_ENTROPY_C_WIDTH', '0.060'))
    recon.euler_pressure_contact_entropy_normality_threshold = float(
        os.environ.get('TMLPU_V170_PRESSURE_ENTROPY_N_THRESHOLD', '0.40'))
    recon.euler_pressure_contact_entropy_normality_width = float(
        os.environ.get('TMLPU_V170_PRESSURE_ENTROPY_N_WIDTH', '0.30'))
    return recon


def _tmlpu_v170_v167_pressure_entropy():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v170_v167_pressure_entropy_euler())


def _tmlpu_v171_v167_pressure_jump_limit_euler():
    recon = _tmlpu_v167_v165_curve_restore_euler()
    recon.euler_pressure_face_jump_limiter_on = True
    recon.euler_pressure_face_jump_limiter_strength = float(
        os.environ.get('TMLPU_V171_PRESSURE_JUMP_LIMIT_STRENGTH', '0.85'))
    recon.euler_pressure_face_jump_limiter_growth_cap = float(
        os.environ.get('TMLPU_V171_PRESSURE_JUMP_LIMIT_GROWTH_CAP', '0.10'))
    recon.euler_pressure_face_jump_limiter_abs_floor = float(
        os.environ.get('TMLPU_V171_PRESSURE_JUMP_LIMIT_ABS_FLOOR', '1e-10'))
    recon.euler_pressure_face_jump_limiter_p_jump_threshold = float(
        os.environ.get('TMLPU_V171_PRESSURE_JUMP_LIMIT_P_THRESHOLD', '0.025'))
    recon.euler_pressure_face_jump_limiter_p_jump_width = float(
        os.environ.get('TMLPU_V171_PRESSURE_JUMP_LIMIT_P_WIDTH', '0.070'))
    recon.euler_pressure_face_jump_limiter_compression_threshold = float(
        os.environ.get('TMLPU_V171_PRESSURE_JUMP_LIMIT_C_THRESHOLD', '0.008'))
    recon.euler_pressure_face_jump_limiter_compression_width = float(
        os.environ.get('TMLPU_V171_PRESSURE_JUMP_LIMIT_C_WIDTH', '0.060'))
    recon.euler_pressure_face_jump_limiter_normality_threshold = float(
        os.environ.get('TMLPU_V171_PRESSURE_JUMP_LIMIT_N_THRESHOLD', '0.40'))
    recon.euler_pressure_face_jump_limiter_normality_width = float(
        os.environ.get('TMLPU_V171_PRESSURE_JUMP_LIMIT_N_WIDTH', '0.30'))
    return recon


def _tmlpu_v171_v167_pressure_jump_limit():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v171_v167_pressure_jump_limit_euler())


def _tmlpu_v172_v167_soft_bridge_cut_euler():
    recon = _tmlpu_v167_v165_curve_restore_euler()
    recon.euler_tangential_signed_tail_bridge_cut_on = True
    recon.euler_tangential_signed_tail_bridge_cut_strength = float(
        os.environ.get('TMLPU_V172_BRIDGE_CUT_STRENGTH', '0.12'))
    recon.euler_tangential_signed_tail_bridge_cut_min_factor = float(
        os.environ.get('TMLPU_V172_BRIDGE_CUT_MIN_FACTOR', '0.82'))
    recon.euler_tangential_signed_tail_bridge_cut_q_min = float(
        os.environ.get('TMLPU_V172_BRIDGE_CUT_Q_MIN', '0.16'))
    recon.euler_tangential_signed_tail_bridge_cut_q_full = float(
        os.environ.get('TMLPU_V172_BRIDGE_CUT_Q_FULL', '0.34'))
    recon.euler_tangential_signed_tail_bridge_cut_contact_min = float(
        os.environ.get('TMLPU_V172_BRIDGE_CUT_CONTACT_MIN', '0.35'))
    recon.euler_tangential_signed_tail_bridge_cut_contact_full = float(
        os.environ.get('TMLPU_V172_BRIDGE_CUT_CONTACT_FULL', '0.72'))
    recon.euler_tangential_signed_tail_bridge_cut_omega_lo_pct = float(
        os.environ.get('TMLPU_V172_BRIDGE_CUT_OMEGA_LO_PCT', '78.0'))
    recon.euler_tangential_signed_tail_bridge_cut_omega_hi_pct = float(
        os.environ.get('TMLPU_V172_BRIDGE_CUT_OMEGA_HI_PCT', '97.0'))
    return recon


def _tmlpu_v172_v167_soft_bridge_cut():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v172_v167_soft_bridge_cut_euler())


def _tmlpu_v173_v167_curve_bridge_cut_euler():
    recon = _tmlpu_v172_v167_soft_bridge_cut_euler()
    recon.euler_tangential_density_curve_tail_bridge_cut_on = True
    recon.euler_tangential_density_curve_tail_bridge_cut_strength = float(
        os.environ.get('TMLPU_V173_CURVE_BRIDGE_CUT_STRENGTH', '0.18'))
    recon.euler_tangential_density_curve_tail_bridge_cut_min_factor = float(
        os.environ.get('TMLPU_V173_CURVE_BRIDGE_CUT_MIN_FACTOR', '0.78'))
    recon.euler_tangential_density_curve_tail_bridge_cut_q_min = float(
        os.environ.get('TMLPU_V173_CURVE_BRIDGE_CUT_Q_MIN', '0.14'))
    recon.euler_tangential_density_curve_tail_bridge_cut_q_full = float(
        os.environ.get('TMLPU_V173_CURVE_BRIDGE_CUT_Q_FULL', '0.32'))
    recon.euler_tangential_density_curve_tail_bridge_cut_contact_min = float(
        os.environ.get('TMLPU_V173_CURVE_BRIDGE_CUT_CONTACT_MIN', '0.30'))
    recon.euler_tangential_density_curve_tail_bridge_cut_contact_full = float(
        os.environ.get('TMLPU_V173_CURVE_BRIDGE_CUT_CONTACT_FULL', '0.68'))
    recon.euler_tangential_density_curve_tail_bridge_cut_omega_lo_pct = float(
        os.environ.get('TMLPU_V173_CURVE_BRIDGE_CUT_OMEGA_LO_PCT', '75.0'))
    recon.euler_tangential_density_curve_tail_bridge_cut_omega_hi_pct = float(
        os.environ.get('TMLPU_V173_CURVE_BRIDGE_CUT_OMEGA_HI_PCT', '96.0'))
    return recon


def _tmlpu_v173_v167_curve_bridge_cut():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v45_unified_scalar(),
        _tmlpu_v173_v167_curve_bridge_cut_euler())


def _tmlpu_v41_unified_euler():
    recon = _tmlpu_v40_unified_euler()
    recon.euler_patch_contact_shear_postpass_on = _env_bool(
        'TMLPU_V41_PATCH_ON', True)
    recon.euler_patch_contact_shear_neighbor_blend = float(
        os.environ.get('TMLPU_V41_PATCH_NEIGHBOR_BLEND', '0.30'))
    recon.euler_patch_contact_shear_entropy_alpha = float(
        os.environ.get('TMLPU_V41_ENTROPY_ALPHA', '0.030'))
    recon.euler_patch_contact_shear_tangential_alpha = float(
        os.environ.get('TMLPU_V41_TANGENTIAL_ALPHA', '0.018'))
    recon.euler_patch_contact_shear_entropy_cap = float(
        os.environ.get('TMLPU_V41_ENTROPY_CAP', '0.008'))
    recon.euler_patch_contact_shear_tangential_cap = float(
        os.environ.get('TMLPU_V41_TANGENTIAL_CAP', '0.020'))
    recon.euler_patch_contact_shear_tangential_wave_cap = float(
        os.environ.get('TMLPU_V41_TANGENTIAL_WAVE_CAP', '0.003'))
    recon.euler_patch_contact_shear_min_valid_neighbours = int(
        os.environ.get('TMLPU_V41_MIN_VALID_NEIGHBOURS', '2'))
    recon.euler_patch_contact_shear_pair_spacing_on = _env_bool(
        'TMLPU_V41_PAIR_SPACING_ON', True)
    recon.euler_patch_contact_shear_pair_spacing_beta = float(
        os.environ.get('TMLPU_V41_PAIR_SPACING_BETA', '0.35'))
    recon.euler_patch_contact_shear_gate_cap = float(
        os.environ.get('TMLPU_V41_GATE_CAP', '1.0'))
    recon.euler_patch_contact_shear_pressure_floor_factor = 0.86
    recon.euler_patch_contact_shear_pressure_margin_on = _env_bool(
        'TMLPU_V41_PRESSURE_MARGIN_ON', True)
    return recon


def _tmlpu_v41_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v41_unified_euler())


def _tmlpu_v42_unified_euler():
    recon = _tmlpu_v40_unified_euler()
    recon.euler_patch_contact_shear_postpass_on = _env_bool(
        'TMLPU_V42_PATCH_ON', True)
    recon.euler_patch_contact_shear_neighbor_blend = float(
        os.environ.get('TMLPU_V42_PATCH_NEIGHBOR_BLEND', '0.30'))
    recon.euler_patch_contact_shear_entropy_alpha = float(
        os.environ.get('TMLPU_V42_ENTROPY_ALPHA', '0.030'))
    recon.euler_patch_contact_shear_tangential_alpha = float(
        os.environ.get('TMLPU_V42_TANGENTIAL_ALPHA', '0.018'))
    recon.euler_patch_contact_shear_entropy_cap = float(
        os.environ.get('TMLPU_V42_ENTROPY_CAP', '0.008'))
    recon.euler_patch_contact_shear_tangential_cap = float(
        os.environ.get('TMLPU_V42_TANGENTIAL_CAP', '0.020'))
    recon.euler_patch_contact_shear_tangential_wave_cap = float(
        os.environ.get('TMLPU_V42_TANGENTIAL_WAVE_CAP', '0.003'))
    recon.euler_patch_contact_shear_min_valid_neighbours = int(
        os.environ.get('TMLPU_V42_MIN_VALID_NEIGHBOURS', '2'))
    recon.euler_patch_contact_shear_pair_spacing_on = _env_bool(
        'TMLPU_V42_PAIR_SPACING_ON', False)
    recon.euler_patch_contact_shear_pair_spacing_beta = 0.0
    recon.euler_patch_contact_shear_gate_cap = 1.0
    recon.euler_patch_contact_shear_pressure_margin_on = False
    recon.euler_patch_contact_shear_late_pressure_rollback_on = _env_bool(
        'TMLPU_V42_LATE_PRESSURE_ROLLBACK_ON', True)
    recon.euler_patch_contact_shear_p_floor_abs = float(
        os.environ.get('TMLPU_V42_P_FLOOR_ABS', '0.925'))
    recon.euler_patch_contact_shear_rho_floor_abs = float(
        os.environ.get('TMLPU_V42_RHO_FLOOR_ABS', '0.700'))
    recon.euler_patch_contact_shear_pressure_floor_factor = float(
        os.environ.get('TMLPU_V42_P_FLOOR_REL', '0.86'))
    recon.euler_patch_contact_shear_rho_floor_factor = float(
        os.environ.get('TMLPU_V42_RHO_FLOOR_REL', '0.72'))
    recon.euler_patch_contact_shear_tangential_rollback_theta = float(
        os.environ.get('TMLPU_V42_TANGENTIAL_ROLLBACK_THETA', '0.50'))
    return recon


def _tmlpu_v42_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v42_unified_euler())


def _tmlpu_v43_unified_euler():
    recon = _tmlpu_v40_unified_euler()
    recon.euler_patch_contact_shear_postpass_on = _env_bool(
        'TMLPU_V43_PATCH_ON', True)
    recon.euler_patch_contact_shear_neighbor_blend = float(
        os.environ.get('TMLPU_V43_PATCH_NEIGHBOR_BLEND', '0.34'))
    recon.euler_patch_contact_shear_entropy_alpha = float(
        os.environ.get('TMLPU_V43_ENTROPY_ALPHA', '0.040'))
    recon.euler_patch_contact_shear_tangential_alpha = float(
        os.environ.get('TMLPU_V43_TANGENTIAL_ALPHA', '0.026'))
    recon.euler_patch_contact_shear_entropy_cap = float(
        os.environ.get('TMLPU_V43_ENTROPY_CAP', '0.010'))
    recon.euler_patch_contact_shear_tangential_cap = float(
        os.environ.get('TMLPU_V43_TANGENTIAL_CAP', '0.026'))
    recon.euler_patch_contact_shear_tangential_wave_cap = float(
        os.environ.get('TMLPU_V43_TANGENTIAL_WAVE_CAP', '0.004'))
    recon.euler_patch_contact_shear_min_valid_neighbours = int(
        os.environ.get('TMLPU_V43_MIN_VALID_NEIGHBOURS', '2'))
    recon.euler_patch_contact_shear_pair_spacing_on = _env_bool(
        'TMLPU_V43_PAIR_SPACING_ON', False)
    recon.euler_patch_contact_shear_pair_spacing_beta = float(
        os.environ.get('TMLPU_V43_PAIR_SPACING_BETA', '0.0'))
    recon.euler_patch_contact_shear_gate_cap = float(
        os.environ.get('TMLPU_V43_GATE_CAP', '1.0'))
    recon.euler_patch_contact_shear_pressure_margin_on = _env_bool(
        'TMLPU_V43_PRESSURE_MARGIN_ON', False)
    recon.euler_patch_contact_shear_late_pressure_rollback_on = _env_bool(
        'TMLPU_V43_LATE_PRESSURE_ROLLBACK_ON', True)
    recon.euler_patch_contact_shear_p_floor_abs = float(
        os.environ.get('TMLPU_V43_P_FLOOR_ABS', '0.900'))
    recon.euler_patch_contact_shear_rho_floor_abs = float(
        os.environ.get('TMLPU_V43_RHO_FLOOR_ABS', '0.660'))
    recon.euler_patch_contact_shear_pressure_floor_factor = float(
        os.environ.get('TMLPU_V43_P_FLOOR_REL', '0.84'))
    recon.euler_patch_contact_shear_rho_floor_factor = float(
        os.environ.get('TMLPU_V43_RHO_FLOOR_REL', '0.70'))
    recon.euler_patch_contact_shear_tangential_rollback_theta = float(
        os.environ.get('TMLPU_V43_TANGENTIAL_ROLLBACK_THETA', '0.55'))
    return recon


def _tmlpu_v43_unified():
    return TMLPUUnifiedReconstruction(
        _tmlpu_v3_unified_scalar(), _tmlpu_v43_unified_euler())


def _tmlpu_off_leveque():
    return TMLPU(tvd='pure_downwind', mlp_bound=False,
                 extremum_relax=False, tvb_M=0.0,
                 virtual_uu_gradient=True, stencil='vertex',
                 order=1, idw_p=6.0)


def _tmlpu_euler(idw_p=6.0):
    def _env_bool(name, default='0'):
        return os.environ.get(name, default).lower() in ('1', 'true', 'yes', 'on')
    zero_delta_psi = float(os.environ.get('TMLPU_ZERO_DELTA_PSI', '2.0'))

    # Clean T-MLP-u per user-specified theory: ONE limiter, NO per-variable
    # overrides, NO Hancock, NO shock-flatten, NO acoustic-flatten. Strictly:
    #   ? = max(0, min(α·r, α, ?_TVD)), ?_TVD chosen consistently
    #   skewness-corrected ?��? virtual far-upwind, cell-vertex LMP bound.
    if _env_bool('TMLPU_EULER_CLEAN'):
        clean_tvd = os.environ.get('TMLPU_EULER_CLEAN_TVD', 'van_leer')
        return TMLPU(
            tvd=clean_tvd,
            hancock_courant=0.0,
            mlp_bound=True,
            extremum_relax=False,
            extremum_relax_curved_otsu=False,
            tvb_M=0.0,
            vertex_mlp=True,
            vertex_mlp_cap=2.0,
            euler_shock_flatten=False,
            euler_density_acoustic_flatten=False,
            euler_density_tvd=None,
            euler_density_lsq_increment=False,
            euler_density_no_hancock=False,
            euler_density_entropy_split=False,
            euler_density_entropy_variable=False,
            euler_density_shear_contact=False,
            euler_density_contact_wave_hancock=False,
            euler_density_pressure_entropy=False,
            euler_density_contact_bvd=False,
            euler_density_contact_cell_bvd=False,
            euler_density_first_order=False,
            euler_pressure_first_order=False,
            euler_velocity_no_hancock=False,
            euler_velocity_shock_flatten=False,
            euler_velocity_lsq_increment=False,
            euler_velocity_tvd=None,
            euler_tangential_velocity_tvd=None,
            euler_tangential_velocity_no_hancock=False,
            euler_tangential_contact_wave_hancock=False,
            euler_tangential_velocity_lsq_increment=False,
            euler_local_hancock=False,
            euler_log_positive=False,
            virtual_uu_gradient=True,
            vertex_mlp_augment=_env_bool('TMLPU_EULER_VERTEX_MLP_AUGMENT'),
            face_skew_correction=True,
            phi_LL_unclipped=_env_bool('TMLPU_PHI_LL_UNCLIPPED'),
            zero_delta_psi=zero_delta_psi,
            stencil='vertex',
            order=2,
            idw_p=idw_p,
        )
    density_tvd = os.environ.get('TMLPU_EULER_DENSITY_TVD', 'umist')
    density_contact_wave = _env_bool('TMLPU_EULER_DENSITY_CONTACT_WAVE')
    density_entropy_split = _env_bool('TMLPU_EULER_DENSITY_ENTROPY_SPLIT')
    density_lsq_increment = _env_bool('TMLPU_EULER_DENSITY_LSQ_INCREMENT')
    density_full_lsq_increment = _env_bool(
        'TMLPU_EULER_DENSITY_FULL_LSQ_INCREMENT')
    density_no_hancock = _env_bool('TMLPU_EULER_DENSITY_NO_HANCOCK')
    density_pressure_entropy = _env_bool('TMLPU_EULER_DENSITY_PRESSURE_ENTROPY')
    density_entropy_variable = _env_bool(
        'TMLPU_EULER_DENSITY_ENTROPY_VARIABLE')
    density_shear_contact = _env_bool(
        'TMLPU_EULER_DENSITY_SHEAR_CONTACT')
    density_contact_bvd = _env_bool('TMLPU_EULER_DENSITY_CONTACT_BVD')
    density_contact_cell_bvd = _env_bool('TMLPU_EULER_DENSITY_CONTACT_CELL_BVD')
    density_first_order = _env_bool('TMLPU_EULER_DENSITY_FIRST_ORDER')
    euler_log_positive = _env_bool('TMLPU_EULER_LOG_POSITIVE')
    euler_log_pressure_only = _env_bool('TMLPU_EULER_LOG_PRESSURE_ONLY')
    pressure_first_order = _env_bool('TMLPU_EULER_PRESSURE_FIRST_ORDER')
    pressure_shear_lsq_increment = _env_bool(
        'TMLPU_EULER_PRESSURE_SHEAR_LSQ_INCREMENT')
    pressure_nonshock_lsq_increment = _env_bool(
        'TMLPU_EULER_PRESSURE_NONSHOCK_LSQ_INCREMENT')
    extremum_relax_curved = _env_bool(
        'TMLPU_EULER_EXTREMUM_RELAX_CURVED_OTSU')
    velocity_no_hancock = _env_bool('TMLPU_EULER_VELOCITY_NO_HANCOCK')
    velocity_shock_flatten = _env_bool('TMLPU_EULER_VELOCITY_SHOCK_FLATTEN')
    density_extrema_lmp = _env_bool('TMLPU_EULER_DENSITY_EXTREMA_LMP')
    velocity_extrema_lmp = _env_bool('TMLPU_EULER_VELOCITY_EXTREMA_LMP')
    velocity_lsq_increment = _env_bool(
        'TMLPU_EULER_VELOCITY_LSQ_INCREMENT', '1')
    tangential_no_hancock = _env_bool('TMLPU_EULER_TANGENTIAL_NO_HANCOCK')
    tangential_lsq_increment = _env_bool('TMLPU_EULER_TANGENTIAL_LSQ_INCREMENT')
    velocity_tvd = os.environ.get('TMLPU_EULER_VELOCITY_TVD', 'umist')
    tangential_velocity_tvd = os.environ.get('TMLPU_EULER_TANGENTIAL_TVD')
    if tangential_velocity_tvd is not None:
        tangential_velocity_tvd = tangential_velocity_tvd.strip()
        if tangential_velocity_tvd.lower() in ('', 'none', 'off', 'false', '0'):
            tangential_velocity_tvd = None
    base_tvd = os.environ.get('TMLPU_EULER_BASE_TVD', 'minmod')
    face_skew_correction = _env_bool('TMLPU_FACE_SKEW_CORRECTION', '1')
    face_gradient_correction = os.environ.get(
        'TMLPU_FACE_GRADIENT_CORRECTION', 'beta')
    face_increment = os.environ.get('TMLPU_FACE_INCREMENT', 'tmlpu')
    r_form = os.environ.get('TMLPU_R_FORM', 'far_upwind')
    stencil = os.environ.get('TMLPU_STENCIL', 'vertex')
    vertex_mlp_augment = _env_bool('TMLPU_EULER_VERTEX_MLP_AUGMENT')
    hancock_courant = float(os.environ.get(
        'TMLPU_EULER_HANCOCK_COURANT', '0.4'))
    return TMLPU(tvd=base_tvd, hancock_courant=hancock_courant,
                 mlp_bound=True, extremum_relax=False,
                 extremum_relax_curved_otsu=extremum_relax_curved,
                 tvb_M=0.0, vertex_mlp=True, vertex_mlp_cap=1.0,
                 euler_shock_flatten=True,
                 euler_density_acoustic_flatten=_env_bool(
                     'TMLPU_EULER_DENSITY_ACOUSTIC_FLATTEN', '1'),
                 euler_density_tvd=density_tvd,
                 euler_density_lsq_increment=density_lsq_increment,
                 euler_density_full_lsq_increment=density_full_lsq_increment,
                 euler_density_no_hancock=density_no_hancock,
                 euler_density_entropy_split=density_entropy_split,
                 euler_density_entropy_variable=density_entropy_variable,
                 euler_density_shear_contact=density_shear_contact,
                 euler_density_contact_wave_hancock=density_contact_wave,
                 euler_density_pressure_entropy=density_pressure_entropy,
                 euler_density_contact_bvd=density_contact_bvd,
                 euler_density_contact_cell_bvd=density_contact_cell_bvd,
                 euler_density_first_order=density_first_order,
                 euler_pressure_first_order=pressure_first_order,
                 euler_pressure_shear_lsq_increment=pressure_shear_lsq_increment,
                 euler_pressure_nonshock_lsq_increment=pressure_nonshock_lsq_increment,
                 euler_velocity_no_hancock=velocity_no_hancock,
                 euler_velocity_shock_flatten=velocity_shock_flatten,
                 euler_density_extrema_lmp=density_extrema_lmp,
                 euler_velocity_extrema_lmp=velocity_extrema_lmp,
                 euler_velocity_lsq_increment=velocity_lsq_increment,
                 euler_velocity_tvd=velocity_tvd,
                 euler_tangential_velocity_tvd=tangential_velocity_tvd,
                 euler_tangential_velocity_no_hancock=tangential_no_hancock,
                 euler_tangential_contact_wave_hancock=True,
                 euler_tangential_velocity_lsq_increment=tangential_lsq_increment,
                 euler_local_hancock=False,
                 euler_log_positive=euler_log_positive,
                 euler_log_pressure_only=euler_log_pressure_only,
                 virtual_uu_gradient=True,
                 vertex_mlp_augment=vertex_mlp_augment,
                 face_skew_correction=face_skew_correction,
                 face_gradient_correction=face_gradient_correction,
                 face_increment=face_increment,
                 r_form=r_form,
                 zero_delta_psi=zero_delta_psi,
                 stencil=stencil, order=2,
                 idw_p=idw_p)


class EulerVelocitySmoothSharpBVD:
    """Euler primitive reconstruction with BVD only on velocity variables.

    rho and pressure are taken from a base TMLP-u candidate.  The velocity
    components are selected cell-wise between a smooth TMLP-u candidate and a
    sharp TMLP-u candidate using TBV plus a scalar centroid-distance proxy.
    This keeps the selection cheap and avoids the full centroid/moment BVD
    machinery used by the scalar LeVeque search class.
    """

    def __init__(self, *, idw_p=6.0):
        euler_log_positive = os.environ.get(
            'TMLPU_EULER_LOG_POSITIVE', '0').lower() in (
                '1', 'true', 'yes', 'on')
        face_gradient_correction = os.environ.get(
            'TMLPU_FACE_GRADIENT_CORRECTION', 'jasak')
        common = dict(
            hancock_courant=0.0,
            mlp_bound=True,
            extremum_relax=False,
            tvb_M=0.0,
            vertex_mlp=True,
            vertex_mlp_cap=2.0,
            euler_shock_flatten=True,
            euler_density_acoustic_flatten=True,
            euler_density_first_order=False,
            euler_pressure_first_order=False,
            euler_velocity_no_hancock=True,
            euler_velocity_shock_flatten=True,
            euler_tangential_velocity_no_hancock=True,
            euler_local_hancock=False,
            euler_log_positive=euler_log_positive,
            virtual_uu_gradient=True,
            face_skew_correction=True,
            face_gradient_correction=face_gradient_correction,
            stencil='vertex',
            order=2,
            idw_p=idw_p,
        )
        self.base = TMLPU(
            tvd='minmod',
            euler_density_tvd='minmod',
            euler_velocity_tvd='minmod',
            euler_tangential_velocity_tvd='minmod',
            euler_velocity_lsq_increment=True,
            euler_tangential_velocity_lsq_increment=True,
            face_increment='tmlpu',
            active_vars=(0, 3),
            **common,
        )
        self.smooth = TMLPU(
            tvd='bounded_cd',
            euler_density_tvd='minmod',
            euler_velocity_tvd='bounded_cd',
            euler_tangential_velocity_tvd='bounded_cd',
            euler_velocity_lsq_increment=True,
            euler_tangential_velocity_lsq_increment=True,
            face_increment='lsq',
            active_vars=(1, 2),
            **common,
        )
        self.sharp = TMLPU(
            tvd='superbee',
            euler_density_tvd='minmod',
            euler_velocity_tvd='superbee',
            euler_tangential_velocity_tvd='superbee',
            euler_velocity_lsq_increment=False,
            euler_tangential_velocity_lsq_increment=False,
            face_increment='tmlpu',
            active_vars=(1, 2),
            **common,
        )
        self._bvd_geom_cache = {}
        self._bvd_scratch = {}

    def set_timestep_context(self, dt, *, total_dt=None, quad_weight=None,
                             quad_points=None, quad_weights=None):
        for recon in (self.base, self.smooth, self.sharp):
            if hasattr(recon, 'set_timestep_context'):
                recon.set_timestep_context(
                    dt, total_dt=total_dt, quad_weight=quad_weight,
                    quad_points=quad_points, quad_weights=quad_weights)

    def _get_scratch(self, n_cells):
        scratch = self._bvd_scratch.get(n_cells)
        if scratch is not None:
            return scratch
        scratch = {
            'tbv_s_u': np.zeros(n_cells, dtype=np.float64),
            'tbv_h_u': np.zeros(n_cells, dtype=np.float64),
            'cp_s_u':  np.zeros(n_cells, dtype=np.float64),
            'cp_h_u':  np.zeros(n_cells, dtype=np.float64),
            'tbv_s_v': np.zeros(n_cells, dtype=np.float64),
            'tbv_h_v': np.zeros(n_cells, dtype=np.float64),
            'cp_s_v':  np.zeros(n_cells, dtype=np.float64),
            'cp_h_v':  np.zeros(n_cells, dtype=np.float64),
            'use_sharp_u': np.zeros(n_cells, dtype=np.bool_),
            'use_sharp_v': np.zeros(n_cells, dtype=np.bool_),
        }
        self._bvd_scratch[n_cells] = scratch
        return scratch

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        old_prep_cache = getattr(mesh, '_tmlpu_transient_recon_prep_cache',
                                 None)
        mesh._tmlpu_transient_recon_prep_cache = {'W_cell': W_cell}
        try:
            base_L, base_R = self.base.reconstruct(
                mesh, W_cell, eq, eval_points=eval_points)
            smooth_L, smooth_R = self.smooth.reconstruct(
                mesh, W_cell, eq, eval_points=eval_points)
            sharp_L, sharp_R = self.sharp.reconstruct(
                mesh, W_cell, eq, eval_points=eval_points)
        finally:
            if old_prep_cache is None:
                try:
                    del mesh._tmlpu_transient_recon_prep_cache
                except AttributeError:
                    pass
            else:
                mesh._tmlpu_transient_recon_prep_cache = old_prep_cache

        owner = mesh.face_owner
        nei = mesh.face_neighbour
        points = mesh.face_centers if eval_points is None else eval_points
        points_token = 0 if np.shares_memory(points, mesh.face_centers) else id(points)
        cache_key = (id(mesh), points_token)
        geom = self._bvd_geom_cache.get(cache_key)
        if geom is None:
            interior = np.ascontiguousarray(
                np.where(nei >= 0)[0], dtype=np.int64)
            o_idx = np.ascontiguousarray(owner[interior], dtype=np.int64)
            n_idx = np.ascontiguousarray(nei[interior], dtype=np.int64)
            face_points = points[interior]
            cell_h = np.sqrt(np.maximum(mesh.cell_volumes, 0.0))
            cell_h = np.maximum(cell_h, np.finfo(float).eps)
            d_owner = np.ascontiguousarray(
                np.linalg.norm(face_points - mesh.cell_centers[o_idx], axis=1)
                / cell_h[o_idx], dtype=np.float64)
            d_nei = np.ascontiguousarray(
                np.linalg.norm(face_points - mesh.cell_centers[n_idx], axis=1)
                / cell_h[n_idx], dtype=np.float64)
            boundary_idx = np.ascontiguousarray(
                np.where(nei < 0)[0], dtype=np.int64)
            boundary_owner = (np.ascontiguousarray(owner[boundary_idx],
                                                   dtype=np.int64)
                              if boundary_idx.size else None)
            geom = (interior, o_idx, n_idx, d_owner, d_nei,
                    boundary_idx, boundary_owner)
            self._bvd_geom_cache[cache_key] = geom
        (interior, o_idx, n_idx, d_owner, d_nei,
         boundary_idx, boundary_owner) = geom
        if interior.size == 0 or W_cell.shape[0] < 3:
            return base_L, base_R

        W_L = base_L
        W_R = base_R
        n_cells = W_cell.shape[1]
        sc = self._get_scratch(n_cells)

        if _NUMBA_AVAILABLE:
            _bvd_select_uv_apply_kernel(
                interior, o_idx, n_idx, d_owner, d_nei,
                smooth_L, smooth_R, sharp_L, sharp_R,
                sc['tbv_s_u'], sc['tbv_h_u'], sc['cp_s_u'], sc['cp_h_u'],
                sc['tbv_s_v'], sc['tbv_h_v'], sc['cp_s_v'], sc['cp_h_v'],
                sc['use_sharp_u'], sc['use_sharp_v'],
                W_L, W_R,
            )
        else:
            for vv, use_sharp in ((1, sc['use_sharp_u']),
                                  (2, sc['use_sharp_v'])):
                jump_s = np.abs(
                    smooth_L[vv, interior] - smooth_R[vv, interior])
                jump_h = np.abs(
                    sharp_L[vv, interior] - sharp_R[vv, interior])
                tbv_s = np.zeros(n_cells, dtype=float)
                tbv_h = np.zeros(n_cells, dtype=float)
                cp_s = np.zeros(n_cells, dtype=float)
                cp_h = np.zeros(n_cells, dtype=float)
                np.add.at(tbv_s, o_idx, jump_s)
                np.add.at(tbv_s, n_idx, jump_s)
                np.add.at(tbv_h, o_idx, jump_h)
                np.add.at(tbv_h, n_idx, jump_h)
                np.add.at(cp_s, o_idx, jump_s * d_owner)
                np.add.at(cp_s, n_idx, jump_s * d_nei)
                np.add.at(cp_h, o_idx, jump_h * d_owner)
                np.add.at(cp_h, n_idx, jump_h * d_nei)
                use_sharp[:] = np.hypot(tbv_h, cp_h) < np.hypot(tbv_s, cp_s)
            for vv, use_sharp in ((1, sc['use_sharp_u']),
                                  (2, sc['use_sharp_v'])):
                use_face_sharp = use_sharp[o_idx] & use_sharp[n_idx]
                W_L[vv, interior] = np.where(
                    use_face_sharp,
                    sharp_L[vv, interior], smooth_L[vv, interior])
                W_R[vv, interior] = np.where(
                    use_face_sharp,
                    sharp_R[vv, interior], smooth_R[vv, interior])

        if boundary_idx is not None and boundary_idx.size:
            W_L[:, boundary_idx] = W_cell[:, boundary_owner]
            W_R[:, boundary_idx] = W_cell[:, boundary_owner]
        return W_L, W_R


def _tmlpu_euler_velocity_smooth_sharp_bvd():
    return EulerVelocitySmoothSharpBVD(idw_p=1.0)


def _tmlpu_double_mach():
    return _tmlpu_euler(idw_p=0.0)


def _tmlpu_mach3_step():
    return _tmlpu_euler(idw_p=6.0)


def _tmlpu_off_euler():
    return TMLPU(tvd='modified_superbee', mlp_bound=False, extremum_relax=False,
                 tvb_M=0.0, vertex_mlp=False,
                 virtual_uu_gradient=True, stencil='face', order=1)


def _comparison_specs(kind):
    if kind == 'leveque':
        t_off = 'tmlpu_leveque_off'
    elif kind == 'double_mach':
        t_off = 'tmlpu_euler_off'
    elif kind == 'mach3_step':
        t_off = 'tmlpu_euler_off'
    else:
        t_off = 'tmlpu_euler_off'
    t_on = os.environ.get('TMLPU_COMMON_RECON_KEY', 'tmlpu_v3_unified_on')
    if os.environ.get('TMLPU_INCLUDE_DIAGNOSTIC_BASELINES', '0').lower() in (
            '1', 'true', 'yes', 'on'):
        return [
            ('first_order', 'first_order'),
            ('Barth-Jespersen', 'barth_jespersen'),
            ('Venkatakrishnan', 'venkatakrishnan'),
            ('MLP-u1', 'mlp_u1'),
            ('MLP-u2', 'mlp_u2'),
            ('T-MLP-u OFF', t_off),
            ('T-MLP-u ON', t_on),
        ]
    return [
        ('MLP-u1', 'mlp_u1'),
        ('T-MLP-u ON', t_on),
    ]


def _diagnostic_comparison_specs(kind):
    if kind == 'leveque':
        t_off = 'tmlpu_leveque_off'
    else:
        t_off = 'tmlpu_euler_off'
    return [
        ('first_order', 'first_order'),
        ('Barth-Jespersen', 'barth_jespersen'),
        ('Venkatakrishnan', 'venkatakrishnan'),
        ('MLP-u1', 'mlp_u1'),
        ('MLP-u2', 'mlp_u2'),
        ('T-MLP-u OFF', t_off),
        ('T-MLP-u ON', os.environ.get('TMLPU_COMMON_RECON_KEY',
                                      'tmlpu_v3_unified_on')),
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
    if key == 'mlp_u1_tmlpu_contact':
        return MLPU1TMLPUContact(
            contact_compress=float(os.environ.get(
                'TMLPU_CONTACT_COMPRESS', '0.5')),
            contact_compress_rho_lo=float(os.environ.get(
                'TMLPU_CONTACT_RHO_LO', '0.06')),
            contact_compress_rho_hi=float(os.environ.get(
                'TMLPU_CONTACT_RHO_HI', '0.30')),
            contact_compress_p_tol=float(os.environ.get(
                'TMLPU_CONTACT_P_TOL', '0.18')),
        )
    if key == 'mlp_u2':
        return MLPU2()
    if key in ('tmlpu_v3_unified_on', 'tmlpu_leveque_on',
               'tmlpu_double_mach_on', 'tmlpu_mach3_step_on'):
        return _tmlpu_v3_unified()
    if key == 'tmlpu_v4_unified_on':
        return _tmlpu_v4_unified()
    if key == 'tmlpu_v4_1_unified_on':
        return _tmlpu_v4_1_unified()
    if key == 'tmlpu_v4_2_unified_on':
        return _tmlpu_v4_2_unified()
    if key == 'tmlpu_v4_3_unified_on':
        return _tmlpu_v4_3_unified()
    if key == 'tmlpu_v5_unified_on':
        return _tmlpu_v5_unified()
    if key == 'tmlpu_v6_unified_on':
        return _tmlpu_v6_unified()
    if key == 'tmlpu_v7_unified_on':
        return _tmlpu_v7_unified()
    if key == 'tmlpu_v8_unified_on':
        return _tmlpu_v8_unified()
    if key == 'tmlpu_v9_unified_on':
        return _tmlpu_v9_unified()
    if key == 'tmlpu_v10_unified_on':
        return _tmlpu_v10_unified()
    if key == 'tmlpu_v11_unified_on':
        return _tmlpu_v11_unified()
    if key == 'tmlpu_v12_unified_on':
        return _tmlpu_v12_unified()
    if key == 'tmlpu_v13_unified_on':
        return _tmlpu_v13_unified()
    if key == 'tmlpu_v14_unified_on':
        return _tmlpu_v14_unified()
    if key == 'tmlpu_v15_unified_on':
        return _tmlpu_v15_unified()
    if key == 'tmlpu_v16_unified_on':
        return _tmlpu_v16_unified()
    if key == 'tmlpu_v17_unified_on':
        return _tmlpu_v17_unified()
    if key == 'tmlpu_v18_unified_on':
        return _tmlpu_v18_unified()
    if key == 'tmlpu_v19_unified_on':
        return _tmlpu_v19_unified()
    if key == 'tmlpu_v20_unified_on':
        return _tmlpu_v20_unified()
    if key == 'tmlpu_v21_unified_on':
        return _tmlpu_v21_unified()
    if key == 'tmlpu_v22_unified_on':
        return _tmlpu_v22_unified()
    if key == 'tmlpu_v23_unified_on':
        return _tmlpu_v23_unified()
    if key == 'tmlpu_v24_unified_on':
        return _tmlpu_v24_unified()
    if key == 'tmlpu_v25_unified_on':
        return _tmlpu_v25_unified()
    if key == 'tmlpu_v26_unified_on':
        return _tmlpu_v26_unified()
    if key == 'tmlpu_v27_unified_on':
        return _tmlpu_v27_unified()
    if key == 'tmlpu_v28_unified_on':
        return _tmlpu_v28_unified()
    if key == 'tmlpu_v29_unified_on':
        return _tmlpu_v29_unified()
    if key == 'tmlpu_v30_unified_on':
        return _tmlpu_v30_unified()
    if key == 'tmlpu_v31_unified_on':
        return _tmlpu_v31_unified()
    if key == 'tmlpu_v32_unified_on':
        return _tmlpu_v32_unified()
    if key == 'tmlpu_v33_unified_on':
        return _tmlpu_v33_unified()
    if key == 'tmlpu_v34_unified_on':
        return _tmlpu_v34_unified()
    if key == 'tmlpu_v35_unified_on':
        return _tmlpu_v35_unified()
    if key == 'tmlpu_v36_unified_on':
        return _tmlpu_v36_unified()
    if key == 'tmlpu_v37_unified_on':
        return _tmlpu_v37_unified()
    if key == 'tmlpu_v38_unified_on':
        return _tmlpu_v38_unified()
    if key == 'tmlpu_v39_unified_on':
        return _tmlpu_v39_unified()
    if key == 'tmlpu_v40_unified_on':
        return _tmlpu_v40_unified()
    if key == 'tmlpu_v41_unified_on':
        return _tmlpu_v41_unified()
    if key == 'tmlpu_v42_unified_on':
        return _tmlpu_v42_unified()
    if key == 'tmlpu_v43_unified_on':
        return _tmlpu_v43_unified()
    if key == 'tmlpu_v45_unified_on':
        return _tmlpu_v45_unified()
    if key == 'tmlpu_v46_unified_on':
        return _tmlpu_v46_unified()
    if key == 'tmlpu_v47_unified_on':
        return _tmlpu_v47_unified()
    if key == 'tmlpu_v48_unified_on':
        return _tmlpu_v48_unified()
    if key == 'tmlpu_v49_unified_on':
        return _tmlpu_v49_unified()
    if key == 'tmlpu_v50_unified_on':
        return _tmlpu_v50_unified()
    if key == 'tmlpu_v51_unified_on':
        return _tmlpu_v51_unified()
    if key == 'tmlpu_v52_unified_on':
        return _tmlpu_v52_unified()
    if key == 'tmlpu_v53_unified_on':
        return _tmlpu_v53_unified()
    if key == 'tmlpu_v54_unified_on':
        return _tmlpu_v54_unified()
    if key == 'tmlpu_v55_unified_on':
        return _tmlpu_v55_unified()
    if key == 'tmlpu_v56_unified_on':
        return _tmlpu_v56_unified()
    if key == 'tmlpu_v57_unified_on':
        return _tmlpu_v57_unified()
    if key == 'tmlpu_v58_unified_on':
        return _tmlpu_v58_unified()
    if key == 'tmlpu_v59_unified_on':
        return _tmlpu_v59_unified()
    if key == 'tmlpu_v60_unified_on':
        return _tmlpu_v60_unified()
    if key == 'tmlpu_v61_unified_on':
        return _tmlpu_v61_unified()
    if key == 'tmlpu_v62_unified_on':
        return _tmlpu_v62_unified()
    if key == 'tmlpu_v63_unified_on':
        return _tmlpu_v63_unified()
    if key == 'tmlpu_v64_unified_on':
        return _tmlpu_v64_unified()
    if key == 'tmlpu_v65_unified_on':
        return _tmlpu_v65_unified()
    if key == 'tmlpu_v66_unified_on':
        return _tmlpu_v66_unified()
    if key == 'tmlpu_v67_unified_on':
        return _tmlpu_v67_unified()
    if key == 'tmlpu_v68_unified_on':
        return _tmlpu_v68_unified()
    if key == 'tmlpu_v69_unified_on':
        return _tmlpu_v69_unified()
    if key == 'tmlpu_v70_unified_on':
        return _tmlpu_v70_unified()
    if key == 'tmlpu_v71_unified_on':
        return _tmlpu_v71_unified()
    if key == 'tmlpu_v72_unified_on':
        return _tmlpu_v72_unified()
    if key == 'tmlpu_v73_unified_on':
        return _tmlpu_v73_unified()
    if key == 'tmlpu_v74_unified_on':
        return _tmlpu_v74_unified()
    if key == 'tmlpu_v75_unified_on':
        return _tmlpu_v75_unified()
    if key == 'tmlpu_v76_unified_on':
        return _tmlpu_v76_unified()
    if key == 'tmlpu_v77_unified_on':
        return _tmlpu_v77_unified()
    if key == 'tmlpu_v78_unified_on':
        return _tmlpu_v78_unified()
    if key == 'tmlpu_v79_unified_on':
        return _tmlpu_v79_unified()
    if key == 'tmlpu_v80_unified_on':
        return _tmlpu_v80_unified()
    if key == 'tmlpu_v81_unified_on':
        return _tmlpu_v81_unified()
    if key == 'tmlpu_v82_unified_on':
        return _tmlpu_v82_unified()
    if key == 'tmlpu_v83_unified_on':
        return _tmlpu_v83_unified()
    if key == 'tmlpu_v84_unified_on':
        return _tmlpu_v84_unified()
    if key == 'tmlpu_v85_unified_on':
        return _tmlpu_v85_unified()
    if key == 'tmlpu_v86_unified_on':
        return _tmlpu_v86_unified()
    if key == 'tmlpu_v87_unified_on':
        return _tmlpu_v87_unified()
    if key == 'tmlpu_v88_unified_on':
        return _tmlpu_v88_unified()
    if key == 'tmlpu_v89_unified_on':
        return _tmlpu_v89_unified()
    if key == 'tmlpu_v90_unified_on':
        return _tmlpu_v90_unified()
    if key == 'tmlpu_v91_unified_on':
        return _tmlpu_v91_unified()
    if key == 'tmlpu_v92_unified_on':
        return _tmlpu_v92_unified()
    if key == 'tmlpu_v93_diag_unified_on':
        return _tmlpu_v93_diag_unified()
    if key == 'tmlpu_v95_legacy_pair_target_on':
        return _tmlpu_v95_legacy_pair_target()
    if key == 'tmlpu_v96_legacy_pair_target_half_on':
        return _tmlpu_v96_legacy_pair_target_half()
    if key == 'tmlpu_v97_legacy_pair_target_075_on':
        return _tmlpu_v97_legacy_pair_target_075()
    if key == 'tmlpu_v98_legacy_pair_target_0875_on':
        return _tmlpu_v98_legacy_pair_target_0875()
    if key == 'tmlpu_v99_split_legacy_target_on':
        return _tmlpu_v99_split_legacy_target()
    if key == 'tmlpu_v100_safe_legacy_target_on':
        return _tmlpu_v100_safe_legacy_target()
    if key == 'tmlpu_v101_safe_legacy_pressure014_on':
        return _tmlpu_v101_safe_legacy_pressure014()
    if key == 'tmlpu_v102_safe_legacy_contact035_on':
        return _tmlpu_v102_safe_legacy_contact035()
    if key == 'tmlpu_v103_safe_legacy_shear072_on':
        return _tmlpu_v103_safe_legacy_shear072()
    if key == 'tmlpu_v104_safe_legacy_shear072_norm020_on':
        return _tmlpu_v104_safe_legacy_shear072_norm020()
    if key == 'tmlpu_v105_safe_legacy_coherence_on':
        return _tmlpu_v105_safe_legacy_coherence()
    if key == 'tmlpu_v106_safe_legacy_qcurv_on':
        return _tmlpu_v106_safe_legacy_qcurv()
    if key == 'tmlpu_v107_safe_legacy_capboost_on':
        return _tmlpu_v107_safe_legacy_capboost()
    if key == 'tmlpu_v108_signed_density_support018_on':
        return _tmlpu_v108_signed_density_support018()
    if key == 'tmlpu_v109_tail_density_support_mid_on':
        return _tmlpu_v109_tail_density_support_mid()
    if key == 'tmlpu_v110_tail_density_shockdamp_on':
        return _tmlpu_v110_tail_density_shockdamp()
    if key == 'tmlpu_v111_tail_density_min015_on':
        return _tmlpu_v111_tail_density_min015()
    if key == 'tmlpu_v112_tail_density_min0145_on':
        return _tmlpu_v112_tail_density_min0145()
    if key == 'tmlpu_v113_tail_density_min015_full055_on':
        return _tmlpu_v113_tail_density_min015_full055()
    if key == 'tmlpu_v114_tail_density_beta115_on':
        return _tmlpu_v114_tail_density_beta115()
    if key == 'tmlpu_v115_v111_taildiag_on':
        return _tmlpu_v115_v111_taildiag()
    if key == 'tmlpu_v116_tail_safe_floor_on':
        return _tmlpu_v116_tail_safe_floor()
    if key == 'tmlpu_v117_v111_featurediag_on':
        return _tmlpu_v117_v111_featurediag()
    if key == 'tmlpu_v118_shear_contact_relief_on':
        return _tmlpu_v118_shear_contact_relief()
    if key == 'tmlpu_v119_shear_contact_relief_floor04_on':
        return _tmlpu_v119_shear_contact_relief_floor04()
    if key == 'tmlpu_v120_shear_contact_relief_floor02_on':
        return _tmlpu_v120_shear_contact_relief_floor02()
    if key == 'tmlpu_v121_shear_contact_relief_p006_on':
        return _tmlpu_v121_shear_contact_relief_p006()
    if key == 'tmlpu_v122_shear_contact_relief_c0010_on':
        return _tmlpu_v122_shear_contact_relief_c0010()
    if key == 'tmlpu_v123_shear_contact_relief_floor03_on':
        return _tmlpu_v123_shear_contact_relief_floor03()
    if key == 'tmlpu_v124_shear_contact_relief_floor035_on':
        return _tmlpu_v124_shear_contact_relief_floor035()
    if key == 'tmlpu_v125_shear_contact_relief_floor0375_on':
        return _tmlpu_v125_shear_contact_relief_floor0375()
    if key == 'tmlpu_v126_curve_only_relief_floor04_on':
        return _tmlpu_v126_curve_only_relief_floor04()
    if key == 'tmlpu_v127_signed_only_relief_floor04_on':
        return _tmlpu_v127_signed_only_relief_floor04()
    if key == 'tmlpu_v128_asym_relief_signed04_curve02_on':
        return _tmlpu_v128_asym_relief_signed04_curve02()
    if key == 'tmlpu_v129_signed_relief_density014_on':
        return _tmlpu_v129_signed_relief_density014()
    if key == 'tmlpu_v130_signed_only_relief_floor06_on':
        return _tmlpu_v130_signed_only_relief_floor06()
    if key == 'tmlpu_v131_signed_gate_decay_relief_on':
        return _tmlpu_v131_signed_gate_decay_relief()
    if key == 'tmlpu_v132_signed_gate_decay_floor07_on':
        return _tmlpu_v132_signed_gate_decay_floor07()
    if key == 'tmlpu_v133_signed_decay_floor10_capboost_on':
        return _tmlpu_v133_signed_decay_floor10_capboost()
    if key == 'tmlpu_v134_signed_postrollback_preserve_on':
        return _tmlpu_v134_signed_postrollback_preserve()
    if key == 'tmlpu_v135_signed_anchored_curve_assist_on':
        return _tmlpu_v135_signed_anchored_curve_assist()
    if key == 'tmlpu_v136_signed_aligned_curve_assist_on':
        return _tmlpu_v136_signed_aligned_curve_assist()
    if key == 'tmlpu_v137_signed_anchor_curve_floor06_on':
        return _tmlpu_v137_signed_anchor_curve_floor06()
    if key == 'tmlpu_v138_signed_anchor_curve_keep_signed_on':
        return _tmlpu_v138_signed_anchor_curve_keep_signed()
    if key == 'tmlpu_v139_signed_anchor_density_trace_on':
        return _tmlpu_v139_signed_anchor_density_trace()
    if key == 'tmlpu_v140_v131_signed_anchor_curve_gate_diag_on':
        return _tmlpu_v140_v131_signed_anchor_curve_gate_diag()
    if key == 'tmlpu_v141_anchor_curve_diag_epsfix_on':
        return _tmlpu_v141_anchor_curve_diag_epsfix()
    if key == 'tmlpu_v142_highsafe_raw_curve_microassist_on':
        return _tmlpu_v142_highsafe_raw_curve_microassist()
    if key == 'tmlpu_v143_signed_sidecar_decay_on':
        return _tmlpu_v143_signed_sidecar_decay()
    if key == 'tmlpu_v144_signed_sidecar_decay_blend015_on':
        return _tmlpu_v144_signed_sidecar_decay_blend015()
    if key == 'tmlpu_v145_signed_decay_floor12_on':
        return _tmlpu_v145_signed_decay_floor12()
    if key == 'tmlpu_v146_signed_gate_shadow_diag_on':
        return _tmlpu_v146_signed_gate_shadow_diag()
    if key == 'tmlpu_v147_signed_beta044_on':
        return _tmlpu_v147_signed_beta044()
    if key == 'tmlpu_v148_signed_beta038_on':
        return _tmlpu_v148_signed_beta038()
    if key == 'tmlpu_v149_v135_downstream_density_micro_on':
        return _tmlpu_v149_v135_downstream_density_micro()
    if key == 'tmlpu_v150_v135_pair_extend_micro_on':
        return _tmlpu_v150_v135_pair_extend_micro()
    if key == 'tmlpu_v153_v131_pair_extend_micro_on':
        return _tmlpu_v153_v131_pair_extend_micro()
    if key == 'tmlpu_v155_v131_reduced_signed_tail_on':
        return _tmlpu_v155_v131_reduced_signed_tail()
    if key == 'tmlpu_v157_v131_antisheet_on':
        return _tmlpu_v157_v131_antisheet()
    if key == 'tmlpu_v158_v131_strong_antisheet_on':
        return _tmlpu_v158_v131_strong_antisheet()
    if key == 'tmlpu_v159_v131_qcore_gate_on':
        return _tmlpu_v159_v131_qcore_gate()
    if key == 'tmlpu_v160_v131_signed_tail_hffilter_on':
        return _tmlpu_v160_v131_signed_tail_hffilter()
    if key == 'tmlpu_v161_v131_bridge_cut_on':
        return _tmlpu_v161_v131_bridge_cut()
    if key == 'tmlpu_v162_v131_conservative_bridge_cut_on':
        return _tmlpu_v162_v131_conservative_bridge_cut()
    if key == 'tmlpu_v163_v131_shock_ridge_guard_on':
        return _tmlpu_v163_v131_shock_ridge_guard()
    if key == 'tmlpu_v164_v131_density_support_damp_on':
        return _tmlpu_v164_v131_density_support_damp()
    if key == 'tmlpu_v165_v131_signed_tail_ablation_on':
        return _tmlpu_v165_v131_signed_tail_ablation()
    if key == 'tmlpu_v166_v165_micro_signed_restore_on':
        return _tmlpu_v166_v165_micro_signed_restore()
    if key == 'tmlpu_v167_v165_curve_restore_on':
        return _tmlpu_v167_v165_curve_restore()
    if key == 'tmlpu_v168_v167_curve_hffilter_on':
        return _tmlpu_v168_v167_curve_hffilter()
    if key == 'tmlpu_v206_v168_fast_weakoff_on':
        return _tmlpu_v206_v168_fast_weakoff()
    if key == 'tmlpu_v207_v206_fast_shockridge_on':
        return _tmlpu_v207_v206_fast_shockridge()
    if key == 'tmlpu_v208_v206_fast_curve_guard_on':
        return _tmlpu_v208_v206_fast_curve_guard()
    if key == 'tmlpu_v209_v206_fast_pressure_jump_on':
        return _tmlpu_v209_v206_fast_pressure_jump()
    if key == 'tmlpu_v210_v168_fast_weakmlp_no_valuescale_on':
        return _tmlpu_v210_v168_fast_weakmlp_no_valuescale()
    if key == 'tmlpu_v211_v210_fast_weakmlp_valuescale_on':
        return _tmlpu_v211_v210_fast_weakmlp_valuescale()
    if key == 'tmlpu_v212_v210_fast_wallshock_flatten_on':
        return _tmlpu_v212_v210_fast_wallshock_flatten()
    if key == 'tmlpu_v213_v212_wallshock_flatten035_on':
        return _tmlpu_v213_v212_wallshock_flatten035()
    if key == 'tmlpu_v214_v212_wallpressure_flatten_on':
        return _tmlpu_v214_v212_wallpressure_flatten()
    if key == 'tmlpu_v216_v210_fast_shockline_rollback_on':
        return _tmlpu_v216_v210_fast_shockline_rollback()
    if key == 'tmlpu_v217_v212_wallshock_tailboost_on':
        return _tmlpu_v217_v212_wallshock_tailboost()
    if key == 'tmlpu_v174_v168_roi_strength_on':
        return _tmlpu_v174_v168_roi_strength()
    if key == 'tmlpu_v175_v174_stronger_filtered_roi_on':
        return _tmlpu_v175_v174_stronger_filtered_roi()
    if key == 'tmlpu_v176_v174_pair_extend_roi_on':
        return _tmlpu_v176_v174_pair_extend_roi()
    if key == 'tmlpu_v177_v174_mid_strength_roi_on':
        return _tmlpu_v177_v174_mid_strength_roi()
    if key == 'tmlpu_v178_v174_dual_bridge_cut_on':
        return _tmlpu_v178_v174_dual_bridge_cut()
    if key == 'tmlpu_v179_v174_antisheet_on':
        return _tmlpu_v179_v174_antisheet()
    if key == 'tmlpu_v180_v174_swirl_core_on':
        return _tmlpu_v180_v174_swirl_core()
    if key == 'tmlpu_v181_v174_qbridge_cut_on':
        return _tmlpu_v181_v174_qbridge_cut()
    if key == 'tmlpu_v182_v174_total_qbridge_damp_on':
        return _tmlpu_v182_v174_total_qbridge_damp()
    if key == 'tmlpu_v183_v174_micro_pair_extend_on':
        return _tmlpu_v183_v174_micro_pair_extend()
    if key == 'tmlpu_v184_v174_midq_cell_blend_on':
        return _tmlpu_v184_v174_midq_cell_blend()
    if key == 'tmlpu_v185_v174_soft_midq_cell_blend_on':
        return _tmlpu_v185_v174_soft_midq_cell_blend()
    if key == 'tmlpu_v186_v184_shockridge_guard_on':
        return _tmlpu_v186_v184_shockridge_guard()
    if key == 'tmlpu_v187_v186_curve_shockridge_guard_on':
        return _tmlpu_v187_v186_curve_shockridge_guard()
    if key == 'tmlpu_v188_v126_soft_midq_rollup_on':
        return _tmlpu_v188_v126_soft_midq_rollup()
    if key == 'tmlpu_v189_v188_checker_safe_midq_on':
        return _tmlpu_v189_v188_checker_safe_midq()
    if key == 'tmlpu_v195_v188_curve_hffilter_on':
        return _tmlpu_v195_v188_curve_hffilter()
    if key == 'tmlpu_v196_v188_curve_qbridge_on':
        return _tmlpu_v196_v188_curve_qbridge()
    if key == 'tmlpu_v190_v45scalar_v3fast_euler_on':
        return _tmlpu_v190_v45scalar_v3fast_euler()
    if key == 'tmlpu_v191_fast_scalar_balanced_on':
        return _tmlpu_v191_fast_scalar_balanced()
    if key == 'tmlpu_v192_fast_scalar_jump_damped_on':
        return _tmlpu_v192_fast_scalar_jump_damped()
    if key == 'tmlpu_v193_fast_scalar_binary_guard_on':
        return _tmlpu_v193_fast_scalar_binary_guard()
    if key == 'tmlpu_v200_fast_scalar_smooth_body_guard_on':
        return _tmlpu_v200_fast_scalar_smooth_body_guard()
    if key == 'tmlpu_v201_fast_scalar_range_smooth_guard_on':
        return _tmlpu_v201_fast_scalar_range_smooth_guard()
    if key == 'tmlpu_v202_fast_scalar_self_bvd_smooth_guard_on':
        return _tmlpu_v202_fast_scalar_self_bvd_smooth_guard()
    if key == 'tmlpu_v218_fast_scalar_selective_balance_on':
        return _tmlpu_v218_fast_scalar_selective_balance()
    if key == 'tmlpu_v204_mlpu1_bounded_increment_boost_on':
        return _tmlpu_v204_mlpu1_bounded_increment_boost()
    if key == 'tmlpu_v220_exact_beta_on':
        return _tmlpu_v220_exact_beta()
    if key == 'tmlpu_v221_bvd_unified_on':
        return _tmlpu_v221_bvd_unified()
    if key == 'tmlpu_v205_fast_scalar_bvd_on':
        return _tmlpu_v205_fast_scalar_bvd()
    if key == 'tmlpu_v194_mlpu1_jump_sharpener_on':
        return _tmlpu_v194_mlpu1_jump_sharpener()
    if key == 'tmlpu_v199_fast_scalar_edge_antidiff_on':
        return _tmlpu_v199_fast_scalar_edge_antidiff()
    if key == 'tmlpu_v203_mlpu1_range_edge_antidiff_on':
        return _tmlpu_v203_mlpu1_range_edge_antidiff()
    if key == 'tmlpu_v197_fast_scalar_hancock_jump_damped_on':
        return _tmlpu_v197_fast_scalar_hancock_jump_damped()
    if key == 'tmlpu_v198_fast_scalar_cell_balanced_on':
        return _tmlpu_v198_fast_scalar_cell_balanced()
    if key == 'tmlpu_v169_v167_qtight_core_on':
        return _tmlpu_v169_v167_qtight_core()
    if key == 'tmlpu_v170_v167_pressure_entropy_on':
        return _tmlpu_v170_v167_pressure_entropy()
    if key == 'tmlpu_v171_v167_pressure_jump_limit_on':
        return _tmlpu_v171_v167_pressure_jump_limit()
    if key == 'tmlpu_v172_v167_soft_bridge_cut_on':
        return _tmlpu_v172_v167_soft_bridge_cut()
    if key == 'tmlpu_v173_v167_curve_bridge_cut_on':
        return _tmlpu_v173_v167_curve_bridge_cut()
    if key == 'tmlpu_leveque_legacy_on':
        return _tmlpu_leveque()
    if key == 'tmlpu_unified_on':
        return _tmlpu_unified()
    if key == 'tmlpu_leveque_off':
        return _tmlpu_off_leveque()
    if key == 'tmlpu_double_mach_legacy_on':
        return _tmlpu_double_mach()
    if key == 'tmlpu_mach3_step_legacy_on':
        return _tmlpu_mach3_step()
    if key == 'tmlpu_euler_on':
        return _tmlpu_euler()
    if key == 'tmlpu_euler_velocity_smooth_sharp_bvd':
        return _tmlpu_euler_velocity_smooth_sharp_bvd()
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
        restart_t = 0.0
        restart_in = os.environ.get('TMLPU_RESTART_IN', '').strip()
        if restart_in:
            data = np.load(restart_in)
            U0 = np.asarray(data['U'], dtype=float)
            restart_t = float(data.get('t', 0.0))
        remaining_t = max(0.0, float(t_end) - restart_t)
        res = solve(mesh, eq, U0, reconstruction=recon, flux=flux,
                    integrator=integrator, bc=bc, cfl=cfl, t_end=remaining_t,
                    max_steps=500_000, n_face_quad=n_face_quad,
                    face_velocity_mode=face_velocity_mode)
        wall = time.time() - t0
        U = res['U_final']
        actual_t = restart_t + float(res.get('t', 0.0))
        stopped_by_wall = bool(res.get('stopped_by_wall', False))
        restart_out = os.environ.get('TMLPU_RESTART_OUT', '').strip()
        if restart_out:
            np.savez_compressed(restart_out, U=U, t=actual_t,
                                steps=int(res['n_steps']))
        W = eq.cons_to_prim(U)
        finite = bool(np.all(np.isfinite(W)))
        positive = True
        if getattr(eq, 'prim_names', ()) and 'rho' in eq.prim_names:
            positive = bool(np.min(W[0]) > 0.0 and np.min(W[-1]) > 0.0)
        final_reached = actual_t >= float(t_end) - 1.0e-12
        ok = bool(finite and positive and final_reached and not stopped_by_wall)
        err = None
        if stopped_by_wall or not final_reached:
            err = (f"checkpoint t={actual_t:.12g}/"
                   f"{float(t_end):.12g}")
        return dict(ok=ok, U=U if final_reached else None,
                    W=W if final_reached else None, steps=res['n_steps'],
                    wall=wall, error=err, t=actual_t,
                    checkpoint=bool(stopped_by_wall or not final_reached))
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
    primary_diagnostics_finite = all(
        _finite_value(row, 'global_E1') is not None
        for row in (on, mlp_u1))
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
        and on['undershoot'] <= LEVEQUE_WIGGLE_PASS_LIMIT
        and on['overshoot'] <= LEVEQUE_WIGGLE_PASS_LIMIT
        and on_range_violation <= LEVEQUE_WIGGLE_PASS_LIMIT)
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
    tmlpu_better_than_mlp_u1 = bool(
        mlp_u1 is not None
        and _lt_vs_mlp_u1('global_E1'))
    off_unstable_or_worse = bool(
        not off['ok'] or on_range_violation <= off_range_violation)
    passed = bool(on['ok']
                  and primary_diagnostics_finite
                  and tmlpu_better_than_mlp_u1)
    on.update({
        'leveque_gate_diagnostics_finite': diagnostics_finite,
        'leveque_gate_primary_diagnostics_finite': (
            primary_diagnostics_finite),
        'leveque_gate_bounded': bounded_ok,
        'leveque_gate_mass': mass_ok,
        'leveque_gate_global_mass_abs': global_mass_ok,
        'leveque_gate_error': error_ok,
        'leveque_gate_slot': slot_ok,
        'leveque_gate_body_shape': body_shape_ok,
        'leveque_gate_off_unstable_or_worse': off_unstable_or_worse,
        'leveque_gate_mlp_u1_dominance': mlp_u1_dominance_ok,
        'leveque_gate_tmlpu_better_than_mlp_u1': (
            tmlpu_better_than_mlp_u1),
        'leveque_gate_wiggle_limit': LEVEQUE_WIGGLE_PASS_LIMIT,
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
            leveque_integrator = os.environ.get(
                'TMLPU_LEVEQUE_INTEGRATOR', 'ssp_rk3')
            leveque_cfl = float(os.environ.get('TMLPU_LEVEQUE_CFL', '0.4'))
            leveque_n_face_quad = int(os.environ.get(
                'TMLPU_LEVEQUE_FACE_QUAD', '2'))
            leveque_face_velocity_mode = os.environ.get(
                'TMLPU_LEVEQUE_FACE_VELOCITY_MODE', 'central_avg')
            r = _run_safely(mesh, eq, U0, recon, bc, flux='upwind',
                            integrator=leveque_integrator, cfl=leveque_cfl,
                            t_end=1.0, n_face_quad=leveque_n_face_quad,
                            face_velocity_mode=leveque_face_velocity_mode)
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
            double_mach_flux = os.environ.get(
                'TMLPU_DOUBLE_MACH_FLUX', 'roe_rotated_hybrid')
            r = _run_safely(mesh, eq, U0, recon, bc, flux=double_mach_flux,
                            integrator='forward_euler', cfl=0.35, t_end=0.2)
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
                W = r.get('W')
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
                       double_mach_flux=double_mach_flux,
                       vortex_proxy=vortex, vorticity_p95=vort_p95,
                       enstrophy_proxy=enstrophy, checker=checker,
                       steps=r['steps'], wall=r['wall'], error=r['error'])
            if r.get('W') is not None:
                row['rho_min'] = float(np.nanmin(r['W'][0]))
                row['p_min'] = float(np.nanmin(r['W'][-1]))
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

            mach3_mesh_kind = os.environ.get(
                'TMLPU_MACH3_MESH', 'auto').lower()
            if mach3_mesh_kind == 'default':
                mach3_mesh_kind = 'tri_alternating'
            if mach3_mesh_kind == 'auto':
                mach3_mesh_kind = ('roi_graded' if quick
                                   else 'tri_alternating')
            if mach3_mesh_kind == 'quad':
                mesh_builder = _quad_mesh
            elif mach3_mesh_kind == 'roi_graded':
                mesh_builder = triangulate_box_roi_graded
            elif mach3_mesh_kind == 'tri_alternating':
                mesh_builder = _tri_mesh
            else:
                mesh_builder = _tri_mesh
            mesh = mesh_builder(nx, ny, Lx, Ly, keep=keep,
                                classifier=classify,
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
            mach3_flux = os.environ.get('TMLPU_MACH3_FLUX', 'roe_rotated_hybrid')
            mach3_cfl = float(os.environ.get('TMLPU_MACH3_CFL', '0.45'))
            mach3_n_face_quad = int(os.environ.get('TMLPU_MACH3_FACE_QUAD', '1'))
            mach3_integrator = os.environ.get('TMLPU_MACH3_INTEGRATOR', 'forward_euler')
            mach3_t_end = float(os.environ.get('TMLPU_MACH3_TEND', '4.0'))
            r = _run_safely(mesh, eq, U0, recon, bc, flux=mach3_flux,
                            integrator=mach3_integrator, cfl=mach3_cfl,
                            t_end=mach3_t_end, n_face_quad=mach3_n_face_quad)
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
                W = r.get('W')
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
            mesh_name = (
                'quad' if mach3_mesh_kind == 'quad'
                else ('triangulate_box_roi_graded'
                      if mach3_mesh_kind == 'roi_graded'
                      else 'tri_alternating'))
            row = dict(case='mach3_step', method=name, ok=r['ok'],
                       mesh=mesh_name, logical_nx=nx,
                       logical_ny=ny, mesh_cells=mesh.n_cells,
                       mesh_faces=mesh.n_faces,
                       flag_proxy=flag, flag_vorticity_p95=flag_vort,
                       flag_enstrophy_proxy=flag_enstrophy,
                       transverse_velocity_rms=transverse,
                       carbuncle=carbuncle,
                       steps=r['steps'], wall=r['wall'], error=r['error'],
                       mach3_step_flux=mach3_flux,
                       mach3_step_cfl=mach3_cfl)
            if r.get('W') is not None:
                row['rho_min'] = float(np.nanmin(r['W'][0]))
                row['p_min'] = float(np.nanmin(r['W'][-1]))
            return order, name, row, field, vtk

        raise ValueError(f"unknown case {case!r}")
    except Exception as exc:
        row = dict(case=case, method=name, ok=False, steps=-1,
                   wall=0.0, error=repr(exc),
                   error_traceback=traceback.format_exc(),
                   error_type=type(exc).__name__,
                   error_errno=getattr(exc, 'errno', None),
                   error_strerror=getattr(exc, 'strerror', None))
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

    mach3_mesh_kind = os.environ.get(
        'TMLPU_MACH3_MESH', 'auto').lower()
    if mach3_mesh_kind == 'default':
        mach3_mesh_kind = 'tri_alternating'
    if mach3_mesh_kind == 'auto':
        mach3_mesh_kind = ('roi_graded' if quick else 'tri_alternating')
    if mach3_mesh_kind == 'quad':
        mesh_builder = _quad_mesh
    elif mach3_mesh_kind == 'roi_graded':
        mesh_builder = triangulate_box_roi_graded
    elif mach3_mesh_kind == 'tri_alternating':
        mesh_builder = _tri_mesh
    else:
        mesh_builder = _tri_mesh
    mesh = mesh_builder(nx, ny, Lx, Ly, keep=keep, classifier=classify,
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
                'requires a finite T-MLP-u ON run and lower global E1 than '
                'MLP-u1 under identical numerical settings.  Boundedness, '
                'body-wise shape metrics, slot metrics, and T-MLP-u OFF '
                'behavior are reported as diagnostics but do not decide the '
                'LeVeque PASS state.  Any failed diagnostic comparison is '
                'written in leveque_gate_mlp_u1_*_failures.'),
            'leveque_flux': (
                'upwind with central-averaged face velocity and 3-point '
                'Gauss face quadrature'),
            'leveque_mesh': (
                f'unstructured criss-cross triangles, N={LEVEQUE_PAPER_N} '
                f'({4 * LEVEQUE_PAPER_N * LEVEQUE_PAPER_N} triangles; '
                f'quick N={LEVEQUE_QUICK_N})'),
            'double_mach_flux': 'hllc_rotated_hybrid',
            'mach3_step_flux': 'hllc_rotated_hybrid',
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
                'ROI-graded unstructured triangles from '
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
