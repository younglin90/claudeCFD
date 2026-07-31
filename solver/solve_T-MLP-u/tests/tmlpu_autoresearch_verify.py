"""Verify one LeVeque TMLP-u BVD candidate against MLP-u1.

This script is intentionally narrow: it runs the LeVeque rotation benchmark
for MLP-u1 and one all-TMLP-u smooth/sharp BVD candidate, then emits a final
JSON metrics line.  The primary metric is

    ratio = weighted_E1(TMLP-u) / weighted_E1(MLP-u1)

where weighted_E1 = 0.10*slot + 0.45*smooth_hump + 0.45*cone.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from _pkgshim import setup_paths
setup_paths()

from boundary import BoundaryCondition
from equations import Advection
from mesh import criss_cross_box
from reconstruction import MLPU1, TMLPUSmoothSharpBVD
from solver import solve

import test_2d_tmlpu_paper_benchmarks as bench


def _weighted(row):
    return (0.10 * row['slotted_cylinder_E1']
            + 0.45 * row['smooth_hump_E1']
            + 0.45 * row['cone_E1'])


def _candidate_reconstruction(args):
    return TMLPUSmoothSharpBVD(
        smooth_mode='tmlpu',
        smooth_tvd=args['smooth_tvd'],
        smooth_face_increment=args['smooth_face_increment'],
        sharp_tvd=args['sharp_tvd'],
        sharp_face_increment=args['sharp_face_increment'],
        stencil='vertex',
        order=1,
        idw_p=args['idw_p'],
        vertex_mlp_cap=args['vertex_mlp_cap'],
        face_skew_correction=bool(args['face_skew_correction']),
        face_gradient_correction=args['face_gradient_correction'],
        vertex_mlp_augment=bool(args['vertex_mlp_augment']),
        r_form=args['r_form'],
        moment_bvd=bool(args['moment_bvd']),
    )


def _run_one(payload):
    method = payload['method']
    n = payload['n']
    cfg = payload['cfg']
    mesh = criss_cross_box(n, L=1.0)
    eq = Advection(velocity=bench._rotation_velocity)
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    exact = bench._leveque_phi0(x, y)
    U0 = exact[None, :]
    bc = {p: BoundaryCondition('dirichlet', state=(0.0,))
          for p in mesh.bc_patches}
    recon = MLPU1() if method == 'mlp_u1' else _candidate_reconstruction(cfg)
    t0 = time.time()
    try:
        res = solve(mesh, eq, U0, reconstruction=recon, flux='upwind',
                    integrator='ssp_rk3', bc=bc, cfl=cfg['cfl'],
                    t_end=1.0, max_steps=500_000, n_face_quad=3,
                    face_velocity_mode='central_avg')
        wall = time.time() - t0
        f = eq.cons_to_prim(res['U_final'])[0]
        finite = bool(np.all(np.isfinite(f)))
        row = {
            'method': method,
            'ok': finite,
            'run_ok': True,
            'finite': finite,
            'N': n,
            'steps': int(res['n_steps']),
            'wall': wall,
            'error': None,
        }
        if finite:
            row.update(bench._leveque_shape_metrics(mesh, exact, f))
            row['range_min'] = float(np.min(f))
            row['range_max'] = float(np.max(f))
            row['wiggle'] = max(0.0, -row['range_min']) + max(
                0.0, row['range_max'] - 1.0)
            row['weighted_E1'] = _weighted(row)
            row['ok'] = bool(np.isfinite(row['weighted_E1'])
                             and row['wiggle'] <= cfg['max_wiggle'])
        return method, row, f if finite else None
    except Exception as exc:
        return method, {
            'method': method,
            'ok': False,
            'run_ok': False,
            'finite': False,
            'N': n,
            'steps': -1,
            'wall': time.time() - t0,
            'error': repr(exc),
        }, None


def _write_plot(path, n, rows, fields):
    mesh = criss_cross_box(n, L=1.0)
    exact = bench._leveque_phi0(mesh.cell_centers[:, 0],
                                mesh.cell_centers[:, 1])
    plot_fields = {'Initial phi0': exact}
    plot_fields.update(fields)
    plot_rows = [{'method': 'Initial phi0', 'ok': True}] + rows
    bench._plot_scheme_contours(
        mesh, plot_fields, plot_rows, path,
        f'LeVeque N={n}: MLP-u1 vs all-TMLP-u BVD',
        vmin=0.0, vmax=1.0,
        metric_keys=('weighted_E1', 'smooth_hump_E1', 'cone_E1',
                     'slotted_cylinder_E1', 'wiggle'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=18)
    ap.add_argument('--smooth-tvd', default='mc')
    ap.add_argument('--smooth-face-increment', default='tmlpu')
    ap.add_argument('--sharp-tvd', default='cicsam_co34')
    ap.add_argument('--sharp-face-increment', default='tmlpu')
    ap.add_argument('--face-gradient-correction', default='jasak')
    ap.add_argument('--r-form', default='far_upwind',
                    choices=('far_upwind', 'nvf'))
    ap.add_argument('--face-skew-correction', action='store_true',
                    default=True)
    ap.add_argument('--no-face-skew-correction', dest='face_skew_correction',
                    action='store_false')
    ap.add_argument('--moment-bvd', action='store_true')
    ap.add_argument('--vertex-mlp-augment', action='store_true')
    ap.add_argument('--idw-p', type=float, default=0.0)
    ap.add_argument('--vertex-mlp-cap', type=float, default=2.0)
    ap.add_argument('--cfl', type=float, default=0.4)
    ap.add_argument('--max-wiggle', type=float, default=1.0e-8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--plot', default=None)
    args = ap.parse_args()
    cfg = vars(args)
    payloads = [
        {'method': 'mlp_u1', 'n': args.n, 'cfg': cfg},
        {'method': 'tmlpu', 'n': args.n, 'cfg': cfg},
    ]
    rows_by_method = {}
    fields = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_run_one, payload) for payload in payloads]
        for fut in as_completed(futures):
            method, row, field = fut.result()
            rows_by_method[method] = row
            if field is not None:
                fields[method] = field
    mlp = rows_by_method['mlp_u1']
    tmlp = rows_by_method['tmlpu']
    ratio = None
    if mlp.get('weighted_E1') and tmlp.get('weighted_E1') is not None:
        ratio = float(tmlp['weighted_E1'] / max(mlp['weighted_E1'], 1.0e-300))
    summary = {
        'N': args.n,
        'ratio': ratio,
        'target_metric': ratio,
        'ok': bool(mlp.get('ok') and tmlp.get('ok') and ratio is not None),
        'smooth_tvd': args.smooth_tvd,
        'smooth_face_increment': args.smooth_face_increment,
        'sharp_tvd': args.sharp_tvd,
        'sharp_face_increment': args.sharp_face_increment,
        'face_gradient_correction': args.face_gradient_correction,
        'r_form': args.r_form,
        'face_skew_correction': args.face_skew_correction,
        'moment_bvd': args.moment_bvd,
        'vertex_mlp_augment': args.vertex_mlp_augment,
        'idw_p': args.idw_p,
        'vertex_mlp_cap': args.vertex_mlp_cap,
        'tmlp_weighted': tmlp.get('weighted_E1'),
        'mlp_u1_weighted': mlp.get('weighted_E1'),
        'tmlp_smooth_E1': tmlp.get('smooth_hump_E1'),
        'mlp_u1_smooth_E1': mlp.get('smooth_hump_E1'),
        'tmlp_cone_E1': tmlp.get('cone_E1'),
        'mlp_u1_cone_E1': mlp.get('cone_E1'),
        'tmlp_slot_E1': tmlp.get('slotted_cylinder_E1'),
        'mlp_u1_slot_E1': mlp.get('slotted_cylinder_E1'),
        'tmlp_iou': tmlp.get('slotted_cylinder_iou'),
        'mlp_u1_iou': mlp.get('slotted_cylinder_iou'),
        'tmlp_wiggle': tmlp.get('wiggle'),
        'mlp_u1_wiggle': mlp.get('wiggle'),
        'tmlp_error': tmlp.get('error'),
        'mlp_u1_error': mlp.get('error'),
        'rows': [mlp, tmlp],
    }
    if args.plot:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _write_plot(plot_path, args.n, summary['rows'], fields)
        summary['plot'] = str(plot_path)
    print(json.dumps(bench._json_ready(summary), sort_keys=True,
                     allow_nan=False))
    return 0 if summary['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
