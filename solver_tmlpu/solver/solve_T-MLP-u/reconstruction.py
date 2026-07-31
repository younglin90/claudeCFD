"""Face-state reconstruction for FVM — primitive-variable based.

A `Reconstruction` builds, for every interior face f, the primitive
state W on the *owner* and *neighbour* sides of the face:

    reconstruct(mesh, W_cell, eq) → (W_face_owner, W_face_neighbour)

with both arrays shaped (nvar, n_faces).  Boundary faces are filled by
`boundary.apply_bc` *before* reconstruction (they live as ghost cells).

T-MLP-u (the user's method) is the headline reconstruction that this
package is intended to validate.  We expose:

    FirstOrder            — piecewise constant (no reconstruction)
    MinmodTVD1D           — classical 1D structured TVD (any limiter from limiters.py)
    MLPU                  — Park-Yoon-Kim 2010 baseline (PLACEHOLDER for now)
    TMLPU                 — user's T-MLP-u (PLACEHOLDER — to be filled by the user)

A Reconstruction object is *grid-aware* (Mesh) and *equation-aware*
(Equation) — same interface for 1D / 2D / structured / unstructured,
which matches the user's stated scope.
"""
from __future__ import annotations
import atexit
from dataclasses import dataclass, field
from typing import Callable
import json
import os
import numpy as np

from limiters import (minmod, minmod2, superbee, t_mlp_u_face_value,
                      TVD_LIMITERS)


_TMLPU_V93_DIAG = None
_TMLPU_V93_X_BANDS = ((0.55, 0.8), (0.8, 1.0), (1.0, 1.3),
                      (1.3, 1.6), (1.6, 2.0))
_TMLPU_V93_Y_BANDS = ((0.6, 0.72), (0.72, 0.85), (0.85, 1.0))
_TMLPU_V93_SUM_KEYS = (
    'base_pair_gate', 'signed_pair_tail_gate', 'density_curve_tail_gate',
    'pressure_jump', 'compression', 'normality_here', 'shear_frac',
    'omega_o_abs', 'omega_n_abs', 'qratio_o', 'qratio_n',
    'density_support', 'signed_tail_dut_abs', 'density_curve_tail_dut_abs')


def _tmlpu_v93_diag_enabled():
    return os.environ.get(
        'TMLPU_V93_GATE_DIAGNOSTICS', '0').lower() in (
            '1', 'true', 'yes', 'on')


def _tmlpu_v93_diag_init():
    bins = []
    for x0, x1 in _TMLPU_V93_X_BANDS:
        for y0, y1 in _TMLPU_V93_Y_BANDS:
            entry = {
                'x_band': [float(x0), float(x1)],
                'y_band': [float(y0), float(y1)],
                'count': 0,
                'signed_pair_count': 0,
                'same_sign_pair_count': 0,
                'signed_gate_active_count': 0,
                'density_curve_gate_active_count': 0,
                'shocklike_count': 0,
                'sums': {k: 0.0 for k in _TMLPU_V93_SUM_KEYS},
                'max': {k: 0.0 for k in _TMLPU_V93_SUM_KEYS},
            }
            bins.append(entry)
    return {
        'schema': 'tmlpu_v93_gate_diagnostics_v1',
        'calls': 0,
        'total_faces': 0,
        'x_bands': [[float(a), float(b)] for a, b in _TMLPU_V93_X_BANDS],
        'y_bands': [[float(a), float(b)] for a, b in _TMLPU_V93_Y_BANDS],
        'bins': bins,
    }


def _tmlpu_v93_diag_write():
    global _TMLPU_V93_DIAG
    if not _TMLPU_V93_DIAG:
        return
    path = os.environ.get(
        'TMLPU_V93_GATE_DIAGNOSTICS_PATH',
        'results/T-MLP-u/current_mach3_quick/v93_gate_diagnostics.json')
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(_TMLPU_V93_DIAG, fh, indent=2, sort_keys=True)
            fh.write('\n')
    except Exception:
        pass


atexit.register(_tmlpu_v93_diag_write)


_TMLPU_V115_TAIL_DIAG = None
_TMLPU_V115_SUM_KEYS = (
    'signed_gate_raw', 'signed_gate_final',
    'density_curve_gate_raw', 'density_curve_gate_final',
    'tail_density_support', 'safe_legacy_gate',
    'signed_dut_raw_abs', 'signed_dut_clipped_abs',
    'density_curve_dut_raw_abs', 'density_curve_dut_clipped_abs',
    'tail_delta_abs', 'pressure_jump', 'compression', 'normality_here',
    'shear_frac', 'density_signal', 'qratio_o', 'qratio_n',
    'density_curve')


def _tmlpu_v115_tail_diag_enabled():
    return os.environ.get(
        'TMLPU_V115_TAIL_DIAGNOSTICS', '0').lower() in (
            '1', 'true', 'yes', 'on')


def _tmlpu_v115_tail_diag_init():
    return {
        'schema': 'tmlpu_v115_tail_diagnostics_v1',
        'calls': 0,
        'total_faces': 0,
        'signed_gate_active_count': 0,
        'density_curve_gate_active_count': 0,
        'signed_cap_hit_count': 0,
        'density_curve_cap_hit_count': 0,
        'sums': {k: 0.0 for k in _TMLPU_V115_SUM_KEYS},
        'max': {k: 0.0 for k in _TMLPU_V115_SUM_KEYS},
        'counts': {k: 0 for k in _TMLPU_V115_SUM_KEYS},
    }


def _tmlpu_v115_tail_diag_write():
    global _TMLPU_V115_TAIL_DIAG
    if not _TMLPU_V115_TAIL_DIAG:
        return
    path = os.environ.get(
        'TMLPU_V115_TAIL_DIAGNOSTICS_PATH',
        'results/T-MLP-u/current_mach3_quick/v115_tail_diag.json')
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(_TMLPU_V115_TAIL_DIAG, fh, indent=2, sort_keys=True)
            fh.write('\n')
    except Exception:
        pass


atexit.register(_tmlpu_v115_tail_diag_write)


def _tmlpu_v115_tail_diag_update(
        signed_gate_raw, signed_gate_final,
        density_curve_gate_raw, density_curve_gate_final,
        tail_density_support, safe_legacy_gate, signed_dut_raw_abs,
        signed_dut_clipped_abs, density_curve_dut_raw_abs,
        density_curve_dut_clipped_abs, signed_cap_hit,
        density_curve_cap_hit, pressure_jump, compression, normality_here,
        shear_frac, density_signal, qratio_o, qratio_n, density_curve):
    global _TMLPU_V115_TAIL_DIAG
    if not _tmlpu_v115_tail_diag_enabled():
        return
    if _TMLPU_V115_TAIL_DIAG is None:
        _TMLPU_V115_TAIL_DIAG = _tmlpu_v115_tail_diag_init()
    diag = _TMLPU_V115_TAIL_DIAG
    diag['calls'] += 1
    n_faces = int(np.asarray(signed_gate_final).size)
    diag['total_faces'] += n_faces
    if n_faces == 0:
        return

    def _as_face_array(value):
        arr = np.asarray(value, dtype=float)
        if arr.size == 1 and n_faces != 1:
            arr = np.full(n_faces, float(arr.reshape(-1)[0]), dtype=float)
        return arr

    arrays = {
        'signed_gate_raw': _as_face_array(signed_gate_raw),
        'signed_gate_final': _as_face_array(signed_gate_final),
        'density_curve_gate_raw': _as_face_array(density_curve_gate_raw),
        'density_curve_gate_final': _as_face_array(density_curve_gate_final),
        'tail_density_support': _as_face_array(tail_density_support),
        'safe_legacy_gate': _as_face_array(safe_legacy_gate),
        'signed_dut_raw_abs': _as_face_array(signed_dut_raw_abs),
        'signed_dut_clipped_abs': _as_face_array(signed_dut_clipped_abs),
        'density_curve_dut_raw_abs': _as_face_array(
            density_curve_dut_raw_abs),
        'density_curve_dut_clipped_abs': _as_face_array(
            density_curve_dut_clipped_abs),
        'tail_delta_abs': np.maximum(
            _as_face_array(signed_dut_clipped_abs),
            _as_face_array(density_curve_dut_clipped_abs)),
        'pressure_jump': _as_face_array(pressure_jump),
        'compression': _as_face_array(compression),
        'normality_here': _as_face_array(normality_here),
        'shear_frac': _as_face_array(shear_frac),
        'density_signal': _as_face_array(density_signal),
        'qratio_o': _as_face_array(qratio_o),
        'qratio_n': _as_face_array(qratio_n),
        'density_curve': _as_face_array(density_curve),
    }
    diag['signed_gate_active_count'] += int(np.count_nonzero(
        arrays['signed_gate_final'] > 0.0))
    diag['density_curve_gate_active_count'] += int(np.count_nonzero(
        arrays['density_curve_gate_final'] > 0.0))
    diag['signed_cap_hit_count'] += int(np.count_nonzero(
        np.asarray(signed_cap_hit, dtype=bool)))
    diag['density_curve_cap_hit_count'] += int(np.count_nonzero(
        np.asarray(density_curve_cap_hit, dtype=bool)))
    for key, arr in arrays.items():
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            continue
        diag['sums'][key] += float(np.sum(vals))
        diag['max'][key] = max(diag['max'][key], float(np.max(vals)))
        diag['counts'][key] += int(vals.size)


_TMLPU_V146_SIGNED_GATE_DIAG = None
_TMLPU_V146_FEATURE_BINS = {
    'safe_gate': (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0, np.inf),
    'pressure': (0.0, 0.005, 0.010, 0.020, 0.040, 0.080, np.inf),
    'compression': (0.0, 0.001, 0.002, 0.004, 0.008, 0.020, np.inf),
    'normality': (0.0, 0.08, 0.14, 0.20, 0.35, 0.55, 1.0, np.inf),
    'shear': (0.0, 0.40, 0.60, 0.72, 0.82, 0.92, 1.0, np.inf),
    'density_support': (0.0, 0.01, 0.015, 0.02, 0.04, 0.08, np.inf),
}


def _tmlpu_v146_signed_gate_diag_enabled():
    return os.environ.get(
        'TMLPU_V146_SIGNED_GATE_DIAGNOSTICS', '0').lower() in (
            '1', 'true', 'yes', 'on')


def _tmlpu_v146_signed_gate_diag_init():
    return {
        'schema': 'tmlpu_v146_signed_gate_shadow_diag_v1',
        'calls': 0,
        'total_faces': 0,
        'raw_count': 0,
        'v127_shadow_count': 0,
        'v131_actual_count': 0,
        'overlap_count': 0,
        'v127_only_count': 0,
        'v131_only_count': 0,
        'raw_sum': 0.0,
        'v127_sum': 0.0,
        'v131_sum': 0.0,
        'd_ut_signed_abs_sum_before_cap': 0.0,
        'd_ut_signed_abs_sum_after_cap': 0.0,
        'feature_bins': {
            name: [{
                'lo': float(edges[i]),
                'hi': float(edges[i + 1]),
                'count': 0,
                'raw_count': 0,
                'v127_shadow_count': 0,
                'v131_actual_count': 0,
                'raw_sum': 0.0,
                'v127_sum': 0.0,
                'v131_sum': 0.0,
            } for i in range(len(edges) - 1)]
            for name, edges in _TMLPU_V146_FEATURE_BINS.items()
        },
    }


def _tmlpu_v146_signed_gate_diag_write():
    global _TMLPU_V146_SIGNED_GATE_DIAG
    if not _TMLPU_V146_SIGNED_GATE_DIAG:
        return
    path = os.environ.get(
        'TMLPU_V146_SIGNED_GATE_DIAGNOSTICS_PATH',
        'results/T-MLP-u/current_mach3_quick/'
        'v146_signed_gate_shadow_diag.json')
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(
                _TMLPU_V146_SIGNED_GATE_DIAG, fh,
                indent=2, sort_keys=True)
            fh.write('\n')
    except Exception:
        pass


atexit.register(_tmlpu_v146_signed_gate_diag_write)


def _tmlpu_v146_signed_gate_diag_update(
        raw_gate, v127_shadow_gate, v131_actual_gate,
        d_ut_signed_abs_before_cap, d_ut_signed_abs_after_cap,
        safe_gate, pressure_jump, compression, normality_here,
        shear_frac, density_support):
    global _TMLPU_V146_SIGNED_GATE_DIAG
    if not _tmlpu_v146_signed_gate_diag_enabled():
        return
    if _TMLPU_V146_SIGNED_GATE_DIAG is None:
        _TMLPU_V146_SIGNED_GATE_DIAG = (
            _tmlpu_v146_signed_gate_diag_init())
    diag = _TMLPU_V146_SIGNED_GATE_DIAG
    diag['calls'] += 1
    n_faces = int(np.asarray(v131_actual_gate).size)
    diag['total_faces'] += n_faces
    if n_faces == 0:
        return

    def _as_face_array(value):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1 and n_faces != 1:
            arr = np.full(n_faces, float(arr[0]), dtype=float)
        return arr

    raw = _as_face_array(raw_gate)
    v127 = _as_face_array(v127_shadow_gate)
    v131 = _as_face_array(v131_actual_gate)
    before = _as_face_array(d_ut_signed_abs_before_cap)
    after = _as_face_array(d_ut_signed_abs_after_cap)
    eps = 1.0e-30
    raw_active = raw > eps
    v127_active = v127 > eps
    v131_active = v131 > eps
    overlap = v127_active & v131_active
    diag['raw_count'] += int(np.count_nonzero(raw_active))
    diag['v127_shadow_count'] += int(np.count_nonzero(v127_active))
    diag['v131_actual_count'] += int(np.count_nonzero(v131_active))
    diag['overlap_count'] += int(np.count_nonzero(overlap))
    diag['v127_only_count'] += int(np.count_nonzero(
        v127_active & ~v131_active))
    diag['v131_only_count'] += int(np.count_nonzero(
        v131_active & ~v127_active))
    diag['raw_sum'] += float(np.sum(raw[np.isfinite(raw)]))
    diag['v127_sum'] += float(np.sum(v127[np.isfinite(v127)]))
    diag['v131_sum'] += float(np.sum(v131[np.isfinite(v131)]))
    diag['d_ut_signed_abs_sum_before_cap'] += float(np.sum(
        before[np.isfinite(before)]))
    diag['d_ut_signed_abs_sum_after_cap'] += float(np.sum(
        after[np.isfinite(after)]))

    features = {
        'safe_gate': _as_face_array(safe_gate),
        'pressure': _as_face_array(pressure_jump),
        'compression': _as_face_array(compression),
        'normality': _as_face_array(normality_here),
        'shear': _as_face_array(shear_frac),
        'density_support': _as_face_array(density_support),
    }
    for name, values in features.items():
        edges = _TMLPU_V146_FEATURE_BINS[name]
        bins = diag['feature_bins'][name]
        finite = np.isfinite(values)
        for i, entry in enumerate(bins):
            lo = edges[i]
            hi = edges[i + 1]
            mask = finite & (values >= lo) & (values < hi)
            if not np.any(mask):
                continue
            entry['count'] += int(np.count_nonzero(mask))
            entry['raw_count'] += int(np.count_nonzero(raw_active & mask))
            entry['v127_shadow_count'] += int(np.count_nonzero(
                v127_active & mask))
            entry['v131_actual_count'] += int(np.count_nonzero(
                v131_active & mask))
            entry['raw_sum'] += float(np.sum(raw[mask]))
            entry['v127_sum'] += float(np.sum(v127[mask]))
            entry['v131_sum'] += float(np.sum(v131[mask]))


_TMLPU_V117_FEATURE_DIAG = None
_TMLPU_V117_FEATURE_BINS = {
    'density_signal': (0.0, 0.01, 0.015, 0.02, 0.04, 0.08, np.inf),
    'safe_gate': (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0),
    'pressure_jump': (0.0, 0.005, 0.010, 0.020, 0.040, 0.080, np.inf),
    'compression': (0.0, 0.001, 0.002, 0.004, 0.008, 0.020, np.inf),
    'normality_here': (0.0, 0.08, 0.14, 0.20, 0.35, 0.55, 1.0),
    'shear_frac': (0.0, 0.40, 0.60, 0.72, 0.82, 0.92, 1.0),
    'signed_pair': (0.0, 0.5, 1.0),
    'density_curve_raw_active': (0.0, 0.5, 1.0),
}


_TMLPU_V140_ANCHOR_CURVE_DIAG = None
_TMLPU_V140_FEATURE_BINS = {
    'safe_gate': (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0),
    'pressure_jump': (0.0, 0.005, 0.010, 0.020, 0.040, 0.080, np.inf),
    'compression': (0.0, 0.001, 0.002, 0.004, 0.008, 0.020, np.inf),
    'normality_here': (0.0, 0.08, 0.14, 0.20, 0.35, 0.55, 1.0),
    'shear_frac': (0.0, 0.40, 0.60, 0.72, 0.82, 0.92, 1.0),
    'density_signal': (0.0, 0.01, 0.015, 0.02, 0.04, 0.08, np.inf),
}


def _tmlpu_v140_anchor_curve_diag_enabled():
    return os.environ.get(
        'TMLPU_V140_ANCHOR_CURVE_DIAGNOSTICS', '0').lower() in (
            '1', 'true', 'yes', 'on')


def _tmlpu_v140_anchor_curve_diag_init():
    bins = {}
    for feature, edges in _TMLPU_V140_FEATURE_BINS.items():
        entries = []
        for i in range(len(edges) - 1):
            entries.append({
                'feature': feature,
                'lo': None if np.isneginf(edges[i]) else float(edges[i]),
                'hi': None if np.isposinf(edges[i + 1])
                else float(edges[i + 1]),
                'count': 0,
                'signed_anchor_count': 0,
                'curve_assist_raw_count': 0,
                'curve_assist_final_count': 0,
                'curve_assist_floor_limited_count': 0,
                'curve_assist_cap_hit_count': 0,
                'highsafe_curve_count': 0,
                'curve_assist_delta_abs_sum': 0.0,
                'microassist_delta_abs_sum': 0.0,
            })
        bins[feature] = entries
    return {
        'schema': 'tmlpu_v140_anchor_curve_gate_diagnostics_v1',
        'calls': 0,
        'total_faces': 0,
        'signed_anchor_count': 0,
        'curve_assist_raw_count': 0,
        'curve_assist_final_count': 0,
        'curve_assist_floor_limited_count': 0,
        'curve_assist_cap_hit_count': 0,
        'highsafe_curve_count': 0,
        'curve_assist_delta_abs_sum': 0.0,
        'microassist_delta_abs_sum': 0.0,
        'signed_anchor_with_curve_count': 0,
        'signed_anchor_no_curve_count': 0,
        'bins': bins,
    }


def _tmlpu_v140_anchor_curve_diag_write():
    global _TMLPU_V140_ANCHOR_CURVE_DIAG
    if not _TMLPU_V140_ANCHOR_CURVE_DIAG:
        return
    path = os.environ.get(
        'TMLPU_V140_ANCHOR_CURVE_DIAGNOSTICS_PATH',
        'results/T-MLP-u/current_mach3_quick/v140_anchor_curve_diag.json')
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(
                _TMLPU_V140_ANCHOR_CURVE_DIAG, fh, indent=2,
                sort_keys=True)
            fh.write('\n')
    except Exception:
        pass


atexit.register(_tmlpu_v140_anchor_curve_diag_write)


def _tmlpu_v140_anchor_curve_diag_update(
        signed_pair_tail_gate, curve_gate_raw, curve_gate_pre_assist,
        curve_gate_final, curve_cap_hit, curve_delta_abs, pressure_jump,
        compression, normality_here, shear_frac, density_signal,
        safe_legacy_gate, highsafe_curve=None, microassist_delta_abs=None):
    global _TMLPU_V140_ANCHOR_CURVE_DIAG
    if not _tmlpu_v140_anchor_curve_diag_enabled():
        return
    if _TMLPU_V140_ANCHOR_CURVE_DIAG is None:
        _TMLPU_V140_ANCHOR_CURVE_DIAG = (
            _tmlpu_v140_anchor_curve_diag_init())
    diag = _TMLPU_V140_ANCHOR_CURVE_DIAG

    signed_gate = np.asarray(signed_pair_tail_gate, dtype=float).copy()
    n_faces = int(signed_gate.size)
    diag['calls'] += 1
    diag['total_faces'] += n_faces
    if n_faces == 0:
        return

    def _as_face_array(value):
        arr = np.asarray(value, dtype=float).copy()
        if arr.size == 1 and n_faces != 1:
            arr = np.full(n_faces, float(arr.reshape(-1)[0]), dtype=float)
        return arr

    curve_raw = _as_face_array(curve_gate_raw)
    curve_pre = _as_face_array(curve_gate_pre_assist)
    curve_final = _as_face_array(curve_gate_final)
    cap_hit = np.asarray(curve_cap_hit, dtype=bool).copy()
    delta_abs = _as_face_array(curve_delta_abs)
    eps = 1.0e-30
    signed_anchor = signed_gate > eps
    raw_active = curve_raw > eps
    final_active = curve_final > eps
    floor_limited = signed_anchor & (curve_final > np.maximum(curve_pre, eps))
    if highsafe_curve is None:
        highsafe = np.zeros(n_faces, dtype=bool)
    else:
        highsafe = np.asarray(highsafe_curve, dtype=bool).copy()
    if microassist_delta_abs is None:
        microassist_abs = np.zeros(n_faces, dtype=float)
    else:
        microassist_abs = _as_face_array(microassist_delta_abs)

    diag['signed_anchor_count'] += int(np.count_nonzero(signed_anchor))
    diag['curve_assist_raw_count'] += int(np.count_nonzero(raw_active))
    diag['curve_assist_final_count'] += int(np.count_nonzero(final_active))
    diag['curve_assist_floor_limited_count'] += int(np.count_nonzero(
        floor_limited))
    diag['curve_assist_cap_hit_count'] += int(np.count_nonzero(cap_hit))
    diag['highsafe_curve_count'] += int(np.count_nonzero(highsafe))
    diag['curve_assist_delta_abs_sum'] += float(np.sum(
        delta_abs[np.isfinite(delta_abs)]))
    diag['microassist_delta_abs_sum'] += float(np.sum(
        microassist_abs[np.isfinite(microassist_abs)]))
    diag['signed_anchor_with_curve_count'] += int(np.count_nonzero(
        signed_anchor & raw_active))
    diag['signed_anchor_no_curve_count'] += int(np.count_nonzero(
        signed_anchor & ~raw_active))

    features = {
        'safe_gate': _as_face_array(safe_legacy_gate),
        'pressure_jump': _as_face_array(pressure_jump),
        'compression': _as_face_array(compression),
        'normality_here': _as_face_array(normality_here),
        'shear_frac': _as_face_array(shear_frac),
        'density_signal': _as_face_array(density_signal),
    }
    finite_base = (
        np.isfinite(curve_raw) & np.isfinite(curve_final)
        & np.isfinite(delta_abs) & np.isfinite(signed_gate))
    for feature, values in features.items():
        edges = _TMLPU_V140_FEATURE_BINS[feature]
        entries = diag['bins'][feature]
        values = np.asarray(values, dtype=float)
        for i, entry in enumerate(entries):
            lo = edges[i]
            hi = edges[i + 1]
            if i == len(entries) - 1:
                mask = (values >= lo) & (values <= hi)
            else:
                mask = (values >= lo) & (values < hi)
            mask = mask & np.isfinite(values) & finite_base
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            entry['count'] += count
            entry['signed_anchor_count'] += int(np.count_nonzero(
                signed_anchor & mask))
            entry['curve_assist_raw_count'] += int(np.count_nonzero(
                raw_active & mask))
            entry['curve_assist_final_count'] += int(np.count_nonzero(
                final_active & mask))
            entry['curve_assist_floor_limited_count'] += int(np.count_nonzero(
                floor_limited & mask))
            entry['curve_assist_cap_hit_count'] += int(np.count_nonzero(
                cap_hit & mask))
            entry['highsafe_curve_count'] += int(np.count_nonzero(
                highsafe & mask))
            entry['curve_assist_delta_abs_sum'] += float(np.sum(
                delta_abs[mask]))
            entry['microassist_delta_abs_sum'] += float(np.sum(
                microassist_abs[mask]))


def _tmlpu_v117_feature_diag_enabled():
    return os.environ.get(
        'TMLPU_V117_FEATURE_DIAGNOSTICS', '0').lower() in (
            '1', 'true', 'yes', 'on')


def _tmlpu_v117_feature_diag_entry(feature, lo, hi):
    return {
        'feature': feature,
        'lo': None if np.isneginf(lo) else float(lo),
        'hi': None if np.isposinf(hi) else float(hi),
        'count': 0,
        'signed_pair_count': 0,
        'density_curve_raw_active_count': 0,
        'shocklike_count': 0,
        'signed_raw_gate_sum': 0.0,
        'signed_final_gate_sum': 0.0,
        'density_curve_raw_gate_sum': 0.0,
        'density_curve_final_gate_sum': 0.0,
        'raw_gate_sum': 0.0,
        'final_gate_sum': 0.0,
        'd_ut_raw_abs_sum': 0.0,
        'd_ut_final_abs_sum': 0.0,
        'final_raw_ratio': 0.0,
    }


def _tmlpu_v117_feature_diag_init():
    bins = {}
    for feature, edges in _TMLPU_V117_FEATURE_BINS.items():
        feature_bins = []
        for i in range(len(edges) - 1):
            feature_bins.append(
                _tmlpu_v117_feature_diag_entry(
                    feature, edges[i], edges[i + 1]))
        bins[feature] = feature_bins
    return {
        'schema': 'tmlpu_v117_feature_diagnostics_v1',
        'calls': 0,
        'total_faces': 0,
        'bins': bins,
    }


def _tmlpu_v117_feature_diag_finalize(diag):
    for feature_bins in diag.get('bins', {}).values():
        for entry in feature_bins:
            raw = entry.get('raw_gate_sum', 0.0)
            entry['final_raw_ratio'] = (
                float(entry.get('final_gate_sum', 0.0) / raw)
                if raw > 0.0 else 0.0)


def _tmlpu_v117_feature_diag_write():
    global _TMLPU_V117_FEATURE_DIAG
    if not _TMLPU_V117_FEATURE_DIAG:
        return
    path = os.environ.get(
        'TMLPU_V117_FEATURE_DIAGNOSTICS_PATH',
        'results/T-MLP-u/current_mach3_quick/v117_feature_diag.json')
    try:
        _tmlpu_v117_feature_diag_finalize(_TMLPU_V117_FEATURE_DIAG)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(_TMLPU_V117_FEATURE_DIAG, fh, indent=2, sort_keys=True)
            fh.write('\n')
    except Exception:
        pass


atexit.register(_tmlpu_v117_feature_diag_write)


def _tmlpu_v117_feature_diag_update(
        signed_gate_raw, signed_gate_final,
        density_curve_gate_raw, density_curve_gate_final,
        signed_dut_raw_abs, signed_dut_clipped_abs,
        density_curve_dut_raw_abs, density_curve_dut_clipped_abs,
        pressure_jump, compression, normality_here, shear_frac,
        density_signal, safe_legacy_gate, omega_o, omega_n):
    global _TMLPU_V117_FEATURE_DIAG
    if not _tmlpu_v117_feature_diag_enabled():
        return
    if _TMLPU_V117_FEATURE_DIAG is None:
        _TMLPU_V117_FEATURE_DIAG = _tmlpu_v117_feature_diag_init()
    diag = _TMLPU_V117_FEATURE_DIAG
    diag['calls'] += 1
    n_faces = int(np.asarray(signed_gate_final).size)
    diag['total_faces'] += n_faces
    if n_faces == 0:
        return

    def _as_face_array(value):
        arr = np.asarray(value, dtype=float)
        if arr.size == 1 and n_faces != 1:
            arr = np.full(n_faces, float(arr.reshape(-1)[0]), dtype=float)
        return arr

    signed_raw = _as_face_array(signed_gate_raw)
    signed_final = _as_face_array(signed_gate_final)
    curve_raw = _as_face_array(density_curve_gate_raw)
    curve_final = _as_face_array(density_curve_gate_final)
    signed_dut_raw = _as_face_array(signed_dut_raw_abs)
    signed_dut_final = _as_face_array(signed_dut_clipped_abs)
    curve_dut_raw = _as_face_array(density_curve_dut_raw_abs)
    curve_dut_final = _as_face_array(density_curve_dut_clipped_abs)
    pressure_jump = _as_face_array(pressure_jump)
    compression = _as_face_array(compression)
    normality_here = _as_face_array(normality_here)
    shear_frac = _as_face_array(shear_frac)
    density_signal = _as_face_array(density_signal)
    safe_gate = _as_face_array(safe_legacy_gate)
    omega_o = _as_face_array(omega_o)
    omega_n = _as_face_array(omega_n)
    signed_pair = omega_o * omega_n < 0.0
    curve_active = curve_raw > 0.0
    shocklike = (
        (pressure_jump >= 0.012)
        & (compression >= 0.0025)
        & (normality_here >= 0.18))
    raw_gate_sum = signed_raw + curve_raw
    final_gate_sum = signed_final + curve_final
    dut_raw_sum = signed_dut_raw + curve_dut_raw
    dut_final_sum = signed_dut_final + curve_dut_final
    features = {
        'density_signal': density_signal,
        'safe_gate': safe_gate,
        'pressure_jump': pressure_jump,
        'compression': compression,
        'normality_here': normality_here,
        'shear_frac': shear_frac,
        'signed_pair': signed_pair.astype(float),
        'density_curve_raw_active': curve_active.astype(float),
    }
    finite_base = (
        np.isfinite(raw_gate_sum) & np.isfinite(final_gate_sum)
        & np.isfinite(dut_raw_sum) & np.isfinite(dut_final_sum))
    for feature, values in features.items():
        values = np.asarray(values, dtype=float)
        edges = _TMLPU_V117_FEATURE_BINS[feature]
        entries = diag['bins'][feature]
        for i, entry in enumerate(entries):
            lo = edges[i]
            hi = edges[i + 1]
            if i == len(entries) - 1:
                mask = (values >= lo) & (values <= hi)
            else:
                mask = (values >= lo) & (values < hi)
            mask = mask & np.isfinite(values) & finite_base
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            entry['count'] += count
            entry['signed_pair_count'] += int(np.count_nonzero(
                signed_pair & mask))
            entry['density_curve_raw_active_count'] += int(np.count_nonzero(
                curve_active & mask))
            entry['shocklike_count'] += int(np.count_nonzero(shocklike & mask))
            entry['signed_raw_gate_sum'] += float(np.sum(signed_raw[mask]))
            entry['signed_final_gate_sum'] += float(np.sum(signed_final[mask]))
            entry['density_curve_raw_gate_sum'] += float(np.sum(
                curve_raw[mask]))
            entry['density_curve_final_gate_sum'] += float(np.sum(
                curve_final[mask]))
            entry['raw_gate_sum'] += float(np.sum(raw_gate_sum[mask]))
            entry['final_gate_sum'] += float(np.sum(final_gate_sum[mask]))
            entry['d_ut_raw_abs_sum'] += float(np.sum(dut_raw_sum[mask]))
            entry['d_ut_final_abs_sum'] += float(np.sum(dut_final_sum[mask]))


def _tmlpu_v93_diag_update(
        x_face, y_face, base_pair_gate, signed_pair_tail_gate,
        density_curve_tail_gate, pressure_jump, compression, normality_here,
        shear_frac, omega_o, omega_n, qratio_o, qratio_n, density_support,
        signed_tail_dut_abs, density_curve_tail_dut_abs):
    global _TMLPU_V93_DIAG
    if not _tmlpu_v93_diag_enabled():
        return
    if _TMLPU_V93_DIAG is None:
        _TMLPU_V93_DIAG = _tmlpu_v93_diag_init()
    diag = _TMLPU_V93_DIAG
    diag['calls'] += 1
    n_faces = int(np.asarray(x_face).size)
    diag['total_faces'] += n_faces
    if n_faces == 0:
        return

    arrays = {
        'base_pair_gate': np.asarray(base_pair_gate, dtype=float),
        'signed_pair_tail_gate': np.asarray(signed_pair_tail_gate, dtype=float),
        'density_curve_tail_gate': np.asarray(density_curve_tail_gate, dtype=float),
        'pressure_jump': np.asarray(pressure_jump, dtype=float),
        'compression': np.asarray(compression, dtype=float),
        'normality_here': np.asarray(normality_here, dtype=float),
        'shear_frac': np.asarray(shear_frac, dtype=float),
        'omega_o_abs': np.abs(np.asarray(omega_o, dtype=float)),
        'omega_n_abs': np.abs(np.asarray(omega_n, dtype=float)),
        'qratio_o': np.asarray(qratio_o, dtype=float),
        'qratio_n': np.asarray(qratio_n, dtype=float),
        'density_support': np.asarray(density_support, dtype=float),
        'signed_tail_dut_abs': np.asarray(signed_tail_dut_abs, dtype=float),
        'density_curve_tail_dut_abs': np.asarray(
            density_curve_tail_dut_abs, dtype=float),
    }
    signed_pair = np.asarray(omega_o, dtype=float) * np.asarray(
        omega_n, dtype=float) < 0.0
    same_sign = ~signed_pair
    shocklike = (
        (arrays['pressure_jump'] >= 0.012)
        & (arrays['compression'] >= 0.0025)
        & (arrays['normality_here'] >= 0.18))
    x_face = np.asarray(x_face, dtype=float)
    y_face = np.asarray(y_face, dtype=float)

    idx = 0
    for x0, x1 in _TMLPU_V93_X_BANDS:
        for y0, y1 in _TMLPU_V93_Y_BANDS:
            mask = (
                (x_face >= x0) & (x_face < x1)
                & (y_face >= y0) & (y_face < y1))
            entry = diag['bins'][idx]
            idx += 1
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            entry['count'] += count
            entry['signed_pair_count'] += int(np.count_nonzero(
                signed_pair & mask))
            entry['same_sign_pair_count'] += int(np.count_nonzero(
                same_sign & mask))
            entry['signed_gate_active_count'] += int(np.count_nonzero(
                (arrays['signed_pair_tail_gate'] > 0.0) & mask))
            entry['density_curve_gate_active_count'] += int(np.count_nonzero(
                (arrays['density_curve_tail_gate'] > 0.0) & mask))
            entry['shocklike_count'] += int(np.count_nonzero(shocklike & mask))
            for key, arr in arrays.items():
                vals = arr[mask]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                entry['sums'][key] += float(np.sum(vals))
                entry['max'][key] = max(entry['max'][key], float(np.max(vals)))

try:  # Optional threaded kernels for expensive unstructured TMLP-u paths.
    from numba import njit, prange, set_num_threads
    _NUMBA_AVAILABLE = True
    _thread_env = (
        os.environ.get('TMLPU_SOLVER_THREADS')
        or os.environ.get('TMLPU_FLUX_THREADS')
        or os.environ.get('NUMBA_NUM_THREADS'))
    _default_threads = min(32, os.cpu_count() or 1)
    set_num_threads(max(1, min(_default_threads, int(
        _thread_env or _default_threads))))
except Exception:  # pragma: no cover - numba is an optional accelerator.
    njit = None
    prange = range
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _patch_contact_shear_side_postpass_kernel(
            W_cell, coeffs, interior, side_idx, dx_side,
            grad_nb_safe, grad_valid_nb, face_n_o,
            clean_contact, shock_off, rho_floor, p_floor,
            rho_lo, rho_hi, rho_avg, c_sum, W_side,
            gamma, entropy_alpha, tangential_alpha, neighbor_blend,
            s_cap, ut_cap, wave_cap, min_valid, roughness_cap,
            pair_spacing_on, pair_spacing_beta, gate_cap, pressure_margin_on,
            late_pressure_rollback_on, tangential_rollback_theta):
        n_face = interior.shape[0]
        max_nb = grad_nb_safe.shape[1]
        eps = 1.0e-30
        for k in prange(n_face):
            f = interior[k]
            c = side_idx[k]
            nx = face_n_o[k, 0]
            ny = face_n_o[k, 1]
            tx = -ny
            ty = nx

            rho_c = W_cell[0, c]
            if rho_c < eps:
                rho_c = eps
            p_c = W_cell[3, c]
            if p_c < eps:
                p_c = eps
            grad_s_x = coeffs[0, c, 0] / rho_c - coeffs[3, c, 0] / (gamma * p_c)
            grad_s_y = coeffs[0, c, 1] / rho_c - coeffs[3, c, 1] / (gamma * p_c)
            grad_ut_x = tx * coeffs[1, c, 0] + ty * coeffs[2, c, 0]
            grad_ut_y = tx * coeffs[1, c, 1] + ty * coeffs[2, c, 1]

            s_cell_t = grad_s_x * tx + grad_s_y * ty
            ut_cell_t = grad_ut_x * tx + grad_ut_y * ty
            ut_cell_n = grad_ut_x * nx + grad_ut_y * ny

            valid_count = 0
            consistent_count = 0
            same_stream_count = 0
            opposite_normal_count = 0
            avg_s_x = 0.0
            avg_s_y = 0.0
            avg_ut_x = 0.0
            avg_ut_y = 0.0
            for j in range(max_nb):
                if not grad_valid_nb[c, j]:
                    continue
                valid_count += 1
                nb = grad_nb_safe[c, j]
                nb_rho = W_cell[0, nb]
                if nb_rho < eps:
                    nb_rho = eps
                nb_p = W_cell[3, nb]
                if nb_p < eps:
                    nb_p = eps
                grad_s_nb_x = (
                    coeffs[0, nb, 0] / nb_rho
                    - coeffs[3, nb, 0] / (gamma * nb_p))
                grad_s_nb_y = (
                    coeffs[0, nb, 1] / nb_rho
                    - coeffs[3, nb, 1] / (gamma * nb_p))
                grad_ut_nb_x = (
                    tx * coeffs[1, nb, 0] + ty * coeffs[2, nb, 0])
                grad_ut_nb_y = (
                    tx * coeffs[1, nb, 1] + ty * coeffs[2, nb, 1])
                s_nb_t = grad_s_nb_x * tx + grad_s_nb_y * ty
                ut_nb_t = grad_ut_nb_x * tx + grad_ut_nb_y * ty
                ut_nb_n = grad_ut_nb_x * nx + grad_ut_nb_y * ny

                same_s = (
                    abs(s_cell_t) > eps and abs(s_nb_t) > eps
                    and ((s_cell_t > 0.0 and s_nb_t > 0.0)
                         or (s_cell_t < 0.0 and s_nb_t < 0.0)))
                same_ut_t = (
                    abs(ut_cell_t) > eps and abs(ut_nb_t) > eps
                    and ((ut_cell_t > 0.0 and ut_nb_t > 0.0)
                         or (ut_cell_t < 0.0 and ut_nb_t < 0.0)))
                if same_s:
                    same_stream_count += 1
                if (abs(ut_cell_n) > eps and abs(ut_nb_n) > eps
                        and ((ut_cell_n > 0.0 and ut_nb_n < 0.0)
                             or (ut_cell_n < 0.0 and ut_nb_n > 0.0))):
                    opposite_normal_count += 1
                if same_s and same_ut_t:
                    consistent_count += 1
                    avg_s_x += grad_s_nb_x
                    avg_s_y += grad_s_nb_y
                    avg_ut_x += grad_ut_nb_x
                    avg_ut_y += grad_ut_nb_y

            denom_count = valid_count
            if denom_count < 1:
                denom_count = 1
            patch_coherence = consistent_count / denom_count
            same_stream_contact = same_stream_count / denom_count
            opposite_normal_shear = opposite_normal_count / denom_count
            if consistent_count > 0:
                inv_count = 1.0 / consistent_count
                avg_s_x *= inv_count
                avg_s_y *= inv_count
                avg_ut_x *= inv_count
                avg_ut_y *= inv_count

            grad_s_patch_x = (1.0 - neighbor_blend) * grad_s_x + neighbor_blend * avg_s_x
            grad_s_patch_y = (1.0 - neighbor_blend) * grad_s_y + neighbor_blend * avg_s_y
            grad_ut_patch_x = (1.0 - neighbor_blend) * grad_ut_x + neighbor_blend * avg_ut_x
            grad_ut_patch_y = (1.0 - neighbor_blend) * grad_ut_y + neighbor_blend * avg_ut_y
            delta_s = grad_s_patch_x * dx_side[k, 0] + grad_s_patch_y * dx_side[k, 1]
            delta_ut = grad_ut_patch_x * dx_side[k, 0] + grad_ut_patch_y * dx_side[k, 1]

            t = (patch_coherence - 0.50) / 0.30
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            coherence_gate = t * t * (3.0 - 2.0 * t)
            gate = clean_contact[k] * shock_off[k] * coherence_gate
            if valid_count < min_valid:
                gate = 0.0
            if pair_spacing_on and pair_spacing_beta > 0.0:
                t1 = (same_stream_contact - 0.45) / 0.30
                if t1 < 0.0:
                    t1 = 0.0
                elif t1 > 1.0:
                    t1 = 1.0
                ssg = t1 * t1 * (3.0 - 2.0 * t1)
                t2 = (opposite_normal_shear - 0.35) / 0.30
                if t2 < 0.0:
                    t2 = 0.0
                elif t2 > 1.0:
                    t2 = 1.0
                ong = t2 * t2 * (3.0 - 2.0 * t2)
                gate *= 1.0 + pair_spacing_beta * ssg * ong
                if gate_cap > 0.0 and gate > gate_cap:
                    gate = gate_cap
            if gate <= 0.0:
                continue

            rho_anchor = W_side[0, f]
            if rho_anchor < eps:
                rho_anchor = eps
            u_anchor = W_side[1, f]
            v_anchor = W_side[2, f]
            p_anchor = W_side[3, f]
            if p_anchor < eps:
                p_anchor = eps
            if pair_spacing_on and pressure_margin_on:
                p_margin = p_anchor / p_floor[k] if p_floor[k] > eps else p_anchor / eps
                pm = (p_margin - 1.05) / 0.20
                if pm < 0.0:
                    pm = 0.0
                elif pm > 1.0:
                    pm = 1.0
                gate *= pm * pm * (3.0 - 2.0 * pm)
                if gate <= 0.0:
                    continue

            s_anchor = np.log(rho_anchor) - np.log(p_anchor) / gamma
            un_anchor = u_anchor * nx + v_anchor * ny
            ut_anchor = u_anchor * tx + v_anchor * ty

            if s_cap > 0.0:
                if delta_s < -s_cap:
                    delta_s = -s_cap
                elif delta_s > s_cap:
                    delta_s = s_cap
            else:
                delta_s = 0.0

            local_ut_cap = 0.0
            if ut_cap > 0.0 and wave_cap > 0.0:
                local_ut_cap = ut_cap
                wc = wave_cap * c_sum[k]
                if wc < local_ut_cap:
                    local_ut_cap = wc
            elif ut_cap > 0.0:
                local_ut_cap = ut_cap
            elif wave_cap > 0.0:
                local_ut_cap = wave_cap * c_sum[k]
            if local_ut_cap > 0.0:
                if delta_ut < -local_ut_cap:
                    delta_ut = -local_ut_cap
                elif delta_ut > local_ut_cap:
                    delta_ut = local_ut_cap
            else:
                delta_ut = 0.0

            s_trial = s_anchor + entropy_alpha * gate * delta_s
            ut_trial = ut_anchor + tangential_alpha * gate * delta_ut
            rho_trial = np.exp(s_trial + np.log(p_anchor) / gamma)
            u_trial = un_anchor * nx + ut_trial * tx
            v_trial = un_anchor * ny + ut_trial * ty

            rough_anchor_hi = rho_anchor - rho_hi[k]
            if rough_anchor_hi < 0.0:
                rough_anchor_hi = 0.0
            rough_anchor_lo = rho_lo[k] - rho_anchor
            if rough_anchor_lo < 0.0:
                rough_anchor_lo = 0.0
            rough_anchor = rough_anchor_hi
            if rough_anchor_lo > rough_anchor:
                rough_anchor = rough_anchor_lo
            rough_anchor /= rho_avg[k]

            rough_trial_hi = rho_trial - rho_hi[k]
            if rough_trial_hi < 0.0:
                rough_trial_hi = 0.0
            rough_trial_lo = rho_lo[k] - rho_trial
            if rough_trial_lo < 0.0:
                rough_trial_lo = 0.0
            rough_trial = rough_trial_hi
            if rough_trial_lo > rough_trial:
                rough_trial = rough_trial_lo
            rough_trial /= rho_avg[k]

            floor_bad = rho_trial < rho_floor[k] or p_anchor < p_floor[k]
            hard_reject = (
                shock_off[k] < 0.20
                or patch_coherence < 0.50
                or rough_trial > rough_anchor + roughness_cap)

            rho_final = rho_trial
            u_final = u_trial
            v_final = v_trial
            if late_pressure_rollback_on:
                rho_entropy_rollback = rho_anchor
                u_entropy_rollback = un_anchor * nx + ut_trial * tx
                v_entropy_rollback = un_anchor * ny + ut_trial * ty
                entropy_still_bad = (
                    rho_entropy_rollback < rho_floor[k]
                    or p_anchor < p_floor[k]
                    or hard_reject)
                ut_theta = ut_anchor + tangential_rollback_theta * (ut_trial - ut_anchor)
                u_tangent_rollback = un_anchor * nx + ut_theta * tx
                v_tangent_rollback = un_anchor * ny + ut_theta * ty
                tangent_still_bad = (
                    rho_entropy_rollback < rho_floor[k]
                    or p_anchor < p_floor[k]
                    or hard_reject)
                if floor_bad:
                    rho_final = rho_entropy_rollback
                    u_final = u_entropy_rollback
                    v_final = v_entropy_rollback
                if floor_bad and entropy_still_bad:
                    rho_final = rho_entropy_rollback
                    u_final = u_tangent_rollback
                    v_final = v_tangent_rollback
                if floor_bad and entropy_still_bad and tangent_still_bad:
                    rho_final = rho_anchor
                    u_final = u_anchor
                    v_final = v_anchor
            elif pair_spacing_on:
                s_rollback = (
                    floor_bad or rough_trial > rough_anchor + roughness_cap)
                full_rollback = (
                    p_anchor < p_floor[k]
                    or shock_off[k] < 0.20
                    or patch_coherence < 0.50)
                if s_rollback:
                    rho_final = rho_anchor
                    u_final = un_anchor * nx + ut_trial * tx
                    v_final = un_anchor * ny + ut_trial * ty
                if full_rollback:
                    rho_final = rho_anchor
                    u_final = u_anchor
                    v_final = v_anchor
            else:
                denom = rho_anchor - rho_trial
                theta = 1.0
                if rho_trial < rho_floor[k] and denom > eps:
                    theta = (rho_anchor - rho_floor[k]) / denom
                if theta < 0.0:
                    theta = 0.0
                elif theta > 1.0:
                    theta = 1.0
                if p_anchor < p_floor[k]:
                    theta = 0.0
                if hard_reject and not floor_bad:
                    theta = 0.0
                rho_final = rho_anchor + theta * (rho_trial - rho_anchor)
                u_final = u_anchor + theta * (u_trial - u_anchor)
                v_final = v_anchor + theta * (v_trial - v_anchor)

            W_side[0, f] = rho_final
            W_side[1, f] = u_final
            W_side[2, f] = v_final
            W_side[3, f] = p_anchor

    @njit(parallel=True, cache=True)
    def _tmlpu_vertex_psi_nodes_kernel(
            phi_l, delta_plus, tstar, d_lr, grad_corr_x, grad_corr_y,
            r_tmlpu, psi_tvd, node_safe, node_valid, d_vi,
            vertex_min, vertex_max, tvb_eps, phys_mask,
            physical_vertex_bounds, physical_value_bounds,
            use_augment, vertex_grad_x, vertex_grad_y, tvd_bounded):
        n_face = phi_l.shape[0]
        n_node = node_safe.shape[1]
        out = np.empty(n_face, dtype=np.float64)
        eps = 1.0e-30
        for i in prange(n_face):
            cap = psi_tvd[i]
            if not np.isfinite(cap):
                cap = 2.0 if cap > 0.0 else 0.0
            if cap < 0.0:
                cap = 0.0
            elif cap > 2.0:
                cap = 2.0
            r = r_tmlpu[i]
            r_pos = r if r > 0.0 else 0.0
            min1r = r_pos if r_pos < 1.0 else 1.0
            monotone = r > eps
            best = cap
            for j in range(n_node):
                if not node_valid[i, j]:
                    psi = cap
                else:
                    node = node_safe[i, j]
                    d_corr_x = d_vi[i, j, 0] - tstar[i] * d_lr[i, 0]
                    d_corr_y = d_vi[i, j, 1] - tstar[i] * d_lr[i, 1]
                    delta_vi = (tstar[i] * delta_plus[i]
                                + grad_corr_x[i] * d_corr_x
                                + grad_corr_y[i] * d_corr_y)
                    if physical_vertex_bounds or (physical_value_bounds and phys_mask[i]):
                        vmin = 0.0
                        vmax = 1.0
                    else:
                        vmin = vertex_min[node] - tvb_eps
                        vmax = vertex_max[node] + tvb_eps
                    if delta_vi >= 0.0:
                        allowed = vmax - phi_l[i]
                    else:
                        allowed = vmin - phi_l[i]

                    if abs(delta_vi) > eps:
                        base = allowed / delta_vi
                        if base <= 0.0 or not np.isfinite(base):
                            base = 0.0
                    else:
                        base = 2.0
                    if tvd_bounded:
                        psi = base if base < cap else cap
                    else:
                        denom = delta_vi * min1r
                        if abs(denom) > eps:
                            alpha = allowed / denom
                            if alpha <= 0.0 or not np.isfinite(alpha):
                                alpha = 0.0
                        else:
                            alpha = 2.0
                        if r_pos <= 1.0:
                            alpha_r = base
                        else:
                            alpha_r = base * r_pos
                        psi = alpha_r if alpha_r < alpha else alpha
                        if cap < psi:
                            psi = cap
                        if not monotone:
                            psi = 0.0

                    if use_augment:
                        avg_proj = (vertex_grad_x[node] * d_vi[i, j, 0]
                                    + vertex_grad_y[node] * d_vi[i, j, 1])
                        augment_monotone = tvd_bounded or monotone
                        tol = eps * (1.0 + abs(phi_l[i]) + abs(vmin) + abs(vmax))
                        if (avg_proj * delta_vi > 0.0
                                and abs(avg_proj) >= 0.5 * abs(delta_vi)
                                and augment_monotone
                                and abs(allowed) <= tol):
                            psi = 1.0
                    if not np.isfinite(psi):
                        psi = 0.0
                    if psi < 0.0:
                        psi = 0.0
                    elif psi > 2.0:
                        psi = 2.0
                if psi < best:
                    best = psi
            out[i] = best
        return out


    @njit(parallel=True, cache=True)
    def _tmlpu_cell_bounds_coeffs_kernel(
            W_cell, nb_safe, valid_nb,
            grad_nb_safe, grad_valid_nb, grad_sqrt_w, grad_lsq_op,
            phi_min_cell, phi_max_cell, coeffs):
        nvar = W_cell.shape[0]
        n_cell = W_cell.shape[1]
        max_nb = nb_safe.shape[1]
        grad_max_nb = grad_nb_safe.shape[1]
        nbasis = coeffs.shape[2]
        total = nvar * n_cell
        for idx in prange(total):
            v = idx // n_cell
            c = idx - v * n_cell
            phi_c = W_cell[v, c]
            lo = phi_c
            hi = phi_c
            for b in range(nbasis):
                coeffs[v, c, b] = 0.0
            for k in range(max_nb):
                if valid_nb[c, k]:
                    nb = nb_safe[c, k]
                    phi_nb = W_cell[v, nb]
                    if phi_nb < lo:
                        lo = phi_nb
                    elif phi_nb > hi:
                        hi = phi_nb
            phi_min_cell[v, c] = lo
            phi_max_cell[v, c] = hi
            for k in range(grad_max_nb):
                if grad_valid_nb[c, k]:
                    nb = grad_nb_safe[c, k]
                    delta_w = (W_cell[v, nb] - phi_c) * grad_sqrt_w[c, k]
                    for b in range(nbasis):
                        coeffs[v, c, b] += grad_lsq_op[c, b, k] * delta_w


    @njit(parallel=True, cache=True)
    def _tmlpu_active_cell_bounds_coeffs_kernel(
            W_cell, active_vars, nb_safe, valid_nb,
            grad_nb_safe, grad_valid_nb, grad_sqrt_w, grad_lsq_op,
            phi_min_cell, phi_max_cell, coeffs):
        n_cell = W_cell.shape[1]
        max_nb = nb_safe.shape[1]
        grad_max_nb = grad_nb_safe.shape[1]
        nbasis = coeffs.shape[2]
        total = active_vars.shape[0] * n_cell
        for idx in prange(total):
            av = idx // n_cell
            c = idx - av * n_cell
            v = active_vars[av]
            phi_c = W_cell[v, c]
            lo = phi_c
            hi = phi_c
            for b in range(nbasis):
                coeffs[v, c, b] = 0.0
            for k in range(max_nb):
                if valid_nb[c, k]:
                    phi_nb = W_cell[v, nb_safe[c, k]]
                    if phi_nb < lo:
                        lo = phi_nb
                    elif phi_nb > hi:
                        hi = phi_nb
            phi_min_cell[v, c] = lo
            phi_max_cell[v, c] = hi
            for k in range(grad_max_nb):
                if grad_valid_nb[c, k]:
                    nb = grad_nb_safe[c, k]
                    delta_w = (W_cell[v, nb] - phi_c) * grad_sqrt_w[c, k]
                    for b in range(nbasis):
                        coeffs[v, c, b] += grad_lsq_op[c, b, k] * delta_w


    @njit(parallel=True, cache=True)
    def _tmlpu_vertex_minmax_kernel(
            W_cell, v2c_safe, v2c_valid, vertex_min_values,
            vertex_max_values):
        nvar = W_cell.shape[0]
        n_node = v2c_safe.shape[0]
        max_v2c = v2c_safe.shape[1]
        total = nvar * n_node
        for idx in prange(total):
            v = idx // n_node
            node = idx - v * n_node
            fill_cell = v2c_safe[node, 0]
            fill_value = W_cell[v, fill_cell]
            lo = fill_value
            hi = fill_value
            for k in range(max_v2c):
                if v2c_valid[node, k]:
                    phi = W_cell[v, v2c_safe[node, k]]
                else:
                    phi = fill_value
                if phi < lo:
                    lo = phi
                elif phi > hi:
                    hi = phi
            vertex_min_values[v, node] = lo
            vertex_max_values[v, node] = hi


    @njit(parallel=True, cache=True)
    def _tmlpu_minmod_vertex_faces_kernel(
            W_cell, coeffs, phi_min_cell, phi_max_cell,
            owner_int, nei_int, face_ids, d_o_int, alpha_o, alpha_n,
            dx_fo, dx_fn,
            cell_node_safe, cell_node_valid, vertex_offsets,
            vertex_min_values, vertex_max_values, shock_flatten,
            pressure_flatten, velocity_flatten, density_flatten,
            density_contact_weight,
            density_contact_hancock_scale, one_minus_C_face, tvb_eps,
            apply_pressure_flatten, apply_density_flatten,
            use_density_lsq_increment, use_density_no_hancock,
            use_density_entropy_split, use_density_contact_wave_hancock,
            use_density_first_order,
            use_pressure_first_order,
            use_velocity_no_hancock, use_velocity_shock_flatten,
            use_velocity_lsq_increment,
            velocity_tvd_mode, density_tvd_mode,
            use_density_extrema_lmp, use_velocity_extrema_lmp,
            gamma, W_L, W_R):
        n_face = owner_int.shape[0]
        nvar = W_cell.shape[0]
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        for k in prange(n_face):
            f = face_ids[k]
            o = owner_int[k]
            n = nei_int[k]
            dx = d_o_int[k, 0]
            dy = d_o_int[k, 1]
            ao = alpha_o[k]
            an = alpha_n[k]
            sh = shock_flatten[k]
            p_sh = pressure_flatten[k]
            vel_sh = velocity_flatten[k]
            rho_sh = density_flatten[k]
            rho_contact = density_contact_weight[k]
            for v in range(nvar):
                # Owner side: U=o, D=n, d_LR=o->n.
                phi_u = W_cell[v, o]
                delta_plus = W_cell[v, n] - phi_u
                gx_u = coeffs[v, o, 0]
                gy_u = coeffs[v, o, 1]
                gx_corr = 0.5 * (coeffs[v, o, 0] + coeffs[v, n, 0])
                gy_corr = 0.5 * (coeffs[v, o, 1] + coeffs[v, n, 1])
                density_var = (v == 0)
                pressure_var = (v == 3)
                velocity_var = (v == 1 or v == 2)
                scale = one_minus_C_face[k]
                if use_density_no_hancock and density_var:
                    scale = 1.0 - (1.0 - one_minus_C_face[k]) * rho_sh
                if use_density_contact_wave_hancock and density_var:
                    cscale = density_contact_hancock_scale[k]
                    scale = scale + rho_contact * (cscale - scale)
                if use_velocity_no_hancock and velocity_var:
                    scale = 1.0 - (1.0 - one_minus_C_face[k]) * sh
                if use_density_first_order and density_var:
                    delta = 0.0
                elif use_pressure_first_order and pressure_var:
                    delta = 0.0
                elif use_density_entropy_split and density_var:
                    p_u = W_cell[3, o]
                    if p_u < eps:
                        p_u = eps
                    rho_u = W_cell[0, o]
                    if rho_u < eps:
                        rho_u = eps
                    c_sq = gamma * p_u / rho_u
                    if c_sq < eps:
                        c_sq = eps
                    delta_line = ao * delta_plus
                    delta_p = ao * (W_cell[3, n] - W_cell[3, o])
                    delta_acoustic = delta_p / c_sq
                    delta_entropy = delta_line - delta_acoustic
                    if delta_entropy * delta_line <= 0.0:
                        delta_entropy = 0.0
                    elif abs(delta_entropy) > abs(delta_line):
                        delta_entropy = delta_line
                    w_contact = rho_contact
                    if w_contact < 0.0:
                        w_contact = 0.0
                    elif w_contact > 1.0:
                        w_contact = 1.0
                    entropy_scale = scale + (1.0 - scale) * w_contact
                    delta = scale * delta_acoustic + entropy_scale * delta_entropy
                    if delta * delta_line <= 0.0:
                        delta = 0.0
                    elif abs(delta) > abs(delta_line):
                        delta = delta_line
                elif use_density_lsq_increment and density_var:
                    delta_line = ao * delta_plus
                    delta_lsq = gx_u * dx_fo[k, 0] + gy_u * dx_fo[k, 1]
                    w_contact = rho_contact
                    if w_contact < 0.0:
                        w_contact = 0.0
                    elif w_contact > 1.0:
                        w_contact = 1.0
                    delta = scale * (
                        delta_line + w_contact * (delta_lsq - delta_line))
                elif use_velocity_lsq_increment and velocity_var:
                    delta = scale * (
                        gx_u * dx_fo[k, 0] + gy_u * dx_fo[k, 1])
                else:
                    delta = scale * ao * delta_plus
                phi_ll = phi_u - (gx_u * dx + gy_u * dy)
                lo = phi_min_cell[v, o] - tvb_eps
                hi = phi_max_cell[v, o] + tvb_eps
                if phi_ll < lo:
                    phi_ll = lo
                elif phi_ll > hi:
                    phi_ll = hi
                delta_minus = phi_u - phi_ll
                den_floor = (64.0 * macheps
                             * (1.0 + abs(phi_u) + abs(phi_ll)
                                + abs(delta_plus)))
                if abs(delta_minus) > den_floor:
                    den = delta_minus
                else:
                    den = den_floor if delta_minus >= 0.0 else -den_floor
                r = delta_plus / den
                if abs(delta_plus) <= eps:
                    psi_tvd = 2.0
                else:
                    if ((velocity_tvd_mode == 1 and (v == 1 or v == 2))
                            or (density_tvd_mode == 1 and v == 0)):
                        a = 2.0 * r
                        b = 0.5 * (1.0 + r)
                        psi_tvd = a if a < b else b
                        if psi_tvd > 2.0:
                            psi_tvd = 2.0
                        if psi_tvd < 0.0:
                            psi_tvd = 0.0
                    elif velocity_tvd_mode == 2 and (v == 1 or v == 2):
                        ar = abs(r)
                        psi_tvd = (r + ar) / (1.0 + ar)
                    elif density_tvd_mode == 2 and v == 0:
                        psi_min = r
                        if psi_min < 0.0:
                            psi_min = 0.0
                        elif psi_min > 1.0:
                            psi_min = 1.0
                        ar = abs(r)
                        psi_vl = (r + ar) / (1.0 + ar)
                        w = rho_contact
                        if w < 0.0:
                            w = 0.0
                        elif w > 1.0:
                            w = 1.0
                        psi_tvd = psi_min + w * (psi_vl - psi_min)
                    elif density_tvd_mode == 3 and v == 0:
                        psi_min = r
                        if psi_min < 0.0:
                            psi_min = 0.0
                        elif psi_min > 1.0:
                            psi_min = 1.0
                        a = 2.0 * r
                        b = 0.25 + 0.75 * r
                        c = 0.75 + 0.25 * r
                        psi_umist = a
                        if b < psi_umist:
                            psi_umist = b
                        if c < psi_umist:
                            psi_umist = c
                        if psi_umist > 2.0:
                            psi_umist = 2.0
                        if psi_umist < 0.0:
                            psi_umist = 0.0
                        w = rho_contact
                        if w < 0.0:
                            w = 0.0
                        elif w > 1.0:
                            w = 1.0
                        psi_tvd = psi_min + w * (psi_umist - psi_min)
                    elif density_tvd_mode == 4 and v == 0:
                        psi_min = r
                        if psi_min < 0.0:
                            psi_min = 0.0
                        elif psi_min > 1.0:
                            psi_min = 1.0
                        a = 2.0 * r
                        if a > 1.0:
                            a = 1.0
                        b = r
                        if b > 2.0:
                            b = 2.0
                        psi_sb = a if a > b else b
                        if psi_sb < 0.0:
                            psi_sb = 0.0
                        elif psi_sb > 2.0:
                            psi_sb = 2.0
                        w = rho_contact
                        if w < 0.0:
                            w = 0.0
                        elif w > 1.0:
                            w = 1.0
                        psi_tvd = psi_min + w * (psi_sb - psi_min)
                    elif velocity_tvd_mode == 3 and (v == 1 or v == 2):
                        a = 2.0 * r
                        b = 0.25 + 0.75 * r
                        c = 0.75 + 0.25 * r
                        psi_tvd = a
                        if b < psi_tvd:
                            psi_tvd = b
                        if c < psi_tvd:
                            psi_tvd = c
                        if psi_tvd > 2.0:
                            psi_tvd = 2.0
                        if psi_tvd < 0.0:
                            psi_tvd = 0.0
                    elif velocity_tvd_mode == 4 and (v == 1 or v == 2):
                        a = 2.0 * r
                        if a > 1.0:
                            a = 1.0
                        b = r
                        if b > 2.0:
                            b = 2.0
                        psi_tvd = a if a > b else b
                        if psi_tvd < 0.0:
                            psi_tvd = 0.0
                        elif psi_tvd > 2.0:
                            psi_tvd = 2.0
                    else:
                        psi_tvd = r
                        if psi_tvd < 0.0:
                            psi_tvd = 0.0
                        elif psi_tvd > 1.0:
                            psi_tvd = 1.0
                if psi_tvd < 0.0:
                    psi_tvd = 0.0
                elif psi_tvd > 2.0:
                    psi_tvd = 2.0
                if use_velocity_extrema_lmp and velocity_var and r <= eps:
                    psi_tvd = 1.0
                if use_density_extrema_lmp and density_var and r <= eps:
                    psi_tvd = 1.0
                r_pos = r if r > 0.0 else 0.0
                min1r = r_pos if r_pos < 1.0 else 1.0
                monotone = r > eps
                psi_best = psi_tvd
                for j in range(cell_node_safe.shape[1]):
                    if not cell_node_valid[o, j]:
                        continue
                    node = cell_node_safe[o, j]
                    dvi_x = vertex_offsets[o, j, 0]
                    dvi_y = vertex_offsets[o, j, 1]
                    dcor_x = dvi_x - ao * dx
                    dcor_y = dvi_y - ao * dy
                    delta_vi = (ao * delta_plus
                                + gx_corr * dcor_x + gy_corr * dcor_y)
                    vmin = vertex_min_values[v, node] - tvb_eps
                    vmax = vertex_max_values[v, node] + tvb_eps
                    allowed = vmax - phi_u if delta_vi >= 0.0 else vmin - phi_u
                    if abs(delta_vi) > eps:
                        base = allowed / delta_vi
                        if base <= 0.0 or not np.isfinite(base):
                            base = 0.0
                    else:
                        base = 2.0
                    if ((use_velocity_extrema_lmp and velocity_var)
                            or (use_density_extrema_lmp and density_var)):
                        psi = base if base < psi_tvd else psi_tvd
                    else:
                        denom = delta_vi * min1r
                        if abs(denom) > eps:
                            alpha = allowed / denom
                            if alpha <= 0.0 or not np.isfinite(alpha):
                                alpha = 0.0
                        else:
                            alpha = 2.0
                        alpha_r = base if r_pos <= 1.0 else base * r_pos
                        psi = alpha_r if alpha_r < alpha else alpha
                        if psi_tvd < psi:
                            psi = psi_tvd
                        if not monotone:
                            psi = 0.0
                    if psi < 0.0:
                        psi = 0.0
                    elif psi > 2.0:
                        psi = 2.0
                    if psi < psi_best:
                        psi_best = psi
                if apply_density_flatten and v == 0:
                    psi_best = psi_best * (1.0 - rho_sh)
                elif apply_pressure_flatten and v == 3:
                    psi_best = psi_best * (1.0 - p_sh)
                if use_velocity_shock_flatten and velocity_var:
                    psi_best = psi_best * (1.0 - vel_sh)
                W_L[v, f] = phi_u + psi_best * delta

                # Neighbour side: U=n, D=o, d_LR=n->o.
                phi_u = W_cell[v, n]
                delta_plus = W_cell[v, o] - phi_u
                gx_u = coeffs[v, n, 0]
                gy_u = coeffs[v, n, 1]
                gx_corr = 0.5 * (coeffs[v, n, 0] + coeffs[v, o, 0])
                gy_corr = 0.5 * (coeffs[v, n, 1] + coeffs[v, o, 1])
                density_var = (v == 0)
                pressure_var = (v == 3)
                velocity_var = (v == 1 or v == 2)
                scale = one_minus_C_face[k]
                if use_density_no_hancock and density_var:
                    scale = 1.0 - (1.0 - one_minus_C_face[k]) * rho_sh
                if use_density_contact_wave_hancock and density_var:
                    cscale = density_contact_hancock_scale[k]
                    scale = scale + rho_contact * (cscale - scale)
                if use_velocity_no_hancock and velocity_var:
                    scale = 1.0 - (1.0 - one_minus_C_face[k]) * sh
                if use_density_first_order and density_var:
                    delta = 0.0
                elif use_pressure_first_order and pressure_var:
                    delta = 0.0
                elif use_density_entropy_split and density_var:
                    p_u = W_cell[3, n]
                    if p_u < eps:
                        p_u = eps
                    rho_u = W_cell[0, n]
                    if rho_u < eps:
                        rho_u = eps
                    c_sq = gamma * p_u / rho_u
                    if c_sq < eps:
                        c_sq = eps
                    delta_line = an * delta_plus
                    delta_p = an * (W_cell[3, o] - W_cell[3, n])
                    delta_acoustic = delta_p / c_sq
                    delta_entropy = delta_line - delta_acoustic
                    if delta_entropy * delta_line <= 0.0:
                        delta_entropy = 0.0
                    elif abs(delta_entropy) > abs(delta_line):
                        delta_entropy = delta_line
                    w_contact = rho_contact
                    if w_contact < 0.0:
                        w_contact = 0.0
                    elif w_contact > 1.0:
                        w_contact = 1.0
                    entropy_scale = scale + (1.0 - scale) * w_contact
                    delta = scale * delta_acoustic + entropy_scale * delta_entropy
                    if delta * delta_line <= 0.0:
                        delta = 0.0
                    elif abs(delta) > abs(delta_line):
                        delta = delta_line
                elif use_density_lsq_increment and density_var:
                    delta_line = an * delta_plus
                    delta_lsq = gx_u * dx_fn[k, 0] + gy_u * dx_fn[k, 1]
                    w_contact = rho_contact
                    if w_contact < 0.0:
                        w_contact = 0.0
                    elif w_contact > 1.0:
                        w_contact = 1.0
                    delta = scale * (
                        delta_line + w_contact * (delta_lsq - delta_line))
                elif use_velocity_lsq_increment and velocity_var:
                    delta = scale * (
                        gx_u * dx_fn[k, 0] + gy_u * dx_fn[k, 1])
                else:
                    delta = scale * an * delta_plus
                ndx = -dx
                ndy = -dy
                phi_ll = phi_u - (gx_u * ndx + gy_u * ndy)
                lo = phi_min_cell[v, n] - tvb_eps
                hi = phi_max_cell[v, n] + tvb_eps
                if phi_ll < lo:
                    phi_ll = lo
                elif phi_ll > hi:
                    phi_ll = hi
                delta_minus = phi_u - phi_ll
                den_floor = (64.0 * macheps
                             * (1.0 + abs(phi_u) + abs(phi_ll)
                                + abs(delta_plus)))
                if abs(delta_minus) > den_floor:
                    den = delta_minus
                else:
                    den = den_floor if delta_minus >= 0.0 else -den_floor
                r = delta_plus / den
                if abs(delta_plus) <= eps:
                    psi_tvd = 2.0
                else:
                    if ((velocity_tvd_mode == 1 and (v == 1 or v == 2))
                            or (density_tvd_mode == 1 and v == 0)):
                        a = 2.0 * r
                        b = 0.5 * (1.0 + r)
                        psi_tvd = a if a < b else b
                        if psi_tvd > 2.0:
                            psi_tvd = 2.0
                        if psi_tvd < 0.0:
                            psi_tvd = 0.0
                    elif velocity_tvd_mode == 2 and (v == 1 or v == 2):
                        ar = abs(r)
                        psi_tvd = (r + ar) / (1.0 + ar)
                    elif density_tvd_mode == 2 and v == 0:
                        psi_min = r
                        if psi_min < 0.0:
                            psi_min = 0.0
                        elif psi_min > 1.0:
                            psi_min = 1.0
                        ar = abs(r)
                        psi_vl = (r + ar) / (1.0 + ar)
                        w = rho_contact
                        if w < 0.0:
                            w = 0.0
                        elif w > 1.0:
                            w = 1.0
                        psi_tvd = psi_min + w * (psi_vl - psi_min)
                    elif density_tvd_mode == 3 and v == 0:
                        psi_min = r
                        if psi_min < 0.0:
                            psi_min = 0.0
                        elif psi_min > 1.0:
                            psi_min = 1.0
                        a = 2.0 * r
                        b = 0.25 + 0.75 * r
                        c = 0.75 + 0.25 * r
                        psi_umist = a
                        if b < psi_umist:
                            psi_umist = b
                        if c < psi_umist:
                            psi_umist = c
                        if psi_umist > 2.0:
                            psi_umist = 2.0
                        if psi_umist < 0.0:
                            psi_umist = 0.0
                        w = rho_contact
                        if w < 0.0:
                            w = 0.0
                        elif w > 1.0:
                            w = 1.0
                        psi_tvd = psi_min + w * (psi_umist - psi_min)
                    elif density_tvd_mode == 4 and v == 0:
                        psi_min = r
                        if psi_min < 0.0:
                            psi_min = 0.0
                        elif psi_min > 1.0:
                            psi_min = 1.0
                        a = 2.0 * r
                        if a > 1.0:
                            a = 1.0
                        b = r
                        if b > 2.0:
                            b = 2.0
                        psi_sb = a if a > b else b
                        if psi_sb < 0.0:
                            psi_sb = 0.0
                        elif psi_sb > 2.0:
                            psi_sb = 2.0
                        w = rho_contact
                        if w < 0.0:
                            w = 0.0
                        elif w > 1.0:
                            w = 1.0
                        psi_tvd = psi_min + w * (psi_sb - psi_min)
                    elif velocity_tvd_mode == 3 and (v == 1 or v == 2):
                        a = 2.0 * r
                        b = 0.25 + 0.75 * r
                        c = 0.75 + 0.25 * r
                        psi_tvd = a
                        if b < psi_tvd:
                            psi_tvd = b
                        if c < psi_tvd:
                            psi_tvd = c
                        if psi_tvd > 2.0:
                            psi_tvd = 2.0
                        if psi_tvd < 0.0:
                            psi_tvd = 0.0
                    elif velocity_tvd_mode == 4 and (v == 1 or v == 2):
                        a = 2.0 * r
                        if a > 1.0:
                            a = 1.0
                        b = r
                        if b > 2.0:
                            b = 2.0
                        psi_tvd = a if a > b else b
                        if psi_tvd < 0.0:
                            psi_tvd = 0.0
                        elif psi_tvd > 2.0:
                            psi_tvd = 2.0
                    else:
                        psi_tvd = r
                        if psi_tvd < 0.0:
                            psi_tvd = 0.0
                        elif psi_tvd > 1.0:
                            psi_tvd = 1.0
                if psi_tvd < 0.0:
                    psi_tvd = 0.0
                elif psi_tvd > 2.0:
                    psi_tvd = 2.0
                if use_velocity_extrema_lmp and velocity_var and r <= eps:
                    psi_tvd = 1.0
                if use_density_extrema_lmp and density_var and r <= eps:
                    psi_tvd = 1.0
                r_pos = r if r > 0.0 else 0.0
                min1r = r_pos if r_pos < 1.0 else 1.0
                monotone = r > eps
                psi_best = psi_tvd
                for j in range(cell_node_safe.shape[1]):
                    if not cell_node_valid[n, j]:
                        continue
                    node = cell_node_safe[n, j]
                    dvi_x = vertex_offsets[n, j, 0]
                    dvi_y = vertex_offsets[n, j, 1]
                    dcor_x = dvi_x - an * ndx
                    dcor_y = dvi_y - an * ndy
                    delta_vi = (an * delta_plus
                                + gx_corr * dcor_x + gy_corr * dcor_y)
                    vmin = vertex_min_values[v, node] - tvb_eps
                    vmax = vertex_max_values[v, node] + tvb_eps
                    allowed = vmax - phi_u if delta_vi >= 0.0 else vmin - phi_u
                    if abs(delta_vi) > eps:
                        base = allowed / delta_vi
                        if base <= 0.0 or not np.isfinite(base):
                            base = 0.0
                    else:
                        base = 2.0
                    if ((use_velocity_extrema_lmp and velocity_var)
                            or (use_density_extrema_lmp and density_var)):
                        psi = base if base < psi_tvd else psi_tvd
                    else:
                        denom = delta_vi * min1r
                        if abs(denom) > eps:
                            alpha = allowed / denom
                            if alpha <= 0.0 or not np.isfinite(alpha):
                                alpha = 0.0
                        else:
                            alpha = 2.0
                        alpha_r = base if r_pos <= 1.0 else base * r_pos
                        psi = alpha_r if alpha_r < alpha else alpha
                        if psi_tvd < psi:
                            psi = psi_tvd
                        if not monotone:
                            psi = 0.0
                    if psi < 0.0:
                        psi = 0.0
                    elif psi > 2.0:
                        psi = 2.0
                    if psi < psi_best:
                        psi_best = psi
                if apply_density_flatten and v == 0:
                    psi_best = psi_best * (1.0 - rho_sh)
                elif apply_pressure_flatten and v == 3:
                    psi_best = psi_best * (1.0 - p_sh)
                if use_velocity_shock_flatten and velocity_var:
                    psi_best = psi_best * (1.0 - vel_sh)
                W_R[v, f] = phi_u + psi_best * delta


    @njit(cache=True)
    def _tmlpu_fast_tvd_value(r, v, base_mode, velocity_mode, density_mode,
                              density_contact):
        """Small numba-side TVD dispatch used by face-loop kernels.

        mode: 0=minmod, 1=MC, 2=van Leer, 3=UMIST, 4=superbee,
              5=bounded central/downwind cap, 6=Koren,
              16=modified-SUPERBEE.
        """
        mode = base_mode
        if (v == 1 or v == 2) and velocity_mode > 0:
            mode = velocity_mode
        elif v == 0 and density_mode > 0:
            mode = density_mode

        if mode == 5:
            return 1.0
        if mode == 1:
            psi = 2.0 * r
            b = 0.5 * (1.0 + r)
            if b < psi:
                psi = b
        elif mode == 2:
            ar = abs(r)
            psi = (r + ar) / (1.0 + ar)
        elif mode == 3:
            psi = 2.0 * r
            b = 0.25 + 0.75 * r
            c = 0.75 + 0.25 * r
            if b < psi:
                psi = b
            if c < psi:
                psi = c
        elif mode == 4:
            a = 2.0 * r
            if a > 1.0:
                a = 1.0
            b = r
            if b > 2.0:
                b = 2.0
            psi = a if a > b else b
        elif mode == 6:
            a = 2.0 * r
            b = (1.0 + 2.0 * r) / 3.0
            psi = a if a < b else b
        elif mode == 7:
            psi = 2.0 * r
            if psi > 2.0:
                psi = 2.0
        elif mode == 16:
            a = 1.5 * r
            if a > 1.0:
                a = 1.0
            b = r
            if b > 1.5:
                b = 1.5
            psi = a if a > b else b
        else:
            psi = r
            if psi > 1.0:
                psi = 1.0

        if v == 0 and density_mode in (2, 3, 4):
            psi_min = r
            if psi_min < 0.0:
                psi_min = 0.0
            elif psi_min > 1.0:
                psi_min = 1.0
            w = density_contact
            if w < 0.0:
                w = 0.0
            elif w > 1.0:
                w = 1.0
            psi = psi_min + w * (psi - psi_min)

        if psi < 0.0:
            psi = 0.0
        elif psi > 2.0:
            psi = 2.0
        return psi


    @njit(parallel=True, cache=True)
    def _tmlpu_jasak_vertex_faces_kernel(
            W_cell, coeffs, phi_min_cell, phi_max_cell,
            owner_int, nei_int, face_ids, d_o_int, tstar_o, tstar_n,
            dx_fo, dx_fn, d_len, e_o, e_n, face_n_o,
            face_gradient_mode, theta_min, zero_delta_psi,
            use_bound_tvd_separate,
            active_vars, cell_node_safe, cell_node_valid, vertex_offsets,
            vertex_min_values, vertex_max_values, shock_flatten,
            pressure_flatten, velocity_flatten, density_flatten,
            density_contact_weight, tangential_contact_weight,
            density_contact_hancock_scale, one_minus_C_face,
            face_hancock_courant, tvb_eps,
            use_density_no_hancock, use_density_contact_wave_hancock,
            density_contact_hancock_boost, density_contact_hancock_boost_cap,
            use_density_first_order, use_pressure_first_order,
            use_velocity_no_hancock, use_velocity_shock_flatten,
            use_density_lsq_increment, use_density_full_lsq_increment,
            use_velocity_lsq_increment, use_pressure_shear_lsq_increment,
            use_pressure_nonshock_lsq_increment,
            face_increment_mode, base_tvd_mode, velocity_tvd_mode,
            density_tvd_mode, use_density_extrema_lmp,
            use_velocity_extrema_lmp, density_delta_out,
            density_psi_tvd_out, density_psi_base_out, W_L, W_R):
        n_face = owner_int.shape[0]
        n_active = active_vars.shape[0]
        n_node = cell_node_safe.shape[1]
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        total = n_face * n_active * 2
        for idx in prange(total):
            side = idx & 1
            tmp = idx >> 1
            av = tmp % n_active
            k = tmp // n_active
            v = active_vars[av]
            f = face_ids[k]
            o = owner_int[k]
            n = nei_int[k]

            if side == 0:
                ucell = o
                dcell = n
                dx = d_o_int[k, 0]
                dy = d_o_int[k, 1]
                dx_f0 = dx_fo[k, 0]
                dy_f0 = dx_fo[k, 1]
                tstar = tstar_o[k]
                ex = e_o[k, 0]
                ey = e_o[k, 1]
                nx = face_n_o[k, 0]
                ny = face_n_o[k, 1]
            else:
                ucell = n
                dcell = o
                dx = -d_o_int[k, 0]
                dy = -d_o_int[k, 1]
                dx_f0 = dx_fn[k, 0]
                dy_f0 = dx_fn[k, 1]
                tstar = tstar_n[k]
                ex = e_n[k, 0]
                ey = e_n[k, 1]
                nx = -face_n_o[k, 0]
                ny = -face_n_o[k, 1]

            phi_u = W_cell[v, ucell]
            delta_plus = W_cell[v, dcell] - phi_u
            gx_u = coeffs[v, ucell, 0]
            gy_u = coeffs[v, ucell, 1]
            gx_d = coeffs[v, dcell, 0]
            gy_d = coeffs[v, dcell, 1]
            grad_bar_x = (1.0 - tstar) * gx_u + tstar * gx_d
            grad_bar_y = (1.0 - tstar) * gy_u + tstar * gy_d

            dl = d_len[k]
            if dl < eps:
                dl = eps
            cell_slope = delta_plus / dl
            cos_no = ex * nx + ey * ny
            if abs(cos_no) > eps:
                cos_safe = cos_no
            else:
                cos_safe = eps if cos_no >= 0.0 else -eps
            if face_gradient_mode == 1:
                grad_n = grad_bar_x * nx + grad_bar_y * ny
                corr = cell_slope / cos_safe - grad_n
                grad_corr_x = grad_bar_x + corr * nx
                grad_corr_y = grad_bar_y + corr * ny
            else:
                grad_proj = grad_bar_x * ex + grad_bar_y * ey
                beta = max(cos_no, 0.0) / theta_min
                if beta > 1.0:
                    beta = 1.0
                corr = beta * (grad_proj - cell_slope)
                grad_corr_x = grad_bar_x - corr * ex
                grad_corr_y = grad_bar_y - corr * ey

            d_face_corr_x = dx_f0 - tstar * dx
            d_face_corr_y = dy_f0 - tstar * dy
            delta_face = (tstar * delta_plus
                          + grad_corr_x * d_face_corr_x
                          + grad_corr_y * d_face_corr_y)
            if face_increment_mode == 1:
                delta_face = gx_u * dx_f0 + gy_u * dy_f0
            elif use_density_full_lsq_increment and v == 0:
                delta_face = gx_u * dx_f0 + gy_u * dy_f0
            elif use_density_lsq_increment and v == 0:
                delta_lsq = gx_u * dx_f0 + gy_u * dy_f0
                w_density = density_contact_weight[k]
                if w_density < 0.0:
                    w_density = 0.0
                elif w_density > 1.0:
                    w_density = 1.0
                delta_face = delta_face + w_density * (delta_lsq - delta_face)
            elif use_velocity_lsq_increment and (v == 1 or v == 2):
                delta_face = gx_u * dx_f0 + gy_u * dy_f0
            elif use_pressure_nonshock_lsq_increment and v == 3:
                delta_lsq = gx_u * dx_f0 + gy_u * dy_f0
                w_pressure = 1.0 - pressure_flatten[k]
                if w_pressure < 0.0:
                    w_pressure = 0.0
                elif w_pressure > 1.0:
                    w_pressure = 1.0
                delta_face = delta_face + w_pressure * (delta_lsq - delta_face)
            elif use_pressure_shear_lsq_increment and v == 3:
                delta_lsq = gx_u * dx_f0 + gy_u * dy_f0
                w_pressure = tangential_contact_weight[k]
                if w_pressure < 0.0:
                    w_pressure = 0.0
                elif w_pressure > 1.0:
                    w_pressure = 1.0
                delta_face = delta_face + w_pressure * (delta_lsq - delta_face)

            velocity_var = (v == 1 or v == 2)
            scale = one_minus_C_face[k]
            if v == 0 and use_density_no_hancock:
                scale = 1.0 - face_hancock_courant[k] * density_flatten[k]
            elif v == 0 and use_density_contact_wave_hancock:
                w_density_scale = density_contact_weight[k]
                if w_density_scale < 0.0:
                    w_density_scale = 0.0
                elif w_density_scale > 1.0:
                    w_density_scale = 1.0
                scale = (one_minus_C_face[k]
                         + w_density_scale
                         * (density_contact_hancock_scale[k]
                            - one_minus_C_face[k]))
                if density_contact_hancock_boost > 0.0:
                    clean_shear = (
                        w_density_scale
                        * max(0.0, min(1.0, tangential_contact_weight[k]))
                        * ((1.0 - max(0.0, min(1.0, pressure_flatten[k])))
                           * (1.0 - max(0.0, min(1.0, pressure_flatten[k]))))
                        * ((1.0 - max(0.0, min(1.0, velocity_flatten[k])))
                           * (1.0 - max(0.0, min(1.0, velocity_flatten[k])))))
                    scale = scale + density_contact_hancock_boost * clean_shear * (
                        1.0 - scale)
                    if scale > density_contact_hancock_boost_cap:
                        scale = density_contact_hancock_boost_cap
            if use_velocity_no_hancock and velocity_var:
                scale = 1.0 - (1.0 - one_minus_C_face[k]) * shock_flatten[k]

            if use_density_first_order and v == 0:
                delta = 0.0
            elif use_pressure_first_order and v == 3:
                delta = 0.0
            else:
                delta = scale * delta_face

            phi_ll = phi_u - (gx_u * dx + gy_u * dy)
            lo = phi_min_cell[v, ucell] - tvb_eps
            hi = phi_max_cell[v, ucell] + tvb_eps
            if phi_ll < lo:
                phi_ll = lo
            elif phi_ll > hi:
                phi_ll = hi
            delta_minus = phi_u - phi_ll
            den_floor = (64.0 * macheps
                         * (1.0 + abs(phi_u) + abs(phi_ll)
                            + abs(delta_plus)))
            if abs(delta_minus) > den_floor:
                den = delta_minus
            else:
                den = den_floor if delta_minus >= 0.0 else -den_floor
            r = delta_plus / den
            if abs(delta_plus) <= eps:
                psi_tvd = zero_delta_psi
            else:
                psi_tvd = _tmlpu_fast_tvd_value(
                    r, v, base_tvd_mode, velocity_tvd_mode,
                    density_tvd_mode, density_contact_weight[k])
            if use_velocity_extrema_lmp and velocity_var and r <= eps:
                psi_tvd = 1.0
            if use_density_extrema_lmp and v == 0 and r <= eps:
                psi_tvd = 1.0
            if v == 0:
                density_delta_out[side, k] = delta
                density_psi_tvd_out[side, k] = psi_tvd

            bounded_tvd = base_tvd_mode == 5
            if velocity_var and velocity_tvd_mode > 0:
                bounded_tvd = velocity_tvd_mode == 5
            elif v == 0 and density_tvd_mode > 0:
                bounded_tvd = density_tvd_mode == 5
            if ((use_velocity_extrema_lmp and velocity_var)
                    or (use_density_extrema_lmp and v == 0)):
                bounded_tvd = True
            if use_bound_tvd_separate:
                bounded_tvd = True

            r_pos = r if r > 0.0 else 0.0
            min1r = r_pos if r_pos < 1.0 else 1.0
            monotone = r > eps
            psi_best = psi_tvd
            for j in range(n_node):
                if not cell_node_valid[ucell, j]:
                    continue
                node = cell_node_safe[ucell, j]
                dvi_x = vertex_offsets[ucell, j, 0]
                dvi_y = vertex_offsets[ucell, j, 1]
                dcor_x = dvi_x - tstar * dx
                dcor_y = dvi_y - tstar * dy
                delta_vi = (tstar * delta_plus
                            + grad_corr_x * dcor_x
                            + grad_corr_y * dcor_y)
                vmin = vertex_min_values[v, node] - tvb_eps
                vmax = vertex_max_values[v, node] + tvb_eps
                allowed = vmax - phi_u if delta_vi >= 0.0 else vmin - phi_u
                if abs(delta_vi) > eps:
                    base = allowed / delta_vi
                    if base <= 0.0 or not np.isfinite(base):
                        base = 0.0
                else:
                    base = 2.0
                if bounded_tvd:
                    psi = base if base < psi_tvd else psi_tvd
                else:
                    denom = delta_vi * min1r
                    if abs(denom) > eps:
                        alpha = allowed / denom
                        if alpha <= 0.0 or not np.isfinite(alpha):
                            alpha = 0.0
                    else:
                        alpha = 2.0
                    alpha_r = base if r_pos <= 1.0 else base * r_pos
                    psi = alpha_r if alpha_r < alpha else alpha
                    if psi_tvd < psi:
                        psi = psi_tvd
                    if not monotone:
                        psi = 0.0
                if psi < 0.0:
                    psi = 0.0
                elif psi > 2.0:
                    psi = 2.0
                if psi < psi_best:
                    psi_best = psi

            if v == 0:
                density_psi_base_out[side, k] = psi_best
                psi_best = psi_best * (1.0 - density_flatten[k])
            elif v == 3:
                psi_best = psi_best * (1.0 - pressure_flatten[k])
            if use_velocity_shock_flatten and velocity_var:
                psi_best = psi_best * (1.0 - velocity_flatten[k])

            if side == 0:
                W_L[v, f] = phi_u + psi_best * delta
            else:
                W_R[v, f] = phi_u + psi_best * delta


    @njit(cache=True)
    def _smoothstep_scalar(lo, hi, x):
        width = hi - lo
        if width < 1.0e-30:
            width = 1.0e-30
        t = (x - lo) / width
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        return t * t * (3.0 - 2.0 * t)


    @njit(parallel=True, cache=True)
    def _euler_density_micro_restore_kernel(
            W_cell, coeffs, owner_int, nei_int, face_ids, face_n_o,
            face_node_safe, face_node_valid, vertex_min_values,
            vertex_max_values, density_delta, density_psi_tvd,
            density_psi_base, density_contact_weight, density_flatten,
            pressure_flatten, velocity_flatten, pressure_jump,
            compression, normality,
            gamma, tvb_eps, alpha, bound_pad, increment_cap,
            require_coherent_shear, artifact_reject, mode,
            weak_mlp_cap, weak_shock_power, W_L, W_R):
        n_face = owner_int.shape[0]
        n_node = face_node_safe.shape[1]
        eps = 1.0e-30
        for idx in prange(n_face * 2):
            side = idx & 1
            k = idx >> 1
            f = face_ids[k]
            o = owner_int[k]
            n = nei_int[k]
            if side == 0:
                ucell = o
            else:
                ucell = n
            phi_u = W_cell[0, ucell]
            delta = density_delta[side, k]
            psi_tvd = density_psi_tvd[side, k]
            psi_base = density_psi_base[side, k]
            if abs(delta) <= eps:
                continue

            count = 0
            qmin = 0.0
            qmax = 0.0
            for j in range(n_node):
                if face_node_valid[k, j]:
                    node = face_node_safe[k, j]
                    qmin += vertex_min_values[0, node]
                    qmax += vertex_max_values[0, node]
                    count += 1
            if count > 0:
                qmin /= count
                qmax /= count
            else:
                qmin = phi_u
                qmax = phi_u

            if delta >= 0.0:
                allowed = qmax - phi_u + tvb_eps
            else:
                allowed = qmin - phi_u - tvb_eps
            delta_eps = (64.0 * np.finfo(np.float64).eps
                         * (1.0 + abs(phi_u) + abs(qmin)
                            + abs(qmax) + abs(delta)))
            psi_bound = 1.0
            if delta > delta_eps:
                denom = delta if delta > delta_eps else delta_eps
                psi_bound = allowed / denom
            elif delta < -delta_eps:
                denom = delta if delta < -delta_eps else -delta_eps
                psi_bound = allowed / denom
            if psi_bound <= 0.0 or not np.isfinite(psi_bound):
                psi_bound = 0.0
            elif psi_bound > 2.0:
                psi_bound = 2.0
            psi_cap = psi_tvd
            if psi_cap < 0.0:
                psi_cap = 0.0
            elif psi_cap > 2.0:
                psi_cap = 2.0
            psi_weak = psi_bound if psi_bound < psi_cap else psi_cap
            rho_off = phi_u + psi_base * delta
            rho_on = phi_u + psi_weak * delta

            mode_local = mode
            weak_then_micro_mode = 0
            if mode >= 10:
                weak_then_micro_mode = mode - 10

            if mode_local == 0 or weak_then_micro_mode > 0:
                pflat = min(max(pressure_flatten[k], 0.0), 1.0)
                vflat = min(max(velocity_flatten[k], 0.0), 1.0)
                shock_gate = (1.0 - pflat) * (1.0 - vflat)
                if weak_shock_power != 1.0:
                    shock_gate = shock_gate ** weak_shock_power
                relax_w = min(max(density_contact_weight[k], 0.0), 1.0)
                relax_w = relax_w * shock_gate
                if relax_w > weak_mlp_cap:
                    relax_w = weak_mlp_cap
                elif relax_w < 0.0:
                    relax_w = 0.0
                psi_relaxed = psi_base + relax_w * (psi_weak - psi_base)
                psi_relaxed = psi_relaxed * (1.0 - density_flatten[k])
                rho_final_flat = phi_u + psi_relaxed * delta
                if weak_then_micro_mode <= 0:
                    if side == 0:
                        W_L[0, f] = rho_final_flat
                    else:
                        W_R[0, f] = rho_final_flat
                    continue
                psi_weak = psi_relaxed
                rho_on = rho_final_flat
                mode_local = weak_then_micro_mode

            rho_o = W_cell[0, o]
            rho_n = W_cell[0, n]
            if rho_o < eps:
                rho_o = eps
            if rho_n < eps:
                rho_n = eps
            shear_fraction = 1.0 - min(max(normality[k], 0.0), 1.0)

            if mode_local == 1:
                contact_gate = _smoothstep_scalar(
                    0.25, 0.55,
                    min(max(density_contact_weight[k], 0.0), 1.0))
                shear_gate = _smoothstep_scalar(0.65, 0.85, shear_fraction)
                p_gate = _smoothstep_scalar(0.035, 0.085, pressure_jump[k])
                c_gate = _smoothstep_scalar(0.010, 0.050, compression[k])
                n_gate = _smoothstep_scalar(0.45, 0.70, normality[k])
                gate = (contact_gate * shear_gate
                        * (1.0 - p_gate) * (1.0 - c_gate)
                        * (1.0 - n_gate))
            else:
                contact_gate = _smoothstep_scalar(
                    0.30, 0.60,
                    min(max(density_contact_weight[k], 0.0), 1.0))
                shear_gate = _smoothstep_scalar(0.70, 0.90, shear_fraction)
                p_gate = _smoothstep_scalar(0.030, 0.075, pressure_jump[k])
                c_gate = _smoothstep_scalar(0.008, 0.040, compression[k])
                n_gate = _smoothstep_scalar(0.40, 0.65, normality[k])
                dr_cell = rho_n - rho_o
                dr_face = rho_on - rho_off
                coherent = 1.0
                if (np.sign(dr_cell) != 0.0 and np.sign(dr_face) != 0.0
                        and np.sign(dr_cell) * np.sign(dr_face) <= 0.0):
                    coherent = 0.0
                coherent_shear_support = (
                    _smoothstep_scalar(0.03, 0.10, shear_fraction)
                    * coherent)
                if not require_coherent_shear:
                    coherent_shear_support = 1.0
                shock_off = ((1.0 - p_gate) * (1.0 - c_gate)
                             * (1.0 - n_gate))
                local_roughness = max(
                    max(rho_on - max(rho_o, rho_n), eps),
                    max(min(rho_o, rho_n) - rho_on, eps))
                local_span = max(abs(rho_n - rho_o), eps)
                local_density_roughness = min(
                    max(local_roughness / local_span, 0.0), 1.0)
                artifact_gate = 1.0 - _smoothstep_scalar(
                    0.06, 0.14, local_density_roughness)
                if not artifact_reject:
                    artifact_gate = 1.0
                gate = (contact_gate * shear_gate * shock_off
                        * coherent_shear_support * artifact_gate)

            if gate <= 0.0 or alpha <= 0.0:
                continue
            if gate > 1.0:
                gate = 1.0

            rho_avg = 0.5 * (rho_o + rho_n)
            p_o = W_cell[3, o]
            p_n = W_cell[3, n]
            if p_o < eps:
                p_o = eps
            if p_n < eps:
                p_n = eps
            c_o = np.sqrt(max(gamma * p_o / rho_o, eps))
            c_n = np.sqrt(max(gamma * p_n / rho_n, eps))
            _ = max(c_o + c_n, eps)
            pad = bound_pad * max(1.0, rho_avg)
            lo = min(rho_o, rho_n) - pad
            hi = max(rho_o, rho_n) + pad
            density_inc = rho_on - rho_off
            if increment_cap > 0.0:
                cap = increment_cap * max(1.0, rho_avg)
                if density_inc < -cap:
                    density_inc = -cap
                elif density_inc > cap:
                    density_inc = cap
            rho_micro = rho_off + alpha * gate * density_inc
            rho_final = rho_micro
            if rho_final < lo:
                rho_final = lo
            elif rho_final > hi:
                rho_final = hi
            if rho_final < eps:
                rho_final = eps
            psi_micro = (rho_final - phi_u) / delta
            psi_micro = psi_micro * (1.0 - density_flatten[k])
            rho_final_flat = phi_u + psi_micro * delta
            if side == 0:
                W_L[0, f] = rho_final_flat
            else:
                W_R[0, f] = rho_final_flat


    @njit(parallel=True, cache=True)
    def _euler_tangential_velocity_mc_kernel(
            W_cell, coeffs, owner_int, nei_int, face_ids, d_o_int,
            dx_fo, dx_fn, alpha_o, alpha_n, face_n_o, velocity_face_scale,
            normal_velocity_flatten, tangential_contact_weight,
            tangential_tvd_mode, tangential_micro_blend,
            tangential_micro_cap, tangential_wavespeed_growth_cap,
            tangential_jump_growth_cap, use_lsq_increment, W_L, W_R):
        n_face = owner_int.shape[0]
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        for k in prange(n_face):
            f = face_ids[k]
            o = owner_int[k]
            n = nei_int[k]
            nx = face_n_o[k, 0]
            ny = face_n_o[k, 1]
            tx = -ny
            ty = nx
            dx = d_o_int[k, 0]
            dy = d_o_int[k, 1]

            u_o = W_cell[1, o]
            v_o = W_cell[2, o]
            u_n = W_cell[1, n]
            v_n = W_cell[2, n]

            qt_u = u_o * tx + v_o * ty
            qt_d = u_n * tx + v_n * ty
            delta_plus = qt_d - qt_u
            gtx = tx * coeffs[1, o, 0] + ty * coeffs[2, o, 0]
            gty = tx * coeffs[1, o, 1] + ty * coeffs[2, o, 1]
            phi_ll = qt_u - (gtx * dx + gty * dy)
            delta_minus = qt_u - phi_ll
            den_floor = (64.0 * macheps
                         * (1.0 + abs(qt_u) + abs(phi_ll)
                            + abs(delta_plus)))
            if abs(delta_minus) > den_floor:
                den = delta_minus
            else:
                den = den_floor if delta_minus >= 0.0 else -den_floor
            r = delta_plus / den
            if abs(delta_plus) <= eps:
                psi = 2.0
            else:
                if (tangential_tvd_mode == 2 or tangential_tvd_mode == 6
                        or tangential_tvd_mode == 7
                        or tangential_tvd_mode == 8
                        or tangential_tvd_mode == 9
                        or tangential_tvd_mode == 10
                        or tangential_tvd_mode == 11):
                    ar = abs(r)
                    psi = (r + ar) / (1.0 + ar)
                elif tangential_tvd_mode == 3:
                    a = 2.0 * r
                    b = 0.25 + 0.75 * r
                    c = 0.75 + 0.25 * r
                    psi = a
                    if b < psi:
                        psi = b
                    if c < psi:
                        psi = c
                elif (tangential_tvd_mode == 4
                        or tangential_tvd_mode == 12
                        or tangential_tvd_mode == 13):
                    a = 2.0 * r
                    b = 0.5 * (1.0 + r)
                    psi_mc = a if a < b else b
                    b = 0.25 + 0.75 * r
                    c = 0.75 + 0.25 * r
                    psi_umist = a
                    if b < psi_umist:
                        psi_umist = b
                    if c < psi_umist:
                        psi_umist = c
                    if psi_mc > 2.0:
                        psi_mc = 2.0
                    if psi_mc < 0.0:
                        psi_mc = 0.0
                    if psi_umist > 2.0:
                        psi_umist = 2.0
                    if psi_umist < 0.0:
                        psi_umist = 0.0
                    w = tangential_contact_weight[k]
                    if w < 0.0:
                        w = 0.0
                    elif w > 1.0:
                        w = 1.0
                    psi = psi_mc + w * (psi_umist - psi_mc)
                elif tangential_tvd_mode == 16:
                    a = 1.5 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 1.5:
                        b = 1.5
                    psi = a if a > b else b
                elif tangential_tvd_mode == 17:
                    a = 2.0 * r
                    b = (1.0 + 2.0 * r) / 3.0
                    psi = a if a < b else b
                elif tangential_tvd_mode == 18:
                    a = 1.5 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 1.5:
                        b = 1.5
                    psi_mod = a if a > b else b
                    a = 2.0 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 2.0:
                        b = 2.0
                    psi_super = a if a > b else b
                    sh_l = normal_velocity_flatten[k]
                    if sh_l < 0.0:
                        sh_l = 0.0
                    elif sh_l > 1.0:
                        sh_l = 1.0
                    shear_free = 1.0 - sh_l
                    psi = psi_mod + shear_free * (psi_super - psi_mod)
                elif tangential_tvd_mode == 19:
                    a = 1.5 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 1.5:
                        b = 1.5
                    psi_mod = a if a > b else b
                    a = 2.0 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 2.0:
                        b = 2.0
                    psi_super = a if a > b else b
                    sh_l = normal_velocity_flatten[k]
                    if sh_l < 0.0:
                        sh_l = 0.0
                    elif sh_l > 1.0:
                        sh_l = 1.0
                    w = tangential_contact_weight[k] * (1.0 - sh_l)
                    if w < 0.0:
                        w = 0.0
                    elif w > 1.0:
                        w = 1.0
                    psi = psi_mod + w * (psi_super - psi_mod)
                elif tangential_tvd_mode == 20:
                    a = 1.5 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 1.5:
                        b = 1.5
                    psi_mod = a if a > b else b
                    a = 2.0 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 2.0:
                        b = 2.0
                    psi_super = a if a > b else b
                    sh_l = normal_velocity_flatten[k]
                    if sh_l < 0.0:
                        sh_l = 0.0
                    elif sh_l > 1.0:
                        sh_l = 1.0
                    contact_w = tangential_contact_weight[k]
                    if contact_w < 0.0:
                        contact_w = 0.0
                    elif contact_w > 1.0:
                        contact_w = 1.0
                    shock_free = 1.0 - sh_l
                    if shock_free < 0.0:
                        shock_free = 0.0
                    w = np.sqrt(contact_w) * np.sqrt(shock_free)
                    if w > 1.0:
                        w = 1.0
                    psi = psi_mod + w * (psi_super - psi_mod)
                elif tangential_tvd_mode == 21 or tangential_tvd_mode == 22:
                    a = 1.5 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 1.5:
                        b = 1.5
                    psi_mod = a if a > b else b
                    a = 2.0 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 2.0:
                        b = 2.0
                    psi_super = a if a > b else b
                    sh_l = normal_velocity_flatten[k]
                    if sh_l < 0.0:
                        sh_l = 0.0
                    elif sh_l > 1.0:
                        sh_l = 1.0
                    contact_w = tangential_contact_weight[k]
                    if contact_w < 0.0:
                        contact_w = 0.0
                    elif contact_w > 1.0:
                        contact_w = 1.0
                    shock_free = 1.0 - sh_l
                    if shock_free < 0.0:
                        shock_free = 0.0
                    w = np.sqrt(contact_w) * np.sqrt(shock_free)
                    micro = tangential_micro_blend * contact_w * shock_free * shock_free
                    if micro > tangential_micro_cap:
                        micro = tangential_micro_cap
                    w = w + micro
                    if w > 1.0:
                        w = 1.0
                    psi = psi_mod + w * (psi_super - psi_mod)
                elif (tangential_tvd_mode == 5
                        or tangential_tvd_mode == 14
                        or tangential_tvd_mode == 15):
                    a = 2.0 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 2.0:
                        b = 2.0
                    psi = a if a > b else b
                else:
                    a = 2.0 * r
                    b = 0.5 * (1.0 + r)
                    psi = a if a < b else b
                if psi > 2.0:
                    psi = 2.0
                if psi < 0.0:
                    psi = 0.0
            delta_face = alpha_o[k] * delta_plus
            if use_lsq_increment:
                delta_face = gtx * dx_fo[k, 0] + gty * dx_fo[k, 1]
            qt_high = qt_u + psi * velocity_face_scale[k] * delta_face
            qt_f = qt_high
            if (tangential_tvd_mode == 6 or tangential_tvd_mode == 7
                    or tangential_tvd_mode == 8
                    or tangential_tvd_mode == 9
                    or tangential_tvd_mode == 10
                    or tangential_tvd_mode == 11
                    or tangential_tvd_mode == 12
                    or tangential_tvd_mode == 13
                    or tangential_tvd_mode == 14
                    or tangential_tvd_mode == 15):
                base_qt = W_L[1, f] * tx + W_L[2, f] * ty
                if tangential_tvd_mode == 9:
                    shock_free = 1.0 - normal_velocity_flatten[k]
                    contact_w = shock_free * shock_free
                    contact_w = contact_w * contact_w
                elif tangential_tvd_mode == 11:
                    shock_free = 1.0 - normal_velocity_flatten[k]
                    contact_w = shock_free * shock_free * shock_free
                elif tangential_tvd_mode == 8:
                    shock_free = 1.0 - normal_velocity_flatten[k]
                    contact_w = shock_free * shock_free
                elif tangential_tvd_mode == 10:
                    contact_w = np.sqrt(np.sqrt(
                        max(0.0, tangential_contact_weight[k])))
                elif tangential_tvd_mode == 13:
                    shock_free = 1.0 - normal_velocity_flatten[k]
                    if shock_free < 0.0:
                        shock_free = 0.0
                    contact_w = tangential_contact_weight[k] * np.sqrt(shock_free)
                elif tangential_tvd_mode == 15:
                    shock_free = 1.0 - normal_velocity_flatten[k]
                    if shock_free < 0.0:
                        shock_free = 0.0
                    contact_w = tangential_contact_weight[k] * shock_free
                elif tangential_tvd_mode == 7:
                    contact_w = np.sqrt(max(0.0, tangential_contact_weight[k]))
                else:
                    contact_w = tangential_contact_weight[k]
                if (tangential_tvd_mode == 8 or tangential_tvd_mode == 9
                        or tangential_tvd_mode == 11
                        or tangential_tvd_mode == 14
                        or tangential_tvd_mode == 15):
                    w = contact_w
                else:
                    w = contact_w * (1.0 - normal_velocity_flatten[k])
                if w < 0.0:
                    w = 0.0
                elif w > 1.0:
                    w = 1.0
                qt_f = base_qt + w * (qt_high - base_qt)
            base_u_face = W_L[1, f]
            base_v_face = W_L[2, f]
            base_qt_face = base_u_face * tx + base_v_face * ty
            qn_f = base_u_face * nx + base_v_face * ny
            qn_cell = u_o * nx + v_o * ny
            sh = normal_velocity_flatten[k]
            if sh < 0.0:
                sh = 0.0
            elif sh > 1.0:
                sh = 1.0
            qn_f = (1.0 - sh) * qn_f + sh * qn_cell
            if tangential_tvd_mode != 22:
                W_L[1, f] = qn_f * nx + qt_f * tx
                W_L[2, f] = qn_f * ny + qt_f * ty
            else:
                cand_u_face = qn_f * nx + qt_f * tx
                cand_v_face = qn_f * ny + qt_f * ty
                base_speed = np.sqrt(
                    base_u_face * base_u_face + base_v_face * base_v_face)
                cand_speed = np.sqrt(
                    cand_u_face * cand_u_face + cand_v_face * cand_v_face)
                vel_min = u_o if u_o < u_n else u_n
                vel_max = u_n if u_o < u_n else u_o
                u_margin = tangential_jump_growth_cap * (
                    abs(u_n - u_o) + 1.0e-12)
                v_min = v_o if v_o < v_n else v_n
                v_max = v_n if v_o < v_n else v_o
                v_margin = tangential_jump_growth_cap * (
                    abs(v_n - v_o) + 1.0e-12)
                jump_base = abs(base_qt_face - (u_n * tx + v_n * ty))
                jump_cand = abs(qt_f - (u_n * tx + v_n * ty))
                reject = (
                    sh > 0.02
                    or cand_speed > base_speed * (
                        1.0 + tangential_wavespeed_growth_cap) + 1.0e-12
                    or jump_cand > jump_base * (
                        1.0 + tangential_jump_growth_cap) + 1.0e-12
                    or cand_u_face < vel_min - u_margin
                    or cand_u_face > vel_max + u_margin
                    or cand_v_face < v_min - v_margin
                    or cand_v_face > v_max + v_margin)
                if reject:
                    cand_u_face = base_u_face
                    cand_v_face = base_v_face
                W_L[1, f] = cand_u_face
                W_L[2, f] = cand_v_face

            qt_u = u_n * tx + v_n * ty
            qt_d = u_o * tx + v_o * ty
            delta_plus = qt_d - qt_u
            ndx = -dx
            ndy = -dy
            gtx = tx * coeffs[1, n, 0] + ty * coeffs[2, n, 0]
            gty = tx * coeffs[1, n, 1] + ty * coeffs[2, n, 1]
            phi_ll = qt_u - (gtx * ndx + gty * ndy)
            delta_minus = qt_u - phi_ll
            den_floor = (64.0 * macheps
                         * (1.0 + abs(qt_u) + abs(phi_ll)
                            + abs(delta_plus)))
            if abs(delta_minus) > den_floor:
                den = delta_minus
            else:
                den = den_floor if delta_minus >= 0.0 else -den_floor
            r = delta_plus / den
            if abs(delta_plus) <= eps:
                psi = 2.0
            else:
                if (tangential_tvd_mode == 2 or tangential_tvd_mode == 6
                        or tangential_tvd_mode == 7
                        or tangential_tvd_mode == 8
                        or tangential_tvd_mode == 9
                        or tangential_tvd_mode == 10
                        or tangential_tvd_mode == 11):
                    ar = abs(r)
                    psi = (r + ar) / (1.0 + ar)
                elif tangential_tvd_mode == 3:
                    a = 2.0 * r
                    b = 0.25 + 0.75 * r
                    c = 0.75 + 0.25 * r
                    psi = a
                    if b < psi:
                        psi = b
                    if c < psi:
                        psi = c
                elif (tangential_tvd_mode == 4
                        or tangential_tvd_mode == 12
                        or tangential_tvd_mode == 13):
                    a = 2.0 * r
                    b = 0.5 * (1.0 + r)
                    psi_mc = a if a < b else b
                    b = 0.25 + 0.75 * r
                    c = 0.75 + 0.25 * r
                    psi_umist = a
                    if b < psi_umist:
                        psi_umist = b
                    if c < psi_umist:
                        psi_umist = c
                    if psi_mc > 2.0:
                        psi_mc = 2.0
                    if psi_mc < 0.0:
                        psi_mc = 0.0
                    if psi_umist > 2.0:
                        psi_umist = 2.0
                    if psi_umist < 0.0:
                        psi_umist = 0.0
                    w = tangential_contact_weight[k]
                    if w < 0.0:
                        w = 0.0
                    elif w > 1.0:
                        w = 1.0
                    psi = psi_mc + w * (psi_umist - psi_mc)
                elif tangential_tvd_mode == 16:
                    a = 1.5 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 1.5:
                        b = 1.5
                    psi = a if a > b else b
                elif tangential_tvd_mode == 17:
                    a = 2.0 * r
                    b = (1.0 + 2.0 * r) / 3.0
                    psi = a if a < b else b
                elif tangential_tvd_mode == 18:
                    a = 1.5 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 1.5:
                        b = 1.5
                    psi_mod = a if a > b else b
                    a = 2.0 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 2.0:
                        b = 2.0
                    psi_super = a if a > b else b
                    sh_l = normal_velocity_flatten[k]
                    if sh_l < 0.0:
                        sh_l = 0.0
                    elif sh_l > 1.0:
                        sh_l = 1.0
                    shear_free = 1.0 - sh_l
                    psi = psi_mod + shear_free * (psi_super - psi_mod)
                elif tangential_tvd_mode == 19:
                    a = 1.5 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 1.5:
                        b = 1.5
                    psi_mod = a if a > b else b
                    a = 2.0 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 2.0:
                        b = 2.0
                    psi_super = a if a > b else b
                    sh_l = normal_velocity_flatten[k]
                    if sh_l < 0.0:
                        sh_l = 0.0
                    elif sh_l > 1.0:
                        sh_l = 1.0
                    w = tangential_contact_weight[k] * (1.0 - sh_l)
                    if w < 0.0:
                        w = 0.0
                    elif w > 1.0:
                        w = 1.0
                    psi = psi_mod + w * (psi_super - psi_mod)
                elif tangential_tvd_mode == 20:
                    a = 1.5 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 1.5:
                        b = 1.5
                    psi_mod = a if a > b else b
                    a = 2.0 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 2.0:
                        b = 2.0
                    psi_super = a if a > b else b
                    sh_l = normal_velocity_flatten[k]
                    if sh_l < 0.0:
                        sh_l = 0.0
                    elif sh_l > 1.0:
                        sh_l = 1.0
                    contact_w = tangential_contact_weight[k]
                    if contact_w < 0.0:
                        contact_w = 0.0
                    elif contact_w > 1.0:
                        contact_w = 1.0
                    shock_free = 1.0 - sh_l
                    if shock_free < 0.0:
                        shock_free = 0.0
                    w = np.sqrt(contact_w) * np.sqrt(shock_free)
                    if w > 1.0:
                        w = 1.0
                    psi = psi_mod + w * (psi_super - psi_mod)
                elif tangential_tvd_mode == 21 or tangential_tvd_mode == 22:
                    a = 1.5 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 1.5:
                        b = 1.5
                    psi_mod = a if a > b else b
                    a = 2.0 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 2.0:
                        b = 2.0
                    psi_super = a if a > b else b
                    sh_l = normal_velocity_flatten[k]
                    if sh_l < 0.0:
                        sh_l = 0.0
                    elif sh_l > 1.0:
                        sh_l = 1.0
                    contact_w = tangential_contact_weight[k]
                    if contact_w < 0.0:
                        contact_w = 0.0
                    elif contact_w > 1.0:
                        contact_w = 1.0
                    shock_free = 1.0 - sh_l
                    if shock_free < 0.0:
                        shock_free = 0.0
                    w = np.sqrt(contact_w) * np.sqrt(shock_free)
                    micro = tangential_micro_blend * contact_w * shock_free * shock_free
                    if micro > tangential_micro_cap:
                        micro = tangential_micro_cap
                    w = w + micro
                    if w > 1.0:
                        w = 1.0
                    psi = psi_mod + w * (psi_super - psi_mod)
                elif (tangential_tvd_mode == 5
                        or tangential_tvd_mode == 14
                        or tangential_tvd_mode == 15):
                    a = 2.0 * r
                    if a > 1.0:
                        a = 1.0
                    b = r
                    if b > 2.0:
                        b = 2.0
                    psi = a if a > b else b
                else:
                    a = 2.0 * r
                    b = 0.5 * (1.0 + r)
                    psi = a if a < b else b
                if psi > 2.0:
                    psi = 2.0
                if psi < 0.0:
                    psi = 0.0
            delta_face = alpha_n[k] * delta_plus
            if use_lsq_increment:
                delta_face = gtx * dx_fn[k, 0] + gty * dx_fn[k, 1]
            qt_high = qt_u + psi * velocity_face_scale[k] * delta_face
            qt_f = qt_high
            if (tangential_tvd_mode == 6 or tangential_tvd_mode == 7
                    or tangential_tvd_mode == 8
                    or tangential_tvd_mode == 9
                    or tangential_tvd_mode == 10
                    or tangential_tvd_mode == 11
                    or tangential_tvd_mode == 12
                    or tangential_tvd_mode == 13
                    or tangential_tvd_mode == 14
                    or tangential_tvd_mode == 15):
                base_qt = W_R[1, f] * tx + W_R[2, f] * ty
                if tangential_tvd_mode == 9:
                    shock_free = 1.0 - normal_velocity_flatten[k]
                    contact_w = shock_free * shock_free
                    contact_w = contact_w * contact_w
                elif tangential_tvd_mode == 11:
                    shock_free = 1.0 - normal_velocity_flatten[k]
                    contact_w = shock_free * shock_free * shock_free
                elif tangential_tvd_mode == 8:
                    shock_free = 1.0 - normal_velocity_flatten[k]
                    contact_w = shock_free * shock_free
                elif tangential_tvd_mode == 10:
                    contact_w = np.sqrt(np.sqrt(
                        max(0.0, tangential_contact_weight[k])))
                elif tangential_tvd_mode == 13:
                    shock_free = 1.0 - normal_velocity_flatten[k]
                    if shock_free < 0.0:
                        shock_free = 0.0
                    contact_w = tangential_contact_weight[k] * np.sqrt(shock_free)
                elif tangential_tvd_mode == 15:
                    shock_free = 1.0 - normal_velocity_flatten[k]
                    if shock_free < 0.0:
                        shock_free = 0.0
                    contact_w = tangential_contact_weight[k] * shock_free
                elif tangential_tvd_mode == 7:
                    contact_w = np.sqrt(max(0.0, tangential_contact_weight[k]))
                else:
                    contact_w = tangential_contact_weight[k]
                if (tangential_tvd_mode == 8 or tangential_tvd_mode == 9
                        or tangential_tvd_mode == 11
                        or tangential_tvd_mode == 14
                        or tangential_tvd_mode == 15):
                    w = contact_w
                else:
                    w = contact_w * (1.0 - normal_velocity_flatten[k])
                if w < 0.0:
                    w = 0.0
                elif w > 1.0:
                    w = 1.0
                qt_f = base_qt + w * (qt_high - base_qt)
            base_u_face = W_R[1, f]
            base_v_face = W_R[2, f]
            base_qt_face = base_u_face * tx + base_v_face * ty
            qn_f = base_u_face * nx + base_v_face * ny
            qn_cell = u_n * nx + v_n * ny
            sh = normal_velocity_flatten[k]
            if sh < 0.0:
                sh = 0.0
            elif sh > 1.0:
                sh = 1.0
            qn_f = (1.0 - sh) * qn_f + sh * qn_cell
            if tangential_tvd_mode != 22:
                W_R[1, f] = qn_f * nx + qt_f * tx
                W_R[2, f] = qn_f * ny + qt_f * ty
            else:
                cand_u_face = qn_f * nx + qt_f * tx
                cand_v_face = qn_f * ny + qt_f * ty
                base_speed = np.sqrt(
                    base_u_face * base_u_face + base_v_face * base_v_face)
                cand_speed = np.sqrt(
                    cand_u_face * cand_u_face + cand_v_face * cand_v_face)
                vel_min = u_o if u_o < u_n else u_n
                vel_max = u_n if u_o < u_n else u_o
                u_margin = tangential_jump_growth_cap * (
                    abs(u_n - u_o) + 1.0e-12)
                v_min = v_o if v_o < v_n else v_n
                v_max = v_n if v_o < v_n else v_o
                v_margin = tangential_jump_growth_cap * (
                    abs(v_n - v_o) + 1.0e-12)
                jump_base = abs(base_qt_face - (u_o * tx + v_o * ty))
                jump_cand = abs(qt_f - (u_o * tx + v_o * ty))
                reject = (
                    sh > 0.02
                    or cand_speed > base_speed * (
                        1.0 + tangential_wavespeed_growth_cap) + 1.0e-12
                    or jump_cand > jump_base * (
                        1.0 + tangential_jump_growth_cap) + 1.0e-12
                    or cand_u_face < vel_min - u_margin
                    or cand_u_face > vel_max + u_margin
                    or cand_v_face < v_min - v_margin
                    or cand_v_face > v_max + v_margin)
                if reject:
                    cand_u_face = base_u_face
                    cand_v_face = base_v_face
                W_R[1, f] = cand_u_face
                W_R[2, f] = cand_v_face


    @njit(parallel=True, cache=True)
    def _euler_density_pressure_entropy_kernel(
            W_cell, coeffs, phi_min_cell, phi_max_cell,
            owner_int, nei_int, face_ids, dx_fo, dx_fn,
            density_contact_weight, density_flatten, gamma,
            pressure_face_is_log, entropy_residual_accept, W_L, W_R):
        n_face = owner_int.shape[0]
        eps = 1.0e-30
        for k in prange(n_face):
            f = face_ids[k]
            o = owner_int[k]
            n = nei_int[k]
            w = density_contact_weight[k] * (1.0 - density_flatten[k])
            if w < 0.0:
                w = 0.0
            elif w > 1.0:
                w = 1.0

            base_l = W_L[0, f]
            base_r = W_R[0, f]

            rho_u = W_cell[0, o]
            p_u = W_cell[3, o]
            if rho_u < eps:
                rho_u = eps
            if p_u < eps:
                p_u = eps
            drho_lsq = (coeffs[0, o, 0] * dx_fo[k, 0]
                         + coeffs[0, o, 1] * dx_fo[k, 1])
            dp_lsq = (coeffs[3, o, 0] * dx_fo[k, 0]
                      + coeffs[3, o, 1] * dx_fo[k, 1])
            lo = phi_min_cell[0, o]
            hi = phi_max_cell[0, o]
            if lo < eps:
                lo = eps
            if hi < lo:
                hi = lo
            p_face = W_L[3, f]
            if pressure_face_is_log:
                p_face = np.exp(p_face)
            if p_face < eps:
                p_face = eps
            s_lsq = (np.log(p_u) - gamma * np.log(rho_u)
                     + dp_lsq / p_u - gamma * drho_lsq / rho_u)
            s_min = np.log(p_face) - gamma * np.log(hi)
            s_max = np.log(p_face) - gamma * np.log(lo)
            if s_lsq < s_min:
                s_lsq = s_min
            elif s_lsq > s_max:
                s_lsq = s_max
            rho_entropy = np.exp((np.log(p_face) - s_lsq) / gamma)
            cand_l = base_l + w * (rho_entropy - base_l)
            if cand_l < lo:
                cand_l = lo
            elif cand_l > hi:
                cand_l = hi
            if cand_l < eps:
                cand_l = eps
            base_s_l = np.log(p_face) - gamma * np.log(max(base_l, eps))
            cand_s_l = np.log(p_face) - gamma * np.log(cand_l)
            base_res_l = abs(base_s_l - s_lsq)
            cand_res_l = abs(cand_s_l - s_lsq)

            rho_u = W_cell[0, n]
            p_u = W_cell[3, n]
            if rho_u < eps:
                rho_u = eps
            if p_u < eps:
                p_u = eps
            drho_lsq = (coeffs[0, n, 0] * dx_fn[k, 0]
                         + coeffs[0, n, 1] * dx_fn[k, 1])
            dp_lsq = (coeffs[3, n, 0] * dx_fn[k, 0]
                      + coeffs[3, n, 1] * dx_fn[k, 1])
            lo = phi_min_cell[0, n]
            hi = phi_max_cell[0, n]
            if lo < eps:
                lo = eps
            if hi < lo:
                hi = lo
            p_face = W_R[3, f]
            if pressure_face_is_log:
                p_face = np.exp(p_face)
            if p_face < eps:
                p_face = eps
            s_lsq = (np.log(p_u) - gamma * np.log(rho_u)
                     + dp_lsq / p_u - gamma * drho_lsq / rho_u)
            s_min = np.log(p_face) - gamma * np.log(hi)
            s_max = np.log(p_face) - gamma * np.log(lo)
            if s_lsq < s_min:
                s_lsq = s_min
            elif s_lsq > s_max:
                s_lsq = s_max
            rho_entropy = np.exp((np.log(p_face) - s_lsq) / gamma)
            cand_r = base_r + w * (rho_entropy - base_r)
            if cand_r < lo:
                cand_r = lo
            elif cand_r > hi:
                cand_r = hi
            if cand_r < eps:
                cand_r = eps
            base_s_r = np.log(p_face) - gamma * np.log(max(base_r, eps))
            cand_s_r = np.log(p_face) - gamma * np.log(cand_r)
            base_res_r = abs(base_s_r - s_lsq)
            cand_res_r = abs(cand_s_r - s_lsq)

            accept = abs(cand_l - cand_r) < abs(base_l - base_r)
            if entropy_residual_accept:
                accept = (cand_res_l <= base_res_l
                          and cand_res_r <= base_res_r)
            if np.isfinite(cand_l + cand_r) and accept:
                W_L[0, f] = cand_l
                W_R[0, f] = cand_r


    @njit(parallel=True, cache=True)
    def _euler_density_contact_bvd_kernel(
            W_cell, phi_min_cell, phi_max_cell,
            owner_int, nei_int, face_ids,
            density_contact_weight, density_contact_hancock_scale,
            shock_flatten,
            one_minus_C, bvd_cap, W_L, W_R):
        n_face = owner_int.shape[0]
        eps = 1.0e-30
        for k in prange(n_face):
            w = density_contact_weight[k] * (1.0 - shock_flatten[k])
            if w > bvd_cap:
                w = bvd_cap
            if w <= eps:
                continue
            f = face_ids[k]
            o = owner_int[k]
            n = nei_int[k]
            base_l = W_L[0, f]
            base_r = W_R[0, f]
            bv_base = abs(base_l - base_r)
            cand_l = base_l + w * (W_cell[0, o] - base_l)
            cand_r = base_r + w * (W_cell[0, n] - base_r)

            lo = phi_min_cell[0, o]
            hi = phi_max_cell[0, o]
            if cand_l < lo:
                cand_l = lo
            elif cand_l > hi:
                cand_l = hi
            lo = phi_min_cell[0, n]
            hi = phi_max_cell[0, n]
            if cand_r < lo:
                cand_r = lo
            elif cand_r > hi:
                cand_r = hi
            if cand_l < eps:
                cand_l = eps
            if cand_r < eps:
                cand_r = eps

            bv_cand = abs(cand_l - cand_r)
            if bv_cand >= bv_base:
                W_L[0, f] = cand_l
                W_R[0, f] = cand_r


    @njit(cache=True)
    def _euler_density_contact_cell_bvd_kernel(
            W_cell, phi_min_cell, phi_max_cell,
            owner_int, nei_int, face_ids,
            density_contact_weight, density_contact_hancock_scale,
            one_minus_C, W_L, W_R):
        n_face = owner_int.shape[0]
        n_cell = W_cell.shape[1]
        eps = 1.0e-30
        cand_l_arr = np.empty(n_face, dtype=np.float64)
        cand_r_arr = np.empty(n_face, dtype=np.float64)
        tbv_base = np.zeros(n_cell, dtype=np.float64)
        tbv_cand = np.zeros(n_cell, dtype=np.float64)
        for k in range(n_face):
            f = face_ids[k]
            o = owner_int[k]
            n = nei_int[k]
            base_l = W_L[0, f]
            base_r = W_R[0, f]
            w = density_contact_weight[k]
            cand_l = base_l
            cand_r = base_r
            if w > eps and one_minus_C > eps:
                cscale = density_contact_hancock_scale[k]
                contact_scale = one_minus_C + w * (cscale - one_minus_C)
                ratio = contact_scale / one_minus_C
                cand_l = W_cell[0, o] + ratio * (base_l - W_cell[0, o])
                cand_r = W_cell[0, n] + ratio * (base_r - W_cell[0, n])

                lo = phi_min_cell[0, o]
                hi = phi_max_cell[0, o]
                if cand_l < lo:
                    cand_l = lo
                elif cand_l > hi:
                    cand_l = hi
                lo = phi_min_cell[0, n]
                hi = phi_max_cell[0, n]
                if cand_r < lo:
                    cand_r = lo
                elif cand_r > hi:
                    cand_r = hi
                if cand_l < eps:
                    cand_l = eps
                if cand_r < eps:
                    cand_r = eps
            cand_l_arr[k] = cand_l
            cand_r_arr[k] = cand_r
            jump = abs(base_l - base_r)
            tbv_base[o] += jump
            tbv_base[n] += jump
            tbv_cand[o] += abs(cand_l - base_r)
            tbv_cand[n] += abs(cand_r - base_l)

        for k in range(n_face):
            f = face_ids[k]
            o = owner_int[k]
            n = nei_int[k]
            if tbv_cand[o] < tbv_base[o]:
                W_L[0, f] = cand_l_arr[k]
            if tbv_cand[n] < tbv_base[n]:
                W_R[0, f] = cand_r_arr[k]


# ─── Base interface ────────────────────────────────────────────────────────
class Reconstruction:
    name: str = 'base'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        """Return (W_owner_side, W_neighbour_side) at face evaluation points.

        Parameters
        ----------
        mesh, W_cell, eq : as usual.
        eval_points : (n_faces, 2) array, optional
            Face-side evaluation point per face.  When None (default) the
            face *centres* are used — this matches midpoint quadrature.
            The high-order solver path passes Gauss-quadrature points
            here to maintain spatial order ≥ 3.

        The two returned arrays have shape (nvar, n_faces) and represent
        W reconstructed from the owner side and neighbour side at the
        chosen evaluation points.
        """
        raise NotImplementedError


# ─── 1st-order (piecewise constant) ────────────────────────────────────────
@dataclass
class FirstOrder(Reconstruction):
    name: str = 'first_order'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        # First-order reconstruction is constant in each cell, so the
        # `eval_points` argument has no effect.
        n_faces = mesh.n_faces
        nvar = W_cell.shape[0]
        W_L = np.empty((nvar, n_faces), dtype=float)
        W_R = np.empty((nvar, n_faces), dtype=float)
        owner = mesh.face_owner
        nei   = mesh.face_neighbour
        for v in range(nvar):
            W_L[v] = W_cell[v, owner]
            W_R[v] = np.where(nei >= 0, W_cell[v, np.maximum(nei, 0)], W_cell[v, owner])
        return W_L, W_R


# ─── 1D structured TVD (any limiter) ───────────────────────────────────────
@dataclass
class MinmodTVD1D(Reconstruction):
    """Classical 1D MUSCL-Hancock TVD reconstruction with a swappable
    slope limiter.  Works on `mesh.dim == 1`.

    For each cell C with left neighbour L and right neighbour R, define:
        Δ_L = W_C − W_L,   Δ_R = W_R − W_C
        Δ   = limiter2(Δ_L, Δ_R)              # symmetric minmod2 form
    Face values:
        W_at_face_to_left_of_C  = W_C − ½ Δ
        W_at_face_to_right_of_C = W_C + ½ Δ

    `limiter2(a, b)` defaults to the symmetric minmod (`limiters.minmod2`)
    which is universally TVD with **zero** free parameters.
    """
    limiter2: Callable = minmod2
    name: str = 'minmod_tvd_1d'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        # 1D structured uses face centres ⇒ eval_points is ignored.
        assert mesh.dim == 1, "MinmodTVD1D requires a 1D mesh."
        nvar, N = W_cell.shape
        # Cell-centred slopes (TVD-limited)
        slopes = np.zeros_like(W_cell)
        # Use cell_neighbours to find L/R; for periodic 1D mesh both exist.
        # For non-periodic 1D, end cells have only one neighbour — keep slope = 0
        # there (degenerates to first-order at the boundary, standard practice).
        for i in range(N):
            nbrs = mesh.cell_neighbours[i]
            valid = [n for n in nbrs if n >= 0]
            if len(valid) < 2:
                continue
            # Sort by x to identify left vs right
            xs = mesh.cell_centers[valid, 0]
            order = np.argsort(xs)
            left  = valid[order[0]]
            right = valid[order[-1]]
            dL = W_cell[:, i]    - W_cell[:, left]
            dR = W_cell[:, right] - W_cell[:, i]
            slopes[:, i] = self.limiter2(dL, dR)

        owner = mesh.face_owner
        nei   = mesh.face_neighbour
        n_faces = mesh.n_faces
        W_L = np.empty((nvar, n_faces), dtype=float)
        W_R = np.empty((nvar, n_faces), dtype=float)

        # Decide sign of half-slope based on which side of the owner the face is.
        # For a 1D structured mesh face_normals is +1 (pointing owner→neighbour).
        # owner contributes +½·slope on its outgoing-normal side.
        sign_owner = mesh.face_normals[:, 0]      # +1 always for our 1D builder
        for v in range(nvar):
            W_L[v] = W_cell[v, owner] + 0.5 * sign_owner * slopes[v, owner]
            # Neighbour-side reconstruction: face_normal points owner→neighbour,
            # so neighbour reconstructs *backward* by −½ slope.
            n_idx = np.maximum(nei, 0)
            W_R[v] = np.where(
                nei >= 0,
                W_cell[v, n_idx] - 0.5 * sign_owner * slopes[v, n_idx],
                W_cell[v, owner],   # boundary placeholder; solver overwrites
            )
        return W_L, W_R


# ─── MLP-u (Park-Yoon-Kim 2010) — placeholder ──────────────────────────────
@dataclass
class MLPU(Reconstruction):
    """MLP-u Multi-dimensional Limiter for unstructured grids
    (Park, Yoon, Kim, J. Comput. Phys. 2010).

    Steps (for each cell):
      1. Compute least-squares unlimited gradient ∇W in the cell.
      2. For each *vertex* of the cell, compute the projected value
         W_vertex = W_cell + ∇W · (x_vertex − x_cell).
      3. Vertex-based MLP slope limiter (Hubbard 1999 + Park-Yoon-Kim 2010
         u-correction) constrains W_vertex into [W_min^vtx, W_max^vtx]
         where the bounds are taken over all cells sharing that vertex.

    Implementation deferred to a follow-up commit — included here as a slot
    so the test infrastructure can switch reconstruction by name.
    """
    name: str = 'mlp_u'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        return _limited_linear_2d(
            mesh, W_cell, eq, eval_points=eval_points,
            limiter='bj',
            stencil='vertex',
            vertex_bounds=True,
            n_rings=1,
        )


@dataclass
class BarthJespersen(Reconstruction):
    """Barth-Jespersen cell-wise linear limiter for 2D unstructured grids."""
    stencil: str = 'face'
    name: str = 'barth_jespersen'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        return _limited_linear_2d(
            mesh, W_cell, eq, eval_points=eval_points,
            limiter='bj',
            stencil=self.stencil,
            vertex_bounds=False,
            n_rings=1,
        )


@dataclass
class Venkatakrishnan(Reconstruction):
    """Venkatakrishnan smooth cell-wise linear limiter for 2D grids."""
    stencil: str = 'face'
    K: float = 1.0
    name: str = 'venkatakrishnan'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        return _limited_linear_2d(
            mesh, W_cell, eq, eval_points=eval_points,
            limiter='venkat',
            stencil=self.stencil,
            vertex_bounds=False,
            n_rings=1,
            venkat_K=self.K,
        )


@dataclass
class MLPU1(Reconstruction):
    """Park-Yoon-Kim-style MLP-u1: vertex bounds, one vertex-neighbour ring."""
    name: str = 'mlp_u1'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        return _limited_linear_2d(
            mesh, W_cell, eq, eval_points=eval_points,
            limiter='bj',
            stencil='vertex',
            vertex_bounds=True,
            n_rings=1,
        )


@dataclass
class MLPU1TMLPU(Reconstruction):
    """MLP-u1 reconstruction with an additional T-MLP-u face LMP bound."""
    name: str = 'mlp_u1_tmlpu'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        return _limited_linear_2d(
            mesh, W_cell, eq, eval_points=eval_points,
            limiter='bj',
            stencil='vertex',
            vertex_bounds=True,
            n_rings=1,
            tmlpu_face_bound=True,
        )


@dataclass
class MLPU1TMLPUContact(Reconstruction):
    """Enhanced T-MLP-u: BJ multidimensional vertex-bounded linear base
    (vortex/shear-preserving, identical to mlp_u1_tmlpu) plus a single-pass
    contact-gated artificial compression that sharpens contact/slip-line/slot
    interfaces (Double-Mach KH cores, LeVeque slot) without touching shocks or
    smooth extrema.  One reconstruction pass -> no BVD wall penalty."""
    name: str = 'mlp_u1_tmlpu_contact'
    contact_compress: float = 0.0
    contact_compress_rho_lo: float = 0.06
    contact_compress_rho_hi: float = 0.30
    contact_compress_p_tol: float = 0.18

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        return _limited_linear_2d(
            mesh, W_cell, eq, eval_points=eval_points,
            limiter='bj',
            stencil='vertex',
            vertex_bounds=True,
            n_rings=1,
            tmlpu_face_bound=True,
            contact_compress=self.contact_compress,
            contact_compress_rho_lo=self.contact_compress_rho_lo,
            contact_compress_rho_hi=self.contact_compress_rho_hi,
            contact_compress_p_tol=self.contact_compress_p_tol,
        )


@dataclass
class MLPU2(Reconstruction):
    """MLP-u2 comparison: one-ring vertex stencil with Venkat smooth limiter."""
    name: str = 'mlp_u2'

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        return _limited_linear_2d(
            mesh, W_cell, eq, eval_points=eval_points,
            limiter='venkat',
            stencil='vertex',
            vertex_bounds=True,
            n_rings=1,
            venkat_K=1.0,
        )


# ─── T-MLP-u (user's method) ───────────────────────────────────────────────
@dataclass
class TMLPU(Reconstruction):
    """T-MLP-u — primitive-variable high-order reconstruction wrapping any
    classical TVD limiter (minmod / van Leer / superbee / MC / van Albada /
    UMIST) with a Local Maximum Principle (LMP) bound on top.

    Parameters
    ----------
    tvd : str | callable, default 'superbee'
        Base TVD limiter ψ_TVD wrapped by T-MLP-u.  String keys come
        from `limiters.TVD_LIMITERS`.  Callable must take r → ψ.
    hancock_courant : float, default 0.0
        Hancock factor C_f (0 ⇒ plain MUSCL).
    stencil : {'face', 'vertex'}, default 'face'
        Local stencil used for the LSQ gradient/Hessian *and* for the
        LMP φ_min / φ_max bounds (only relevant on 2D unstructured grids):
          'face'   — 1-ring face neighbours (current default; tightest bound)
          'vertex' — Park-Yoon-Kim 2010 MLP-u stencil: every cell sharing
                     any vertex with C; wider, more accurate gradient and
                     a less-restrictive bound on smooth regions.
    order : {1, 2}, default 1
        Polynomial order of the reconstruction inside each cell.
          1 — linear LSQ  (∇W only, 2nd-order accurate face value)
          2 — quadratic LSQ  (∇W + Hessian, 3rd-order on smooth data)
        order=2 implies stencil='vertex' (face-only triangle stencil has
        too few neighbours for the quadratic system); the wrapper enforces
        this automatically.

    Implementation status:
      - 1D structured: vectorised, working.
      - 2D Cartesian (axis-aligned): linear path only — `stencil`/`order`
        flags are honoured silently.
      - 2D unstructured: both stencils, both orders, vectorised.
    """
    tvd: object = 'superbee'
    hancock_courant: float = 0.0
    euler_shock_flatten: bool = False
    euler_density_acoustic_flatten: bool = False
    euler_density_tvd: object = None
    euler_density_lsq_increment: bool = False
    euler_density_full_lsq_increment: bool = False
    euler_density_no_hancock: bool = False
    euler_density_entropy_split: bool = False
    euler_density_entropy_variable: bool = False
    euler_density_shear_contact: bool = False
    euler_density_contact_wave_hancock: bool = False
    euler_density_contact_hancock_boost: float = 0.0
    euler_density_contact_hancock_boost_cap: float = 1.0
    euler_density_contact_lsq_root_blend: float = 0.0
    euler_density_contact_lsq_root_blend_cap: float = 1.0
    euler_density_contact_lsq_shear_floor: float = 0.0
    euler_density_contact_lsq_shear_floor_cap: float = 1.0
    euler_density_pressure_entropy: bool = False
    euler_density_contact_bvd: bool = False
    euler_density_contact_bvd_cap: float = 1.0
    euler_density_contact_cell_bvd: bool = False
    euler_density_contact_weak_face_mlp: bool = False
    euler_density_contact_weak_face_mlp_cap: float = 1.0
    euler_density_contact_weak_face_shock_power: float = 1.0
    euler_density_contact_weak_face_legacy_order: bool = False
    euler_density_contact_weak_face_legacy_relax: bool = False
    euler_density_contact_weak_face_legacy_relax_cap: float = 1.0
    euler_density_contact_weak_face_legacy_tvd_after_weak: bool = False
    euler_density_contact_weak_face_head_generic: bool = False
    euler_density_contact_weak_face_disable_specialized_relax: bool = False
    euler_density_contact_weak_face_head_generic_blend_cap: float = 1.0
    euler_density_contact_weak_face_root_blend: float = 0.0
    euler_density_contact_weak_face_swirl_extra: float = 0.0
    euler_density_contact_weak_face_value_scaling: bool = False
    euler_density_contact_weak_face_rho_floor_factor: float = 0.88
    euler_density_contact_weak_face_theta_floor: float = 0.25
    euler_density_contact_weak_face_value_scaling_mode: str = 'global_floor'
    euler_density_contact_weak_face_value_scaling_strength: float = 1.0
    euler_density_contact_weak_face_value_scaling_shear_blend_alpha: float = 0.0
    euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad: float = 0.0
    euler_density_contact_weak_face_hard_rho_floor_factor: float = 0.0
    euler_density_contact_weak_face_hard_p_floor_factor: float = 0.0
    euler_density_contact_weak_face_value_scaling_p_floor_factor: float = 1.0
    euler_density_contact_weak_face_value_scaling_risk_width: float = 0.08
    euler_density_contact_weak_face_value_scaling_p_threshold: float = 0.04
    euler_density_contact_weak_face_value_scaling_p_width: float = 0.06
    euler_density_contact_weak_face_value_scaling_compression_threshold: float = 0.015
    euler_density_contact_weak_face_value_scaling_compression_width: float = 0.065
    euler_density_contact_weak_face_value_scaling_normality_threshold: float = 0.35
    euler_density_contact_weak_face_value_scaling_normality_width: float = 0.35
    euler_density_contact_weak_face_value_scaling_contact_threshold: float = 0.25
    euler_density_contact_weak_face_value_scaling_contact_width: float = 0.35
    euler_density_contact_weak_face_value_scaling_shear_threshold: float = 0.60
    euler_density_contact_weak_face_value_scaling_shear_width: float = 0.25
    euler_density_contact_weak_face_value_scaling_pressure_clean_threshold: float = 0.04
    euler_density_contact_weak_face_value_scaling_pressure_clean_width: float = 0.06
    euler_density_contact_weak_face_value_scaling_hard_protect_cutoff: float = 0.65
    euler_density_contact_weak_face_value_scaling_require_coherent_shear: bool = True
    euler_density_contact_weak_face_value_scaling_artifact_reject: bool = True
    euler_density_contact_weak_face_contour_continuity_on: bool = False
    euler_density_contact_weak_face_contour_continuity_min: float = 0.55
    euler_density_contact_weak_face_contour_continuity_full: float = 0.85
    euler_density_contact_weak_face_density_increment_cap: float = 0.0
    euler_density_contact_weak_face_admissibility_damp: bool = False
    euler_density_contact_weak_face_rho_floor: float = 0.0
    euler_density_contact_weak_face_p_floor: float = 0.0
    euler_density_contact_weak_face_admissibility_strength: float = 1.0
    euler_density_contact_weak_face_admissibility_shear_protect: bool = False
    euler_density_contact_weak_face_entropy_accept: bool = False
    euler_density_contact_weak_face_entropy_accept_eps: float = 0.05
    euler_density_contact_weak_face_entropy_reject_scale: float = 0.35
    euler_density_contact_weak_face_shock_gate: bool = False
    euler_density_contact_weak_face_shock_gate_strength: float = 0.65
    euler_density_contact_weak_face_shock_gate_floor: float = 0.35
    euler_pressure_contact_entropy_blend: bool = False
    euler_pressure_contact_entropy_beta: float = 0.18
    euler_pressure_contact_entropy_cap: float = 0.18
    euler_pressure_contact_entropy_downscale: float = 0.25
    euler_pressure_contact_entropy_p_jump_threshold: float = 0.04
    euler_pressure_contact_entropy_p_jump_width: float = 0.08
    euler_pressure_contact_entropy_compression_threshold: float = 0.01
    euler_pressure_contact_entropy_compression_width: float = 0.07
    euler_pressure_contact_entropy_normality_threshold: float = 0.45
    euler_pressure_contact_entropy_normality_width: float = 0.30
    euler_pressure_face_jump_limiter_on: bool = False
    euler_pressure_face_jump_limiter_strength: float = 0.0
    euler_pressure_face_jump_limiter_growth_cap: float = 0.10
    euler_pressure_face_jump_limiter_abs_floor: float = 1e-10
    euler_pressure_face_jump_limiter_p_jump_threshold: float = 0.025
    euler_pressure_face_jump_limiter_p_jump_width: float = 0.070
    euler_pressure_face_jump_limiter_compression_threshold: float = 0.008
    euler_pressure_face_jump_limiter_compression_width: float = 0.060
    euler_pressure_face_jump_limiter_normality_threshold: float = 0.40
    euler_pressure_face_jump_limiter_normality_width: float = 0.30
    euler_density_contact_weak_face_shock_gate_mode: str = 'wide'
    euler_density_contact_weak_face_shock_gate_p_threshold: float = 0.06
    euler_density_contact_weak_face_shock_gate_p_width: float = 0.24
    euler_density_contact_weak_face_shock_gate_compression_threshold: float = 0.015
    euler_density_contact_weak_face_shock_gate_compression_width: float = 0.12
    euler_density_contact_weak_face_shock_gate_normality_threshold: float = 0.45
    euler_density_contact_weak_face_shock_gate_normality_width: float = 0.35
    euler_density_contact_weak_face_shock_gate_shear_threshold: float = 0.65
    euler_density_contact_weak_face_shock_gate_shear_width: float = 0.25
    euler_density_contact_weak_face_shock_gate_contact_threshold: float = 0.25
    euler_density_contact_weak_face_shock_gate_contact_width: float = 0.45
    euler_density_first_order: bool = False
    euler_pressure_first_order: bool = False
    euler_log_pressure_only: bool = False
    euler_pressure_shear_lsq_increment: bool = False
    euler_pressure_nonshock_lsq_increment: bool = False
    euler_velocity_no_hancock: bool = False
    euler_velocity_shock_flatten: bool = False
    euler_velocity_lsq_increment: bool = False
    euler_density_extrema_lmp: bool = False
    euler_velocity_extrema_lmp: bool = False
    euler_velocity_tvd: object = None
    euler_velocity_flatten_sensor: str = 'pressure'
    euler_tangential_velocity_tvd: object = None
    euler_tangential_ducros: bool = False
    euler_tangential_flatten_mode: str = ''
    euler_wall_tangential_flatten: bool = False
    euler_wall_tangential_flatten_mode: str = 'all'
    euler_wall_tangential_flatten_strength: float = 1.0
    euler_tangential_contact_relax_flatten: bool = False
    euler_tangential_shear_micro_blend: float = 0.0
    euler_tangential_shear_micro_cap: float = 0.0
    euler_tangential_mood_wavespeed_growth_cap: float = 0.0
    euler_tangential_mood_jump_growth_cap: float = 0.0
    euler_tangential_pair_restore_on: bool = False
    euler_tangential_pair_restore_alpha: float = 0.0
    euler_tangential_pair_restore_cap: float = 0.0
    euler_tangential_pair_restore_wave_cap: float = 0.0
    euler_tangential_pair_gate_contact_min: float = 0.30
    euler_tangential_pair_gate_contact_full: float = 0.60
    euler_tangential_pair_gate_shear_min: float = 0.70
    euler_tangential_pair_gate_shear_full: float = 0.90
    euler_tangential_pair_gate_density_support_min: float = 0.02
    euler_tangential_pair_gate_density_support_full: float = 0.08
    euler_tangential_tail_density_support_min: float = -1.0
    euler_tangential_tail_density_support_full: float = -1.0
    euler_tangential_signed_tail_density_support_min: float = -1.0
    euler_tangential_signed_tail_density_support_full: float = -1.0
    euler_tangential_tail_density_shock_damp_on: bool = False
    euler_tangential_tail_density_shock_damp_theta: float = 0.65
    euler_tangential_tail_density_shock_damp_pressure_min: float = 0.010
    euler_tangential_tail_density_shock_damp_compression_min: float = 0.002
    euler_tangential_tail_density_shock_damp_normality_min: float = 0.16
    euler_tangential_tail_safe_floor_on: bool = False
    euler_tangential_tail_safe_floor: float = 0.18
    euler_tangential_tail_shear_contact_relief_on: bool = False
    euler_tangential_tail_shear_contact_relief_floor: float = 0.08
    euler_tangential_tail_shear_contact_shear_min: float = 0.94
    euler_tangential_tail_shear_contact_normality_max: float = 0.08
    euler_tangential_tail_shear_contact_pressure_max: float = 0.008
    euler_tangential_tail_shear_contact_compression_max: float = 0.0015
    euler_tangential_tail_shear_contact_relief_apply_signed: bool = True
    euler_tangential_tail_shear_contact_relief_apply_curve: bool = True
    euler_tangential_tail_shear_contact_signed_floor: float = -1.0
    euler_tangential_tail_shear_contact_curve_floor: float = -1.0
    euler_tangential_signed_tail_safe_decay_relief_on: bool = False
    euler_tangential_signed_tail_safe_floor: float = 0.10
    euler_tangential_signed_tail_sidecar_decay_on: bool = False
    euler_tangential_signed_tail_sidecar_safe_floor: float = 0.10
    euler_tangential_signed_tail_sidecar_blend: float = 0.35
    euler_tangential_signed_tail_postrollback_preserve_on: bool = False
    euler_tangential_signed_tail_postrollback_theta: float = 0.35
    euler_tangential_tail_signed_anchored_curve_assist_on: bool = False
    euler_tangential_tail_signed_anchored_curve_floor: float = 0.04
    euler_tangential_tail_signed_anchored_curve_align_on: bool = False
    euler_tangential_tail_signed_anchored_curve_preserve_signed_on: bool = False
    euler_tangential_highsafe_raw_curve_microassist_on: bool = False
    euler_tangential_highsafe_raw_curve_microassist_floor: float = 0.015
    euler_tangential_highsafe_raw_curve_microassist_cap: float = 0.020
    euler_tangential_highsafe_raw_curve_microassist_wave_cap: float = 0.0025
    euler_tangential_highsafe_raw_curve_safe_min: float = 0.40
    euler_tangential_highsafe_raw_curve_shear_min: float = 0.94
    euler_tangential_highsafe_raw_curve_normality_max: float = 0.08
    euler_tangential_highsafe_raw_curve_pressure_max: float = 0.008
    euler_tangential_highsafe_raw_curve_compression_max: float = 0.0015
    euler_tangential_pair_gate_shock_reject_keep_v32: bool = True
    euler_tangential_pair_ignore_normality_gate: bool = False
    euler_tangential_pair_extend_on: bool = False
    euler_tangential_pair_extend_beta: float = 0.0
    euler_tangential_pair_extend_cap: float = 0.0
    euler_tangential_pair_extend_wave_cap: float = 0.0
    euler_tangential_pair_extend_alignment_min: float = 0.65
    euler_tangential_pair_extend_alignment_full: float = 0.90
    euler_tangential_pair_extend_shock_exclude: bool = False
    euler_tangential_pair_restore_stream_coherence_on: bool = False
    euler_tangential_pair_restore_stream_coherence_min: float = 0.2
    euler_tangential_pair_restore_stream_coherence_full: float = 0.6
    euler_contact_characteristic_postpass_on: bool = False
    euler_contact_characteristic_entropy_alpha: float = 0.0
    euler_contact_characteristic_tangential_alpha: float = 0.0
    euler_contact_characteristic_entropy_cap: float = 0.0
    euler_contact_characteristic_tangential_cap: float = 0.0
    euler_contact_characteristic_tangential_wave_cap: float = 0.0
    euler_contact_characteristic_pressure_alpha: float = 0.0
    euler_contact_characteristic_normal_alpha: float = 0.0
    euler_contact_characteristic_mood_fallback_on: bool = True
    euler_patch_contact_shear_postpass_on: bool = False
    euler_patch_contact_shear_neighbor_blend: float = 0.30
    euler_patch_contact_shear_entropy_alpha: float = 0.0
    euler_patch_contact_shear_tangential_alpha: float = 0.0
    euler_patch_contact_shear_entropy_cap: float = 0.0
    euler_patch_contact_shear_tangential_cap: float = 0.0
    euler_patch_contact_shear_tangential_wave_cap: float = 0.0
    euler_patch_contact_shear_min_valid_neighbours: int = 2
    euler_patch_contact_shear_roughness_cap: float = 0.08
    euler_patch_contact_shear_pair_spacing_on: bool = False
    euler_patch_contact_shear_pair_spacing_beta: float = 0.0
    euler_patch_contact_shear_gate_cap: float = 1.0
    euler_patch_contact_shear_pressure_floor_factor: float = 0.80
    euler_patch_contact_shear_pressure_margin_on: bool = True
    euler_patch_contact_shear_rho_floor_factor: float = 0.72
    euler_patch_contact_shear_late_pressure_rollback_on: bool = False
    euler_patch_contact_shear_p_floor_abs: float = 0.0
    euler_patch_contact_shear_rho_floor_abs: float = 0.0
    euler_patch_contact_shear_tangential_rollback_theta: float = 0.50
    euler_density_contact_weak_face_downstream_rho_beta: float = 0.035
    euler_density_contact_weak_face_downstream_rho_cap: float = 0.006
    euler_density_contact_weak_face_downstream_rho_wave_cap: float = 0.004
    euler_density_contact_weak_face_downstream_tangential_beta: float = 0.020
    euler_density_contact_weak_face_downstream_tangential_cap: float = 0.030
    euler_density_contact_weak_face_downstream_tangential_wave_cap: float = 0.004
    euler_density_contact_weak_face_stream_coherence_on: bool = False
    euler_density_contact_weak_face_stream_coherence_min: float = 0.2
    euler_density_contact_weak_face_stream_coherence_full: float = 0.6
    euler_density_signed_tail_trace_on: bool = False
    euler_density_signed_tail_trace_beta: float = 0.0
    euler_density_signed_tail_trace_cap: float = 0.0
    euler_density_signed_tail_trace_wave_cap: float = 0.0
    euler_tangential_downstream_shock_exclude: bool = False
    euler_tangential_downstream_shock_pressure_min: float = 0.025
    euler_tangential_downstream_shock_compression_min: float = 0.006
    euler_tangential_downstream_shock_normality_min: float = 0.35
    euler_tangential_downstream_branch_damp_on: bool = False
    euler_tangential_downstream_branch_pressure_min: float = 0.010
    euler_tangential_downstream_branch_compression_min: float = 0.002
    euler_tangential_downstream_branch_normality_min: float = 0.18
    euler_tangential_downstream_branch_floor: float = 0.78
    euler_tangential_clean_contact_tail_on: bool = False
    euler_tangential_clean_contact_tail_beta: float = 0.0
    euler_tangential_clean_contact_tail_cap: float = 0.0
    euler_tangential_clean_contact_tail_wave_cap: float = 0.0
    euler_tangential_clean_contact_tail_stream_full: float = 0.18
    euler_tangential_clean_contact_tail_pressure_lo: float = 0.006
    euler_tangential_clean_contact_tail_pressure_hi: float = 0.020
    euler_tangential_clean_contact_tail_compression_lo: float = 0.001
    euler_tangential_clean_contact_tail_compression_hi: float = 0.006
    euler_tangential_clean_contact_tail_normality_lo: float = 0.10
    euler_tangential_clean_contact_tail_normality_hi: float = 0.24
    euler_tangential_swirl_tail_on: bool = False
    euler_tangential_swirl_tail_beta: float = 0.0
    euler_tangential_swirl_tail_cap: float = 0.0
    euler_tangential_swirl_tail_wave_cap: float = 0.0
    euler_tangential_swirl_tail_q_min: float = 0.015
    euler_tangential_swirl_tail_q_full: float = 0.055
    euler_tangential_swirl_tail_pressure_hi: float = 0.018
    euler_tangential_swirl_tail_compression_hi: float = 0.004
    euler_tangential_swirl_tail_normality_hi: float = 0.20
    euler_tangential_signed_pair_tail_on: bool = False
    euler_tangential_signed_pair_tail_beta: float = 0.0
    euler_tangential_signed_pair_tail_cap: float = 0.0
    euler_tangential_signed_pair_tail_wave_cap: float = 0.0
    euler_tangential_signed_pair_tail_q_min: float = 0.010
    euler_tangential_signed_pair_tail_q_full: float = 0.040
    euler_tangential_signed_pair_tail_pressure_hi: float = 0.016
    euler_tangential_signed_pair_tail_compression_hi: float = 0.004
    euler_tangential_signed_pair_tail_normality_hi: float = 0.18
    euler_tangential_signed_tail_antisheet_on: bool = False
    euler_tangential_signed_tail_antisheet_strength: float = 0.0
    euler_tangential_signed_tail_antisheet_min_factor: float = 0.45
    euler_tangential_signed_tail_antisheet_q_hi: float = 0.070
    euler_tangential_signed_tail_antisheet_contact_min: float = 0.25
    euler_tangential_signed_tail_antisheet_contact_full: float = 0.60
    euler_tangential_signed_tail_bridge_cut_on: bool = False
    euler_tangential_signed_tail_bridge_cut_strength: float = 0.0
    euler_tangential_signed_tail_bridge_cut_min_factor: float = 0.25
    euler_tangential_signed_tail_bridge_cut_q_min: float = 0.08
    euler_tangential_signed_tail_bridge_cut_q_full: float = 0.22
    euler_tangential_signed_tail_bridge_cut_contact_min: float = 0.25
    euler_tangential_signed_tail_bridge_cut_contact_full: float = 0.60
    euler_tangential_signed_tail_bridge_cut_omega_lo_pct: float = 70.0
    euler_tangential_signed_tail_bridge_cut_omega_hi_pct: float = 92.0
    euler_tangential_signed_tail_qbridge_cut_on: bool = False
    euler_tangential_signed_tail_qbridge_cut_strength: float = 0.0
    euler_tangential_signed_tail_qbridge_cut_min_factor: float = 0.60
    euler_tangential_signed_tail_qbridge_cut_q_lo_pct: float = 30.0
    euler_tangential_signed_tail_qbridge_cut_q_mid_pct: float = 65.0
    euler_tangential_signed_tail_qbridge_cut_q_core_pct: float = 84.0
    euler_tangential_signed_tail_qbridge_cut_q_top_pct: float = 96.0
    euler_tangential_signed_tail_qbridge_cut_contact_min: float = 0.25
    euler_tangential_signed_tail_qbridge_cut_contact_full: float = 0.60
    euler_tangential_signed_tail_shock_ridge_clean_on: bool = False
    euler_tangential_signed_tail_shock_ridge_strength: float = 0.0
    euler_tangential_signed_tail_shock_ridge_min_factor: float = 0.60
    euler_tangential_signed_tail_shock_ridge_density_min: float = 0.35
    euler_tangential_signed_tail_shock_ridge_density_full: float = 0.85
    euler_tangential_signed_tail_shock_ridge_q_keep_min: float = 0.10
    euler_tangential_signed_tail_shock_ridge_q_keep_full: float = 0.25
    euler_tangential_signed_tail_hf_filter_on: bool = False
    euler_tangential_signed_tail_hf_filter_strength: float = 0.0
    euler_tangential_signed_tail_hf_filter_min_weight: float = 1e-12
    euler_tangential_signed_tail_hf_filter_shock_exclude: bool = True
    euler_tangential_density_curve_tail_hf_filter_on: bool = False
    euler_tangential_density_curve_tail_hf_filter_strength: float = 0.0
    euler_tangential_density_curve_tail_hf_filter_min_weight: float = 1e-12
    euler_tangential_density_curve_tail_hf_filter_shock_exclude: bool = True
    euler_tangential_density_curve_tail_bridge_cut_on: bool = False
    euler_tangential_density_curve_tail_bridge_cut_strength: float = 0.0
    euler_tangential_density_curve_tail_bridge_cut_min_factor: float = 0.50
    euler_tangential_density_curve_tail_bridge_cut_q_min: float = 0.08
    euler_tangential_density_curve_tail_bridge_cut_q_full: float = 0.22
    euler_tangential_density_curve_tail_bridge_cut_contact_min: float = 0.25
    euler_tangential_density_curve_tail_bridge_cut_contact_full: float = 0.60
    euler_tangential_density_curve_tail_bridge_cut_omega_lo_pct: float = 70.0
    euler_tangential_density_curve_tail_bridge_cut_omega_hi_pct: float = 92.0
    euler_tangential_density_curve_tail_qbridge_cut_on: bool = False
    euler_tangential_density_curve_tail_qbridge_cut_strength: float = 0.0
    euler_tangential_density_curve_tail_qbridge_cut_min_factor: float = 0.60
    euler_tangential_density_curve_tail_qbridge_cut_q_lo_pct: float = 30.0
    euler_tangential_density_curve_tail_qbridge_cut_q_mid_pct: float = 65.0
    euler_tangential_density_curve_tail_qbridge_cut_q_core_pct: float = 84.0
    euler_tangential_density_curve_tail_qbridge_cut_q_top_pct: float = 96.0
    euler_tangential_density_curve_tail_qbridge_cut_contact_min: float = 0.25
    euler_tangential_density_curve_tail_qbridge_cut_contact_full: float = 0.60
    euler_tangential_density_curve_tail_shock_ridge_clean_on: bool = False
    euler_tangential_density_curve_tail_shock_ridge_strength: float = 0.0
    euler_tangential_density_curve_tail_shock_ridge_min_factor: float = 0.60
    euler_tangential_density_curve_tail_shock_ridge_density_min: float = 0.35
    euler_tangential_density_curve_tail_shock_ridge_density_full: float = 0.85
    euler_tangential_density_curve_tail_shock_ridge_q_keep_min: float = 0.10
    euler_tangential_density_curve_tail_shock_ridge_q_keep_full: float = 0.25
    euler_tangential_total_qbridge_damp_on: bool = False
    euler_tangential_total_qbridge_damp_strength: float = 0.0
    euler_tangential_total_qbridge_damp_min_factor: float = 0.65
    euler_tangential_total_qbridge_damp_q_lo_pct: float = 30.0
    euler_tangential_total_qbridge_damp_q_mid_pct: float = 65.0
    euler_tangential_total_qbridge_damp_q_core_pct: float = 84.0
    euler_tangential_total_qbridge_damp_q_top_pct: float = 96.0
    euler_tangential_total_qbridge_damp_contact_min: float = 0.25
    euler_tangential_total_qbridge_damp_contact_full: float = 0.60
    euler_tangential_midq_cell_blend_on: bool = False
    euler_tangential_midq_cell_blend_strength: float = 0.0
    euler_tangential_midq_cell_blend_q_lo_pct: float = 20.0
    euler_tangential_midq_cell_blend_q_mid_pct: float = 55.0
    euler_tangential_midq_cell_blend_q_core_pct: float = 82.0
    euler_tangential_midq_cell_blend_q_top_pct: float = 96.0
    euler_tangential_midq_cell_blend_contact_min: float = 0.20
    euler_tangential_midq_cell_blend_contact_full: float = 0.55
    euler_tangential_density_curve_pair_tail_on: bool = False
    euler_tangential_density_curve_pair_tail_beta: float = 0.0
    euler_tangential_density_curve_pair_tail_cap: float = 0.0
    euler_tangential_density_curve_pair_tail_wave_cap: float = 0.0
    euler_tangential_density_curve_pair_tail_curve_min: float = 0.18
    euler_tangential_density_curve_pair_tail_curve_full: float = 0.45
    euler_tangential_density_curve_pair_tail_q_min: float = 0.008
    euler_tangential_density_curve_pair_tail_q_full: float = 0.035
    euler_tangential_density_curve_pair_tail_pressure_hi: float = 0.014
    euler_tangential_density_curve_pair_tail_compression_hi: float = 0.0035
    euler_tangential_density_curve_pair_tail_normality_hi: float = 0.18
    euler_tangential_legacy_pair_target_on: bool = False
    euler_tangential_legacy_pair_target_blend: float = 0.0
    euler_tangential_signed_pair_legacy_target_blend: float = -1.0
    euler_tangential_density_curve_legacy_target_blend: float = -1.0
    euler_tangential_safe_legacy_gate_on: bool = False
    euler_tangential_safe_legacy_pressure_hi: float = 0.010
    euler_tangential_safe_legacy_compression_hi: float = 0.002
    euler_tangential_safe_legacy_normality_hi: float = 0.14
    euler_tangential_safe_legacy_shear_min: float = 0.82
    euler_tangential_safe_legacy_contact_min: float = 0.45
    euler_tangential_safe_legacy_coherence_on: bool = False
    euler_tangential_safe_legacy_coherence_beta: float = 0.0
    euler_tangential_safe_legacy_coherence_floor: float = 0.08
    euler_tangential_safe_legacy_coherence_cap: float = 0.35
    euler_tangential_safe_legacy_qcurv_on: bool = False
    euler_tangential_safe_legacy_qcurv_beta: float = 0.18
    euler_tangential_safe_legacy_qcurv_q_min: float = 0.012
    euler_tangential_safe_legacy_qcurv_q_full: float = 0.045
    euler_tangential_safe_legacy_qcurv_curve_min: float = 0.20
    euler_tangential_safe_legacy_qcurv_curve_full: float = 0.50
    euler_tangential_shockline_rollback_on: bool = False
    euler_tangential_shockline_rollback_theta: float = 0.55
    euler_tangential_shockline_pressure_min: float = 0.012
    euler_tangential_shockline_compression_min: float = 0.0025
    euler_tangential_shockline_normality_min: float = 0.18
    euler_tangential_shockline_shear_max: float = 0.86
    euler_tangential_velocity_no_hancock: bool = False
    euler_tangential_contact_wave_hancock: bool = False
    euler_tangential_velocity_lsq_increment: bool = False
    euler_local_hancock: bool = False
    euler_log_positive: bool = False
    euler_face_positivity_limiter: bool = False
    euler_face_rho_abs_floor: float = 0.0
    euler_face_p_abs_floor: float = 0.0
    # PPM/HLLC-ADC style shock flattening for Euler primitives.  At strong
    # compressive pressure jumps it damps the high-order face increment,
    # reducing carbuncle-prone transverse antidiffusion while leaving shear
    # and smooth regions under the normal T-MLP-u limiter.
    stencil: str = 'face'
    order: int = 1
    # Optional performance knob for composite reconstructions: compute
    # high-order face states only for these primitive-variable indices and
    # leave all other variables at their first-order face values.
    active_vars: object = None
    mlp_bound: bool = True   # False ⇒ pure TVD limiter (no LMP wrapper)
    extremum_relax: bool = False   # smooth-region LMP relaxation
    extremum_relax_curved_otsu: bool = False
    vertex_mlp: bool = False    # PYG2010 vertex-projected polynomial bound
    vertex_mlp_cap: float = 1.0
    # Maximum ψ allowed by the vertex-MLP projection.  The canonical
    # PYG2010 slope limiter uses 1.0.  For T-MLP-u wrapped compressive
    # TVD schemes, values up to 2.0 can be used while still respecting
    # the vertex bounds; this preserves SUPERBEE/pure-downwind sharpness
    # where the local maximum principle permits it.
    vertex_mlp_face_local: bool = False
    vertex_mlp_face_local_otsu: bool = False
    vertex_mlp_face_local_otsu_mode: str = 'range'
    # Use an exact Otsu split of the cell-stencil variation to apply
    # face-local vertex caps only on high-variation cells.  Low-variation
    # smooth cells retain the cell-wide PYG vertex constraint, avoiding a
    # directional face-by-face limiter choice at smooth extrema.
    # When face nodes are available, compute the vertex-MLP cap from the
    # two vertices of the reconstructed face rather than the cell-wide
    # minimum over every vertex.  This keeps the PYG vertex bound but avoids
    # a remote vertex over-limiting unrelated face-side TVD increments.
    vertex_mlp_face_relax: float = 1.0
    # Safety factor for the face-local vertex cap.  1.0 reproduces the
    # conservative cell-wide PYG cap; >1 allows limited face-local relaxation
    # while keeping the original cell-wide cap as a positivity/stability guard.
    vertex_mlp_augment: bool = False
    physical_vertex_bounds: bool = False
    # Park-Kim MLP-u2-style smooth-extremum augmenting condition.  When the
    # vertex area-weighted average gradient is consistent with the local
    # T-MLP-u vertex increment, the vertex box is relaxed to ψ=1.  This is a
    # parameter-free way to avoid clipping smooth cone/hump extrema while
    # keeping discontinuous slot vertices bounded.
    physical_vertex_bounds_value_continuous_otsu: bool = False
    # Extended-bounds variant: use the physical [0,1] vertex box only on
    # lower-value cells whose stencil is not separated by a dominant gap.
    # This follows the extended-bounds idea for smooth extrema while keeping
    # high peaks and two-state interfaces under the local MLP vertex box.
    physical_vertex_bounds_value_upper_otsu: bool = False
    # Use the physical [0,1] vertex box on the lower side of a two-level Otsu
    # split of positive local maxima.  This is a data-derived intermediate
    # between global physical bounds and the dominant-gap continuous gate.
    weak_face_mlp: bool = False
    # Zhang-Liu-Chen weak-MLP condition: constrain the reconstructed face
    # centre by the arithmetic average of the MLP bounds at the face vertices,
    # instead of constraining every cell vertex.  This is less dissipative in
    # continuous regions while retaining the local maximum/minimum principle.
    weak_face_mlp_smooth_otsu: bool = False
    # Apply weak-face MLP only on locally linear cells identified by an exact
    # Otsu split of the scale-free LSQ residual.  This keeps the weak MLP's
    # smooth-extrema benefit while preserving sharper vertex constraints near
    # cone apexes, corners, and discontinuities.
    weak_face_mlp_range_otsu: bool = False
    # Alternative parameter-free weak-face gate: retain weak MLP only on the
    # low-variation class from an exact Otsu split of the local stencil range.
    # This targets smooth, low-amplitude extrema without relaxing sharp peaks.
    weak_face_mlp_high_range_otsu: bool = False
    # Complementary gate: retain weak-face MLP only on the high-variation
    # class from the same exact Otsu split.  This uses weak MLP as an
    # interface/discontinuity relaxation while leaving smooth extrema under
    # the stricter cell-wide vertex MLP constraint.
    weak_face_mlp_dominant_gap_or_high_range: bool = False
    # When the high-range gate is active, also allow weak-face MLP on cells
    # whose local stencil is split by one dominant value gap.  This keeps a
    # parameter-free two-state interface path for thin slots without opening
    # smooth ramps that only have gradual neighbour-to-neighbour changes.
    weak_face_mlp_value_otsu: bool = False
    # Use the positive local-maximum distribution to keep weak-face MLP on
    # submax smooth features while preserving stricter vertex caps on the
    # high-value phase-like class.  The split is exact Otsu, not a fixed value.
    weak_face_mlp_value_upper_otsu: bool = False
    # Two-level value Otsu gate. First separate background/features, then split
    # the positive feature tail and apply weak-face MLP below the upper split.
    # This guards mid-amplitude smooth extrema without damping the strongest
    # discontinuities or peaks.
    weak_face_mlp_curved_value_otsu: bool = False
    # Apply weak-face MLP only where two parameter-free tests agree: the local
    # LSQ residual is in the curved/nonlinear Otsu class and the positive local
    # maximum is in the lower-value Otsu class.  This targets rounded humps
    # without relaxing linear cone sides or high-value discontinuities.
    weak_face_mlp_value_continuous_otsu: bool = False
    # Apply weak-face MLP on lower-value feature cells only when their local
    # stencil is not split by one dominant value gap.  This keeps smooth,
    # continuous extrema under weak MLP while excluding two-state slot-like
    # discontinuities without using a case-tuned threshold.
    tvb_M: float = 0.0   # Cockburn-Shu TVB modulus (M·h² LMP tolerance)
    virtual_uu_gradient: bool = False
    # When True, the slope ratio r = (φ_U − φ_UU)/(φ_D − φ_U) uses a
    # *virtual* far-upwind value derived from the LSQ gradient at the
    # upwind cell — Darwish-Moukalled (2003), Jasak (1996).  Avoids the
    # geometric face-neighbour search and works on any unstructured mesh.
    #     φ_UU_virt = φ_D − 2·∇φ_U · (x_D − x_U)
    #     ⇒ φ_U − φ_UU_virt = −Δ⁺ + 2·∇φ_U · d_UD
    face_skew_correction: bool = False
    # Use the same arithmetic-average-gradient skew correction for the
    # reconstructed face increment that the T-MLP-u vertex criterion uses:
    #   δ_f = 0.5(φ_R-φ_L) + \bar∇φ·(d_f - 0.5 d_LR).
    # On orthogonal uniform grids this reduces exactly to α_f=0.5.
    face_gradient_correction: str = 'beta'
    face_gradient_shock_damping: str = ''
    # 'beta' uses the Mathur-Murthy style LR projection blend.  'jasak'
    # uses an over-relaxed normal correction that enforces the face-normal
    # gradient from the L/R jump while retaining the averaged tangential part.
    face_increment: str = 'tmlpu'
    # 'tmlpu' uses the t* plus corrected face-gradient increment from the
    # T-MLP-u vertex criterion.  'lsq' uses the upwind cell's WLSQ
    # projection grad_L·(x_f-x_L) for the final face value while keeping the
    # T-MLP-u vertex box as the limiter constraint.
    r_form: str = 'far_upwind'
    tmlpu_bound_tvd_separate: bool = False
    clamp_tstar: bool = True
    phi_LL_unclipped: bool = False
    # Theory-strict: phi_LL = phi_L - grad_L . d_LR with no clipping to
    # [phi_min, phi_max].  Disabled by default (legacy clip preserved).
    # When True, removes the artificial loosening of r at strong shocks
    # so the TVD limiter naturally drops psi -> 0 at shock faces.
    zero_delta_psi: float = 2.0
    # TVD cap used when the direct upwind-downwind jump Δ+ is at roundoff.
    # The legacy value 2.0 matches the previous downwind/compressive branch.
    # Setting 1.0 gives a scale-consistent non-compressive cap for skewed
    # faces where Δ+≈0 but the transverse T-MLP-u increment is not zero.
    # 'far_upwind' uses r=(phi_R-phi_L)/(phi_L-phi_LL) with a clipped
    # gradient-based far-upwind value.  'nvf' uses the Darwish-Moukalled
    # inverted normalized-variable form before mapping back to Sweby r.
    cicsam_full: bool = False
    # Full CICSAM (Ubbink 1997) in NVD framework:
    #   • Hyper-C arm (sharp): φ̃_f^HC = min(1, φ̃_C/Co)
    #   • Ultimate-QUICKEST arm (smooth):
    #       φ̃_f^UQ = (8·Co·φ̃_C + (1−Co)·(6·φ̃_C+3))/8
    #     clipped to [φ̃_C, φ̃_f^HC]
    #   • Blend: φ̃_f = γ·HC + (1−γ)·UQ
    #     γ = cos²(2θ), θ = ∠(∇φ_C, d_UD).  γ→1 sharp interface aligned
    #     with face → maximum compression; γ→0 gradient tangential →
    #     gentler UQ.  Outside monotone region [φ̃_C ∈ 0..1] falls back
    #     to first-order.  Bypasses the standard ψ_TVD path; uses the
    #     virtual-UU formulation for φ̃_C natively.
    cicsam_courant: float = 0.4   # Co for both HC and UQ formulae
    tvd_smooth: object = None
    # Optional secondary TVD limiter used only when explicitly requested
    # with extremum_relax=True.  The default T-MLP-u path uses a single
    # primary ψ_TVD and does not switch or blend limiters by threshold.
    smoothness_threshold: float = 0.1
    # LSQ-residual ratio below which a cell is classified `smooth`
    # (extremum_relax / tvd_smooth dispatch).  Lower = stricter, more
    # cells stay under LMP.  0.1 is the iter-15 default.
    tvd_smooth2: object = None
    # Optional tertiary limiter for *very* smooth cells, used when
    # residual < smoothness_threshold2 < smoothness_threshold.  Three
    # tiers: sharp (LMP+TVD primary) > moderate (tvd_smooth) >
    # very-smooth (tvd_smooth2).  Set to None for 2-tier (default).
    smoothness_threshold2: float = 0.05
    # Secondary threshold for tvd_smooth2; ignored if tvd_smooth2=None.
    idw_p: float = 6.0
    # Inverse-distance weighting exponent for the LSQ.  weight = 1/d^p.
    # Larger p emphasises closer cells more heavily.  6.0 is the
    # iter-N=50 winner.
    name: str = 't_mlp_u'

    def __post_init__(self):
        if isinstance(self.tvd, str):
            key = self.tvd.lower()
            if key in ('vanleer', 'van-leer'):
                key = 'van_leer'
            if key not in TVD_LIMITERS:
                raise ValueError(
                    f"unknown TVD limiter '{self.tvd}'; "
                    f"available: {list(TVD_LIMITERS)}")
            self._psi_tvd = TVD_LIMITERS[key]
            self._tvd_name = key
        elif callable(self.tvd):
            self._psi_tvd = self.tvd
            self._tvd_name = getattr(self.tvd, '__name__', 'custom')
        else:
            raise TypeError("`tvd` must be a string or a callable.")
        self._psi_tvd_velocity = None
        self._velocity_tvd_name = None
        self._psi_tvd_density = None
        self._density_tvd_name = None
        if self.euler_density_tvd is not None:
            if isinstance(self.euler_density_tvd, str):
                key_rho = self.euler_density_tvd.lower()
                if key_rho in ('vanleer', 'van-leer'):
                    key_rho = 'van_leer'
                if key_rho not in TVD_LIMITERS:
                    raise ValueError(
                        f"unknown euler_density_tvd '{self.euler_density_tvd}'")
                self._psi_tvd_density = TVD_LIMITERS[key_rho]
                self._density_tvd_name = key_rho
            elif callable(self.euler_density_tvd):
                self._psi_tvd_density = self.euler_density_tvd
                self._density_tvd_name = getattr(
                    self.euler_density_tvd, '__name__', 'custom')
            else:
                raise TypeError(
                    "`euler_density_tvd` must be None, a string, or callable.")
        if self.euler_velocity_tvd is not None:
            if isinstance(self.euler_velocity_tvd, str):
                key_v = self.euler_velocity_tvd.lower()
                if key_v in ('vanleer', 'van-leer'):
                    key_v = 'van_leer'
                if key_v in (
                        'shock_downwind_modified_superbee',
                        'shock-aware-downwind-modified-superbee',
                        'shock_downwind_superbee15',
                        'shock_downwind_superbee',
                        'shock-aware-downwind-superbee'):
                    self._psi_tvd_velocity = TVD_LIMITERS['downwind']
                    self._velocity_tvd_name = key_v
                elif key_v not in TVD_LIMITERS:
                    raise ValueError(
                        f"unknown euler_velocity_tvd '{self.euler_velocity_tvd}'")
                else:
                    self._psi_tvd_velocity = TVD_LIMITERS[key_v]
                    self._velocity_tvd_name = key_v
            elif callable(self.euler_velocity_tvd):
                self._psi_tvd_velocity = self.euler_velocity_tvd
                self._velocity_tvd_name = getattr(
                    self.euler_velocity_tvd, '__name__', 'custom')
            else:
                raise TypeError(
                    "`euler_velocity_tvd` must be None, a string, or callable.")
        self._psi_tvd_tangential_velocity = None
        self._tangential_velocity_tvd_name = None
        if self.euler_tangential_velocity_tvd is not None:
            if isinstance(self.euler_tangential_velocity_tvd, str):
                key_t = self.euler_tangential_velocity_tvd.lower()
                if key_t in ('vanleer', 'van-leer'):
                    key_t = 'van_leer'
                if key_t == 'contact_umist':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['umist']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'contact_umist_shock':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['umist']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'contact_umist_shock_root':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['umist']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'contact_superbee':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['superbee']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'contact_superbee_shock':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['superbee']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'superbee_shock_blend':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['superbee']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'shear_superbee_blend':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['superbee']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'shear_superbee_root_blend':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['superbee']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'shear_superbee_root_micro':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['superbee']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'shear_superbee_root_mood':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['superbee']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'contact_van_leer':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['van_leer']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'contact_van_leer_linear':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['van_leer']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'contact_van_leer_root':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['van_leer']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'shock_van_leer':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['van_leer']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'shock_van_leer_cubic':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['van_leer']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t == 'shock_van_leer_strict':
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS['van_leer']
                    self._tangential_velocity_tvd_name = key_t
                elif key_t not in TVD_LIMITERS:
                    raise ValueError(
                        f"unknown euler_tangential_velocity_tvd "
                        f"'{self.euler_tangential_velocity_tvd}'")
                else:
                    self._psi_tvd_tangential_velocity = TVD_LIMITERS[key_t]
                    self._tangential_velocity_tvd_name = key_t
            elif callable(self.euler_tangential_velocity_tvd):
                self._psi_tvd_tangential_velocity = (
                    self.euler_tangential_velocity_tvd)
                self._tangential_velocity_tvd_name = getattr(
                    self.euler_tangential_velocity_tvd, '__name__', 'custom')
            else:
                raise TypeError(
                    "`euler_tangential_velocity_tvd` must be None, a string, "
                    "or callable.")
        # Resolve secondary smooth-cell limiter (optional).
        self._psi_tvd_smooth = None
        if self.tvd_smooth is not None:
            if isinstance(self.tvd_smooth, str):
                key2 = self.tvd_smooth.lower()
                if key2 in ('vanleer', 'van-leer'):
                    key2 = 'van_leer'
                if key2 not in TVD_LIMITERS:
                    raise ValueError(
                        f"unknown tvd_smooth '{self.tvd_smooth}'")
                self._psi_tvd_smooth = TVD_LIMITERS[key2]
            elif callable(self.tvd_smooth):
                self._psi_tvd_smooth = self.tvd_smooth
            else:
                raise TypeError("`tvd_smooth` must be a string or callable.")
        # Resolve tertiary very-smooth limiter (optional, 3-tier).
        self._psi_tvd_smooth2 = None
        if self.tvd_smooth2 is not None:
            if isinstance(self.tvd_smooth2, str):
                key3 = self.tvd_smooth2.lower()
                if key3 in ('vanleer', 'van-leer'):
                    key3 = 'van_leer'
                if key3 not in TVD_LIMITERS:
                    raise ValueError(
                        f"unknown tvd_smooth2 '{self.tvd_smooth2}'")
                self._psi_tvd_smooth2 = TVD_LIMITERS[key3]
            elif callable(self.tvd_smooth2):
                self._psi_tvd_smooth2 = self.tvd_smooth2
            else:
                raise TypeError("`tvd_smooth2` must be a string or callable.")
        if self.stencil not in ('face', 'vertex', 'vertex2'):
            raise ValueError(
                f"stencil must be 'face' / 'vertex' / 'vertex2', got {self.stencil!r}")
        if self.order not in (1, 2, 3):
            raise ValueError(f"order must be 1, 2, or 3, got {self.order!r}")
        if self.order == 2 and self.stencil == 'face':
            # Quadratic needs 5 unknowns — face stencil (3 nbrs on triangles)
            # is under-determined.  Promote to vertex stencil silently.
            self.stencil = 'vertex'
        if self.order == 3 and self.stencil != 'vertex2':
            # Cubic LSQ needs 9 unknowns; vertex (≈ 7–10 cells) is borderline,
            # vertex2 (≈ 25 cells) is comfortably over-determined.
            self.stencil = 'vertex2'
        self._timestep_dt = None

    def set_timestep_context(self, dt: float | None, *, total_dt=None,
                             quad_weight=None, quad_points=None,
                             quad_weights=None):
        """Store the current physical time step for local Hancock scaling."""
        dt_value = total_dt if total_dt is not None else dt
        try:
            dt_float = float(dt_value)
        except (TypeError, ValueError):
            self._timestep_dt = None
            return
        self._timestep_dt = dt_float if np.isfinite(dt_float) and dt_float > 0.0 else None

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        if mesh.dim == 1:
            return self._reconstruct_1d(mesh, W_cell, eq, eval_points)
        if mesh.kind == 'structured_2d':
            return self._reconstruct_2d_axis_aligned(mesh, W_cell, eq, eval_points)
        if mesh.kind == 'unstructured_2d':
            return self._reconstruct_unstructured_2d(mesh, W_cell, eq, eval_points)
        raise NotImplementedError(
            f"TMLPU.reconstruct: unsupported mesh "
            f"(dim={mesh.dim}, kind={mesh.kind})."
        )

    # --- 1D structured implementation ---------------------------------------
    def _reconstruct_1d(self, mesh, W_cell, eq, eval_points=None):
        nvar, N = W_cell.shape
        n_faces = mesh.n_faces
        owner = mesh.face_owner
        nei   = mesh.face_neighbour

        # Build "left / right" neighbour tables for every cell using x-coords.
        left = np.full(N, -1, dtype=int)
        right = np.full(N, -1, dtype=int)
        xs = mesh.cell_centers[:, 0]
        for i in range(N):
            for nb in mesh.cell_neighbours[i]:
                if nb < 0:
                    continue
                if xs[nb] < xs[i]:
                    left[i] = nb
                else:
                    right[i] = nb

        W_L = np.empty((nvar, n_faces), dtype=float)
        W_R = np.empty((nvar, n_faces), dtype=float)

        # First-order default (overridden for interior faces below;
        # boundary slots are then patched by `boundary.apply_patch_bcs`).
        n_idx = np.maximum(nei, 0)
        for v in range(nvar):
            W_L[v] = W_cell[v, owner]
            W_R[v] = np.where(nei >= 0, W_cell[v, n_idx], W_cell[v, owner])

        # Interior faces: full T-MLP-u
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R

        # For each interior face, work out the UU on each side.
        # Rule: nei is to the right of owner  ⇒ UU_owner = left[owner],
        #                                       UU_nei   = right[nei].
        #       nei is to the left  of owner  ⇒ UU_owner = right[owner],
        #                                       UU_nei   = left[nei].
        n_face_int = interior.size
        UU_o = np.empty(n_face_int, dtype=int)
        UU_n = np.empty(n_face_int, dtype=int)
        for k, f in enumerate(interior):
            o = int(owner[f]); n = int(nei[f])
            if xs[n] > xs[o]:
                UU_o[k] = left[o]
                UU_n[k] = right[n]
            else:
                UU_o[k] = right[o]
                UU_n[k] = left[n]

        # Treat missing UU (at-boundary cell) by falling back to U itself
        # — equivalent to first-order on that face side.  W_L/W_R already
        # holds first-order values, so we only overwrite with the T-MLP-u
        # value when both sides have a valid UU.
        valid_o = UU_o >= 0
        valid_n = UU_n >= 0

        o_idx = owner[interior]
        n_idx_int = nei[interior]
        UU_o_safe = np.where(valid_o, UU_o, o_idx)
        UU_n_safe = np.where(valid_n, UU_n, n_idx_int)

        # Vectorised gather over variables and faces.
        for v in range(nvar):
            phi_U_o = W_cell[v, o_idx]
            phi_D_o = W_cell[v, n_idx_int]
            phi_UU_o = W_cell[v, UU_o_safe]
            recon_owner = t_mlp_u_face_value(phi_UU_o, phi_U_o, phi_D_o,
                                             self._psi_tvd,
                                             hancock_courant=self.hancock_courant)
            W_L[v, interior] = np.where(valid_o, recon_owner, phi_U_o)

            phi_U_n = W_cell[v, n_idx_int]
            phi_D_n = W_cell[v, o_idx]
            phi_UU_n = W_cell[v, UU_n_safe]
            recon_nei = t_mlp_u_face_value(phi_UU_n, phi_U_n, phi_D_n,
                                           self._psi_tvd,
                                           hancock_courant=self.hancock_courant)
            W_R[v, interior] = np.where(valid_n, recon_nei, phi_U_n)

        return W_L, W_R

    # --- 2D structured Cartesian implementation -----------------------------
    def _reconstruct_2d_axis_aligned(self, mesh, W_cell, eq, eval_points=None):
        """T-MLP-u for axis-aligned Cartesian grids.

        For every face the dominant axis is read off the face normal; the
        upstream cell UU is then the same-axis neighbour of U on the side
        opposite to D.  The per-face formula is identical to the 1D path.
        """
        nvar, N = W_cell.shape
        n_faces = mesh.n_faces
        owner = mesh.face_owner
        nei = mesh.face_neighbour

        xs = mesh.cell_centers[:, 0]
        ys = mesh.cell_centers[:, 1]

        # Per-cell axis neighbour tables.
        xneg = np.full(N, -1, dtype=int)
        xpos = np.full(N, -1, dtype=int)
        yneg = np.full(N, -1, dtype=int)
        ypos = np.full(N, -1, dtype=int)
        for i in range(N):
            for nb in mesh.cell_neighbours[i]:
                if nb < 0:
                    continue
                ddx = xs[nb] - xs[i]
                ddy = ys[nb] - ys[i]
                if abs(ddx) >= abs(ddy):
                    if ddx < 0:
                        xneg[i] = nb
                    else:
                        xpos[i] = nb
                else:
                    if ddy < 0:
                        yneg[i] = nb
                    else:
                        ypos[i] = nb

        # First-order default
        W_L = np.empty((nvar, n_faces), dtype=float)
        W_R = np.empty((nvar, n_faces), dtype=float)
        n_idx_def = np.maximum(nei, 0)
        for v in range(nvar):
            W_L[v] = W_cell[v, owner]
            W_R[v] = np.where(nei >= 0, W_cell[v, n_idx_def], W_cell[v, owner])

        # Per-face UU lookup (axis decided by face normal sign)
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return W_L, W_R
        n_face_int = interior.size
        UU_o = np.empty(n_face_int, dtype=int)
        UU_n = np.empty(n_face_int, dtype=int)
        nx = mesh.face_normals[interior, 0]
        ny = mesh.face_normals[interior, 1]
        for k in range(n_face_int):
            f = interior[k]; o = int(owner[f]); nb = int(nei[f])
            if abs(nx[k]) >= abs(ny[k]):
                # x-axis face — nei is on +x or -x of owner depending on sign(nx).
                if nx[k] >= 0:
                    UU_o[k] = xneg[o]
                    UU_n[k] = xpos[nb]
                else:
                    UU_o[k] = xpos[o]
                    UU_n[k] = xneg[nb]
            else:
                if ny[k] >= 0:
                    UU_o[k] = yneg[o]
                    UU_n[k] = ypos[nb]
                else:
                    UU_o[k] = ypos[o]
                    UU_n[k] = yneg[nb]

        valid_o = UU_o >= 0
        valid_n = UU_n >= 0
        o_idx = owner[interior]
        n_idx_int = nei[interior]
        UU_o_safe = np.where(valid_o, UU_o, o_idx)
        UU_n_safe = np.where(valid_n, UU_n, n_idx_int)

        for v in range(nvar):
            phi_U_o = W_cell[v, o_idx]
            phi_D_o = W_cell[v, n_idx_int]
            phi_UU_o = W_cell[v, UU_o_safe]
            recon_owner = t_mlp_u_face_value(phi_UU_o, phi_U_o, phi_D_o,
                                             self._psi_tvd,
                                             hancock_courant=self.hancock_courant)
            W_L[v, interior] = np.where(valid_o, recon_owner, phi_U_o)

            phi_U_n = W_cell[v, n_idx_int]
            phi_D_n = W_cell[v, o_idx]
            phi_UU_n = W_cell[v, UU_n_safe]
            recon_nei = t_mlp_u_face_value(phi_UU_n, phi_U_n, phi_D_n,
                                           self._psi_tvd,
                                           hancock_courant=self.hancock_courant)
            W_R[v, interior] = np.where(valid_n, recon_nei, phi_U_n)

        return W_L, W_R

    # --- Unstructured 2D implementation (vectorised) ------------------------
    def _reconstruct_unstructured_2d(self, mesh, W_cell, eq, eval_points=None):
        """T-MLP-u extended to unstructured 2D grids — fully vectorised.

        Per cell/face side L→R, an unlimited least-squares polynomial is
        computed from the chosen stencil.  The T-MLP-u path follows the
        vertex formula:

            φ_LL = clip(φ_R − 2∇φ_L·d_LR, φ_L^min, φ_L^max)
            r    = (φ_R − φ_L) / (φ_L − φ_LL)
            Δφ_V = 0.5(φ_R − φ_L) + \bar{∇φ}·(d_V − 0.5d_LR)
            α_V  = bound_V / (Δφ_V min(1,r))
            ψ    = min_V min(α_V r, α_V, ψ_TVD)
            φ_f  = φ_L + ψ (1 − C_f) α_f (φ_R − φ_L)

        α_f is the geometric face-location coefficient.  For pure downwind
        stress tests it is forced to 0.5 so ψ_TVD=2 gives φ_f=φ_R when the
        MLP bound is disabled.

        UU is chosen per face as the face-neighbour of C whose centroid
        offset is most *opposite* the downstream direction (x_D − x_C).
        For unstructured grids without such a "opposite" neighbour
        (boundary cells, acute fans) UU defaults to C itself, giving
        Δ_- = 0 and falling back to the LMP-only limiter.

        Mesh-dependent quantities (UU per face, neighbour padding, A⁻¹
        for the gradient, face-displacement vectors) are cached on the
        mesh so they are computed only once.
        """
        nvar, N = W_cell.shape
        n_faces = mesh.n_faces
        owner = mesh.face_owner
        nei = mesh.face_neighbour

        def _finish_faces(W_L, W_R):
            """Optionally enforce Euler face-state admissibility.

            This is a uniform face-local scaling limiter: if a reconstructed
            primitive density or pressure would leave the admissible set, scale
            the high-order face increment back toward the owning cell average.
            The floor is machine-precision based; it is not a case threshold.
            """
            enabled = (
                bool(self.euler_face_positivity_limiter)
                or os.environ.get(
                    'TMLPU_EULER_FACE_POSITIVITY_LIMITER', '0'
                ).lower() in ('1', 'true', 'yes', 'on'))
            if (not enabled or nvar < 4
                    or getattr(eq, '__class__', type(eq)).__name__ != 'Euler2D'):
                return W_L, W_R
            tiny = np.finfo(float).tiny
            eps = np.finfo(float).eps
            n_idx_def = np.maximum(nei, 0)
            abs_floors = {
                0: max(float(self.euler_face_rho_abs_floor), 0.0),
                3: max(float(self.euler_face_p_abs_floor), 0.0),
            }
            # Physical relative-floor admissibility (Zhang-Shu style): when a
            # relative floor factor is set, scale the WHOLE primitive increment
            # of a face back toward its owning cell average by a single theta so
            # density and pressure stay at >= factor * cell value.  A single
            # theta keeps the velocity consistent with the limited rho/p.  This
            # is a global, feature-free positivity layer (no ROI, no case
            # coefficient); it only activates where the high-order face state
            # would otherwise leave the admissible set at strong shocks.
            rho_ff = float(os.environ.get('TMLPU_EULER_RHO_FLOOR_FACTOR', '0.0'))
            p_ff = float(os.environ.get('TMLPU_EULER_P_FLOOR_FACTOR', '0.0'))
            if rho_ff > 0.0 or p_ff > 0.0:
                rel = {0: max(rho_ff, 0.0), 3: max(p_ff, 0.0)}

                def _theta(Wf, cellv):
                    theta = np.ones(Wf.shape[1])
                    for var in (0, 3):
                        cell = cellv(var)
                        floor = np.maximum(
                            np.maximum(tiny, eps * np.abs(cell)),
                            np.maximum(abs_floors[var], rel[var] * np.abs(cell)))
                        denom = cell - Wf[var]
                        th = np.where(
                            Wf[var] < floor,
                            np.where(denom > tiny,
                                     (cell - floor) / np.maximum(denom, tiny),
                                     0.0),
                            1.0)
                        theta = np.minimum(theta, np.clip(th, 0.0, 1.0))
                    return theta

                cellL = lambda var: W_cell[var, owner]
                cellR = lambda var: np.where(
                    nei >= 0, W_cell[var, n_idx_def], W_cell[var, owner])
                theta_L = _theta(W_L, cellL)
                theta_R = _theta(W_R, cellR)
                for v in range(nvar):
                    cl = W_cell[v, owner]
                    W_L[v] = cl + theta_L * (W_L[v] - cl)
                    cr = np.where(nei >= 0, W_cell[v, n_idx_def],
                                  W_cell[v, owner])
                    W_R[v] = cr + theta_R * (W_R[v] - cr)
                return W_L, W_R
            for var in (0, 3):
                cell_l = W_cell[var, owner]
                abs_floor = abs_floors[var]
                floor_l = np.maximum(
                    np.maximum(tiny, eps * np.abs(cell_l)), abs_floor)
                mask_l = W_L[var] < floor_l
                if np.any(mask_l):
                    denom_l = cell_l - W_L[var]
                    theta_l = np.where(
                        denom_l > 0.0,
                        (cell_l - floor_l) / np.maximum(denom_l, tiny),
                        0.0)
                    theta_l = np.clip(theta_l, 0.0, 1.0)
                    limited_l = cell_l + theta_l * (W_L[var] - cell_l)
                    W_L[var] = np.where(mask_l, limited_l, W_L[var])

                cell_r = np.where(nei >= 0, W_cell[var, n_idx_def], cell_l)
                floor_r = np.maximum(
                    np.maximum(tiny, eps * np.abs(cell_r)), abs_floor)
                mask_r = W_R[var] < floor_r
                if np.any(mask_r):
                    denom_r = cell_r - W_R[var]
                    theta_r = np.where(
                        denom_r > 0.0,
                        (cell_r - floor_r) / np.maximum(denom_r, tiny),
                        0.0)
                    theta_r = np.clip(theta_r, 0.0, 1.0)
                    limited_r = cell_r + theta_r * (W_R[var] - cell_r)
                    W_R[var] = np.where(mask_r, limited_r, W_R[var])
            return W_L, W_R

        if self.active_vars is None:
            active_vars = tuple(range(nvar))
        else:
            active_vars = tuple(
                int(v) for v in self.active_vars
                if 0 <= int(v) < nvar)
            if not active_vars:
                W_L = np.empty((nvar, n_faces), dtype=float)
                W_R = np.empty((nvar, n_faces), dtype=float)
                n_idx_def = np.maximum(nei, 0)
                for v in range(nvar):
                    W_L[v] = W_cell[v, owner]
                    W_R[v] = np.where(
                        nei >= 0, W_cell[v, n_idx_def], W_cell[v, owner])
                return _finish_faces(W_L, W_R)
        all_vars_active = (len(active_vars) == nvar)
        active_vars_arr = getattr(self, '_active_vars_arr_cache', None)
        active_vars_key = getattr(self, '_active_vars_key_cache', None)
        if active_vars_arr is None or active_vars_key != active_vars:
            active_vars_arr = np.asarray(active_vars, dtype=np.int64)
            self._active_vars_arr_cache = active_vars_arr
            self._active_vars_key_cache = active_vars

        ctx = self._unstructured_cache(mesh)

        nb_padded = ctx['nb_padded']    # (N, max_nb)
        nb_safe   = ctx['nb_safe']
        valid_nb  = ctx['valid_nb']
        A_basis   = ctx['A']            # (N, max_nb, nbasis)  ← already √W·A
        grad_A_basis = ctx.get('grad_A', A_basis)
        ATA_inv   = ctx['ATA_inv']      # (N, nbasis, nbasis)
        nbasis    = ctx['nbasis']
        sqrt_w    = ctx['sqrt_w']       # (N, max_nb) — same √W weighting
        lsq_op    = ctx.get('lsq_op')   # (N, nbasis, max_nb)
        grad_nb_safe = ctx.get('grad_nb_safe', nb_safe)
        grad_valid_nb = ctx.get('grad_valid_nb', valid_nb)
        grad_sqrt_w = ctx.get('grad_sqrt_w', sqrt_w)
        grad_lsq_op = ctx.get('grad_lsq_op', lsq_op)
        UU_o_int  = ctx['UU_o_int']     # interior faces only
        UU_n_int  = ctx['UU_n_int']
        d_o_int   = ctx['d_o_int']      # (Nint, 2)  x_neighbour − x_owner
        interior  = ctx['interior']

        # Per-call evaluation points (face centres by default; high-order
        # solver passes Gauss-quadrature points here to maintain ≥3rd-order
        # face quadrature).
        use_cached_face_geometry = (
            eval_points is None or eval_points is mesh.face_centers)
        if use_cached_face_geometry:
            dx_fo = ctx['dx_fo']
            dx_fn = ctx['dx_fn']
            d_sq = ctx['d_sq']
            alpha_o = ctx['alpha_o']
            alpha_n = ctx['alpha_n']
            face_n_o = ctx['face_n_o']
            tstar_o = ctx['tstar_o']
            tstar_n = ctx['tstar_n']
            d_len = ctx['d_len']
            e_o = ctx['e_o']
            e_n = ctx['e_n']
        else:
            dx_fo = (eval_points[interior]
                     - mesh.cell_centers[mesh.face_owner[interior]])
            dx_fn = (eval_points[interior]
                     - mesh.cell_centers[mesh.face_neighbour[interior]])
            d_sq = np.sum(d_o_int * d_o_int, axis=1)
            safe_d_sq = np.maximum(d_sq, 1e-30)
            alpha_o = np.sum(dx_fo * d_o_int, axis=1) / safe_d_sq
            alpha_n = np.sum(dx_fn * (-d_o_int), axis=1) / safe_d_sq
            alpha_o = np.clip(alpha_o, 0.0, 1.0)
            alpha_n = np.clip(alpha_n, 0.0, 1.0)
            face_n_o = mesh.face_normals[interior]
            norm_den = np.sum(d_o_int * face_n_o, axis=1)
            norm_den_safe = np.where(np.abs(norm_den) > 1e-30,
                                     norm_den, np.copysign(1e-30, norm_den))
            tstar_o = np.sum(dx_fo * face_n_o, axis=1) / norm_den_safe
            tstar_n = np.sum(dx_fn * (-face_n_o), axis=1) / norm_den_safe
            tstar_o = np.where(np.abs(norm_den) > 1e-30, tstar_o, alpha_o)
            tstar_n = np.where(np.abs(norm_den) > 1e-30, tstar_n, alpha_n)
            if self.clamp_tstar:
                tstar_o = np.clip(tstar_o, 0.0, 1.0)
                tstar_n = np.clip(tstar_n, 0.0, 1.0)
            d_len = np.sqrt(np.maximum(d_sq, 1e-30))
            e_o = d_o_int / d_len[:, None]
            e_n = -e_o
        if self._tvd_name in ('bounded_cd', 'central', 'cd',
                              'pure_downwind'):
            alpha_o = np.full_like(alpha_o, 0.5)
            alpha_n = np.full_like(alpha_n, 0.5)
        theta_min = 0.3

        shock_flatten = np.zeros_like(alpha_o)
        pressure_flatten = np.zeros_like(alpha_o)
        tangential_shock_flatten = np.zeros_like(alpha_o)
        velocity_flatten = np.zeros_like(alpha_o)
        density_flatten = np.zeros_like(alpha_o)
        density_contact_weight = np.ones_like(alpha_o)
        tangential_contact_weight = np.ones_like(alpha_o)
        tangential_ducros_default = (
            '1' if self.euler_tangential_ducros else '0')
        tangential_ducros_gate = os.environ.get(
            'TMLPU_EULER_TANGENTIAL_DUCROS', tangential_ducros_default
        ).lower() in ('1', 'true', 'yes', 'on')
        tangential_flatten_default = str(
            self.euler_tangential_flatten_mode or '')
        tangential_flatten_mode = os.environ.get(
            'TMLPU_EULER_TANGENTIAL_FLATTEN_MODE',
            tangential_flatten_default
        ).strip().lower()
        velocity_flatten_default = str(
            self.euler_velocity_flatten_sensor or 'pressure')
        velocity_flatten_mode = os.environ.get(
            'TMLPU_EULER_VELOCITY_FLATTEN_SENSOR',
            velocity_flatten_default
        ).strip().lower()
        wall_tangential_flatten_enabled = (
            bool(self.euler_wall_tangential_flatten)
            or os.environ.get(
                'TMLPU_EULER_WALL_TANGENTIAL_FLATTEN', '0'
            ).lower() in ('1', 'true', 'yes', 'on'))
        tangential_normality_flatten = os.environ.get(
            'TMLPU_EULER_TANGENTIAL_NORMALITY_FLATTEN', '0'
        ).lower() in ('1', 'true', 'yes', 'on')
        if tangential_normality_flatten and not tangential_flatten_mode:
            tangential_flatten_mode = 'normality'
        face_hancock_courant = np.full_like(alpha_o, self.hancock_courant)
        one_minus_C_face = 1.0 - face_hancock_courant
        density_contact_hancock_scale = one_minus_C_face.copy()
        pressure_jump = np.zeros_like(alpha_o)
        compression = np.zeros_like(alpha_o)
        normality = np.zeros_like(alpha_o)
        prep_cache = getattr(mesh, '_tmlpu_transient_recon_prep_cache', None)
        if prep_cache is not None and prep_cache.get('W_cell') is not W_cell:
            prep_cache = None
        if (prep_cache is None
                and os.environ.get('TMLPU_PREP_CACHE', '0').lower()
                in ('1', 'true', 'yes', 'on')):
            # Activate the (otherwise dormant) transient prep cache, keyed by the
            # W_cell object identity. Two reconstructions on the SAME W_cell
            # (e.g. the two BVD branches in one call) then share the expensive
            # LSQ coeffs, vertex min/max bounds, and Euler sensors. Auto-
            # invalidates next substep when W_cell changes (guard above).
            prep_cache = {'W_cell': W_cell}
            mesh._tmlpu_transient_recon_prep_cache = prep_cache
        sensor_key = (
            'euler_face_sensors',
            0 if use_cached_face_geometry else id(eval_points),
            float(self.hancock_courant),
            bool(self.euler_shock_flatten),
            bool(self.euler_local_hancock),
            bool(self.euler_density_acoustic_flatten),
            bool(self.euler_density_shear_contact),
            bool(tangential_ducros_gate),
            tangential_flatten_mode,
            velocity_flatten_mode,
            bool(os.environ.get(
                'TMLPU_EULER_DENSITY_FULL_SHOCK_FLATTEN', '0'
            ).lower() in ('1', 'true', 'yes', 'on')),
        )
        cached_sensors = (
            prep_cache.get(sensor_key)
            if prep_cache is not None else None)
        if cached_sensors is not None:
            if len(cached_sensors) == 13:
                (shock_flatten, pressure_flatten, tangential_shock_flatten,
                 velocity_flatten, density_flatten, density_contact_weight,
                 tangential_contact_weight, face_hancock_courant,
                 one_minus_C_face, density_contact_hancock_scale,
                 pressure_jump, compression, normality) = cached_sensors
            elif len(cached_sensors) == 10:
                (shock_flatten, pressure_flatten, tangential_shock_flatten,
                 velocity_flatten, density_flatten, density_contact_weight,
                 tangential_contact_weight, face_hancock_courant,
                 one_minus_C_face, density_contact_hancock_scale) = cached_sensors
                # Legacy cache shape (pre-v15). Compute shock-only sensors
                # from cached geometric/primitive state.
                rho_o = np.maximum(W_cell[0, owner[interior]], 1.0e-30)
                rho_n = np.maximum(W_cell[0, nei[interior]], 1.0e-30)
                u_o = W_cell[1, owner[interior]]
                v_o = W_cell[2, owner[interior]]
                p_o = np.maximum(W_cell[3, owner[interior]], 1.0e-30)
                u_n = W_cell[1, nei[interior]]
                v_n = W_cell[2, nei[interior]]
                p_n = np.maximum(W_cell[3, nei[interior]], 1.0e-30)
                gamma = float(getattr(eq, 'gamma', 1.4))
                c_o = np.sqrt(np.maximum(gamma * p_o / rho_o, 1.0e-30))
                c_n = np.sqrt(np.maximum(gamma * p_n / rho_n, 1.0e-30))
                du = u_n - u_o
                dv = v_n - v_o
                dun = du * face_n_o[:, 0] + dv * face_n_o[:, 1]
                dut = du * (-face_n_o[:, 1]) + dv * face_n_o[:, 0]
                pressure_jump = np.abs(p_n - p_o) / np.maximum(p_n + p_o, 1.0e-30)
                compression = np.maximum(0.0, dun) / np.maximum(c_o + c_n, 1.0e-30)
                normality = np.abs(dun) / np.maximum(
                    np.abs(dun) + np.abs(dut), 1.0e-30)
            else:
                cached_sensors = None
        if (cached_sensors is None
                and (self.euler_shock_flatten or self.euler_local_hancock
                     or self.euler_density_acoustic_flatten)
                and getattr(eq, '__class__', type(eq)).__name__ == 'Euler2D'
                and nvar >= 4):
            rho_o = np.maximum(W_cell[0, owner[interior]], 1.0e-30)
            rho_n = np.maximum(W_cell[0, nei[interior]], 1.0e-30)
            u_o = W_cell[1, owner[interior]]
            v_o = W_cell[2, owner[interior]]
            p_o = np.maximum(W_cell[3, owner[interior]], 1.0e-30)
            u_n = W_cell[1, nei[interior]]
            v_n = W_cell[2, nei[interior]]
            p_n = np.maximum(W_cell[3, nei[interior]], 1.0e-30)
            gamma = float(getattr(eq, 'gamma', 1.4))
            c_o = np.sqrt(np.maximum(gamma * p_o / rho_o, 1.0e-30))
            c_n = np.sqrt(np.maximum(gamma * p_n / rho_n, 1.0e-30))
            un_o = u_o * face_n_o[:, 0] + v_o * face_n_o[:, 1]
            un_n = u_n * face_n_o[:, 0] + v_n * face_n_o[:, 1]
            un_contact = 0.5 * (np.abs(un_o) + np.abs(un_n))
            c_contact = 0.5 * (c_o + c_n)
            contact_courant_fraction = un_contact / np.maximum(
                un_contact + c_contact, 1.0e-30)
            if self.euler_local_hancock:
                dt_ctx = getattr(self, '_timestep_dt', None)
                if dt_ctx is not None:
                    face_speed = np.maximum(np.abs(un_o) + c_o,
                                            np.abs(un_n) + c_n)
                    face_hancock_courant = np.clip(
                        dt_ctx * face_speed / np.maximum(d_len, 1.0e-30),
                        0.0, 1.0)
                    one_minus_C_face = 1.0 - face_hancock_courant
            density_contact_hancock_scale = (
                1.0 - face_hancock_courant * contact_courant_fraction)
            pressure_jump = np.abs(p_n - p_o) / np.maximum(p_n + p_o, 1.0e-30)
            compression = np.maximum(0.0, un_o - un_n) / np.maximum(c_o + c_n, 1.0e-30)
            pressure_sensor = np.clip((pressure_jump - 0.05) / 0.35, 0.0, 1.0)
            compression_sensor = np.clip(4.0 * compression, 0.0, 1.0)
            shock_sensor = pressure_sensor * np.maximum(
                pressure_sensor, compression_sensor)
            du = u_n - u_o
            dv = v_n - v_o
            dun = du * face_n_o[:, 0] + dv * face_n_o[:, 1]
            dut = du * (-face_n_o[:, 1]) + dv * face_n_o[:, 0]
            normality = np.abs(dun) / np.maximum(
                np.abs(dun) + np.abs(dut), 1.0e-30)
            shear_strength = np.abs(dut) / np.maximum(c_o + c_n, 1.0e-30)
            compression_strength = np.maximum(0.0, un_o - un_n) / np.maximum(
                c_o + c_n, 1.0e-30)
            comp_sq = compression_strength * compression_strength
            shear_sq = shear_strength * shear_strength
            ducros_compression_weight = comp_sq / np.maximum(
                comp_sq + shear_sq, np.finfo(float).eps)
            shock_base = np.maximum(pressure_sensor, compression_sensor)
            pressure_ratio_sensor = np.clip(
                np.abs(p_n - p_o) / np.maximum(np.maximum(p_n, p_o), 1.0e-30),
                0.0, 1.0)
            density_shock_flatten = shock_base * np.sqrt(normality)
            ducros_normal_flatten = density_shock_flatten * ducros_compression_weight
            wave_contact_flatten = density_shock_flatten * (
                pressure_ratio_sensor
                + (1.0 - pressure_ratio_sensor) * ducros_compression_weight)
            shock_flatten = shock_base
            pressure_flatten = shock_base
            tangential_pressure_flatten = np.maximum(
                shock_base, pressure_ratio_sensor)
            acoustic_normal_flatten = (
                np.sqrt(np.maximum(pressure_sensor * compression_sensor, 0.0))
                * normality)
            if tangential_flatten_mode == 'normality':
                tangential_shock_flatten = density_shock_flatten
            elif tangential_flatten_mode == 'geomean':
                tangential_shock_flatten = np.sqrt(
                    np.maximum(tangential_pressure_flatten
                               * density_shock_flatten, 0.0))
            else:
                tangential_shock_flatten = tangential_pressure_flatten
            if velocity_flatten_mode in (
                    'acoustic_normality', 'acoustic-normality',
                    'shock_normality', 'shock-normality'):
                velocity_flatten = acoustic_normal_flatten
            elif velocity_flatten_mode in (
                    'shear_aware', 'shear-aware', 'normality',
                    'tangential', 'ducros'):
                velocity_flatten = tangential_shock_flatten
            elif velocity_flatten_mode in (
                    'ducros_normality', 'ducros-normality',
                    'shear_dilatation', 'shear-dilatation',
                    'vorticity_guard', 'vorticity-guard'):
                velocity_flatten = ducros_normal_flatten
            elif velocity_flatten_mode in (
                    'wave_contact', 'wave-contact',
                    'pressure_contact', 'pressure-contact',
                    'contact_aware', 'contact-aware'):
                velocity_flatten = wave_contact_flatten
            elif velocity_flatten_mode in (
                    'density_soft', 'density-soft',
                    'density_normality_soft', 'density-normality-soft'):
                velocity_flatten = density_shock_flatten * density_shock_flatten
            elif velocity_flatten_mode in (
                    'density_acoustic_soft', 'density-acoustic-soft',
                    'density_normality_acoustic_soft',
                    'density-normality-acoustic-soft'):
                velocity_flatten = np.maximum(
                    density_shock_flatten * density_shock_flatten,
                    acoustic_normal_flatten)
            elif velocity_flatten_mode in ('density', 'density_normality'):
                velocity_flatten = density_shock_flatten
            elif velocity_flatten_mode in (
                    'geomean_normality', 'geomean-normality',
                    'pressure_density_geomean',
                    'pressure-density-geomean'):
                velocity_flatten = np.sqrt(
                    np.maximum(pressure_flatten * density_shock_flatten, 0.0))
            elif velocity_flatten_mode in (
                    'compression_core_geomean',
                    'compression-core-geomean',
                    'ducros_geomean_core',
                    'ducros-geomean-core'):
                geomean_flatten = np.sqrt(
                    np.maximum(pressure_flatten * density_shock_flatten, 0.0))
                shock_core = pressure_sensor * compression_sensor * normality
                shock_core = np.clip(shock_core, 0.0, 1.0)
                velocity_flatten = (
                    ducros_normal_flatten
                    + shock_core * (geomean_flatten - ducros_normal_flatten))
            else:
                velocity_flatten = pressure_flatten
            if self.euler_density_acoustic_flatten:
                rho_avg = 0.5 * (rho_o + rho_n)
                p_avg = 0.5 * (p_o + p_n)
                c_sq = np.maximum(gamma * p_avg / rho_avg, 1.0e-30)
                acoustic_drho = np.abs(p_n - p_o) / c_sq
                density_jump = np.abs(rho_n - rho_o)
                acoustic_fraction = np.minimum(
                    1.0,
                    acoustic_drho / np.maximum(density_jump, 1.0e-30))
                density_flatten = density_shock_flatten * acoustic_fraction
                if (os.environ.get(
                        'TMLPU_EULER_DENSITY_FULL_SHOCK_FLATTEN', '0'
                    ).lower() in ('1', 'true', 'yes', 'on')):
                    density_flatten = shock_base
                contact_fraction = (
                    1.0 - acoustic_fraction) * (1.0 - acoustic_fraction)
                base_contact_weight = contact_fraction * (1.0 - density_flatten)
                density_base_contact_weight = base_contact_weight
                shear_fraction = np.abs(dut) / np.maximum(
                    np.abs(dun) + np.abs(dut), 1.0e-30)
                shear_contact_weight = (
                    shear_fraction
                    * (1.0 - shock_base)
                    * (1.0 - shock_base)
                    * (1.0 - density_flatten))
                if self.euler_density_shear_contact:
                    density_base_contact_weight = np.maximum(
                        density_base_contact_weight, shear_contact_weight)
                density_contact_weight = density_base_contact_weight
                tangential_contact_weight = np.maximum(
                    base_contact_weight, shear_contact_weight)
            if prep_cache is not None:
                prep_cache[sensor_key] = (
                    shock_flatten, pressure_flatten,
                    tangential_shock_flatten, velocity_flatten, density_flatten,
                    density_contact_weight, tangential_contact_weight,
                    face_hancock_courant, one_minus_C_face,
                    density_contact_hancock_scale, pressure_jump, compression,
                    normality)
        if (wall_tangential_flatten_enabled
                and getattr(eq, '__class__', type(eq)).__name__ == 'Euler2D'
                and nvar >= 4 and interior.size):
            wall_cells = getattr(mesh, '_tmlpu_reflective_wall_cells_cache', None)
            if wall_cells is None or wall_cells.shape[0] != N:
                wall_cells = np.zeros(N, dtype=bool)
                face_bc_tag = getattr(mesh, 'face_bc_tag', None)
                bc_patches = tuple(getattr(mesh, 'bc_patches', ()) or ())
                if face_bc_tag is not None and bc_patches:
                    boundary = nei < 0
                    for patch_id, patch_name in enumerate(bc_patches, start=1):
                        name = str(patch_name).lower()
                        if ('wall' not in name and 'reflect' not in name):
                            continue
                        faces = boundary & (face_bc_tag == patch_id)
                        if np.any(faces):
                            wall_cells[owner[faces]] = True
                mesh._tmlpu_reflective_wall_cells_cache = wall_cells
            wall_face = np.logical_or(wall_cells[owner[interior]],
                                      wall_cells[nei[interior]])
            if np.any(wall_face):
                # Uniform slip-wall consistency: only the tangential velocity
                # high-order increment is suppressed on faces touching the
                # first wall-adjacent cell layer.  Later candidates may gate
                # this by local shock sensors to avoid damping shear roll-up.
                wall_weight = wall_face.astype(np.float64)
                wall_mode = str(
                    getattr(self, 'euler_wall_tangential_flatten_mode', 'all')
                    or 'all').strip().lower()
                if wall_mode in ('shock', 'shock_only', 'shock-only'):
                    wall_sensor = np.maximum(pressure_flatten, compression)
                    wall_weight = wall_weight * np.clip(wall_sensor, 0.0, 1.0)
                elif wall_mode in ('pressure', 'pressure_only', 'pressure-only'):
                    wall_weight = wall_weight * np.clip(
                        pressure_flatten, 0.0, 1.0)
                elif wall_mode in (
                        'compression', 'compression_only',
                        'compression-only'):
                    wall_weight = wall_weight * np.clip(compression, 0.0, 1.0)
                wall_strength = float(getattr(
                    self, 'euler_wall_tangential_flatten_strength', 1.0))
                wall_strength = min(1.0, max(0.0, wall_strength))
                wall_weight = wall_strength * wall_weight
                tangential_shock_flatten = np.maximum(
                    tangential_shock_flatten, wall_weight)

        def _ducros_tangential_weight(base_weight, coeffs_local):
            if (not tangential_ducros_gate or nvar < 3
                    or interior.size == 0):
                return base_weight
            grad_u_x = 0.5 * (
                coeffs_local[1, owner[interior], 0]
                + coeffs_local[1, nei[interior], 0])
            grad_u_y = 0.5 * (
                coeffs_local[1, owner[interior], 1]
                + coeffs_local[1, nei[interior], 1])
            grad_v_x = 0.5 * (
                coeffs_local[2, owner[interior], 0]
                + coeffs_local[2, nei[interior], 0])
            grad_v_y = 0.5 * (
                coeffs_local[2, owner[interior], 1]
                + coeffs_local[2, nei[interior], 1])
            div_u = grad_u_x + grad_v_y
            vort_z = grad_v_x - grad_u_y
            vort_sq = vort_z * vort_z
            div_sq = div_u * div_u
            shear_gate = vort_sq / np.maximum(
                vort_sq + div_sq, np.finfo(float).eps)
            return base_weight * shear_gate

        if self.mlp_bound and self.tvb_M > 0.0:
            h_sq = getattr(mesh, '_tvb_h_sq_cache', None)
            if h_sq is None:
                h_sq = float(np.median(mesh.cell_volumes))
                mesh._tvb_h_sq_cache = h_sq
            tvb_eps_fast = self.tvb_M * h_sq
        else:
            tvb_eps_fast = 0.0
        use_extremum_relax_fast = (
            self.extremum_relax or self.extremum_relax_curved_otsu)
        fast_cell_face_loop = (
            _NUMBA_AVAILABLE
            and not os.environ.get('TMLPU_DISABLE_FAST_FACE_RECON')
            and self._tvd_name == 'minmod'
            and (self._velocity_tvd_name is None
                 or self._velocity_tvd_name in (
                     'mc', 'van_leer', 'umist', 'superbee',
                     'bounded_cd', 'central', 'cd', 'pure_downwind',
                     'koren', 'modified_superbee', 'superbee15'))
            and (self._density_tvd_name is None
                 or self._density_tvd_name in (
                     'mc', 'van_leer', 'umist', 'superbee',
                     'bounded_cd', 'central', 'cd', 'pure_downwind',
                     'koren', 'modified_superbee', 'superbee15'))
            and self.mlp_bound
            and self.vertex_mlp
            and self.virtual_uu_gradient
            and str(self.r_form).lower() == 'far_upwind'
            and not self.face_skew_correction
            and not self.cicsam_full
            and not self.weak_face_mlp
            and not use_extremum_relax_fast
            and not self.vertex_mlp_augment
            and not self.physical_vertex_bounds
            and not self.physical_vertex_bounds_value_continuous_otsu
            and lsq_op is not None
            and ctx.get('cell_node_arr') is not None
            and ctx.get('vertex_offsets') is not None
            and ctx.get('v2c_safe') is not None
            and ctx.get('v2c_valid') is not None
        )
        if fast_cell_face_loop:
            euler_log_positive = (
                self.euler_log_positive
                and getattr(eq, '__class__', type(eq)).__name__ == 'Euler2D'
                and nvar >= 4)
            euler_log_pressure_only = (
                self.euler_log_pressure_only
                and not euler_log_positive
                and getattr(eq, '__class__', type(eq)).__name__ == 'Euler2D'
                and nvar >= 4)
            euler_density_entropy_variable = (
                self.euler_density_entropy_variable
                and getattr(eq, '__class__', type(eq)).__name__ == 'Euler2D'
                and nvar >= 4
                and not euler_log_positive
                and not euler_log_pressure_only)
            W_cell_f64 = np.asarray(W_cell, dtype=np.float64)
            W_recon_cell = W_cell_f64
            if (euler_log_positive or euler_log_pressure_only
                    or euler_density_entropy_variable):
                W_recon_cell = W_cell_f64.copy()
            if euler_density_entropy_variable:
                gamma = float(getattr(eq, 'gamma', 1.4))
                W_recon_cell[0] = (
                    np.log(np.maximum(W_cell_f64[0], 1.0e-300))
                    - np.log(np.maximum(W_cell_f64[3], 1.0e-300)) / gamma)
            if euler_log_positive:
                W_recon_cell[0] = np.log(np.maximum(W_cell_f64[0], 1.0e-300))
                W_recon_cell[3] = np.log(np.maximum(W_cell_f64[3], 1.0e-300))
            elif euler_log_pressure_only:
                W_recon_cell[3] = np.log(np.maximum(W_cell_f64[3], 1.0e-300))
            # Opt-in shared cell-coefficient cache (default OFF -> byte-identical
            # for every other scheme). When two reconstructions with identical
            # (stencil, order, idw_p, nbasis, log-flags) run on the SAME W_cell
            # object (e.g. the two BVD branches in one call), the expensive LSQ
            # coefficient kernel is computed once and reused.  Keyed on W_cell
            # identity so it auto-invalidates on the next timestep substep.
            _gc_on = os.environ.get(
                'TMLPU_GRAD_COEFF_CACHE', '0').lower() in (
                    '1', 'true', 'yes', 'on')
            _gc_key = (int(self.order), str(self.stencil), float(self.idw_p),
                       int(nbasis), bool(euler_log_positive),
                       bool(euler_log_pressure_only),
                       bool(euler_density_entropy_variable))
            _gc = (getattr(mesh, '_tmlpu_grad_coeff_cache', None)
                   if _gc_on else None)
            if (_gc is not None and _gc.get('W_cell') is W_cell
                    and _gc.get('key') == _gc_key):
                coeffs = _gc['coeffs']
                phi_min_cell = _gc['phi_min']
                phi_max_cell = _gc['phi_max']
            else:
                coeffs = np.empty((nvar, N, nbasis), dtype=float)
                phi_min_cell = np.empty((nvar, N), dtype=float)
                phi_max_cell = np.empty((nvar, N), dtype=float)
                _tmlpu_cell_bounds_coeffs_kernel(
                    W_recon_cell,
                    np.asarray(nb_safe, dtype=np.int64),
                    np.asarray(valid_nb, dtype=np.bool_),
                    np.asarray(grad_nb_safe, dtype=np.int64),
                    np.asarray(grad_valid_nb, dtype=np.bool_),
                    np.asarray(grad_sqrt_w, dtype=np.float64),
                    np.asarray(grad_lsq_op, dtype=np.float64),
                    phi_min_cell,
                    phi_max_cell,
                    coeffs,
                )
                if _gc_on:
                    mesh._tmlpu_grad_coeff_cache = {
                        'W_cell': W_cell, 'key': _gc_key,
                        'coeffs': coeffs.copy(),
                        'phi_min': phi_min_cell.copy(),
                        'phi_max': phi_max_cell.copy()}
            v2c_safe = ctx['v2c_safe']
            v2c_valid = ctx['v2c_valid']
            vertex_min_values = np.empty((nvar, v2c_safe.shape[0]),
                                         dtype=float)
            vertex_max_values = np.empty((nvar, v2c_safe.shape[0]),
                                         dtype=float)
            _tmlpu_vertex_minmax_kernel(
                W_recon_cell,
                np.asarray(v2c_safe, dtype=np.int64),
                np.asarray(v2c_valid, dtype=np.bool_),
                vertex_min_values,
                vertex_max_values,
            )

            W_L = np.empty((nvar, n_faces), dtype=float)
            W_R = np.empty((nvar, n_faces), dtype=float)
            n_idx_def = np.maximum(nei, 0)
            for v in range(nvar):
                W_L[v] = W_cell[v, owner]
                W_R[v] = np.where(
                    nei >= 0, W_cell[v, n_idx_def], W_cell[v, owner])
            if interior.size == 0:
                return _finish_faces(W_L, W_R)

            o_idx = owner[interior]
            n_idx = nei[interior]
            cell_node_safe_fast = np.where(ctx['cell_node_valid'],
                                           ctx['cell_node_arr'], 0)
            velocity_tvd_mode = (
                1 if self._velocity_tvd_name == 'mc'
                else 2 if self._velocity_tvd_name == 'van_leer'
                else 3 if self._velocity_tvd_name == 'umist'
                else 4 if self._velocity_tvd_name == 'superbee'
                else 5 if self._velocity_tvd_name in (
                    'bounded_cd', 'central', 'cd', 'pure_downwind')
                else 6 if self._velocity_tvd_name == 'koren'
                else 16 if self._velocity_tvd_name in (
                    'modified_superbee', 'superbee15')
                else 0)
            density_tvd_mode = (
                1 if self._density_tvd_name == 'mc'
                else 2 if self._density_tvd_name == 'van_leer'
                else 3 if self._density_tvd_name == 'umist'
                else 4 if self._density_tvd_name == 'superbee'
                else 5 if self._density_tvd_name in (
                    'bounded_cd', 'central', 'cd', 'pure_downwind')
                else 6 if self._density_tvd_name == 'koren'
                else 7 if self._density_tvd_name == 'downwind'
                else 16 if self._density_tvd_name in (
                    'modified_superbee', 'superbee15')
                else 0)
            _tmlpu_minmod_vertex_faces_kernel(
                W_recon_cell,
                coeffs,
                phi_min_cell,
                phi_max_cell,
                np.asarray(o_idx, dtype=np.int64),
                np.asarray(n_idx, dtype=np.int64),
                np.asarray(interior, dtype=np.int64),
                np.asarray(d_o_int, dtype=np.float64),
                np.asarray(alpha_o, dtype=np.float64),
                np.asarray(alpha_n, dtype=np.float64),
                np.asarray(dx_fo, dtype=np.float64),
                np.asarray(dx_fn, dtype=np.float64),
                np.asarray(cell_node_safe_fast, dtype=np.int64),
                np.asarray(ctx['cell_node_valid'], dtype=np.bool_),
                np.asarray(ctx['vertex_offsets'], dtype=np.float64),
                vertex_min_values,
                vertex_max_values,
                np.asarray(shock_flatten, dtype=np.float64),
                np.asarray(pressure_flatten, dtype=np.float64),
                np.asarray(velocity_flatten, dtype=np.float64),
                np.asarray(density_flatten, dtype=np.float64),
                np.asarray(density_contact_weight, dtype=np.float64),
                np.asarray(density_contact_hancock_scale, dtype=np.float64),
                np.asarray(one_minus_C_face, dtype=np.float64),
                float(tvb_eps_fast),
                bool(self.euler_shock_flatten),
                bool(self.euler_density_acoustic_flatten),
                bool(self.euler_density_lsq_increment),
                bool(self.euler_density_no_hancock),
                bool(self.euler_density_entropy_split),
                bool(self.euler_density_contact_wave_hancock),
                bool(self.euler_density_first_order),
                bool(self.euler_pressure_first_order),
                bool(self.euler_velocity_no_hancock),
                bool(self.euler_velocity_shock_flatten),
                bool(self.euler_velocity_lsq_increment),
                int(velocity_tvd_mode),
                int(density_tvd_mode),
                bool(self.euler_density_extrema_lmp),
                bool(self.euler_velocity_extrema_lmp),
                float(getattr(eq, 'gamma', 1.4)),
                W_L,
                W_R,
            )
            if (_NUMBA_AVAILABLE
                    and self.euler_density_pressure_entropy
                    and nvar >= 4
                    and not euler_log_positive
                    and 0 in active_vars):
                _euler_density_pressure_entropy_kernel(
                    W_cell_f64,
                    coeffs,
                    phi_min_cell,
                    phi_max_cell,
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(dx_fo, dtype=np.float64),
                    np.asarray(dx_fn, dtype=np.float64),
                    np.asarray(density_contact_weight, dtype=np.float64),
                    np.asarray(density_flatten, dtype=np.float64),
                    float(getattr(eq, 'gamma', 1.4)),
                    bool(euler_log_pressure_only),
                    bool(os.environ.get(
                        'TMLPU_EULER_DENSITY_PRESSURE_ENTROPY_ACCEPT', ''
                    ).strip().lower() in (
                        'entropy', 'entropy_residual',
                        'entropy-residual', 'residual')),
                    W_L,
                    W_R,
                )
            if (self.euler_density_contact_bvd
                    and nvar >= 4
                    and not euler_log_positive):
                _euler_density_contact_bvd_kernel(
                    W_cell_f64,
                    phi_min_cell,
                    phi_max_cell,
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(density_contact_weight, dtype=np.float64),
                    np.asarray(density_contact_hancock_scale, dtype=np.float64),
                    np.asarray(shock_flatten, dtype=np.float64),
                    float(1.0 - self.hancock_courant),
                    float(self.euler_density_contact_bvd_cap),
                    W_L,
                    W_R,
                )
            if (self.euler_density_contact_cell_bvd
                    and nvar >= 4
                    and not euler_log_positive):
                _euler_density_contact_cell_bvd_kernel(
                    W_cell_f64,
                    phi_min_cell,
                    phi_max_cell,
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(density_contact_weight, dtype=np.float64),
                    np.asarray(density_contact_hancock_scale, dtype=np.float64),
                    float(1.0 - self.hancock_courant),
                    W_L,
                    W_R,
                )
            if (self.euler_density_contact_bvd
                    and nvar >= 4
                    and not euler_log_positive
                    and 0 in active_vars):
                _euler_density_contact_bvd_kernel(
                    np.asarray(W_cell, dtype=np.float64),
                    np.asarray(phi_min_cell, dtype=np.float64),
                    np.asarray(phi_max_cell, dtype=np.float64),
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(density_contact_weight, dtype=np.float64),
                    np.asarray(density_contact_hancock_scale, dtype=np.float64),
                    np.asarray(shock_flatten, dtype=np.float64),
                    float(1.0 - self.hancock_courant),
                    float(self.euler_density_contact_bvd_cap),
                    W_L,
                    W_R,
                )
            if (self.euler_density_contact_cell_bvd
                    and nvar >= 4
                    and not euler_log_positive
                    and 0 in active_vars):
                _euler_density_contact_cell_bvd_kernel(
                    np.asarray(W_cell, dtype=np.float64),
                    np.asarray(phi_min_cell, dtype=np.float64),
                    np.asarray(phi_max_cell, dtype=np.float64),
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(density_contact_weight, dtype=np.float64),
                    np.asarray(density_contact_hancock_scale, dtype=np.float64),
                    float(1.0 - self.hancock_courant),
                    W_L,
                    W_R,
                )
            tangential_tvd_mode = (
                1 if self._tangential_velocity_tvd_name == 'mc'
                else 2 if self._tangential_velocity_tvd_name == 'van_leer'
                else 3 if self._tangential_velocity_tvd_name == 'umist'
                else 17 if self._tangential_velocity_tvd_name == 'koren'
                else 4 if self._tangential_velocity_tvd_name == 'contact_umist'
                else 5 if self._tangential_velocity_tvd_name == 'superbee'
                else 16 if self._tangential_velocity_tvd_name in (
                    'modified_superbee', 'superbee15')
                else 6 if self._tangential_velocity_tvd_name == 'contact_van_leer'
                else 7 if self._tangential_velocity_tvd_name == 'contact_van_leer_linear'
                else 8 if self._tangential_velocity_tvd_name == 'shock_van_leer'
                else 9 if self._tangential_velocity_tvd_name == 'shock_van_leer_strict'
                else 10 if self._tangential_velocity_tvd_name == 'contact_van_leer_root'
                else 11 if self._tangential_velocity_tvd_name == 'shock_van_leer_cubic'
                else 12 if self._tangential_velocity_tvd_name == 'contact_umist_shock'
                else 13 if self._tangential_velocity_tvd_name == 'contact_umist_shock_root'
                else 14 if self._tangential_velocity_tvd_name == 'contact_superbee'
                else 15 if self._tangential_velocity_tvd_name == 'contact_superbee_shock'
                else 18 if self._tangential_velocity_tvd_name == 'superbee_shock_blend'
                else 19 if self._tangential_velocity_tvd_name == 'shear_superbee_blend'
                else 20 if self._tangential_velocity_tvd_name == 'shear_superbee_root_blend'
                else 21 if self._tangential_velocity_tvd_name == 'shear_superbee_root_micro'
                else 22 if self._tangential_velocity_tvd_name == 'shear_superbee_root_mood'
                else 0)
            if tangential_tvd_mode > 0 and nvar >= 4:
                tangential_weight = _ducros_tangential_weight(
                    tangential_contact_weight, coeffs)
                if (self.euler_velocity_no_hancock
                        or self.euler_tangential_velocity_no_hancock):
                    velocity_face_scale = (
                        1.0 - face_hancock_courant * shock_flatten)
                elif self.euler_tangential_contact_wave_hancock:
                    base_scale = one_minus_C_face
                    velocity_face_scale = (
                        base_scale
                        + tangential_weight
                        * (density_contact_hancock_scale - base_scale))
                else:
                    velocity_face_scale = one_minus_C_face
                if self.euler_velocity_shock_flatten:
                    tangential_flatten = tangential_shock_flatten
                    if self.euler_tangential_contact_relax_flatten:
                        tangential_flatten = tangential_flatten * (
                            1.0 - np.clip(tangential_weight, 0.0, 1.0))
                    velocity_face_scale = (
                        velocity_face_scale * (1.0 - tangential_flatten))
                _euler_tangential_velocity_mc_kernel(
                    W_cell_f64,
                    coeffs,
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(d_o_int, dtype=np.float64),
                    np.asarray(dx_fo, dtype=np.float64),
                    np.asarray(dx_fn, dtype=np.float64),
                    np.asarray(alpha_o, dtype=np.float64),
                    np.asarray(alpha_n, dtype=np.float64),
                    np.asarray(face_n_o, dtype=np.float64),
                    np.asarray(velocity_face_scale, dtype=np.float64),
                    np.asarray(tangential_shock_flatten, dtype=np.float64),
                    np.asarray(tangential_weight, dtype=np.float64),
                    int(tangential_tvd_mode),
                    float(self.euler_tangential_shear_micro_blend),
                    float(self.euler_tangential_shear_micro_cap),
                    float(self.euler_tangential_mood_wavespeed_growth_cap),
                    float(self.euler_tangential_mood_jump_growth_cap),
                    bool(self.euler_tangential_velocity_lsq_increment),
                    W_L,
                    W_R,
                )
            if euler_log_positive:
                W_L[0, interior] = np.exp(W_L[0, interior])
                W_R[0, interior] = np.exp(W_R[0, interior])
                W_L[3, interior] = np.exp(W_L[3, interior])
                W_R[3, interior] = np.exp(W_R[3, interior])
            elif euler_log_pressure_only:
                W_L[3, interior] = np.exp(W_L[3, interior])
                W_R[3, interior] = np.exp(W_R[3, interior])
            elif euler_density_entropy_variable:
                gamma = float(getattr(eq, 'gamma', 1.4))
                W_L[0, interior] = np.exp(
                    W_L[0, interior]
                    + np.log(np.maximum(W_L[3, interior], 1.0e-300)) / gamma)
                W_R[0, interior] = np.exp(
                    W_R[0, interior]
                    + np.log(np.maximum(W_R[3, interior], 1.0e-300)) / gamma)
            if self.euler_pressure_contact_entropy_blend:
                _euler_pressure_contact_entropy_blend(W_L)
                _euler_pressure_contact_entropy_blend(W_R)
            if self.euler_pressure_face_jump_limiter_on:
                _euler_pressure_face_jump_limiter(W_L, W_R)
            return _finish_faces(W_L, W_R)

        need_smoothness = (
            self.extremum_relax
            or self.extremum_relax_curved_otsu
            or self.weak_face_mlp_smooth_otsu
            or self.weak_face_mlp_curved_value_otsu
        )
        use_extremum_relax = self.extremum_relax or self.extremum_relax_curved_otsu
        can_reuse_basic_prep = (
            not os.environ.get('TMLPU_DISABLE_SLOW_PREP_CACHE')
            and not need_smoothness
            and not self.weak_face_mlp_high_range_otsu
            and not self.weak_face_mlp_dominant_gap_or_high_range
            and not self.weak_face_mlp_value_otsu
            and not self.weak_face_mlp_value_upper_otsu
            and not self.weak_face_mlp_value_continuous_otsu
            and not self.physical_vertex_bounds_value_continuous_otsu
            and not self.physical_vertex_bounds_value_upper_otsu
        )
        fast_active_prep = (
            _NUMBA_AVAILABLE
            and not all_vars_active
            and not need_smoothness
            and grad_lsq_op is not None
        )
        fast_all_prep = (
            _NUMBA_AVAILABLE
            and all_vars_active
            and not need_smoothness
            and grad_lsq_op is not None
        )
        euler_log_positive = (
            self.euler_log_positive
            and getattr(eq, '__class__', type(eq)).__name__ == 'Euler2D'
            and nvar >= 4)
        euler_log_pressure_only = (
            self.euler_log_pressure_only
            and not euler_log_positive
            and getattr(eq, '__class__', type(eq)).__name__ == 'Euler2D'
            and nvar >= 4)
        euler_density_entropy_variable = (
            self.euler_density_entropy_variable
            and getattr(eq, '__class__', type(eq)).__name__ == 'Euler2D'
            and nvar >= 4
            and not euler_log_positive
            and not euler_log_pressure_only)
        W_recon_cell = W_cell
        recon_space_key = 'primitive'
        if (euler_log_positive or euler_log_pressure_only
                or euler_density_entropy_variable):
            W_recon_cell = np.asarray(W_cell, dtype=float).copy()
            if euler_density_entropy_variable:
                gamma = float(getattr(eq, 'gamma', 1.4))
                W_recon_cell[0] = (
                    np.log(np.maximum(W_cell[0], 1.0e-300))
                    - np.log(np.maximum(W_cell[3], 1.0e-300)) / gamma)
                recon_space_key = 'density_entropy'
            if euler_log_positive:
                W_recon_cell[0] = np.log(np.maximum(W_cell[0], 1.0e-300))
                W_recon_cell[3] = np.log(np.maximum(W_cell[3], 1.0e-300))
                recon_space_key = 'log_positive'
            elif euler_log_pressure_only:
                W_recon_cell[3] = np.log(np.maximum(W_cell[3], 1.0e-300))
                recon_space_key = 'log_pressure'
        basic_prep_key = ('basic_prep', active_vars, recon_space_key)
        cached_basic_prep = (
            prep_cache.get(basic_prep_key)
            if can_reuse_basic_prep and prep_cache is not None else None)

        # 1) phi_min / phi_max per cell over (self ∪ chosen stencil).
        # Avoid np.nanmin/nanmax by replacing invalid neighbour slots with
        # the centre cell's own value (a no-op for min/max).  Much faster
        # than NaN-propagating reductions and gives identical results.
        if cached_basic_prep is not None:
            W_with_self = None
            phi_min_cell, phi_max_cell, coeffs = cached_basic_prep
        elif fast_active_prep:
            W_with_self = None
            phi_min_cell = W_recon_cell.copy()
            phi_max_cell = W_recon_cell.copy()
            coeffs = np.zeros((nvar, N, nbasis), dtype=float)
            _tmlpu_active_cell_bounds_coeffs_kernel(
                np.asarray(W_recon_cell, dtype=np.float64),
                active_vars_arr,
                np.asarray(nb_safe, dtype=np.int64),
                np.asarray(valid_nb, dtype=np.bool_),
                np.asarray(grad_nb_safe, dtype=np.int64),
                np.asarray(grad_valid_nb, dtype=np.bool_),
                np.asarray(grad_sqrt_w, dtype=np.float64),
                np.asarray(grad_lsq_op, dtype=np.float64),
                phi_min_cell,
                phi_max_cell,
                coeffs,
            )
            if can_reuse_basic_prep and prep_cache is not None:
                prep_cache[basic_prep_key] = (
                    phi_min_cell, phi_max_cell, coeffs)
        elif fast_all_prep:
            W_with_self = None
            phi_min_cell = np.empty_like(W_recon_cell)
            phi_max_cell = np.empty_like(W_recon_cell)
            coeffs = np.empty((nvar, N, nbasis), dtype=float)
            _tmlpu_cell_bounds_coeffs_kernel(
                np.asarray(W_recon_cell, dtype=np.float64),
                np.asarray(nb_safe, dtype=np.int64),
                np.asarray(valid_nb, dtype=np.bool_),
                np.asarray(grad_nb_safe, dtype=np.int64),
                np.asarray(grad_valid_nb, dtype=np.bool_),
                np.asarray(grad_sqrt_w, dtype=np.float64),
                np.asarray(grad_lsq_op, dtype=np.float64),
                phi_min_cell,
                phi_max_cell,
                coeffs,
            )
            if can_reuse_basic_prep and prep_cache is not None:
                prep_cache[basic_prep_key] = (
                    phi_min_cell, phi_max_cell, coeffs)
        elif all_vars_active:
            W_self = W_recon_cell[:, :, None]                # (nvar, N, 1)
            W_nb_filled = np.where(
                valid_nb[None, :, :],
                W_recon_cell[:, nb_safe], W_self)            # (nvar, N, max_nb)
            W_with_self = np.concatenate([W_self, W_nb_filled], axis=2)
            phi_min_cell = W_with_self.min(axis=2)
            phi_max_cell = W_with_self.max(axis=2)
        else:
            W_with_self = None
            phi_min_cell = W_cell.copy()
            phi_max_cell = W_cell.copy()
            for v in active_vars:
                W_self_v = W_recon_cell[v, :, None]
                W_nb_v = np.where(valid_nb, W_recon_cell[v, nb_safe], W_self_v)
                W_with_self_v = np.concatenate([W_self_v, W_nb_v], axis=1)
                phi_min_cell[v] = W_with_self_v.min(axis=1)
                phi_max_cell[v] = W_with_self_v.max(axis=1)

        # 2) LSQ polynomial coefficients per cell, per variable.
        #    coeffs[v, c, :] = ATA_inv[c] · (Aᵀ · ΔW)[c]
        # Fused einsum: combine the two separate calls into one
        # (ATA_inv · Aᵀ) · ΔW_w — saves one tensor traversal per variable.
        if cached_basic_prep is None and not fast_active_prep and not fast_all_prep:
            coeffs = np.empty((nvar, N, nbasis), dtype=float)
        is_smooth_cell = None
        is_very_smooth_cell = None
        if need_smoothness:
            is_smooth_cell = np.zeros((nvar, N), dtype=bool)
            is_very_smooth_cell = np.zeros((nvar, N), dtype=bool)
        need_weak_face_smooth = (
            self.weak_face_mlp_smooth_otsu
            or self.weak_face_mlp_range_otsu
            or self.weak_face_mlp_high_range_otsu
            or self.weak_face_mlp_value_otsu
            or self.weak_face_mlp_value_upper_otsu
            or self.weak_face_mlp_curved_value_otsu
            or self.weak_face_mlp_value_continuous_otsu
        )
        weak_face_smooth_cell = (
            np.ones((nvar, N), dtype=bool) if need_weak_face_smooth else None
        )
        physical_vertex_cell = np.full((nvar, N),
                                       bool(self.physical_vertex_bounds),
                                       dtype=bool)
        smoothness_for_weak = (
            np.zeros((nvar, N), dtype=float) if need_smoothness else None
        )
        if cached_basic_prep is None and not fast_active_prep and not fast_all_prep:
            for v in active_vars:
                delta_W = ((W_recon_cell[v, grad_nb_safe]
                            - W_recon_cell[v, :, None])
                           * grad_valid_nb)
                # Weighted RHS: A_basis is already √W·A, multiply ΔW by √W too.
                delta_W_w = delta_W * grad_sqrt_w
                if grad_lsq_op is not None:
                    # coeffs[c, i] = Σ_k (ATA⁻¹Aᵀ)[c, i, k] · ΔW_w[c, k].
                    # The mesh-only LSQ operator is cached once; this avoids
                    # rebuilding the same three-operand contraction every step.
                    coeffs[v] = np.einsum('cik,ck->ci', grad_lsq_op, delta_W_w,
                                          optimize=False)
                else:
                    coeffs[v] = np.einsum('cij,ckj,ck->ci',
                                          ATA_inv, A_basis, delta_W_w,
                                          optimize=True)
                if need_smoothness:
                    # Smoothness indicator: relative LSQ residual norm.  On a
                    # smooth function the k-th-order LSQ polynomial fits
                    # neighbours to O(h^{k+1}); discontinuities give residual
                    # ≈ jump.  We additionally restrict relaxation to cells
                    # that are LOCAL EXTREMA (the only place LMP is binding).
                    # Predicted ΔW (un-weighted): A · p, not √W · A · p.
                    # `A_basis` here is √W·A, so divide by sqrt_w (safe because
                    # sqrt_w > 0 on valid neighbours).
                    delta_W_pred_w = np.einsum(
                        'ckb,cb->ck', grad_A_basis, coeffs[v])
                    delta_W_pred = (
                        delta_W_pred_w / np.maximum(grad_sqrt_w, 1e-30))
                    delta_W_pred = delta_W_pred * grad_valid_nb
                    resid = (delta_W - delta_W_pred) * grad_valid_nb
                    num = np.sqrt(np.sum(resid * resid, axis=1))
                    den = np.sqrt(np.sum(delta_W * delta_W, axis=1))
                    smoothness = num / np.maximum(den, 1e-30)
                    smoothness_for_weak[v] = smoothness
                    if self.extremum_relax:
                        is_smooth_cell[v] = smoothness < self.smoothness_threshold
                        is_very_smooth_cell[v] = (
                            smoothness < self.smoothness_threshold2)
            if can_reuse_basic_prep and prep_cache is not None:
                prep_cache[basic_prep_key] = (phi_min_cell, phi_max_cell, coeffs)
        if self.extremum_relax_curved_otsu:
            for v in range(nvar):
                values = np.sort(smoothness_for_weak[v])
                values = values[np.isfinite(values)]
                if values.size < 2 or values[0] == values[-1]:
                    continue
                prefix = np.cumsum(values)
                total = prefix[-1]
                counts = np.arange(1, values.size)
                left_mean = prefix[:-1] / counts
                right_count = values.size - counts
                right_mean = (total - prefix[:-1]) / right_count
                between = counts * right_count * (left_mean - right_mean) ** 2
                between[values[:-1] == values[1:]] = -1.0
                idx = int(np.argmax(between))
                cutoff = 0.5 * (values[idx] + values[idx + 1])
                is_smooth_cell[v] = smoothness_for_weak[v] >= cutoff
                is_very_smooth_cell[v] = is_smooth_cell[v]
        if self.weak_face_mlp_smooth_otsu:
            for v in range(nvar):
                values = np.sort(smoothness_for_weak[v])
                values = values[np.isfinite(values)]
                if values.size < 2 or values[0] == values[-1]:
                    continue
                prefix = np.cumsum(values)
                total = prefix[-1]
                counts = np.arange(1, values.size)
                left_mean = prefix[:-1] / counts
                right_count = values.size - counts
                right_mean = (total - prefix[:-1]) / right_count
                between = counts * right_count * (left_mean - right_mean) ** 2
                between[values[:-1] == values[1:]] = -1.0
                idx = int(np.argmax(between))
                cutoff = 0.5 * (values[idx] + values[idx + 1])
                weak_face_smooth_cell[v] = smoothness_for_weak[v] <= cutoff
        if self.weak_face_mlp_curved_value_otsu:
            for v in range(nvar):
                values = np.sort(smoothness_for_weak[v])
                values = values[np.isfinite(values)]
                if values.size < 2 or values[0] == values[-1]:
                    weak_face_smooth_cell[v] &= False
                    continue
                prefix = np.cumsum(values)
                total = prefix[-1]
                counts = np.arange(1, values.size)
                left_mean = prefix[:-1] / counts
                right_count = values.size - counts
                right_mean = (total - prefix[:-1]) / right_count
                between = counts * right_count * (left_mean - right_mean) ** 2
                between[values[:-1] == values[1:]] = -1.0
                idx = int(np.argmax(between))
                cutoff = 0.5 * (values[idx] + values[idx + 1])
                weak_face_smooth_cell[v] &= smoothness_for_weak[v] >= cutoff
        if self.weak_face_mlp_range_otsu:
            local_range = phi_max_cell - phi_min_cell
            for v in range(nvar):
                values = np.sort(local_range[v])
                values = values[np.isfinite(values)]
                if values.size < 2 or values[0] == values[-1]:
                    continue
                prefix = np.cumsum(values)
                total = prefix[-1]
                counts = np.arange(1, values.size)
                left_mean = prefix[:-1] / counts
                right_count = values.size - counts
                right_mean = (total - prefix[:-1]) / right_count
                between = counts * right_count * (left_mean - right_mean) ** 2
                between[values[:-1] == values[1:]] = -1.0
                idx = int(np.argmax(between))
                cutoff = 0.5 * (values[idx] + values[idx + 1])
                weak_face_smooth_cell[v] &= local_range[v] <= cutoff
        if self.weak_face_mlp_high_range_otsu:
            local_range = phi_max_cell - phi_min_cell
            for v in range(nvar):
                values = np.sort(local_range[v])
                values = values[np.isfinite(values)]
                if values.size < 2 or values[0] == values[-1]:
                    continue
                prefix = np.cumsum(values)
                total = prefix[-1]
                counts = np.arange(1, values.size)
                left_mean = prefix[:-1] / counts
                right_count = values.size - counts
                right_mean = (total - prefix[:-1]) / right_count
                between = counts * right_count * (left_mean - right_mean) ** 2
                between[values[:-1] == values[1:]] = -1.0
                idx = int(np.argmax(between))
                cutoff = 0.5 * (values[idx] + values[idx + 1])
                high_range = local_range[v] >= cutoff
                if self.weak_face_mlp_dominant_gap_or_high_range:
                    vals = np.sort(W_with_self[v], axis=1)
                    gaps = np.diff(vals, axis=1)
                    imax = np.argmax(gaps, axis=1)
                    max_gap = gaps[np.arange(N), imax]
                    left_hi = vals[np.arange(N), imax]
                    right_lo = vals[np.arange(N), imax + 1]
                    left_range = left_hi - vals[:, 0]
                    right_range = vals[:, -1] - right_lo
                    eps = (64.0 * np.finfo(float).eps
                           * (1.0 + np.abs(vals[:, 0]) + np.abs(vals[:, -1])))
                    dominant_gap = (
                        (local_range[v] > eps)
                        & (max_gap > np.maximum(left_range, right_range) + eps)
                    )
                    high_range |= dominant_gap
                weak_face_smooth_cell[v] &= high_range
        if (self.weak_face_mlp_value_otsu
                or self.weak_face_mlp_value_upper_otsu
                or self.weak_face_mlp_curved_value_otsu
                or self.weak_face_mlp_value_continuous_otsu
                or self.physical_vertex_bounds_value_continuous_otsu
                or self.physical_vertex_bounds_value_upper_otsu):
            local_max = phi_max_cell
            dominant_gap = np.zeros_like(phi_max_cell, dtype=bool)
            if (self.weak_face_mlp_value_continuous_otsu
                    or self.physical_vertex_bounds_value_continuous_otsu):
                local_range = phi_max_cell - phi_min_cell
                vals = np.sort(W_with_self, axis=2)
                gaps = np.diff(vals, axis=2)
                imax = np.argmax(gaps, axis=2)
                max_gap = np.take_along_axis(
                    gaps, imax[:, :, None], axis=2)[:, :, 0]
                left_hi = np.take_along_axis(
                    vals, imax[:, :, None], axis=2)[:, :, 0]
                right_lo = np.take_along_axis(
                    vals, (imax + 1)[:, :, None], axis=2)[:, :, 0]
                left_range = left_hi - vals[:, :, 0]
                right_range = vals[:, :, -1] - right_lo
                eps_gap = (64.0 * np.finfo(float).eps
                           * (1.0 + np.abs(vals[:, :, 0])
                              + np.abs(vals[:, :, -1])))
                dominant_gap = (
                    (local_range > eps_gap)
                    & (max_gap > np.maximum(left_range, right_range) + eps_gap)
                )
            for v in range(nvar):
                finite = np.isfinite(local_max[v])
                scale = (max(float(np.max(np.abs(local_max[v, finite]))), 1.0)
                         if np.any(finite) else 1.0)
                values = np.sort(local_max[v, finite & (
                    local_max[v] > 64.0 * np.finfo(float).eps * scale)])
                if values.size < 2 or values[0] == values[-1]:
                    if self.weak_face_mlp_curved_value_otsu:
                        weak_face_smooth_cell[v] &= False
                    continue
                def _otsu_cut(vals):
                    prefix = np.cumsum(vals)
                    total = prefix[-1]
                    counts = np.arange(1, vals.size)
                    left_mean = prefix[:-1] / counts
                    right_count = vals.size - counts
                    right_mean = (total - prefix[:-1]) / right_count
                    between = counts * right_count * (left_mean - right_mean) ** 2
                    between[vals[:-1] == vals[1:]] = -1.0
                    idx = int(np.argmax(between))
                    return 0.5 * (vals[idx] + vals[idx + 1])
                cutoff = _otsu_cut(values)
                if (self.weak_face_mlp_value_upper_otsu
                        or self.physical_vertex_bounds_value_upper_otsu):
                    tail = values[values >= cutoff]
                    if tail.size >= 2 and tail[0] != tail[-1]:
                        cutoff = _otsu_cut(tail)
                value_ok = local_max[v] <= cutoff
                continuous_value_ok = value_ok & ~dominant_gap[v]
                if self.weak_face_mlp_value_continuous_otsu:
                    weak_face_smooth_cell[v] &= continuous_value_ok
                if self.physical_vertex_bounds_value_continuous_otsu:
                    physical_vertex_cell[v] |= continuous_value_ok
                if self.physical_vertex_bounds_value_upper_otsu:
                    physical_vertex_cell[v] |= value_ok

        # Helper — evaluate the LSQ polynomial at a face displacement vector.
        def _poly_at(coef_per_face, dxs):
            """coef_per_face: (Nf, nbasis), dxs: (Nf, 2). Returns (Nf,)."""
            δx = dxs[:, 0]; δy = dxs[:, 1]
            if nbasis == 2:
                return coef_per_face[:, 0] * δx + coef_per_face[:, 1] * δy
            quad = (coef_per_face[:, 0] * δx +
                    coef_per_face[:, 1] * δy +
                    0.5 * coef_per_face[:, 2] * δx * δx +
                    coef_per_face[:, 3] * δx * δy +
                    0.5 * coef_per_face[:, 4] * δy * δy)
            if nbasis == 5:
                return quad
            # nbasis == 9 (cubic)
            return (quad +
                    coef_per_face[:, 5] * δx * δx * δx / 6.0 +
                    0.5 * coef_per_face[:, 6] * δx * δx * δy +
                    0.5 * coef_per_face[:, 7] * δx * δy * δy +
                    coef_per_face[:, 8] * δy * δy * δy / 6.0)

        # Helper — evaluate the LSQ polynomial of cell C at displacement
        # vectors arranged as (N, V, 2), returning (N, V).
        def _poly_at_cell_offsets(coeffs_v, offsets_NV2):
            """coeffs_v: (N, nbasis), offsets: (N, V, 2). Returns (N, V)."""
            δx = offsets_NV2[:, :, 0]
            δy = offsets_NV2[:, :, 1]
            if nbasis == 2:
                return (coeffs_v[:, None, 0] * δx +
                        coeffs_v[:, None, 1] * δy)
            quad = (coeffs_v[:, None, 0] * δx +
                    coeffs_v[:, None, 1] * δy +
                    0.5 * coeffs_v[:, None, 2] * δx * δx +
                    coeffs_v[:, None, 3] * δx * δy +
                    0.5 * coeffs_v[:, None, 4] * δy * δy)
            if nbasis == 5:
                return quad
            return (quad +
                    coeffs_v[:, None, 5] * δx * δx * δx / 6.0 +
                    0.5 * coeffs_v[:, None, 6] * δx * δx * δy +
                    0.5 * coeffs_v[:, None, 7] * δx * δy * δy +
                    coeffs_v[:, None, 8] * δy * δy * δy / 6.0)

        # TVB tolerance — needed by both vertex-MLP and per-face MLP paths.
        if self.mlp_bound and self.tvb_M > 0.0:
            h_sq = getattr(mesh, '_tvb_h_sq_cache', None)
            if h_sq is None:
                h_sq = float(np.median(mesh.cell_volumes))
                mesh._tvb_h_sq_cache = h_sq
            tvb_eps = self.tvb_M * h_sq
        else:
            tvb_eps = 0.0
        face_gradient_correction_name = str(self.face_gradient_correction).lower()
        face_increment_name = str(self.face_increment).lower()
        r_form_name = str(self.r_form).lower()
        fast_jasak_vertex_face_candidate_pre = (
            _NUMBA_AVAILABLE
            and not os.environ.get('TMLPU_DISABLE_FAST_FACE_RECON')
            and self.mlp_bound
            and self.vertex_mlp
            and self.virtual_uu_gradient
            and self.face_skew_correction
            and face_gradient_correction_name in ('jasak', 'beta')
            and face_increment_name in ('tmlpu', 'lsq')
            and r_form_name == 'far_upwind'
            and not self.cicsam_full
            and not self.weak_face_mlp
            and not use_extremum_relax
            and not self.vertex_mlp_augment
            and not self.physical_vertex_bounds
            and not self.physical_vertex_bounds_value_continuous_otsu
            and not self.phi_LL_unclipped
            and ctx.get('cell_node_arr') is not None
            and ctx.get('vertex_offsets') is not None
            and not self.euler_density_entropy_split
            and not self.euler_density_contact_bvd
            and not self.euler_density_contact_cell_bvd
            and not self.euler_log_positive
        )
        fast_vertex_face_loop_candidate = (
            _NUMBA_AVAILABLE
            and not os.environ.get('TMLPU_DISABLE_FAST_FACE_RECON')
            and self._tvd_name == 'minmod'
            and (self._velocity_tvd_name is None
                 or self._velocity_tvd_name in (
                     'mc', 'van_leer', 'umist', 'superbee',
                     'bounded_cd', 'central', 'cd', 'pure_downwind',
                     'koren', 'modified_superbee', 'superbee15'))
            and (self._density_tvd_name is None
                 or self._density_tvd_name in (
                     'mc', 'van_leer', 'umist', 'superbee',
                     'bounded_cd', 'central', 'cd', 'pure_downwind',
                     'downwind', 'koren', 'modified_superbee', 'superbee15'))
            and self.mlp_bound
            and self.vertex_mlp
            and self.virtual_uu_gradient
            and str(self.r_form).lower() == 'far_upwind'
            and not self.face_skew_correction
            and not self.cicsam_full
            and not self.weak_face_mlp
            and not use_extremum_relax
            and not self.vertex_mlp_augment
            and not self.physical_vertex_bounds
            and not self.physical_vertex_bounds_value_continuous_otsu
            and ctx.get('cell_node_arr') is not None
            and ctx.get('vertex_offsets') is not None
        )
        # Vertex data for the T-MLP-u bound.  The final face limiter below
        # uses the paper formula with Δφ_Vi based on the arithmetic-average
        # gradient, but we also keep the older cell-wise PYG projection
        # available for legacy options.
        psi_vertex_cell = None
        psi_vertex_face_o = None
        psi_vertex_face_n = None
        vertex_min_values = None
        vertex_max_values = None
        vertex_grad_x_values = None
        vertex_grad_y_values = None
        if self.vertex_mlp and ctx['vertex_offsets'] is not None:
            v2c_safe = ctx['v2c_safe']
            v2c_valid = ctx['v2c_valid']
            cell_node_safe = np.where(ctx['cell_node_valid'],
                                      ctx['cell_node_arr'], 0)
            cell_node_valid = ctx['cell_node_valid']
            vertex_offsets = ctx['vertex_offsets']  # (N, V, 2)
            face_node_safe = ctx.get('face_node_int_safe')
            face_node_valid = ctx.get('face_node_int_valid')
            vertex_minmax_key = ('vertex_minmax', active_vars, recon_space_key)
            cached_vertex_minmax = (
                prep_cache.get(vertex_minmax_key)
                if prep_cache is not None else None)

            if (not fast_vertex_face_loop_candidate
                    and not fast_jasak_vertex_face_candidate_pre):
                psi_vertex_cell = np.ones((nvar, N), dtype=float)
            if cached_vertex_minmax is not None:
                vertex_min_values, vertex_max_values = cached_vertex_minmax
            else:
                vertex_min_values = np.empty((nvar, v2c_safe.shape[0]),
                                             dtype=float)
                vertex_max_values = np.empty((nvar, v2c_safe.shape[0]),
                                             dtype=float)
            numba_vertex_minmax_done = False
            if cached_vertex_minmax is None and _NUMBA_AVAILABLE:
                _tmlpu_vertex_minmax_kernel(
                    np.asarray(W_recon_cell, dtype=np.float64),
                    np.asarray(v2c_safe, dtype=np.int64),
                    np.asarray(v2c_valid, dtype=np.bool_),
                    vertex_min_values,
                    vertex_max_values,
                )
                numba_vertex_minmax_done = True
            if self.vertex_mlp_augment:
                vertex_grad_x_values = np.empty((nvar, v2c_safe.shape[0]),
                                                dtype=float)
                vertex_grad_y_values = np.empty((nvar, v2c_safe.shape[0]),
                                                dtype=float)
            if (not fast_vertex_face_loop_candidate
                    and not fast_jasak_vertex_face_candidate_pre
                    and self.vertex_mlp_face_local
                    and face_node_safe is not None
                    and face_node_valid is not None
                    and interior.size > 0):
                psi_vertex_face_o = np.ones((nvar, interior.size), dtype=float)
                psi_vertex_face_n = np.ones((nvar, interior.size), dtype=float)

            if (not fast_vertex_face_loop_candidate
                    and not fast_jasak_vertex_face_candidate_pre):
                def _vertex_psi_from_projection(proj, W_C, phi_min_at_node,
                                                phi_max_at_node, valid_mask,
                                                psi_cap):
                    allowed_max = phi_max_at_node - W_C[..., None] + tvb_eps
                    allowed_min = phi_min_at_node - W_C[..., None] - tvb_eps
                    W_scale = max(float(np.max(np.abs(W_C))), 1e-30)
                    eps = 1e-12 * W_scale
                    psi_each = np.full_like(proj, psi_cap)
                    pos = proj > eps
                    neg = proj < -eps
                    psi_each = np.where(
                        pos,
                        np.minimum(psi_cap, allowed_max / np.maximum(proj, eps)),
                        psi_each)
                    psi_each = np.where(
                        neg,
                        np.minimum(psi_cap, allowed_min / np.minimum(proj, -eps)),
                        psi_each)
                    psi_each = np.where(valid_mask, psi_each, psi_cap)
                    return np.clip(psi_each, 0.0, psi_cap)

            for v in active_vars:
                if cached_vertex_minmax is None and not numba_vertex_minmax_done:
                    W_at_vc = W_recon_cell[v, v2c_safe]       # (Nnodes, max_v2c)
                    # Use np.where + min/max instead of nanmin/max (slow).
                    W_self_v = W_recon_cell[v, v2c_safe[:, :1]]
                    W_at_vc_filled = np.where(v2c_valid, W_at_vc, W_self_v)
                    phi_min_v = W_at_vc_filled.min(axis=1)     # (Nnodes,)
                    phi_max_v = W_at_vc_filled.max(axis=1)
                    vertex_min_values[v] = phi_min_v
                    vertex_max_values[v] = phi_max_v
                else:
                    phi_min_v = vertex_min_values[v]
                    phi_max_v = vertex_max_values[v]
                if (fast_vertex_face_loop_candidate
                        or fast_jasak_vertex_face_candidate_pre):
                    continue
                # ─── Fix 1: linear gradient only (canonical PYG2010) ────
                # Per-cell gradient = (coeffs[0], coeffs[1])
                grad_x = coeffs[v, :, 0]                       # (N,)
                grad_y = coeffs[v, :, 1]
                if self.vertex_mlp_augment:
                    area_at_vc = mesh.cell_volumes[v2c_safe]
                    w_vc = np.where(v2c_valid, area_at_vc, 0.0)
                    w_sum = np.maximum(np.sum(w_vc, axis=1), 1.0e-30)
                    vertex_grad_x_values[v] = (
                        np.sum(w_vc * grad_x[v2c_safe], axis=1) / w_sum)
                    vertex_grad_y_values[v] = (
                        np.sum(w_vc * grad_y[v2c_safe], axis=1) / w_sum)
                proj = (grad_x[:, None] * vertex_offsets[..., 0]
                        + grad_y[:, None] * vertex_offsets[..., 1])
                W_C = W_recon_cell[v]
                phi_min_at_node = phi_min_v[cell_node_safe]    # (N, V)
                phi_max_at_node = phi_max_v[cell_node_safe]
                psi_cap = float(np.clip(self.vertex_mlp_cap, 0.0, 2.0))
                # ─── Fix 2: TVB tolerance M·h² added to bounds ──────────
                psi_v_each = _vertex_psi_from_projection(
                    proj, W_C, phi_min_at_node, phi_max_at_node,
                    cell_node_valid, psi_cap)
                psi_vertex_cell[v] = np.min(psi_v_each, axis=1)
                if psi_vertex_face_o is not None:
                    face_vertex_xy = mesh.nodes[face_node_safe]
                    int_o_idx = owner[interior]
                    int_n_idx = nei[interior]

                    off_o = face_vertex_xy - mesh.cell_centers[int_o_idx, None, :]
                    proj_o = (grad_x[int_o_idx, None] * off_o[..., 0]
                              + grad_y[int_o_idx, None] * off_o[..., 1])
                    phi_min_face = phi_min_v[face_node_safe]
                    phi_max_face = phi_max_v[face_node_safe]
                    psi_o_each = _vertex_psi_from_projection(
                        proj_o, W_recon_cell[v, int_o_idx], phi_min_face,
                        phi_max_face, face_node_valid, psi_cap)
                    psi_vertex_face_o[v] = np.min(psi_o_each, axis=1)

                    off_n = face_vertex_xy - mesh.cell_centers[int_n_idx, None, :]
                    proj_n = (grad_x[int_n_idx, None] * off_n[..., 0]
                              + grad_y[int_n_idx, None] * off_n[..., 1])
                    psi_n_each = _vertex_psi_from_projection(
                        proj_n, W_recon_cell[v, int_n_idx], phi_min_face,
                        phi_max_face, face_node_valid, psi_cap)
                    psi_vertex_face_n[v] = np.min(psi_n_each, axis=1)
            if cached_vertex_minmax is None and prep_cache is not None:
                prep_cache[vertex_minmax_key] = (
                    vertex_min_values, vertex_max_values)

        # 3) Default first-order (overridden for interior below).
        W_L = np.empty((nvar, n_faces), dtype=float)
        W_R = np.empty((nvar, n_faces), dtype=float)
        n_idx_def = np.maximum(nei, 0)
        for v in range(nvar):
            W_L[v] = W_cell[v, owner]
            W_R[v] = np.where(nei >= 0, W_cell[v, n_idx_def], W_cell[v, owner])

        if interior.size == 0:
            return _finish_faces(W_L, W_R)

        # tvb_eps already computed above before vertex-MLP path.

        o_idx = owner[interior]
        n_idx = nei[interior]
        valid_o = UU_o_int >= 0
        valid_n = UU_n_int >= 0
        UU_o_safe = np.where(valid_o, UU_o_int, o_idx)
        UU_n_safe = np.where(valid_n, UU_n_int, n_idx)

        one_minus_C = (1.0 - self.hancock_courant)
        velocity_scale = (
            1.0 - face_hancock_courant * shock_flatten
            if self.euler_velocity_no_hancock
            else one_minus_C_face)
        _EPS = 1e-30

        # Pre-compute side-independent fallback flag once.
        all_valid_when_virt = self.virtual_uu_gradient

        Co_cic = self.cicsam_courant if self.cicsam_full else 0.0
        face_gradient_correction_name = str(self.face_gradient_correction).lower()
        face_increment_name = str(self.face_increment).lower()
        r_form_name = str(self.r_form).lower()
        velocity_pair_active = (1 in active_vars and 2 in active_vars)

        def _apply_tangential_pair_restore(W_L_local, W_R_local):
            if not velocity_pair_active:
                return
            if not bool(self.euler_tangential_pair_restore_on):
                return
            if nvar < 4:
                return
            if (euler_log_positive or euler_log_pressure_only
                    or euler_density_entropy_variable):
                return
            pair_extend_on = bool(self.euler_tangential_pair_extend_on)
            stream_on = bool(
                self.euler_tangential_pair_restore_stream_coherence_on)
            downstream_tan_beta = (
                np.clip(float(
                    self.euler_density_contact_weak_face_downstream_tangential_beta),
                        0.0, 1.0))
            clean_tail_on = bool(self.euler_tangential_clean_contact_tail_on)
            clean_tail_beta = np.clip(
                float(self.euler_tangential_clean_contact_tail_beta), 0.0, 1.0)
            swirl_tail_on = bool(self.euler_tangential_swirl_tail_on)
            swirl_tail_beta = np.clip(
                float(self.euler_tangential_swirl_tail_beta), 0.0, 1.0)
            signed_tail_on = bool(self.euler_tangential_signed_pair_tail_on)
            signed_tail_beta = np.clip(
                float(self.euler_tangential_signed_pair_tail_beta), 0.0, 1.0)
            density_signed_trace_on = bool(
                self.euler_density_signed_tail_trace_on)
            density_signed_trace_beta = np.clip(
                float(self.euler_density_signed_tail_trace_beta), 0.0, 1.0)
            density_curve_tail_on = bool(
                self.euler_tangential_density_curve_pair_tail_on)
            density_curve_tail_beta = np.clip(
                float(self.euler_tangential_density_curve_pair_tail_beta),
                0.0, 1.0)
            if (not bool(self.euler_tangential_pair_restore_on)
                    and not pair_extend_on
                    and not (stream_on and downstream_tan_beta > 0.0)
                    and not (clean_tail_on and clean_tail_beta > 0.0)
                    and not (swirl_tail_on and swirl_tail_beta > 0.0)
                    and not (signed_tail_on and signed_tail_beta > 0.0)
                    and not (
                        density_signed_trace_on
                        and density_signed_trace_beta > 0.0)
                    and not (
                        density_curve_tail_on
                        and density_curve_tail_beta > 0.0)):
                return
            alpha = np.clip(
                float(self.euler_tangential_pair_restore_alpha), 0.0, 1.0)
            if not bool(self.euler_tangential_pair_restore_on):
                alpha = 0.0
            if (alpha <= 0.0 and not pair_extend_on
                    and not (stream_on and downstream_tan_beta > 0.0)
                    and not (clean_tail_on and clean_tail_beta > 0.0)
                    and not (swirl_tail_on and swirl_tail_beta > 0.0)
                    and not (signed_tail_on and signed_tail_beta > 0.0)
                    and not (
                        density_signed_trace_on
                        and density_signed_trace_beta > 0.0)
                    and not (
                        density_curve_tail_on
                        and density_curve_tail_beta > 0.0)):
                return
            pair_cap = max(float(self.euler_tangential_pair_restore_cap), 0.0)
            wave_cap = max(
                float(self.euler_tangential_pair_restore_wave_cap), 0.0)
            if (pair_cap <= 0.0 and wave_cap <= 0.0
                    and not pair_extend_on
                    and not (stream_on and downstream_tan_beta > 0.0)
                    and not (clean_tail_on and clean_tail_beta > 0.0)
                    and not (swirl_tail_on and swirl_tail_beta > 0.0)
                    and not (signed_tail_on and signed_tail_beta > 0.0)
                    and not (
                        density_signed_trace_on
                        and density_signed_trace_beta > 0.0)
                    and not (
                        density_curve_tail_on
                        and density_curve_tail_beta > 0.0)):
                return
            n_face = interior.shape[0]
            if n_face == 0:
                return

            nx = face_n_o[:, 0]
            ny = face_n_o[:, 1]
            tx = -ny
            ty = nx

            def _smoothstep(lo, hi, x):
                width = max(float(hi - lo), _EPS)
                t = np.clip((x - lo) / width, 0.0, 1.0)
                return t * t * (3.0 - 2.0 * t)

            downstream_tan_cap = max(
                float(
                    self.euler_density_contact_weak_face_downstream_tangential_cap),
                0.0)
            downstream_tan_wave_cap = max(
                float(
                    self.euler_density_contact_weak_face_downstream_tangential_wave_cap),
                0.0)
            clean_tail_cap_base = max(
                float(self.euler_tangential_clean_contact_tail_cap), 0.0)
            clean_tail_wave_cap = max(
                float(self.euler_tangential_clean_contact_tail_wave_cap), 0.0)
            swirl_tail_cap_base = max(
                float(self.euler_tangential_swirl_tail_cap), 0.0)
            swirl_tail_wave_cap = max(
                float(self.euler_tangential_swirl_tail_wave_cap), 0.0)
            signed_tail_cap_base = max(
                float(self.euler_tangential_signed_pair_tail_cap), 0.0)
            signed_tail_wave_cap = max(
                float(self.euler_tangential_signed_pair_tail_wave_cap), 0.0)
            density_curve_tail_cap_base = max(
                float(self.euler_tangential_density_curve_pair_tail_cap), 0.0)
            density_curve_tail_wave_cap = max(
                float(
                    self.euler_tangential_density_curve_pair_tail_wave_cap),
                0.0)
            pair_extend_beta = np.clip(
                float(self.euler_tangential_pair_extend_beta), 0.0, 1.0)
            pair_extend_cap = max(
                float(self.euler_tangential_pair_extend_cap), 0.0)
            pair_extend_wave_cap = max(
                float(self.euler_tangential_pair_extend_wave_cap), 0.0)

            rho_o = np.maximum(W_cell[0, o_idx], _EPS)
            rho_n = np.maximum(W_cell[0, n_idx], _EPS)
            rho_avg = 0.5 * (rho_o + rho_n)
            rho_v31 = 0.5 * (
                np.maximum(W_L_local[0, interior], _EPS)
                + np.maximum(W_R_local[0, interior], _EPS))
            rho_off = rho_avg

            rho_o_cell = rho_o
            rho_n_cell = rho_n
            p_o = np.maximum(W_cell[3, o_idx], _EPS)
            p_n = np.maximum(W_cell[3, n_idx], _EPS)
            gamma = float(getattr(eq, 'gamma', 1.4))
            c_o = np.sqrt(np.maximum(gamma * p_o / rho_o_cell, _EPS))
            c_n = np.sqrt(np.maximum(gamma * p_n / rho_n_cell, _EPS))
            c_sum = np.maximum(c_o + c_n, _EPS)

            u_o_cell = W_cell[1, o_idx]
            v_o_cell = W_cell[2, o_idx]
            u_n_cell = W_cell[1, n_idx]
            v_n_cell = W_cell[2, n_idx]
            dux = u_n_cell - u_o_cell
            duy = v_n_cell - v_o_cell
            dun = dux * nx + duy * ny
            dut = dux * tx + duy * ty
            normality_here = np.abs(dun) / np.maximum(
                np.abs(dun) + np.abs(dut), _EPS)
            shear_frac = np.abs(dut) / np.maximum(
                np.abs(dun) + np.abs(dut), _EPS)
            contact_gate = _smoothstep(
                float(self.euler_tangential_pair_gate_contact_min),
                float(self.euler_tangential_pair_gate_contact_full),
                density_contact_weight)
            shear_gate = _smoothstep(
                float(self.euler_tangential_pair_gate_shear_min),
                float(self.euler_tangential_pair_gate_shear_full),
                shear_frac)
            p_gate = _smoothstep(0.025, 0.065, pressure_jump)
            c_gate = _smoothstep(0.006, 0.035, compression)
            n_gate = _smoothstep(0.35, 0.58, normality_here)
            if bool(getattr(
                    self, 'euler_tangential_pair_ignore_normality_gate',
                    False)):
                n_gate = np.zeros_like(n_gate)
            density_support_measure = (
                np.abs(rho_v31 - rho_off) / np.maximum(rho_avg, _EPS))
            density_support = _smoothstep(
                float(self.euler_tangential_pair_gate_density_support_min),
                float(self.euler_tangential_pair_gate_density_support_full),
                density_support_measure)
            tail_density_support_min = float(
                self.euler_tangential_tail_density_support_min)
            tail_density_support_full = float(
                self.euler_tangential_tail_density_support_full)
            if (tail_density_support_min >= 0.0
                    and tail_density_support_full >= tail_density_support_min):
                tail_density_support = _smoothstep(
                    tail_density_support_min, tail_density_support_full,
                    density_support_measure)
            else:
                tail_density_support = density_support
            if bool(self.euler_tangential_tail_density_shock_damp_on):
                tail_damp_theta = np.clip(
                    float(
                        self.euler_tangential_tail_density_shock_damp_theta),
                    0.0, 1.0)
                tail_p_min = float(
                    self
                    .euler_tangential_tail_density_shock_damp_pressure_min)
                tail_c_min = float(
                    self
                    .euler_tangential_tail_density_shock_damp_compression_min)
                tail_n_min = float(
                    self
                    .euler_tangential_tail_density_shock_damp_normality_min)
                tail_p_shock = _smoothstep(
                    tail_p_min, tail_p_min + 0.030, pressure_jump)
                tail_c_shock = _smoothstep(
                    tail_c_min, tail_c_min + 0.020, compression)
                tail_n_shock = _smoothstep(
                    tail_n_min, tail_n_min + 0.18, normality_here)
                tail_shocklike = np.maximum(
                    tail_p_shock * tail_n_shock,
                    tail_c_shock * tail_n_shock)
                tail_damp = np.clip(
                    1.0 - tail_damp_theta * tail_shocklike, 0.0, 1.0)
                tail_density_support = tail_density_support * tail_damp
            signed_tail_density_support_min = float(
                self.euler_tangential_signed_tail_density_support_min)
            signed_tail_density_support_full = float(
                self.euler_tangential_signed_tail_density_support_full)
            if (signed_tail_density_support_min >= 0.0
                    and signed_tail_density_support_full
                    >= signed_tail_density_support_min):
                signed_tail_density_support = _smoothstep(
                    signed_tail_density_support_min,
                    signed_tail_density_support_full,
                    density_support_measure)
            else:
                signed_tail_density_support = tail_density_support
            shock_off = (1.0 - p_gate) * (1.0 - c_gate) * (1.0 - n_gate)
            gate_pair = contact_gate * shear_gate * shock_off * density_support
            shock_exclude_pair = np.zeros_like(gate_pair)
            if bool(self.euler_tangential_downstream_shock_exclude):
                p_min = float(
                    self.euler_tangential_downstream_shock_pressure_min)
                c_min = float(
                    self.euler_tangential_downstream_shock_compression_min)
                n_min = float(
                    self.euler_tangential_downstream_shock_normality_min)
                p_shock = _smoothstep(p_min, p_min + 0.040, pressure_jump)
                c_shock = _smoothstep(c_min, c_min + 0.030, compression)
                n_shock = _smoothstep(n_min, n_min + 0.22, normality_here)
                shock_exclude_pair = np.maximum(
                    p_shock * n_shock, c_shock * n_shock)

            stream = np.zeros_like(gate_pair)
            if stream_on or clean_tail_on:
                u_avg = 0.5 * (u_o_cell + u_n_cell)
                v_avg = 0.5 * (v_o_cell + v_n_cell)
                u_hat = np.abs(u_avg)
                v_hat = np.abs(v_avg)
                vel_mag = np.sqrt(u_avg * u_avg + v_avg * v_avg)
                vel_mag_safe = np.where(vel_mag > _EPS, vel_mag, 1.0)
                e_sx = np.where(
                    vel_mag > _EPS, u_avg / vel_mag_safe, tx)
                e_sy = np.where(
                    vel_mag > _EPS, v_avg / vel_mag_safe, ty)
                grad_rho = 0.5 * (
                    coeffs[0, o_idx, :2] + coeffs[0, n_idx, :2]
                )
                grad_rho_parallel = (
                    grad_rho[:, 0] * e_sx + grad_rho[:, 1] * e_sy)
                grad_rho_norm = (
                    np.abs(grad_rho[:, 0]) + np.abs(grad_rho[:, 1]))
                stream = np.where(
                    grad_rho_norm > _EPS,
                    np.abs(grad_rho_parallel) / np.maximum(
                        grad_rho_norm, _EPS),
                    0.0)
                stream_min = float(
                    self.euler_tangential_pair_restore_stream_coherence_min)
                stream_full = float(
                    self.euler_tangential_pair_restore_stream_coherence_full)
                stream_coherence = _smoothstep(stream_min, stream_full, stream)
            if stream_on:
                gate_pair = gate_pair * stream_coherence
                # preserve the existing pair restore when no downstream coupling.
                # This gate only adds an additional downstream-coherent pass.
                downstream_gate = gate_pair
            else:
                downstream_gate = np.zeros_like(gate_pair)

            ut_pair_cell = 0.5 * (
                (u_o_cell * tx + v_o_cell * ty)
                + (u_n_cell * tx + v_n_cell * ty))
            ut_pair = ut_pair_cell
            uL = W_L_local[1, interior]
            vL = W_L_local[2, interior]
            uR = W_R_local[1, interior]
            vR = W_R_local[2, interior]

            qnL = uL * nx + vL * ny
            qnR = uR * nx + vR * ny
            utL = uL * tx + vL * ty
            utR = uR * tx + vR * ty
            ut_pair_face = 0.5 * (utL + utR)
            legacy_target_on = bool(
                self.euler_tangential_legacy_pair_target_on)
            if legacy_target_on:
                base_blend = np.clip(
                    float(self.euler_tangential_legacy_pair_target_blend),
                    0.0, 1.0)
            else:
                base_blend = 0.0
            signed_blend = float(
                self.euler_tangential_signed_pair_legacy_target_blend)
            if not legacy_target_on:
                signed_blend = 0.0
            elif signed_blend < 0.0:
                signed_blend = base_blend
            signed_blend = np.clip(signed_blend, 0.0, 1.0)
            curve_blend = float(
                self.euler_tangential_density_curve_legacy_target_blend)
            if not legacy_target_on:
                curve_blend = 0.0
            elif curve_blend < 0.0:
                curve_blend = base_blend
            curve_blend = np.clip(curve_blend, 0.0, 1.0)
            ut_signed_target = (
                (1.0 - signed_blend) * ut_pair_cell
                + signed_blend * ut_pair_face)
            ut_curve_target = (
                (1.0 - curve_blend) * ut_pair_cell
                + curve_blend * ut_pair_face)
            if bool(self.euler_tangential_safe_legacy_gate_on):
                safe_p_hi = float(
                    self.euler_tangential_safe_legacy_pressure_hi)
                safe_c_hi = float(
                    self.euler_tangential_safe_legacy_compression_hi)
                safe_n_hi = float(
                    self.euler_tangential_safe_legacy_normality_hi)
                safe_shear_min = float(
                    self.euler_tangential_safe_legacy_shear_min)
                safe_contact_min = float(
                    self.euler_tangential_safe_legacy_contact_min)
                p_safe = 1.0 - _smoothstep(0.0, safe_p_hi, pressure_jump)
                c_safe = 1.0 - _smoothstep(0.0, safe_c_hi, compression)
                n_safe = 1.0 - _smoothstep(0.0, safe_n_hi, normality_here)
                shear_safe = _smoothstep(
                    safe_shear_min, min(0.98, safe_shear_min + 0.10),
                    shear_frac)
                contact_safe = _smoothstep(
                    safe_contact_min, min(0.95, safe_contact_min + 0.20),
                    np.clip(density_contact_weight, 0.0, 1.0))
                safe_legacy_gate_raw = (
                    p_safe * c_safe * n_safe * shear_safe * contact_safe)
                if bool(self.euler_tangential_safe_legacy_coherence_on):
                    n_cells = W_cell.shape[1]
                    cell_safe_sum = (
                        np.bincount(
                            o_idx, weights=safe_legacy_gate_raw,
                            minlength=n_cells)
                        + np.bincount(
                            n_idx, weights=safe_legacy_gate_raw,
                            minlength=n_cells))
                    cell_safe_count = (
                        np.bincount(o_idx, minlength=n_cells)
                        + np.bincount(n_idx, minlength=n_cells))
                    cell_safe_avg = np.divide(
                        cell_safe_sum,
                        np.maximum(cell_safe_count, 1),
                        out=np.zeros_like(cell_safe_sum),
                        where=cell_safe_count > 0)
                    neighbor_safe = 0.5 * (
                        cell_safe_avg[o_idx] + cell_safe_avg[n_idx])
                    coherence = _smoothstep(
                        float(
                            self.euler_tangential_safe_legacy_coherence_floor),
                        float(
                            self.euler_tangential_safe_legacy_coherence_cap),
                        neighbor_safe)
                    coherence_beta = max(
                        float(
                            self.euler_tangential_safe_legacy_coherence_beta),
                        0.0)
                    safe_legacy_gate = np.maximum(
                        safe_legacy_gate_raw,
                        safe_legacy_gate_raw
                        + coherence_beta * coherence
                        * (1.0 - safe_legacy_gate_raw))
                else:
                    safe_legacy_gate = safe_legacy_gate_raw
                if bool(self.euler_tangential_safe_legacy_qcurv_on):
                    grad_u_o = coeffs[1, o_idx, :2]
                    grad_u_n = coeffs[1, n_idx, :2]
                    grad_v_o = coeffs[2, o_idx, :2]
                    grad_v_n = coeffs[2, n_idx, :2]

                    def _qratio_from_grad(grad_u, grad_v):
                        ux = grad_u[:, 0]
                        uy = grad_u[:, 1]
                        vx = grad_v[:, 0]
                        vy = grad_v[:, 1]
                        sxx = ux
                        syy = vy
                        sxy = 0.5 * (uy + vx)
                        omxy = 0.5 * (uy - vx)
                        s2 = sxx * sxx + syy * syy + 2.0 * sxy * sxy
                        o2 = 2.0 * omxy * omxy
                        return np.maximum(0.5 * (o2 - s2), 0.0) / np.maximum(
                            s2 + o2, _EPS)

                    q_face = np.maximum(
                        _qratio_from_grad(grad_u_o, grad_v_o),
                        _qratio_from_grad(grad_u_n, grad_v_n))
                    q_gate = _smoothstep(
                        float(self.euler_tangential_safe_legacy_qcurv_q_min),
                        float(self.euler_tangential_safe_legacy_qcurv_q_full),
                        q_face)
                    gr_o = coeffs[0, o_idx, :2]
                    gr_n = coeffs[0, n_idx, :2]
                    go_norm = np.linalg.norm(gr_o, axis=1)
                    gn_norm = np.linalg.norm(gr_n, axis=1)
                    cos_gr = np.sum(gr_o * gr_n, axis=1) / np.maximum(
                        go_norm * gn_norm, _EPS)
                    curve = np.clip(0.5 * (1.0 - cos_gr), 0.0, 1.0)
                    curve_gate = _smoothstep(
                        float(
                            self.euler_tangential_safe_legacy_qcurv_curve_min),
                        float(
                            self.euler_tangential_safe_legacy_qcurv_curve_full),
                        curve)
                    qcurv = q_gate * curve_gate
                    qcurv_beta = max(
                        float(self.euler_tangential_safe_legacy_qcurv_beta),
                        0.0)
                    safe_legacy_gate = np.minimum(
                        safe_legacy_gate * (1.0 + qcurv_beta * qcurv),
                        1.0)
            else:
                safe_legacy_gate = 1.0

            def _tail_safe_multiplier(
                    raw_gate, apply_relief=True, relief_floor_override=-1.0):
                if bool(self.euler_tangential_tail_safe_floor_on):
                    safe_tail = np.maximum(
                        safe_legacy_gate,
                        np.clip(
                            float(self.euler_tangential_tail_safe_floor),
                            0.0, 1.0))
                else:
                    safe_tail = safe_legacy_gate
                if (apply_relief
                        and bool(
                            self.euler_tangential_tail_shear_contact_relief_on)):
                    relief_floor_value = float(relief_floor_override)
                    if relief_floor_value < 0.0:
                        relief_floor_value = float(
                            self
                            .euler_tangential_tail_shear_contact_relief_floor)
                    relief_floor = np.clip(relief_floor_value, 0.0, 1.0)
                    relief_shear_min = float(
                        self
                        .euler_tangential_tail_shear_contact_shear_min)
                    relief_normality_max = float(
                        self
                        .euler_tangential_tail_shear_contact_normality_max)
                    relief_pressure_max = float(
                        self
                        .euler_tangential_tail_shear_contact_pressure_max)
                    relief_compression_max = float(
                        self
                        .euler_tangential_tail_shear_contact_compression_max)
                    shear_ok = _smoothstep(
                        relief_shear_min,
                        min(0.995, relief_shear_min + 0.04),
                        shear_frac)
                    normal_ok = 1.0 - _smoothstep(
                        relief_normality_max, relief_normality_max + 0.04,
                        normality_here)
                    pressure_ok = 1.0 - _smoothstep(
                        relief_pressure_max, relief_pressure_max + 0.006,
                        pressure_jump)
                    compression_ok = 1.0 - _smoothstep(
                        relief_compression_max,
                        relief_compression_max + 0.002,
                        compression)
                    relief_gate = (
                        shear_ok * normal_ok * pressure_ok * compression_ok)
                    safe_tail = np.where(
                        raw_gate > 0.0,
                        np.maximum(safe_tail, relief_floor * relief_gate),
                        safe_tail)
                return safe_tail

            delta_lim = np.minimum(pair_cap, wave_cap * c_sum)
            if pair_cap <= 0.0:
                delta_lim = wave_cap * c_sum

            d_ut_pair_L = (
                alpha * gate_pair * (ut_pair - utL))
            d_ut_pair_R = (
                alpha * gate_pair * (ut_pair - utR))
            d_ut_pair_L = np.clip(d_ut_pair_L, -delta_lim, delta_lim)
            d_ut_pair_R = np.clip(d_ut_pair_R, -delta_lim, delta_lim)

            utL_new = utL + d_ut_pair_L
            utR_new = utR + d_ut_pair_R

            if (pair_extend_on and pair_extend_beta > 0.0
                    and (pair_extend_cap > 0.0
                         or pair_extend_wave_cap > 0.0)):
                u_avg = 0.5 * (u_o_cell + u_n_cell)
                v_avg = 0.5 * (v_o_cell + v_n_cell)
                vel_mag = np.sqrt(u_avg * u_avg + v_avg * v_avg)
                vel_mag_safe = np.where(vel_mag > _EPS, vel_mag, 1.0)
                flow_x = np.where(vel_mag > _EPS, u_avg / vel_mag_safe, 0.0)
                flow_y = np.where(vel_mag > _EPS, v_avg / vel_mag_safe, 0.0)
                flow_alignment = np.abs(tx * flow_x + ty * flow_y)
                align_gate = _smoothstep(
                    float(self.euler_tangential_pair_extend_alignment_min),
                    float(self.euler_tangential_pair_extend_alignment_full),
                    flow_alignment)
                clean_shear = (
                    _smoothstep(0.68, 0.88, shear_frac)
                    * _smoothstep(
                        0.25, 0.55,
                        np.clip(density_contact_weight, 0.0, 1.0)))
                gate_extend = clean_shear * shock_off * align_gate
                if bool(self.euler_tangential_pair_extend_shock_exclude):
                    gate_extend = gate_extend * (1.0 - shock_exclude_pair)
                extend_cap = np.minimum(
                    pair_extend_cap,
                    pair_extend_wave_cap * c_sum)
                if pair_extend_cap <= 0.0:
                    extend_cap = pair_extend_wave_cap * c_sum
                elif pair_extend_wave_cap <= 0.0:
                    extend_cap = np.full_like(c_sum, pair_extend_cap)
                d_ut_ext_L = (
                    pair_extend_beta * gate_extend * (ut_pair - utL_new))
                d_ut_ext_R = (
                    pair_extend_beta * gate_extend * (ut_pair - utR_new))
                d_ut_ext_L = np.where(
                    extend_cap > _EPS,
                    np.clip(d_ut_ext_L, -extend_cap, extend_cap),
                    0.0)
                d_ut_ext_R = np.where(
                    extend_cap > _EPS,
                    np.clip(d_ut_ext_R, -extend_cap, extend_cap),
                    0.0)
                utL_new = utL_new + d_ut_ext_L
                utR_new = utR_new + d_ut_ext_R

            utL_pre_tail = utL_new.copy()
            utR_pre_tail = utR_new.copy()
            utL_pre_signed = utL_new.copy()
            utR_pre_signed = utR_new.copy()
            signed_pair_tail_gate_raw = np.zeros_like(gate_pair)
            signed_pair_tail_gate = np.zeros_like(gate_pair)
            density_curve_tail_gate_raw = np.zeros_like(gate_pair)
            density_curve_tail_gate = np.zeros_like(gate_pair)
            signed_tail_dut_abs = np.zeros_like(gate_pair)
            signed_tail_dut_clipped_abs = np.zeros_like(gate_pair)
            signed_tail_cap_hit = np.zeros_like(gate_pair, dtype=bool)
            density_curve_tail_dut_abs = np.zeros_like(gate_pair)
            density_curve_tail_dut_clipped_abs = np.zeros_like(gate_pair)
            density_curve_tail_cap_hit = np.zeros_like(gate_pair, dtype=bool)
            omega_o_diag = np.zeros_like(gate_pair)
            omega_n_diag = np.zeros_like(gate_pair)
            qratio_o_diag = np.zeros_like(gate_pair)
            qratio_n_diag = np.zeros_like(gate_pair)
            density_curve_diag = np.zeros_like(gate_pair)

            if signed_tail_on and signed_tail_beta > 0.0:
                grad_u_o = coeffs[1, o_idx, :2]
                grad_u_n = coeffs[1, n_idx, :2]
                grad_v_o = coeffs[2, o_idx, :2]
                grad_v_n = coeffs[2, n_idx, :2]
                omega_o = grad_v_o[:, 0] - grad_u_o[:, 1]
                omega_n = grad_v_n[:, 0] - grad_u_n[:, 1]
                omega_o_diag = omega_o
                omega_n_diag = omega_n
                signed_pair = omega_o * omega_n < 0.0

                def _qratio_from_grad(grad_u, grad_v):
                    ux = grad_u[:, 0]
                    uy = grad_u[:, 1]
                    vx = grad_v[:, 0]
                    vy = grad_v[:, 1]
                    sxx = ux
                    syy = vy
                    sxy = 0.5 * (uy + vx)
                    omxy = 0.5 * (uy - vx)
                    s2 = sxx * sxx + syy * syy + 2.0 * sxy * sxy
                    o2 = 2.0 * omxy * omxy
                    return np.maximum(0.5 * (o2 - s2), 0.0) / np.maximum(
                        s2 + o2, _EPS)

                q_pair = np.minimum(
                    _qratio_from_grad(grad_u_o, grad_v_o),
                    _qratio_from_grad(grad_u_n, grad_v_n))
                qratio_o_diag = _qratio_from_grad(grad_u_o, grad_v_o)
                qratio_n_diag = _qratio_from_grad(grad_u_n, grad_v_n)
                q_gate = _smoothstep(
                    float(self.euler_tangential_signed_pair_tail_q_min),
                    float(self.euler_tangential_signed_pair_tail_q_full),
                    q_pair)
                p_clean = 1.0 - _smoothstep(
                    0.0,
                    float(self.euler_tangential_signed_pair_tail_pressure_hi),
                    pressure_jump)
                c_clean = 1.0 - _smoothstep(
                    0.0,
                    float(self.euler_tangential_signed_pair_tail_compression_hi),
                    compression)
                n_clean = 1.0 - _smoothstep(
                    0.0,
                    float(self.euler_tangential_signed_pair_tail_normality_hi),
                    normality_here)
                pair_core_gate_raw = (
                    signed_pair.astype(float) * q_gate
                    * contact_gate * shear_gate * signed_tail_density_support
                    * p_clean * c_clean * n_clean)
                pair_core_gate = (
                    pair_core_gate_raw
                    * _tail_safe_multiplier(
                        pair_core_gate_raw,
                        bool(
                            self
                            .euler_tangential_tail_shear_contact_relief_apply_signed),
                        float(
                            self
                            .euler_tangential_tail_shear_contact_signed_floor)))
                pair_core_gate = np.where(
                    pair_core_gate_raw > 0.0, pair_core_gate, 0.0)
                pair_core_gate_primary = pair_core_gate.copy()
                if bool(
                        self
                        .euler_tangential_signed_tail_safe_decay_relief_on):
                    signed_decay_active = pair_core_gate > _EPS
                    signed_safe_floor = np.clip(
                        float(self.euler_tangential_signed_tail_safe_floor),
                        0.0, 1.0)
                    pair_core_gate = np.where(
                        signed_decay_active,
                        np.maximum(
                            pair_core_gate,
                            pair_core_gate_raw * signed_safe_floor),
                        pair_core_gate)
                signed_pair_tail_gate_raw = pair_core_gate_raw
                signed_pair_tail_gate = pair_core_gate
                signed_cap = np.maximum(
                    signed_tail_cap_base, signed_tail_wave_cap * c_sum)
                d_ut_signed_L = (
                    signed_tail_beta * pair_core_gate
                    * (ut_signed_target - utL_new))
                d_ut_signed_R = (
                    signed_tail_beta * pair_core_gate
                    * (ut_signed_target - utR_new))
                signed_tail_dut_abs = np.maximum(
                    np.abs(d_ut_signed_L), np.abs(d_ut_signed_R))
                signed_tail_cap_hit = (
                    (signed_cap > _EPS)
                    & ((np.abs(d_ut_signed_L) > signed_cap)
                       | (np.abs(d_ut_signed_R) > signed_cap)))
                d_ut_signed_L = np.where(
                    signed_cap > _EPS,
                    np.clip(d_ut_signed_L, -signed_cap, signed_cap),
                    0.0)
                d_ut_signed_R = np.where(
                    signed_cap > _EPS,
                    np.clip(d_ut_signed_R, -signed_cap, signed_cap),
                    0.0)
                if bool(self.euler_tangential_signed_tail_antisheet_on):
                    antisheet_strength = np.clip(
                        float(
                            self
                            .euler_tangential_signed_tail_antisheet_strength),
                        0.0, 1.0)
                    if antisheet_strength > 0.0:
                        antisheet_min_factor = np.clip(
                            float(
                                self
                                .euler_tangential_signed_tail_antisheet_min_factor),
                            0.0, 1.0)
                        weak_q_gate = 1.0 - _smoothstep(
                            0.0,
                            max(
                                float(
                                    self
                                    .euler_tangential_signed_tail_antisheet_q_hi),
                                _EPS),
                            q_pair)
                        broad_contact_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_signed_tail_antisheet_contact_min),
                            float(
                                self
                                .euler_tangential_signed_tail_antisheet_contact_full),
                            np.clip(density_contact_weight, 0.0, 1.0))
                        sheet_gate = np.clip(
                            pair_core_gate_raw * broad_contact_gate
                            * weak_q_gate,
                            0.0, 1.0)
                        damp = np.maximum(
                            antisheet_min_factor,
                            1.0 - antisheet_strength * sheet_gate)
                        d_ut_signed_L = d_ut_signed_L * damp
                        d_ut_signed_R = d_ut_signed_R * damp
                if bool(self.euler_tangential_signed_tail_sidecar_decay_on):
                    sidecar_safe_floor = np.clip(
                        float(
                            self
                            .euler_tangential_signed_tail_sidecar_safe_floor),
                        0.0, 1.0)
                    sidecar_blend = np.clip(
                        float(
                            self.euler_tangential_signed_tail_sidecar_blend),
                        0.0, 1.0)
                    sidecar_decay_gate = np.where(
                        pair_core_gate_primary > _EPS,
                        np.maximum(
                            pair_core_gate_primary,
                            pair_core_gate_raw * sidecar_safe_floor),
                        pair_core_gate_primary)
                    sidecar_gate = (
                        sidecar_blend
                        * np.maximum(
                            sidecar_decay_gate - pair_core_gate_primary,
                            0.0))
                    utL_sidecar_base = utL_new + d_ut_signed_L
                    utR_sidecar_base = utR_new + d_ut_signed_R
                    d_ut_sidecar_L = (
                        signed_tail_beta * sidecar_gate
                        * (ut_signed_target - utL_sidecar_base))
                    d_ut_sidecar_R = (
                        signed_tail_beta * sidecar_gate
                        * (ut_signed_target - utR_sidecar_base))
                    d_ut_sidecar_L = np.where(
                        signed_cap > _EPS,
                        np.clip(d_ut_sidecar_L, -signed_cap, signed_cap),
                        0.0)
                    d_ut_sidecar_R = np.where(
                        signed_cap > _EPS,
                        np.clip(d_ut_sidecar_R, -signed_cap, signed_cap),
                        0.0)
                    d_ut_signed_L = d_ut_signed_L + d_ut_sidecar_L
                    d_ut_signed_R = d_ut_signed_R + d_ut_sidecar_R
                    d_ut_signed_L = np.where(
                        signed_cap > _EPS,
                        np.clip(d_ut_signed_L, -signed_cap, signed_cap),
                        0.0)
                    d_ut_signed_R = np.where(
                        signed_cap > _EPS,
                        np.clip(d_ut_signed_R, -signed_cap, signed_cap),
                        0.0)
                if bool(self.euler_tangential_signed_tail_shock_ridge_clean_on):
                    ridge_strength = np.clip(
                        float(
                            self
                            .euler_tangential_signed_tail_shock_ridge_strength),
                        0.0, 1.0)
                    if ridge_strength > 0.0:
                        density_ridge_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_signed_tail_shock_ridge_density_min),
                            float(
                                self
                                .euler_tangential_signed_tail_shock_ridge_density_full),
                            np.asarray(signed_tail_density_support,
                                       dtype=float))
                        q_keep_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_signed_tail_shock_ridge_q_keep_min),
                            float(
                                self
                                .euler_tangential_signed_tail_shock_ridge_q_keep_full),
                            q_pair)
                        ridge_gate = np.clip(
                            density_ridge_gate * (1.0 - q_keep_gate),
                            0.0, 1.0)
                        ridge_min = np.clip(
                            float(
                                self
                                .euler_tangential_signed_tail_shock_ridge_min_factor),
                            0.0, 1.0)
                        ridge_damp = np.maximum(
                            ridge_min,
                            1.0 - ridge_strength * ridge_gate)
                        d_ut_signed_L = d_ut_signed_L * ridge_damp
                        d_ut_signed_R = d_ut_signed_R * ridge_damp
                        d_ut_signed_L = np.where(
                            signed_cap > _EPS,
                            np.clip(d_ut_signed_L, -signed_cap, signed_cap),
                            0.0)
                        d_ut_signed_R = np.where(
                            signed_cap > _EPS,
                            np.clip(d_ut_signed_R, -signed_cap, signed_cap),
                            0.0)
                if bool(self.euler_tangential_signed_tail_bridge_cut_on):
                    bridge_strength = np.clip(
                        float(
                            self
                            .euler_tangential_signed_tail_bridge_cut_strength),
                        0.0, 1.0)
                    if bridge_strength > 0.0:
                        omega_pair_abs = np.minimum(
                            np.abs(omega_o), np.abs(omega_n))
                        active_omega = omega_pair_abs[
                            (pair_core_gate_raw > _EPS)
                            & np.isfinite(omega_pair_abs)]
                        if active_omega.size >= 8:
                            lo_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_signed_tail_bridge_cut_omega_lo_pct),
                                0.0, 100.0)
                            hi_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_signed_tail_bridge_cut_omega_hi_pct),
                                lo_pct, 100.0)
                            omega_lo = float(np.percentile(
                                active_omega, lo_pct))
                            omega_hi = float(np.percentile(
                                active_omega, hi_pct))
                            if omega_hi <= omega_lo:
                                omega_hi = omega_lo + _EPS
                            weak_omega_gate = 1.0 - _smoothstep(
                                omega_lo, omega_hi, omega_pair_abs)
                        else:
                            weak_omega_gate = np.zeros_like(pair_core_gate_raw)
                        bridge_q_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_signed_tail_bridge_cut_q_min),
                            float(
                                self
                                .euler_tangential_signed_tail_bridge_cut_q_full),
                            q_pair)
                        bridge_contact_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_signed_tail_bridge_cut_contact_min),
                            float(
                                self
                                .euler_tangential_signed_tail_bridge_cut_contact_full),
                            np.clip(density_contact_weight, 0.0, 1.0))
                        bridge_gate = np.clip(
                            (pair_core_gate_raw > _EPS).astype(float)
                            * bridge_q_gate * bridge_contact_gate
                            * weak_omega_gate * p_clean * c_clean * n_clean,
                            0.0, 1.0)
                        bridge_min = np.clip(
                            float(
                                self
                                .euler_tangential_signed_tail_bridge_cut_min_factor),
                            0.0, 1.0)
                        bridge_damp = np.maximum(
                            bridge_min,
                            1.0 - bridge_strength * bridge_gate)
                        d_ut_signed_L = d_ut_signed_L * bridge_damp
                        d_ut_signed_R = d_ut_signed_R * bridge_damp
                        d_ut_signed_L = np.where(
                            signed_cap > _EPS,
                            np.clip(d_ut_signed_L, -signed_cap, signed_cap),
                            0.0)
                        d_ut_signed_R = np.where(
                            signed_cap > _EPS,
                            np.clip(d_ut_signed_R, -signed_cap, signed_cap),
                            0.0)
                if bool(self.euler_tangential_signed_tail_qbridge_cut_on):
                    qbridge_strength = np.clip(
                        float(
                            self
                            .euler_tangential_signed_tail_qbridge_cut_strength),
                        0.0, 1.0)
                    if qbridge_strength > 0.0:
                        active_q = q_pair[
                            (pair_core_gate_raw > _EPS)
                            & np.isfinite(q_pair)]
                        if active_q.size >= 8:
                            lo_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_signed_tail_qbridge_cut_q_lo_pct),
                                0.0, 100.0)
                            mid_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_signed_tail_qbridge_cut_q_mid_pct),
                                lo_pct, 100.0)
                            core_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_signed_tail_qbridge_cut_q_core_pct),
                                mid_pct, 100.0)
                            top_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_signed_tail_qbridge_cut_q_top_pct),
                                core_pct, 100.0)
                            q_lo = float(np.percentile(active_q, lo_pct))
                            q_mid = float(np.percentile(active_q, mid_pct))
                            q_core = float(np.percentile(active_q, core_pct))
                            q_top = float(np.percentile(active_q, top_pct))
                            if q_mid <= q_lo:
                                q_mid = q_lo + _EPS
                            if q_core <= q_mid:
                                q_core = q_mid + _EPS
                            if q_top <= q_core:
                                q_top = q_core + _EPS
                            mid_q_gate = (
                                _smoothstep(q_lo, q_mid, q_pair)
                                * (1.0 - _smoothstep(
                                    q_core, q_top, q_pair)))
                        else:
                            mid_q_gate = np.zeros_like(pair_core_gate_raw)
                        qbridge_contact_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_signed_tail_qbridge_cut_contact_min),
                            float(
                                self
                                .euler_tangential_signed_tail_qbridge_cut_contact_full),
                            np.clip(density_contact_weight, 0.0, 1.0))
                        qbridge_gate = np.clip(
                            (pair_core_gate_raw > _EPS).astype(float)
                            * mid_q_gate * qbridge_contact_gate
                            * p_clean * c_clean * n_clean,
                            0.0, 1.0)
                        qbridge_min = np.clip(
                            float(
                                self
                                .euler_tangential_signed_tail_qbridge_cut_min_factor),
                            0.0, 1.0)
                        qbridge_damp = np.maximum(
                            qbridge_min,
                            1.0 - qbridge_strength * qbridge_gate)
                        d_ut_signed_L = d_ut_signed_L * qbridge_damp
                        d_ut_signed_R = d_ut_signed_R * qbridge_damp
                        d_ut_signed_L = np.where(
                            signed_cap > _EPS,
                            np.clip(d_ut_signed_L, -signed_cap, signed_cap),
                            0.0)
                        d_ut_signed_R = np.where(
                            signed_cap > _EPS,
                            np.clip(d_ut_signed_R, -signed_cap, signed_cap),
                            0.0)
                if bool(self.euler_tangential_signed_tail_hf_filter_on):
                    hf_strength = np.clip(
                        float(
                            self
                            .euler_tangential_signed_tail_hf_filter_strength),
                        0.0, 1.0)
                    if hf_strength > 0.0:
                        n_cells = W_cell.shape[1]
                        active_weight = np.clip(pair_core_gate, 0.0, 1.0)
                        min_weight = max(
                            float(
                                self
                                .euler_tangential_signed_tail_hf_filter_min_weight),
                            _EPS)
                        cell_weight = (
                            np.bincount(
                                o_idx,
                                weights=active_weight,
                                minlength=n_cells)
                            + np.bincount(
                                n_idx,
                                weights=active_weight,
                                minlength=n_cells))
                        cell_sum = (
                            np.bincount(
                                o_idx,
                                weights=d_ut_signed_L * active_weight,
                                minlength=n_cells)
                            + np.bincount(
                                n_idx,
                                weights=d_ut_signed_R * active_weight,
                                minlength=n_cells))
                        cell_mean = np.divide(
                            cell_sum,
                            np.maximum(cell_weight, min_weight),
                            out=np.zeros_like(cell_sum),
                            where=cell_weight > min_weight)
                        face_mean = 0.5 * (cell_mean[o_idx] + cell_mean[n_idx])
                        support_gate = (
                            (cell_weight[o_idx] > min_weight)
                            & (cell_weight[n_idx] > min_weight)).astype(float)
                        clean_gate = p_clean * c_clean * n_clean
                        if bool(
                                self
                                .euler_tangential_signed_tail_hf_filter_shock_exclude):
                            clean_gate = clean_gate * shock_off
                        hf_gate = np.clip(
                            hf_strength * support_gate * clean_gate,
                            0.0, 1.0)
                        d_ut_signed_L = (
                            (1.0 - hf_gate) * d_ut_signed_L
                            + hf_gate * face_mean)
                        d_ut_signed_R = (
                            (1.0 - hf_gate) * d_ut_signed_R
                            + hf_gate * face_mean)
                        d_ut_signed_L = np.where(
                            signed_cap > _EPS,
                            np.clip(d_ut_signed_L, -signed_cap, signed_cap),
                            0.0)
                        d_ut_signed_R = np.where(
                            signed_cap > _EPS,
                            np.clip(d_ut_signed_R, -signed_cap, signed_cap),
                            0.0)
                signed_tail_dut_clipped_abs = np.maximum(
                    np.abs(d_ut_signed_L), np.abs(d_ut_signed_R))
                if _tmlpu_v146_signed_gate_diag_enabled():
                    _tmlpu_v146_signed_gate_diag_update(
                        np.asarray(pair_core_gate_raw).copy(),
                        np.asarray(pair_core_gate_primary).copy(),
                        np.asarray(pair_core_gate).copy(),
                        np.asarray(signed_tail_dut_abs).copy(),
                        np.asarray(signed_tail_dut_clipped_abs).copy(),
                        np.asarray(safe_legacy_gate).copy(),
                        np.asarray(pressure_jump).copy(),
                        np.asarray(compression).copy(),
                        np.asarray(normality_here).copy(),
                        np.asarray(shear_frac).copy(),
                        np.asarray(density_support_measure).copy())
                utL_new = utL_new + d_ut_signed_L
                utR_new = utR_new + d_ut_signed_R

            if density_curve_tail_on and density_curve_tail_beta > 0.0:
                gr_o = coeffs[0, o_idx, :2]
                gr_n = coeffs[0, n_idx, :2]
                go_norm = np.linalg.norm(gr_o, axis=1)
                gn_norm = np.linalg.norm(gr_n, axis=1)
                cos_gr = np.sum(gr_o * gr_n, axis=1) / np.maximum(
                    go_norm * gn_norm, _EPS)
                curve = np.clip(0.5 * (1.0 - cos_gr), 0.0, 1.0)
                density_curve_diag = curve
                curve_gate = _smoothstep(
                    float(
                        self.euler_tangential_density_curve_pair_tail_curve_min),
                    float(
                        self.euler_tangential_density_curve_pair_tail_curve_full),
                    curve)

                grad_u_o = coeffs[1, o_idx, :2]
                grad_u_n = coeffs[1, n_idx, :2]
                grad_v_o = coeffs[2, o_idx, :2]
                grad_v_n = coeffs[2, n_idx, :2]

                def _qratio_from_grad(grad_u, grad_v):
                    ux = grad_u[:, 0]
                    uy = grad_u[:, 1]
                    vx = grad_v[:, 0]
                    vy = grad_v[:, 1]
                    sxx = ux
                    syy = vy
                    sxy = 0.5 * (uy + vx)
                    omxy = 0.5 * (uy - vx)
                    s2 = sxx * sxx + syy * syy + 2.0 * sxy * sxy
                    o2 = 2.0 * omxy * omxy
                    return np.maximum(0.5 * (o2 - s2), 0.0) / np.maximum(
                        s2 + o2, _EPS)

                q_curve = np.maximum(
                    _qratio_from_grad(grad_u_o, grad_v_o),
                    _qratio_from_grad(grad_u_n, grad_v_n))
                q_gate = _smoothstep(
                    float(self.euler_tangential_density_curve_pair_tail_q_min),
                    float(self.euler_tangential_density_curve_pair_tail_q_full),
                    q_curve)
                p_clean = 1.0 - _smoothstep(
                    0.0,
                    float(
                        self.euler_tangential_density_curve_pair_tail_pressure_hi),
                    pressure_jump)
                c_clean = 1.0 - _smoothstep(
                    0.0,
                    float(
                        self.euler_tangential_density_curve_pair_tail_compression_hi),
                    compression)
                n_clean = 1.0 - _smoothstep(
                    0.0,
                    float(
                        self.euler_tangential_density_curve_pair_tail_normality_hi),
                    normality_here)
                curve_pair_gate_raw = (
                    curve_gate * q_gate * p_clean * c_clean * n_clean
                    * contact_gate * shear_gate * tail_density_support)
                curve_pair_gate = (
                    curve_pair_gate_raw
                    * _tail_safe_multiplier(
                        curve_pair_gate_raw,
                        bool(
                            self
                            .euler_tangential_tail_shear_contact_relief_apply_curve),
                        float(
                            self
                            .euler_tangential_tail_shear_contact_curve_floor)))
                curve_pair_gate = np.where(
                    curve_pair_gate_raw > 0.0, curve_pair_gate, 0.0)
                curve_pair_gate_pre_assist = curve_pair_gate.copy()
                highsafe_curve = np.zeros_like(curve_pair_gate, dtype=bool)
                if bool(
                        self
                        .euler_tangential_highsafe_raw_curve_microassist_on):
                    highsafe_curve = (
                        (curve_pair_gate_raw > _EPS)
                        & (safe_legacy_gate >= float(
                            self
                            .euler_tangential_highsafe_raw_curve_safe_min))
                        & (shear_frac >= float(
                            self
                            .euler_tangential_highsafe_raw_curve_shear_min))
                        & (normality_here <= float(
                            self
                            .euler_tangential_highsafe_raw_curve_normality_max))
                        & (pressure_jump <= float(
                            self
                            .euler_tangential_highsafe_raw_curve_pressure_max))
                        & (compression <= float(
                            self
                            .euler_tangential_highsafe_raw_curve_compression_max)))
                    microassist_floor = np.clip(
                        float(
                            self
                            .euler_tangential_highsafe_raw_curve_microassist_floor),
                        0.0, 1.0)
                    curve_pair_gate = np.where(
                        highsafe_curve,
                        np.maximum(curve_pair_gate_raw, microassist_floor),
                        curve_pair_gate)
                if bool(
                        self
                        .euler_tangential_tail_signed_anchored_curve_assist_on):
                    signed_anchor = signed_pair_tail_gate > _EPS
                    anchored_curve_floor = np.clip(
                        float(
                            self
                            .euler_tangential_tail_signed_anchored_curve_floor),
                        0.0, 1.0)
                    anchored_curve_gate = np.where(
                        signed_anchor,
                        np.maximum(curve_pair_gate, anchored_curve_floor),
                        0.0)
                    if bool(
                            self
                            .euler_tangential_highsafe_raw_curve_microassist_on):
                        curve_pair_gate = np.where(
                            highsafe_curve,
                            np.maximum(curve_pair_gate, anchored_curve_gate),
                            anchored_curve_gate)
                    else:
                        curve_pair_gate = anchored_curve_gate
                density_curve_tail_gate_raw = curve_pair_gate_raw
                density_curve_tail_gate = curve_pair_gate
                curve_cap = np.maximum(
                    density_curve_tail_cap_base,
                    density_curve_tail_wave_cap * c_sum)
                if bool(
                        self
                        .euler_tangential_highsafe_raw_curve_microassist_on):
                    microassist_cap = np.maximum(
                        max(
                            float(
                                self
                                .euler_tangential_highsafe_raw_curve_microassist_cap),
                            0.0),
                        max(
                            float(
                                self
                                .euler_tangential_highsafe_raw_curve_microassist_wave_cap),
                            0.0) * c_sum)
                    curve_cap = np.where(
                        highsafe_curve, microassist_cap, curve_cap)
                d_ut_curve_L = (
                    density_curve_tail_beta * curve_pair_gate
                    * (ut_curve_target - utL_new))
                d_ut_curve_R = (
                    density_curve_tail_beta * curve_pair_gate
                    * (ut_curve_target - utR_new))
                density_curve_tail_dut_abs = np.maximum(
                    np.abs(d_ut_curve_L), np.abs(d_ut_curve_R))
                density_curve_tail_cap_hit = (
                    (curve_cap > _EPS)
                    & ((np.abs(d_ut_curve_L) > curve_cap)
                       | (np.abs(d_ut_curve_R) > curve_cap)))
                d_ut_curve_L = np.where(
                    curve_cap > _EPS,
                    np.clip(d_ut_curve_L, -curve_cap, curve_cap),
                    0.0)
                d_ut_curve_R = np.where(
                    curve_cap > _EPS,
                    np.clip(d_ut_curve_R, -curve_cap, curve_cap),
                    0.0)
                if bool(
                        self
                        .euler_tangential_tail_signed_anchored_curve_align_on):
                    signed_anchor = signed_pair_tail_gate > _EPS
                    align_L = (
                        signed_anchor
                        & ((d_ut_signed_L * d_ut_curve_L) > 0.0))
                    align_R = (
                        signed_anchor
                        & ((d_ut_signed_R * d_ut_curve_R) > 0.0))
                    d_ut_curve_L = np.where(align_L, d_ut_curve_L, 0.0)
                    d_ut_curve_R = np.where(align_R, d_ut_curve_R, 0.0)
                if bool(
                        self
                        .euler_tangential_tail_signed_anchored_curve_preserve_signed_on):
                    signed_anchor = signed_pair_tail_gate > _EPS
                    preserve_L = (
                        signed_anchor
                        & (np.abs(d_ut_signed_L + d_ut_curve_L)
                           >= np.abs(d_ut_signed_L)))
                    preserve_R = (
                        signed_anchor
                        & (np.abs(d_ut_signed_R + d_ut_curve_R)
                           >= np.abs(d_ut_signed_R)))
                    d_ut_curve_L = np.where(preserve_L, d_ut_curve_L, 0.0)
                    d_ut_curve_R = np.where(preserve_R, d_ut_curve_R, 0.0)
                if bool(
                        self
                        .euler_tangential_density_curve_tail_bridge_cut_on):
                    curve_bridge_strength = np.clip(
                        float(
                            self
                            .euler_tangential_density_curve_tail_bridge_cut_strength),
                        0.0, 1.0)
                    if curve_bridge_strength > 0.0:
                        omega_pair_abs = np.minimum(
                            np.abs(omega_o), np.abs(omega_n))
                        active_omega = omega_pair_abs[
                            (curve_pair_gate_raw > _EPS)
                            & np.isfinite(omega_pair_abs)]
                        if active_omega.size >= 8:
                            lo_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_density_curve_tail_bridge_cut_omega_lo_pct),
                                0.0, 100.0)
                            hi_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_density_curve_tail_bridge_cut_omega_hi_pct),
                                lo_pct, 100.0)
                            omega_lo = float(np.percentile(
                                active_omega, lo_pct))
                            omega_hi = float(np.percentile(
                                active_omega, hi_pct))
                            if omega_hi <= omega_lo:
                                omega_hi = omega_lo + _EPS
                            weak_omega_gate = 1.0 - _smoothstep(
                                omega_lo, omega_hi, omega_pair_abs)
                        else:
                            weak_omega_gate = np.zeros_like(curve_pair_gate_raw)
                        bridge_q_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_density_curve_tail_bridge_cut_q_min),
                            float(
                                self
                                .euler_tangential_density_curve_tail_bridge_cut_q_full),
                            q_pair)
                        bridge_contact_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_density_curve_tail_bridge_cut_contact_min),
                            float(
                                self
                                .euler_tangential_density_curve_tail_bridge_cut_contact_full),
                            np.clip(density_contact_weight, 0.0, 1.0))
                        bridge_gate = np.clip(
                            (curve_pair_gate_raw > _EPS).astype(float)
                            * bridge_q_gate * bridge_contact_gate
                            * weak_omega_gate * p_clean * c_clean * n_clean,
                            0.0, 1.0)
                        curve_bridge_min = np.clip(
                            float(
                                self
                                .euler_tangential_density_curve_tail_bridge_cut_min_factor),
                            0.0, 1.0)
                        bridge_damp = np.maximum(
                            curve_bridge_min,
                            1.0 - curve_bridge_strength * bridge_gate)
                        d_ut_curve_L = d_ut_curve_L * bridge_damp
                        d_ut_curve_R = d_ut_curve_R * bridge_damp
                        d_ut_curve_L = np.where(
                            curve_cap > _EPS,
                            np.clip(d_ut_curve_L, -curve_cap, curve_cap),
                            0.0)
                        d_ut_curve_R = np.where(
                            curve_cap > _EPS,
                            np.clip(d_ut_curve_R, -curve_cap, curve_cap),
                            0.0)
                if bool(
                        self
                        .euler_tangential_density_curve_tail_qbridge_cut_on):
                    curve_qbridge_strength = np.clip(
                        float(
                            self
                            .euler_tangential_density_curve_tail_qbridge_cut_strength),
                        0.0, 1.0)
                    if curve_qbridge_strength > 0.0:
                        active_q = q_pair[
                            (curve_pair_gate_raw > _EPS)
                            & np.isfinite(q_pair)]
                        if active_q.size >= 8:
                            lo_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_density_curve_tail_qbridge_cut_q_lo_pct),
                                0.0, 100.0)
                            mid_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_density_curve_tail_qbridge_cut_q_mid_pct),
                                lo_pct, 100.0)
                            core_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_density_curve_tail_qbridge_cut_q_core_pct),
                                mid_pct, 100.0)
                            top_pct = np.clip(
                                float(
                                    self
                                    .euler_tangential_density_curve_tail_qbridge_cut_q_top_pct),
                                core_pct, 100.0)
                            q_lo = float(np.percentile(active_q, lo_pct))
                            q_mid = float(np.percentile(active_q, mid_pct))
                            q_core = float(np.percentile(active_q, core_pct))
                            q_top = float(np.percentile(active_q, top_pct))
                            if q_mid <= q_lo:
                                q_mid = q_lo + _EPS
                            if q_core <= q_mid:
                                q_core = q_mid + _EPS
                            if q_top <= q_core:
                                q_top = q_core + _EPS
                            mid_q_gate = (
                                _smoothstep(q_lo, q_mid, q_pair)
                                * (1.0 - _smoothstep(
                                    q_core, q_top, q_pair)))
                        else:
                            mid_q_gate = np.zeros_like(curve_pair_gate_raw)
                        curve_qbridge_contact_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_density_curve_tail_qbridge_cut_contact_min),
                            float(
                                self
                                .euler_tangential_density_curve_tail_qbridge_cut_contact_full),
                            np.clip(density_contact_weight, 0.0, 1.0))
                        qbridge_gate = np.clip(
                            (curve_pair_gate_raw > _EPS).astype(float)
                            * mid_q_gate * curve_qbridge_contact_gate
                            * p_clean * c_clean * n_clean,
                            0.0, 1.0)
                        curve_qbridge_min = np.clip(
                            float(
                                self
                                .euler_tangential_density_curve_tail_qbridge_cut_min_factor),
                            0.0, 1.0)
                        qbridge_damp = np.maximum(
                            curve_qbridge_min,
                            1.0 - curve_qbridge_strength * qbridge_gate)
                        d_ut_curve_L = d_ut_curve_L * qbridge_damp
                        d_ut_curve_R = d_ut_curve_R * qbridge_damp
                        d_ut_curve_L = np.where(
                            curve_cap > _EPS,
                            np.clip(d_ut_curve_L, -curve_cap, curve_cap),
                            0.0)
                        d_ut_curve_R = np.where(
                            curve_cap > _EPS,
                            np.clip(d_ut_curve_R, -curve_cap, curve_cap),
                            0.0)
                if bool(
                        self
                        .euler_tangential_density_curve_tail_shock_ridge_clean_on):
                    curve_ridge_strength = np.clip(
                        float(
                            self
                            .euler_tangential_density_curve_tail_shock_ridge_strength),
                        0.0, 1.0)
                    if curve_ridge_strength > 0.0:
                        density_ridge_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_density_curve_tail_shock_ridge_density_min),
                            float(
                                self
                                .euler_tangential_density_curve_tail_shock_ridge_density_full),
                            np.asarray(tail_density_support, dtype=float))
                        q_keep_gate = _smoothstep(
                            float(
                                self
                                .euler_tangential_density_curve_tail_shock_ridge_q_keep_min),
                            float(
                                self
                                .euler_tangential_density_curve_tail_shock_ridge_q_keep_full),
                            q_curve)
                        curve_ridge_gate = np.clip(
                            density_ridge_gate * (1.0 - q_keep_gate),
                            0.0, 1.0)
                        curve_ridge_min = np.clip(
                            float(
                                self
                                .euler_tangential_density_curve_tail_shock_ridge_min_factor),
                            0.0, 1.0)
                        curve_ridge_damp = np.maximum(
                            curve_ridge_min,
                            1.0 - curve_ridge_strength * curve_ridge_gate)
                        d_ut_curve_L = d_ut_curve_L * curve_ridge_damp
                        d_ut_curve_R = d_ut_curve_R * curve_ridge_damp
                        d_ut_curve_L = np.where(
                            curve_cap > _EPS,
                            np.clip(d_ut_curve_L, -curve_cap, curve_cap),
                            0.0)
                        d_ut_curve_R = np.where(
                            curve_cap > _EPS,
                            np.clip(d_ut_curve_R, -curve_cap, curve_cap),
                            0.0)
                if bool(self.euler_tangential_density_curve_tail_hf_filter_on):
                    hf_strength = np.clip(
                        float(
                            self
                            .euler_tangential_density_curve_tail_hf_filter_strength),
                        0.0, 1.0)
                    if hf_strength > 0.0:
                        n_cells = W_cell.shape[1]
                        active_weight = np.clip(curve_pair_gate, 0.0, 1.0)
                        min_weight = max(
                            float(
                                self
                                .euler_tangential_density_curve_tail_hf_filter_min_weight),
                            _EPS)
                        cell_weight = (
                            np.bincount(
                                o_idx,
                                weights=active_weight,
                                minlength=n_cells)
                            + np.bincount(
                                n_idx,
                                weights=active_weight,
                                minlength=n_cells))
                        cell_sum = (
                            np.bincount(
                                o_idx,
                                weights=d_ut_curve_L * active_weight,
                                minlength=n_cells)
                            + np.bincount(
                                n_idx,
                                weights=d_ut_curve_R * active_weight,
                                minlength=n_cells))
                        cell_mean = np.divide(
                            cell_sum,
                            np.maximum(cell_weight, min_weight),
                            out=np.zeros_like(cell_sum),
                            where=cell_weight > min_weight)
                        face_mean = 0.5 * (cell_mean[o_idx] + cell_mean[n_idx])
                        support_gate = (
                            (cell_weight[o_idx] > min_weight)
                            & (cell_weight[n_idx] > min_weight)).astype(float)
                        clean_gate = p_clean * c_clean * n_clean
                        if bool(
                                self
                                .euler_tangential_density_curve_tail_hf_filter_shock_exclude):
                            clean_gate = clean_gate * shock_off
                        hf_gate = np.clip(
                            hf_strength * support_gate * clean_gate,
                            0.0, 1.0)
                        d_ut_curve_L = (
                            (1.0 - hf_gate) * d_ut_curve_L
                            + hf_gate * face_mean)
                        d_ut_curve_R = (
                            (1.0 - hf_gate) * d_ut_curve_R
                            + hf_gate * face_mean)
                        d_ut_curve_L = np.where(
                            curve_cap > _EPS,
                            np.clip(d_ut_curve_L, -curve_cap, curve_cap),
                            0.0)
                        d_ut_curve_R = np.where(
                            curve_cap > _EPS,
                            np.clip(d_ut_curve_R, -curve_cap, curve_cap),
                            0.0)
                density_curve_tail_dut_clipped_abs = np.maximum(
                    np.abs(d_ut_curve_L), np.abs(d_ut_curve_R))
                microassist_delta_abs = np.where(
                    highsafe_curve, density_curve_tail_dut_clipped_abs, 0.0)
                if _tmlpu_v140_anchor_curve_diag_enabled():
                    _tmlpu_v140_anchor_curve_diag_update(
                        np.asarray(signed_pair_tail_gate).copy(),
                        np.asarray(curve_pair_gate_raw).copy(),
                        np.asarray(curve_pair_gate_pre_assist).copy(),
                        np.asarray(curve_pair_gate).copy(),
                        np.asarray(density_curve_tail_cap_hit).copy(),
                        np.asarray(density_curve_tail_dut_clipped_abs).copy(),
                        np.asarray(pressure_jump).copy(),
                        np.asarray(compression).copy(),
                        np.asarray(normality_here).copy(),
                        np.asarray(shear_frac).copy(),
                        np.asarray(density_support_measure).copy(),
                        np.asarray(safe_legacy_gate).copy(),
                        np.asarray(highsafe_curve).copy(),
                        np.asarray(microassist_delta_abs).copy())
                utL_new = utL_new + d_ut_curve_L
                utR_new = utR_new + d_ut_curve_R

            if (signed_tail_on and signed_tail_beta > 0.0
                    and bool(self.euler_tangential_total_qbridge_damp_on)):
                total_qbridge_strength = np.clip(
                    float(self.euler_tangential_total_qbridge_damp_strength),
                    0.0, 1.0)
                if total_qbridge_strength > 0.0:
                    total_clean = p_clean * c_clean * n_clean
                    active_seed = (
                        contact_gate * shear_gate * total_clean)
                    active_q = q_pair[
                        (active_seed > _EPS) & np.isfinite(q_pair)]
                    if active_q.size >= 8:
                        lo_pct = np.clip(
                            float(
                                self
                                .euler_tangential_total_qbridge_damp_q_lo_pct),
                            0.0, 100.0)
                        mid_pct = np.clip(
                            float(
                                self
                                .euler_tangential_total_qbridge_damp_q_mid_pct),
                            lo_pct, 100.0)
                        core_pct = np.clip(
                            float(
                                self
                                .euler_tangential_total_qbridge_damp_q_core_pct),
                            mid_pct, 100.0)
                        top_pct = np.clip(
                            float(
                                self
                                .euler_tangential_total_qbridge_damp_q_top_pct),
                            core_pct, 100.0)
                        q_lo = float(np.percentile(active_q, lo_pct))
                        q_mid = float(np.percentile(active_q, mid_pct))
                        q_core = float(np.percentile(active_q, core_pct))
                        q_top = float(np.percentile(active_q, top_pct))
                        if q_mid <= q_lo:
                            q_mid = q_lo + _EPS
                        if q_core <= q_mid:
                            q_core = q_mid + _EPS
                        if q_top <= q_core:
                            q_top = q_core + _EPS
                        mid_q_gate = (
                            _smoothstep(q_lo, q_mid, q_pair)
                            * (1.0 - _smoothstep(q_core, q_top, q_pair)))
                    else:
                        mid_q_gate = np.zeros_like(gate_pair)
                    total_contact_gate = _smoothstep(
                        float(
                            self
                            .euler_tangential_total_qbridge_damp_contact_min),
                        float(
                            self
                            .euler_tangential_total_qbridge_damp_contact_full),
                        np.clip(density_contact_weight, 0.0, 1.0))
                    total_bridge_gate = np.clip(
                        (active_seed > _EPS).astype(float)
                        * mid_q_gate * total_contact_gate * total_clean,
                        0.0, 1.0)
                    total_qbridge_min = np.clip(
                        float(
                            self
                            .euler_tangential_total_qbridge_damp_min_factor),
                        0.0, 1.0)
                    total_damp = np.maximum(
                        total_qbridge_min,
                        1.0 - total_qbridge_strength * total_bridge_gate)
                    utL_new = utL + (utL_new - utL) * total_damp
                    utR_new = utR + (utR_new - utR) * total_damp

            if (signed_tail_on and signed_tail_beta > 0.0
                    and bool(self.euler_tangential_midq_cell_blend_on)):
                cell_blend_strength = np.clip(
                    float(self.euler_tangential_midq_cell_blend_strength),
                    0.0, 1.0)
                if cell_blend_strength > 0.0:
                    cell_clean = p_clean * c_clean * n_clean
                    active_seed = contact_gate * shear_gate * cell_clean
                    active_q = q_pair[
                        (active_seed > _EPS) & np.isfinite(q_pair)]
                    if active_q.size >= 8:
                        lo_pct = np.clip(
                            float(
                                self
                                .euler_tangential_midq_cell_blend_q_lo_pct),
                            0.0, 100.0)
                        mid_pct = np.clip(
                            float(
                                self
                                .euler_tangential_midq_cell_blend_q_mid_pct),
                            lo_pct, 100.0)
                        core_pct = np.clip(
                            float(
                                self
                                .euler_tangential_midq_cell_blend_q_core_pct),
                            mid_pct, 100.0)
                        top_pct = np.clip(
                            float(
                                self
                                .euler_tangential_midq_cell_blend_q_top_pct),
                            core_pct, 100.0)
                        q_lo = float(np.percentile(active_q, lo_pct))
                        q_mid = float(np.percentile(active_q, mid_pct))
                        q_core = float(np.percentile(active_q, core_pct))
                        q_top = float(np.percentile(active_q, top_pct))
                        if q_mid <= q_lo:
                            q_mid = q_lo + _EPS
                        if q_core <= q_mid:
                            q_core = q_mid + _EPS
                        if q_top <= q_core:
                            q_top = q_core + _EPS
                        mid_q_gate = (
                            _smoothstep(q_lo, q_mid, q_pair)
                            * (1.0 - _smoothstep(q_core, q_top, q_pair)))
                    else:
                        mid_q_gate = np.zeros_like(gate_pair)
                    cell_contact_gate = _smoothstep(
                        float(
                            self.euler_tangential_midq_cell_blend_contact_min),
                        float(
                            self.euler_tangential_midq_cell_blend_contact_full),
                        np.clip(density_contact_weight, 0.0, 1.0))
                    cell_blend_gate = np.clip(
                        cell_blend_strength
                        * (active_seed > _EPS).astype(float)
                        * mid_q_gate * cell_contact_gate * cell_clean,
                        0.0, 1.0)
                    ut_cell_o = u_o_cell * tx + v_o_cell * ty
                    ut_cell_n = u_n_cell * tx + v_n_cell * ty
                    utL_new = (
                        (1.0 - cell_blend_gate) * utL_new
                        + cell_blend_gate * ut_cell_o)
                    utR_new = (
                        (1.0 - cell_blend_gate) * utR_new
                        + cell_blend_gate * ut_cell_n)

            if stream_on and downstream_tan_beta > 0.0:
                downstream_gate = downstream_gate * (1.0 - shock_exclude_pair)
                if bool(self.euler_tangential_downstream_branch_damp_on):
                    p0 = float(
                        self.euler_tangential_downstream_branch_pressure_min)
                    c0 = float(
                        self.euler_tangential_downstream_branch_compression_min)
                    n0 = float(
                        self.euler_tangential_downstream_branch_normality_min)
                    branch_floor = np.clip(
                        float(self.euler_tangential_downstream_branch_floor),
                        0.0, 1.0)
                    p_branch = _smoothstep(p0, p0 + 0.035, pressure_jump)
                    c_branch = _smoothstep(c0, c0 + 0.025, compression)
                    n_branch = _smoothstep(n0, n0 + 0.22, normality_here)
                    branch = np.maximum(
                        p_branch * n_branch, c_branch * n_branch)
                    downstream_gate = downstream_gate * (
                        1.0 - (1.0 - branch_floor) * branch)
                d_downstream_cap = max(downstream_tan_cap, 0.0)
                d_downstream_wave = np.maximum(downstream_tan_wave_cap, 0.0) * c_sum
                downstream_cap = np.maximum(d_downstream_cap, d_downstream_wave)
                d_ut_down_L = (
                    downstream_tan_beta * downstream_gate
                    * (ut_pair - utL_new))
                d_ut_down_R = (
                    downstream_tan_beta * downstream_gate
                    * (ut_pair - utR_new))
                d_ut_down_L = np.where(
                    downstream_cap > _EPS,
                    np.clip(d_ut_down_L, -downstream_cap, downstream_cap),
                    0.0)
                d_ut_down_R = np.where(
                    downstream_cap > _EPS,
                    np.clip(d_ut_down_R, -downstream_cap, downstream_cap),
                    0.0)
                utL_new = utL_new + d_ut_down_L
                utR_new = utR_new + d_ut_down_R

            if clean_tail_on and clean_tail_beta > 0.0:
                clean_stream = _smoothstep(
                    0.0,
                    float(self.euler_tangential_clean_contact_tail_stream_full),
                    stream)
                p_eq = 1.0 - _smoothstep(
                    float(self.euler_tangential_clean_contact_tail_pressure_lo),
                    float(self.euler_tangential_clean_contact_tail_pressure_hi),
                    pressure_jump)
                c_eq = 1.0 - _smoothstep(
                    float(self.euler_tangential_clean_contact_tail_compression_lo),
                    float(self.euler_tangential_clean_contact_tail_compression_hi),
                    compression)
                shear_only = 1.0 - _smoothstep(
                    float(self.euler_tangential_clean_contact_tail_normality_lo),
                    float(self.euler_tangential_clean_contact_tail_normality_hi),
                    normality_here)
                clean_gate = (
                    contact_gate * shear_gate * density_support
                    * clean_stream * p_eq * c_eq * shear_only)
                clean_cap = np.maximum(
                    clean_tail_cap_base, clean_tail_wave_cap * c_sum)
                d_ut_clean_L = (
                    clean_tail_beta * clean_gate * (ut_pair - utL_new))
                d_ut_clean_R = (
                    clean_tail_beta * clean_gate * (ut_pair - utR_new))
                d_ut_clean_L = np.where(
                    clean_cap > _EPS,
                    np.clip(d_ut_clean_L, -clean_cap, clean_cap),
                    0.0)
                d_ut_clean_R = np.where(
                    clean_cap > _EPS,
                    np.clip(d_ut_clean_R, -clean_cap, clean_cap),
                    0.0)
                utL_new = utL_new + d_ut_clean_L
                utR_new = utR_new + d_ut_clean_R

            if swirl_tail_on and swirl_tail_beta > 0.0:
                grad_u = 0.5 * (
                    coeffs[1, o_idx, :2] + coeffs[1, n_idx, :2])
                grad_v = 0.5 * (
                    coeffs[2, o_idx, :2] + coeffs[2, n_idx, :2])
                ux = grad_u[:, 0]
                uy = grad_u[:, 1]
                vx = grad_v[:, 0]
                vy = grad_v[:, 1]
                sxx = ux
                syy = vy
                sxy = 0.5 * (uy + vx)
                omxy = 0.5 * (uy - vx)
                s2 = sxx * sxx + syy * syy + 2.0 * sxy * sxy
                o2 = 2.0 * omxy * omxy
                qratio = np.maximum(0.5 * (o2 - s2), 0.0) / np.maximum(
                    s2 + o2, _EPS)
                q_gate = _smoothstep(
                    float(self.euler_tangential_swirl_tail_q_min),
                    float(self.euler_tangential_swirl_tail_q_full),
                    qratio)
                p_clean = 1.0 - _smoothstep(
                    0.0,
                    float(self.euler_tangential_swirl_tail_pressure_hi),
                    pressure_jump)
                c_clean = 1.0 - _smoothstep(
                    0.0,
                    float(self.euler_tangential_swirl_tail_compression_hi),
                    compression)
                n_clean = 1.0 - _smoothstep(
                    0.0,
                    float(self.euler_tangential_swirl_tail_normality_hi),
                    normality_here)
                swirl_gate = (
                    q_gate * p_clean * c_clean * n_clean
                    * contact_gate * shear_gate * density_support)
                swirl_cap = np.maximum(
                    swirl_tail_cap_base, swirl_tail_wave_cap * c_sum)
                d_ut_swirl_L = (
                    swirl_tail_beta * swirl_gate * (ut_pair - utL_new))
                d_ut_swirl_R = (
                    swirl_tail_beta * swirl_gate * (ut_pair - utR_new))
                d_ut_swirl_L = np.where(
                    swirl_cap > _EPS,
                    np.clip(d_ut_swirl_L, -swirl_cap, swirl_cap),
                    0.0)
                d_ut_swirl_R = np.where(
                    swirl_cap > _EPS,
                    np.clip(d_ut_swirl_R, -swirl_cap, swirl_cap),
                    0.0)
                utL_new = utL_new + d_ut_swirl_L
                utR_new = utR_new + d_ut_swirl_R

            if bool(self.euler_tangential_shockline_rollback_on):
                p_min = float(self.euler_tangential_shockline_pressure_min)
                c_min = float(self.euler_tangential_shockline_compression_min)
                n_min = float(self.euler_tangential_shockline_normality_min)
                shear_max = float(self.euler_tangential_shockline_shear_max)
                p_shockline = _smoothstep(
                    p_min, p_min + 0.035, pressure_jump)
                c_shockline = _smoothstep(
                    c_min, c_min + 0.025, compression)
                n_shockline = _smoothstep(
                    n_min, n_min + 0.22, normality_here)
                shear_not_pure = 1.0 - _smoothstep(
                    shear_max, min(0.98, shear_max + 0.08), shear_frac)
                shockline_gate = (
                    p_shockline * c_shockline
                    * n_shockline * shear_not_pure)
                theta = np.clip(
                    float(self.euler_tangential_shockline_rollback_theta),
                    0.0, 1.0)
                rollback = theta * shockline_gate
                utL_new = (1.0 - rollback) * utL_new + rollback * utL_pre_tail
                utR_new = (1.0 - rollback) * utR_new + rollback * utR_pre_tail

            if bool(self.euler_tangential_signed_tail_postrollback_preserve_on):
                signed_preserve_mask = signed_pair_tail_gate > _EPS
                signed_preserve_theta = np.clip(
                    float(
                        self
                        .euler_tangential_signed_tail_postrollback_theta),
                    0.0, 1.0)
                utL_signed_keep = (
                    utL_pre_signed
                    + signed_preserve_theta
                    * (ut_signed_target - utL_pre_signed))
                utR_signed_keep = (
                    utR_pre_signed
                    + signed_preserve_theta
                    * (ut_signed_target - utR_pre_signed))
                utL_new = np.where(
                    signed_preserve_mask, utL_signed_keep, utL_new)
                utR_new = np.where(
                    signed_preserve_mask, utR_signed_keep, utR_new)

            if _tmlpu_v93_diag_enabled():
                centers = mesh.face_centers[interior]
                if centers.shape[1] >= 2:
                    _tmlpu_v93_diag_update(
                        np.asarray(centers[:, 0]).copy(),
                        np.asarray(centers[:, 1]).copy(),
                        np.asarray(gate_pair).copy(),
                        np.asarray(signed_pair_tail_gate).copy(),
                        np.asarray(density_curve_tail_gate).copy(),
                        np.asarray(pressure_jump).copy(),
                        np.asarray(compression).copy(),
                        np.asarray(normality_here).copy(),
                        np.asarray(shear_frac).copy(),
                        np.asarray(omega_o_diag).copy(),
                        np.asarray(omega_n_diag).copy(),
                        np.asarray(qratio_o_diag).copy(),
                        np.asarray(qratio_n_diag).copy(),
                        np.asarray(density_support).copy(),
                        np.asarray(signed_tail_dut_abs).copy(),
                        np.asarray(density_curve_tail_dut_abs).copy())

            if _tmlpu_v115_tail_diag_enabled():
                _tmlpu_v115_tail_diag_update(
                    np.asarray(signed_pair_tail_gate_raw).copy(),
                    np.asarray(signed_pair_tail_gate).copy(),
                    np.asarray(density_curve_tail_gate_raw).copy(),
                    np.asarray(density_curve_tail_gate).copy(),
                    np.asarray(tail_density_support).copy(),
                    np.asarray(safe_legacy_gate).copy(),
                    np.asarray(signed_tail_dut_abs).copy(),
                    np.asarray(signed_tail_dut_clipped_abs).copy(),
                    np.asarray(density_curve_tail_dut_abs).copy(),
                    np.asarray(density_curve_tail_dut_clipped_abs).copy(),
                    np.asarray(signed_tail_cap_hit).copy(),
                    np.asarray(density_curve_tail_cap_hit).copy(),
                    np.asarray(pressure_jump).copy(),
                    np.asarray(compression).copy(),
                    np.asarray(normality_here).copy(),
                    np.asarray(shear_frac).copy(),
                    np.asarray(density_support_measure).copy(),
                    np.asarray(qratio_o_diag).copy(),
                    np.asarray(qratio_n_diag).copy(),
                    np.asarray(density_curve_diag).copy())

            if _tmlpu_v117_feature_diag_enabled():
                _tmlpu_v117_feature_diag_update(
                    np.asarray(signed_pair_tail_gate_raw).copy(),
                    np.asarray(signed_pair_tail_gate).copy(),
                    np.asarray(density_curve_tail_gate_raw).copy(),
                    np.asarray(density_curve_tail_gate).copy(),
                    np.asarray(signed_tail_dut_abs).copy(),
                    np.asarray(signed_tail_dut_clipped_abs).copy(),
                    np.asarray(density_curve_tail_dut_abs).copy(),
                    np.asarray(density_curve_tail_dut_clipped_abs).copy(),
                    np.asarray(pressure_jump).copy(),
                    np.asarray(compression).copy(),
                    np.asarray(normality_here).copy(),
                    np.asarray(shear_frac).copy(),
                    np.asarray(density_support_measure).copy(),
                    np.asarray(safe_legacy_gate).copy(),
                    np.asarray(omega_o_diag).copy(),
                    np.asarray(omega_n_diag).copy())

            if density_signed_trace_on and density_signed_trace_beta > 0.0:
                signed_anchor = signed_pair_tail_gate > _EPS
                rho_trace_gate = np.where(
                    signed_anchor, signed_pair_tail_gate, 0.0)
                rho_scale = np.maximum(1.0, rho_avg)
                density_jump = rho_n_cell - rho_o_cell
                d_rho_L = (
                    -0.5 * density_signed_trace_beta
                    * rho_trace_gate * density_jump)
                d_rho_R = (
                    0.5 * density_signed_trace_beta
                    * rho_trace_gate * density_jump)
                trace_cap = max(
                    float(self.euler_density_signed_tail_trace_cap), 0.0)
                trace_wave_cap = max(
                    float(self.euler_density_signed_tail_trace_wave_cap),
                    0.0)
                delta_cap = np.maximum(
                    trace_cap * rho_scale, trace_wave_cap * c_sum)
                d_rho_L = np.where(
                    delta_cap > _EPS,
                    np.clip(d_rho_L, -delta_cap, delta_cap),
                    0.0)
                d_rho_R = np.where(
                    delta_cap > _EPS,
                    np.clip(d_rho_R, -delta_cap, delta_cap),
                    0.0)
                W_L_local[0, interior] = np.maximum(
                    W_L_local[0, interior] + d_rho_L, _EPS)
                W_R_local[0, interior] = np.maximum(
                    W_R_local[0, interior] + d_rho_R, _EPS)

            W_L_local[1, interior] = qnL * nx + utL_new * tx
            W_L_local[2, interior] = qnL * ny + utL_new * ty
            W_R_local[1, interior] = qnR * nx + utR_new * tx
            W_R_local[2, interior] = qnR * ny + utR_new * ty

        def _apply_contact_characteristic_postpass(W_L_local, W_R_local):
            if not bool(self.euler_contact_characteristic_postpass_on):
                return
            if nvar < 4 or interior.size == 0:
                return
            if getattr(eq, '__class__', type(eq)).__name__ != 'Euler2D':
                return
            if euler_log_positive or euler_log_pressure_only:
                return
            entropy_alpha = np.clip(
                float(self.euler_contact_characteristic_entropy_alpha),
                0.0, 1.0)
            tangential_alpha = np.clip(
                float(self.euler_contact_characteristic_tangential_alpha),
                0.0, 1.0)
            if entropy_alpha <= 0.0 and tangential_alpha <= 0.0:
                return
            s_cap = max(
                float(self.euler_contact_characteristic_entropy_cap), 0.0)
            ut_cap = max(
                float(self.euler_contact_characteristic_tangential_cap), 0.0)
            wave_cap = max(
                float(
                    self.euler_contact_characteristic_tangential_wave_cap),
                0.0)

            nx = face_n_o[:, 0]
            ny = face_n_o[:, 1]
            tx = -ny
            ty = nx
            gamma = float(getattr(eq, 'gamma', 1.4))

            rho_o = np.maximum(W_cell[0, o_idx], _EPS)
            rho_n = np.maximum(W_cell[0, n_idx], _EPS)
            p_o = np.maximum(W_cell[3, o_idx], _EPS)
            p_n = np.maximum(W_cell[3, n_idx], _EPS)
            c_o = np.sqrt(np.maximum(gamma * p_o / rho_o, _EPS))
            c_n = np.sqrt(np.maximum(gamma * p_n / rho_n, _EPS))
            c_sum = np.maximum(c_o + c_n, _EPS)

            def _smoothstep(lo, hi, x):
                width = max(float(hi - lo), _EPS)
                t = np.clip((x - lo) / width, 0.0, 1.0)
                return t * t * (3.0 - 2.0 * t)

            clean_contact = (
                _smoothstep(0.30, 0.60,
                            np.clip(density_contact_weight, 0.0, 1.0))
                * _smoothstep(0.35, 0.65,
                              np.clip(tangential_contact_weight, 0.0, 1.0)))
            shock_off = (
                (1.0 - _smoothstep(0.025, 0.070, pressure_jump))
                * (1.0 - _smoothstep(0.006, 0.040, compression))
                * (1.0 - _smoothstep(0.35, 0.60, normality))
                * (1.0 - np.clip(pressure_flatten, 0.0, 1.0))
                * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)))
            gate = np.clip(clean_contact * shock_off, 0.0, 1.0)
            if not np.any(gate > 0.0):
                return

            grad_s_o = np.empty((interior.shape[0], 2), dtype=float)
            grad_s_n = np.empty((interior.shape[0], 2), dtype=float)
            grad_s_o[:, 0] = (
                coeffs[0, o_idx, 0] / rho_o
                - coeffs[3, o_idx, 0] / (gamma * p_o))
            grad_s_o[:, 1] = (
                coeffs[0, o_idx, 1] / rho_o
                - coeffs[3, o_idx, 1] / (gamma * p_o))
            grad_s_n[:, 0] = (
                coeffs[0, n_idx, 0] / rho_n
                - coeffs[3, n_idx, 0] / (gamma * p_n))
            grad_s_n[:, 1] = (
                coeffs[0, n_idx, 1] / rho_n
                - coeffs[3, n_idx, 1] / (gamma * p_n))
            s_o = np.log(rho_o) - np.log(p_o) / gamma
            s_n = np.log(rho_n) - np.log(p_n) / gamma
            s_high_L = (
                s_o + grad_s_o[:, 0] * dx_fo[:, 0]
                + grad_s_o[:, 1] * dx_fo[:, 1])
            s_high_R = (
                s_n + grad_s_n[:, 0] * dx_fn[:, 0]
                + grad_s_n[:, 1] * dx_fn[:, 1])

            grad_ut_o_x = tx * coeffs[1, o_idx, 0] + ty * coeffs[2, o_idx, 0]
            grad_ut_o_y = tx * coeffs[1, o_idx, 1] + ty * coeffs[2, o_idx, 1]
            grad_ut_n_x = tx * coeffs[1, n_idx, 0] + ty * coeffs[2, n_idx, 0]
            grad_ut_n_y = tx * coeffs[1, n_idx, 1] + ty * coeffs[2, n_idx, 1]
            ut_cell_o = W_cell[1, o_idx] * tx + W_cell[2, o_idx] * ty
            ut_cell_n = W_cell[1, n_idx] * tx + W_cell[2, n_idx] * ty
            ut_high_L = (
                ut_cell_o + grad_ut_o_x * dx_fo[:, 0]
                + grad_ut_o_y * dx_fo[:, 1])
            ut_high_R = (
                ut_cell_n + grad_ut_n_x * dx_fn[:, 0]
                + grad_ut_n_y * dx_fn[:, 1])

            def _apply_side(W_side, s_high, ut_high, rho_cell_side,
                            p_cell_side):
                rho_anchor = np.maximum(W_side[0, interior], _EPS)
                u_anchor = W_side[1, interior]
                v_anchor = W_side[2, interior]
                p_anchor = np.maximum(W_side[3, interior], _EPS)
                s_anchor = (
                    np.log(rho_anchor) - np.log(p_anchor) / gamma)
                un_anchor = u_anchor * nx + v_anchor * ny
                ut_anchor = u_anchor * tx + v_anchor * ty

                delta_s = s_high - s_anchor
                if s_cap > 0.0:
                    delta_s = np.clip(delta_s, -s_cap, s_cap)
                else:
                    delta_s = np.zeros_like(delta_s)

                delta_ut = ut_high - ut_anchor
                if ut_cap > 0.0 and wave_cap > 0.0:
                    local_ut_cap = np.minimum(ut_cap, wave_cap * c_sum)
                elif ut_cap > 0.0:
                    local_ut_cap = np.full_like(c_sum, ut_cap)
                elif wave_cap > 0.0:
                    local_ut_cap = wave_cap * c_sum
                else:
                    local_ut_cap = np.zeros_like(c_sum)
                delta_ut = np.where(
                    local_ut_cap > 0.0,
                    np.clip(delta_ut, -local_ut_cap, local_ut_cap),
                    0.0)

                s_trial = (
                    s_anchor + entropy_alpha * gate * delta_s)
                ut_trial = (
                    ut_anchor + tangential_alpha * gate * delta_ut)
                rho_trial = np.exp(
                    s_trial + np.log(p_anchor) / gamma)
                u_trial = un_anchor * nx + ut_trial * tx
                v_trial = un_anchor * ny + ut_trial * ty

                if bool(self.euler_contact_characteristic_mood_fallback_on):
                    rho_floor = 0.72 * np.minimum(rho_o, rho_n)
                    p_floor = 0.80 * np.minimum(p_o, p_n)
                    rho_bad = rho_trial < rho_floor
                    p_bad = p_anchor < p_floor
                    denom = rho_anchor - rho_trial
                    theta_rho = np.where(
                        (rho_bad & (denom > _EPS)),
                        (rho_anchor - rho_floor) / np.maximum(denom, _EPS),
                        1.0)
                    theta = np.clip(theta_rho, 0.0, 1.0)
                    theta = np.where(p_bad, 0.0, theta)
                    rho_trial = rho_anchor + theta * (rho_trial - rho_anchor)
                    u_trial = u_anchor + theta * (u_trial - u_anchor)
                    v_trial = v_anchor + theta * (v_trial - v_anchor)

                W_side[0, interior] = rho_trial
                W_side[1, interior] = u_trial
                W_side[2, interior] = v_trial
                W_side[3, interior] = p_anchor

            _apply_side(W_L_local, s_high_L, ut_high_L, rho_o, p_o)
            _apply_side(W_R_local, s_high_R, ut_high_R, rho_n, p_n)

        def _apply_patch_contact_shear_postpass(W_L_local, W_R_local):
            if not bool(self.euler_patch_contact_shear_postpass_on):
                return
            if nvar < 4 or interior.size == 0:
                return
            if getattr(eq, '__class__', type(eq)).__name__ != 'Euler2D':
                return
            if euler_log_positive or euler_log_pressure_only:
                return
            entropy_alpha = np.clip(
                float(self.euler_patch_contact_shear_entropy_alpha),
                0.0, 1.0)
            tangential_alpha = np.clip(
                float(self.euler_patch_contact_shear_tangential_alpha),
                0.0, 1.0)
            if entropy_alpha <= 0.0 and tangential_alpha <= 0.0:
                return
            neighbor_blend = np.clip(
                float(self.euler_patch_contact_shear_neighbor_blend),
                0.0, 1.0)
            s_cap = max(
                float(self.euler_patch_contact_shear_entropy_cap), 0.0)
            ut_cap = max(
                float(self.euler_patch_contact_shear_tangential_cap), 0.0)
            wave_cap = max(
                float(self.euler_patch_contact_shear_tangential_wave_cap),
                0.0)
            min_valid = max(
                int(self.euler_patch_contact_shear_min_valid_neighbours), 0)
            roughness_cap = max(
                float(self.euler_patch_contact_shear_roughness_cap), 0.0)
            pair_spacing_on = bool(
                self.euler_patch_contact_shear_pair_spacing_on)
            pair_spacing_beta = np.clip(
                float(self.euler_patch_contact_shear_pair_spacing_beta),
                0.0, 4.0)
            gate_cap = max(
                float(self.euler_patch_contact_shear_gate_cap), 0.0)
            p_floor_factor = max(
                float(self.euler_patch_contact_shear_pressure_floor_factor),
                0.0)
            pressure_margin_on = bool(
                self.euler_patch_contact_shear_pressure_margin_on)
            rho_floor_factor = max(
                float(self.euler_patch_contact_shear_rho_floor_factor), 0.0)
            late_pressure_rollback_on = bool(
                self.euler_patch_contact_shear_late_pressure_rollback_on)
            p_floor_abs = max(
                float(self.euler_patch_contact_shear_p_floor_abs), 0.0)
            rho_floor_abs = max(
                float(self.euler_patch_contact_shear_rho_floor_abs), 0.0)
            tangential_rollback_theta = np.clip(
                float(
                    self.euler_patch_contact_shear_tangential_rollback_theta),
                0.0, 1.0)
            diagnostics_on = os.environ.get(
                'TMLPU_V40_DIAGNOSTICS', '0'
            ).lower() in ('1', 'true', 'yes', 'on')

            nx = face_n_o[:, 0]
            ny = face_n_o[:, 1]
            tx = -ny
            ty = nx
            gamma = float(getattr(eq, 'gamma', 1.4))
            rho_cell = np.maximum(W_cell[0], _EPS)
            p_cell = np.maximum(W_cell[3], _EPS)

            grad_s_all = np.empty((N, 2), dtype=float)
            grad_s_all[:, 0] = (
                coeffs[0, :, 0] / rho_cell
                - coeffs[3, :, 0] / (gamma * p_cell))
            grad_s_all[:, 1] = (
                coeffs[0, :, 1] / rho_cell
                - coeffs[3, :, 1] / (gamma * p_cell))

            def _smoothstep(lo, hi, x):
                width = max(float(hi - lo), _EPS)
                t = np.clip((x - lo) / width, 0.0, 1.0)
                return t * t * (3.0 - 2.0 * t)

            clean_contact = (
                _smoothstep(0.30, 0.60,
                            np.clip(density_contact_weight, 0.0, 1.0))
                * _smoothstep(0.35, 0.65,
                              np.clip(tangential_contact_weight, 0.0, 1.0)))
            shock_off = (
                (1.0 - _smoothstep(0.025, 0.070, pressure_jump))
                * (1.0 - _smoothstep(0.006, 0.040, compression))
                * (1.0 - _smoothstep(0.35, 0.60, normality))
                * (1.0 - np.clip(pressure_flatten, 0.0, 1.0))
                * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)))

            def _record_v40_patch_diagnostics(
                    side_name, gate, patch_coherence, rejected,
                    entropy_increment, tangential_increment):
                if not diagnostics_on:
                    return
                centers = mesh.face_centers[interior]
                if centers.shape[1] < 2:
                    return
                roi = (
                    (centers[:, 0] >= 0.5)
                    & (centers[:, 0] <= 3.0)
                    & (centers[:, 1] >= 0.6)
                    & (centers[:, 1] <= 1.0))
                if not np.any(roi):
                    return
                bins = np.asarray([0.5, 0.9, 1.2, 1.6, 2.1, 3.0],
                                  dtype=float)
                bin_names = [
                    '0.5-0.9', '0.9-1.2', '1.2-1.6',
                    '1.6-2.1', '2.1-3.0']
                stats = getattr(self, '_v40_patch_diag_stats', None)
                if stats is None:
                    stats = {
                        'schema': 'tmlpu_v40_patch_contact_shear_v1',
                        'pid': int(os.getpid()),
                        'calls': 0,
                        'total_samples': 0,
                        'bins': {
                            name: {
                                'samples': 0,
                                'gate_sum': 0.0,
                                'gate_max': 0.0,
                                'gate_gt_0p1': 0,
                                'gate_gt_0p3': 0,
                                'patch_coherence_sum': 0.0,
                                'shock_off_sum': 0.0,
                                'clean_contact_sum': 0.0,
                                'fallback_reject_count': 0,
                                'entropy_increment_abs_sum': 0.0,
                                'tangential_increment_abs_sum': 0.0,
                            }
                            for name in bin_names
                        },
                    }
                    self._v40_patch_diag_stats = stats
                stats['calls'] += 1

                x = centers[:, 0]
                roi_idx = np.where(roi)[0]
                for b in range(len(bin_names)):
                    mask = roi & (x >= bins[b]) & (x < bins[b + 1])
                    if b == len(bin_names) - 1:
                        mask = roi & (x >= bins[b]) & (x <= bins[b + 1])
                    if not np.any(mask):
                        continue
                    row = stats['bins'][bin_names[b]]
                    g = gate[mask]
                    n_samp = int(g.size)
                    row['samples'] += n_samp
                    stats['total_samples'] += n_samp
                    row['gate_sum'] += float(np.sum(g))
                    row['gate_max'] = max(row['gate_max'], float(np.max(g)))
                    row['gate_gt_0p1'] += int(np.count_nonzero(g > 0.1))
                    row['gate_gt_0p3'] += int(np.count_nonzero(g > 0.3))
                    row['patch_coherence_sum'] += float(
                        np.sum(patch_coherence[mask]))
                    row['shock_off_sum'] += float(np.sum(shock_off[mask]))
                    row['clean_contact_sum'] += float(
                        np.sum(clean_contact[mask]))
                    row['fallback_reject_count'] += int(
                        np.count_nonzero(rejected[mask]))
                    row['entropy_increment_abs_sum'] += float(
                        np.sum(np.abs(entropy_increment[mask])))
                    row['tangential_increment_abs_sum'] += float(
                        np.sum(np.abs(tangential_increment[mask])))

                raw = {
                    'side': np.full(roi_idx.shape, side_name),
                    'face_id': interior[roi_idx].copy(),
                    'x': centers[roi_idx, 0].copy(),
                    'y': centers[roi_idx, 1].copy(),
                    'gate': gate[roi_idx].copy(),
                    'patch_coherence': patch_coherence[roi_idx].copy(),
                    'shock_off': shock_off[roi_idx].copy(),
                    'clean_contact': clean_contact[roi_idx].copy(),
                    'rejected': rejected[roi_idx].astype(np.int8),
                    'entropy_increment_abs': np.abs(
                        entropy_increment[roi_idx]),
                    'tangential_increment_abs': np.abs(
                        tangential_increment[roi_idx]),
                }
                prev_raw = getattr(self, '_v40_patch_diag_latest_raw', None)
                if prev_raw is not None:
                    raw = {
                        key: np.concatenate((prev_raw[key], raw[key]))
                        for key in raw
                    }
                self._v40_patch_diag_latest_raw = raw

                out_dir = os.environ.get(
                    'TMLPU_V40_DIAGNOSTICS_DIR',
                    os.path.join('results', 'T-MLP-u', 'v40_diagnostics'))
                os.makedirs(out_dir, exist_ok=True)
                pid = int(os.getpid())
                summary = {
                    'schema': stats['schema'],
                    'pid': pid,
                    'calls': int(stats['calls']),
                    'total_samples': int(stats['total_samples']),
                    'roi': {
                        'x_min': 0.5, 'x_max': 3.0,
                        'y_min': 0.6, 'y_max': 1.0,
                    },
                    'bin_edges': bins.tolist(),
                    'bins': {},
                }
                for name, row in stats['bins'].items():
                    samples = max(int(row['samples']), 1)
                    summary['bins'][name] = {
                        'samples': int(row['samples']),
                        'gate_mean': row['gate_sum'] / samples,
                        'gate_max': row['gate_max'],
                        'gate_gt_0p1_count': int(row['gate_gt_0p1']),
                        'gate_gt_0p3_count': int(row['gate_gt_0p3']),
                        'patch_coherence_mean': (
                            row['patch_coherence_sum'] / samples),
                        'shock_off_mean': row['shock_off_sum'] / samples,
                        'clean_contact_mean': (
                            row['clean_contact_sum'] / samples),
                        'fallback_reject_count': int(
                            row['fallback_reject_count']),
                        'fallback_reject_rate': (
                            row['fallback_reject_count'] / samples),
                        'entropy_increment_abs_mean': (
                            row['entropy_increment_abs_sum'] / samples),
                        'tangential_increment_abs_mean': (
                            row['tangential_increment_abs_sum'] / samples),
                    }
                json_path = os.path.join(
                    out_dir, f'v40_patch_diag_pid{pid}.json')
                with open(json_path, 'w', encoding='utf-8') as fh:
                    json.dump(summary, fh, indent=2, sort_keys=True)
                    fh.write('\n')
                npz_path = os.path.join(
                    out_dir, f'v40_patch_diag_latest_pid{pid}.npz')
                np.savez(npz_path, **raw)

            rho_o = np.maximum(W_cell[0, o_idx], _EPS)
            rho_n = np.maximum(W_cell[0, n_idx], _EPS)
            p_o = np.maximum(W_cell[3, o_idx], _EPS)
            p_n = np.maximum(W_cell[3, n_idx], _EPS)
            c_o = np.sqrt(np.maximum(gamma * p_o / rho_o, _EPS))
            c_n = np.sqrt(np.maximum(gamma * p_n / rho_n, _EPS))
            c_sum = np.maximum(c_o + c_n, _EPS)
            rho_floor = np.maximum(
                rho_floor_abs,
                rho_floor_factor * np.minimum(rho_o, rho_n))
            p_floor = np.maximum(
                p_floor_abs,
                p_floor_factor * np.minimum(p_o, p_n))
            rho_lo = np.minimum(rho_o, rho_n)
            rho_hi = np.maximum(rho_o, rho_n)
            rho_avg = np.maximum(0.5 * (rho_o + rho_n), _EPS)

            use_numba_patch_postpass = (
                _NUMBA_AVAILABLE
                and not diagnostics_on
                and os.environ.get(
                    'TMLPU_V40_POSTPASS_PYTHON', '0'
                ).lower() not in ('1', 'true', 'yes', 'on'))
            if use_numba_patch_postpass:
                _patch_contact_shear_side_postpass_kernel(
                    W_cell,
                    coeffs,
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(dx_fo, dtype=np.float64),
                    np.asarray(grad_nb_safe, dtype=np.int64),
                    np.asarray(grad_valid_nb, dtype=np.bool_),
                    np.asarray(face_n_o, dtype=np.float64),
                    np.asarray(clean_contact, dtype=np.float64),
                    np.asarray(shock_off, dtype=np.float64),
                    np.asarray(rho_floor, dtype=np.float64),
                    np.asarray(p_floor, dtype=np.float64),
                    np.asarray(rho_lo, dtype=np.float64),
                    np.asarray(rho_hi, dtype=np.float64),
                    np.asarray(rho_avg, dtype=np.float64),
                    np.asarray(c_sum, dtype=np.float64),
                    W_L_local,
                    gamma,
                    entropy_alpha,
                    tangential_alpha,
                    neighbor_blend,
                    s_cap,
                    ut_cap,
                    wave_cap,
                    min_valid,
                    roughness_cap,
                    bool(pair_spacing_on),
                    pair_spacing_beta,
                    gate_cap,
                    bool(pressure_margin_on),
                    bool(late_pressure_rollback_on),
                    tangential_rollback_theta,
                )
                _patch_contact_shear_side_postpass_kernel(
                    W_cell,
                    coeffs,
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(dx_fn, dtype=np.float64),
                    np.asarray(grad_nb_safe, dtype=np.int64),
                    np.asarray(grad_valid_nb, dtype=np.bool_),
                    np.asarray(face_n_o, dtype=np.float64),
                    np.asarray(clean_contact, dtype=np.float64),
                    np.asarray(shock_off, dtype=np.float64),
                    np.asarray(rho_floor, dtype=np.float64),
                    np.asarray(p_floor, dtype=np.float64),
                    np.asarray(rho_lo, dtype=np.float64),
                    np.asarray(rho_hi, dtype=np.float64),
                    np.asarray(rho_avg, dtype=np.float64),
                    np.asarray(c_sum, dtype=np.float64),
                    W_R_local,
                    gamma,
                    entropy_alpha,
                    tangential_alpha,
                    neighbor_blend,
                    s_cap,
                    ut_cap,
                    wave_cap,
                    min_valid,
                    roughness_cap,
                    bool(pair_spacing_on),
                    pair_spacing_beta,
                    gate_cap,
                    bool(pressure_margin_on),
                    bool(late_pressure_rollback_on),
                    tangential_rollback_theta,
                )
                return

            def _side_patch(side_idx, dx_side):
                grad_s_cell = grad_s_all[side_idx]
                grad_ut_cell = np.empty_like(grad_s_cell)
                grad_ut_cell[:, 0] = (
                    tx * coeffs[1, side_idx, 0]
                    + ty * coeffs[2, side_idx, 0])
                grad_ut_cell[:, 1] = (
                    tx * coeffs[1, side_idx, 1]
                    + ty * coeffs[2, side_idx, 1])

                nb = grad_nb_safe[side_idx]
                valid = grad_valid_nb[side_idx]
                nb_rho = rho_cell[nb]
                nb_p = p_cell[nb]
                grad_s_nb = np.empty((side_idx.shape[0], nb.shape[1], 2),
                                     dtype=float)
                grad_s_nb[:, :, 0] = (
                    coeffs[0, nb, 0] / nb_rho
                    - coeffs[3, nb, 0] / (gamma * nb_p))
                grad_s_nb[:, :, 1] = (
                    coeffs[0, nb, 1] / nb_rho
                    - coeffs[3, nb, 1] / (gamma * nb_p))
                grad_ut_nb = np.empty_like(grad_s_nb)
                grad_ut_nb[:, :, 0] = (
                    tx[:, None] * coeffs[1, nb, 0]
                    + ty[:, None] * coeffs[2, nb, 0])
                grad_ut_nb[:, :, 1] = (
                    tx[:, None] * coeffs[1, nb, 1]
                    + ty[:, None] * coeffs[2, nb, 1])

                s_cell_t = grad_s_cell[:, 0] * tx + grad_s_cell[:, 1] * ty
                ut_cell_t = (
                    grad_ut_cell[:, 0] * tx + grad_ut_cell[:, 1] * ty)
                ut_cell_n = (
                    grad_ut_cell[:, 0] * nx + grad_ut_cell[:, 1] * ny)
                s_nb_t = grad_s_nb[:, :, 0] * tx[:, None] + (
                    grad_s_nb[:, :, 1] * ty[:, None])
                ut_nb_t = grad_ut_nb[:, :, 0] * tx[:, None] + (
                    grad_ut_nb[:, :, 1] * ty[:, None])
                ut_nb_n = grad_ut_nb[:, :, 0] * nx[:, None] + (
                    grad_ut_nb[:, :, 1] * ny[:, None])
                same_stream = (
                    valid
                    & (np.abs(s_cell_t)[:, None] > _EPS)
                    & (np.abs(s_nb_t) > _EPS)
                    & (np.sign(s_cell_t)[:, None] == np.sign(s_nb_t)))
                opposite_normal = (
                    valid
                    & (np.abs(ut_cell_n)[:, None] > _EPS)
                    & (np.abs(ut_nb_n) > _EPS)
                    & (np.sign(ut_cell_n)[:, None] != np.sign(ut_nb_n)))
                consistent = (
                    valid
                    & (np.abs(s_cell_t)[:, None] > _EPS)
                    & (np.abs(ut_cell_t)[:, None] > _EPS)
                    & (np.abs(s_nb_t) > _EPS)
                    & (np.abs(ut_nb_t) > _EPS)
                    & (np.sign(s_cell_t)[:, None] == np.sign(s_nb_t))
                    & (np.sign(ut_cell_t)[:, None] == np.sign(ut_nb_t)))
                valid_count = np.sum(valid, axis=1)
                consistent_count = np.sum(consistent, axis=1)
                same_stream_count = np.sum(same_stream, axis=1)
                opposite_normal_count = np.sum(opposite_normal, axis=1)
                patch_coherence = consistent_count / np.maximum(
                    valid_count, 1)
                same_stream_contact = same_stream_count / np.maximum(
                    valid_count, 1)
                opposite_normal_shear = opposite_normal_count / np.maximum(
                    valid_count, 1)
                avg_s = np.sum(
                    np.where(consistent[:, :, None], grad_s_nb, 0.0),
                    axis=1) / np.maximum(consistent_count[:, None], 1)
                avg_ut = np.sum(
                    np.where(consistent[:, :, None], grad_ut_nb, 0.0),
                    axis=1) / np.maximum(consistent_count[:, None], 1)
                grad_s_patch = (
                    (1.0 - neighbor_blend) * grad_s_cell
                    + neighbor_blend * avg_s)
                grad_ut_patch = (
                    (1.0 - neighbor_blend) * grad_ut_cell
                    + neighbor_blend * avg_ut)
                delta_s = (
                    grad_s_patch[:, 0] * dx_side[:, 0]
                    + grad_s_patch[:, 1] * dx_side[:, 1])
                delta_ut = (
                    grad_ut_patch[:, 0] * dx_side[:, 0]
                    + grad_ut_patch[:, 1] * dx_side[:, 1])
                coherence_gate = _smoothstep(0.50, 0.80, patch_coherence)
                gate = (
                    clean_contact * shock_off * coherence_gate
                    * (valid_count >= min_valid))
                if pair_spacing_on and pair_spacing_beta > 0.0:
                    pair_spacing_gate = (
                        _smoothstep(0.45, 0.75, same_stream_contact)
                        * _smoothstep(0.35, 0.65, opposite_normal_shear))
                    gate = gate * (1.0 + pair_spacing_beta * pair_spacing_gate)
                    if gate_cap > 0.0:
                        gate = np.minimum(gate, gate_cap)
                return delta_s, delta_ut, patch_coherence, gate

            def _apply_side(W_side, side_idx, dx_side, side_name):
                delta_s, delta_ut, patch_coherence, gate = _side_patch(
                    side_idx, dx_side)
                if not np.any(gate > 0.0) and not diagnostics_on:
                    return
                rho_anchor = np.maximum(W_side[0, interior], _EPS)
                u_anchor = W_side[1, interior]
                v_anchor = W_side[2, interior]
                p_anchor = np.maximum(W_side[3, interior], _EPS)
                p_margin = p_anchor / np.maximum(p_floor, _EPS)
                if pair_spacing_on and pressure_margin_on:
                    pressure_margin_gate = _smoothstep(
                        1.05, 1.25, p_margin)
                    gate = gate * pressure_margin_gate
                s_anchor = (
                    np.log(rho_anchor) - np.log(p_anchor) / gamma)
                un_anchor = u_anchor * nx + v_anchor * ny
                ut_anchor = u_anchor * tx + v_anchor * ty

                if s_cap > 0.0:
                    delta_s = np.clip(delta_s, -s_cap, s_cap)
                else:
                    delta_s = np.zeros_like(delta_s)
                if ut_cap > 0.0 and wave_cap > 0.0:
                    local_ut_cap = np.minimum(ut_cap, wave_cap * c_sum)
                elif ut_cap > 0.0:
                    local_ut_cap = np.full_like(c_sum, ut_cap)
                elif wave_cap > 0.0:
                    local_ut_cap = wave_cap * c_sum
                else:
                    local_ut_cap = np.zeros_like(c_sum)
                delta_ut = np.where(
                    local_ut_cap > 0.0,
                    np.clip(delta_ut, -local_ut_cap, local_ut_cap),
                    0.0)

                s_trial = s_anchor + entropy_alpha * gate * delta_s
                ut_trial = ut_anchor + tangential_alpha * gate * delta_ut
                entropy_increment = entropy_alpha * gate * delta_s
                tangential_increment = tangential_alpha * gate * delta_ut
                rho_trial = np.exp(
                    s_trial + np.log(p_anchor) / gamma)
                u_trial = un_anchor * nx + ut_trial * tx
                v_trial = un_anchor * ny + ut_trial * ty

                rough_anchor = np.maximum(
                    np.maximum(rho_anchor - rho_hi, 0.0),
                    np.maximum(rho_lo - rho_anchor, 0.0)) / rho_avg
                rough_trial = np.maximum(
                    np.maximum(rho_trial - rho_hi, 0.0),
                    np.maximum(rho_lo - rho_trial, 0.0)) / rho_avg
                floor_bad = (rho_trial < rho_floor) | (p_anchor < p_floor)
                hard_reject = (
                    (shock_off < 0.20)
                    | (patch_coherence < 0.50)
                    | (rough_trial > rough_anchor + roughness_cap))
                if late_pressure_rollback_on:
                    rho_entropy_rollback = rho_anchor
                    u_entropy_rollback = un_anchor * nx + ut_trial * tx
                    v_entropy_rollback = un_anchor * ny + ut_trial * ty
                    entropy_still_bad = (
                        (rho_entropy_rollback < rho_floor)
                        | (p_anchor < p_floor)
                        | hard_reject)
                    ut_theta = (
                        ut_anchor
                        + tangential_rollback_theta * (ut_trial - ut_anchor))
                    u_tangent_rollback = un_anchor * nx + ut_theta * tx
                    v_tangent_rollback = un_anchor * ny + ut_theta * ty
                    tangent_still_bad = (
                        (rho_entropy_rollback < rho_floor)
                        | (p_anchor < p_floor)
                        | hard_reject)
                    rho_final = np.where(
                        floor_bad, rho_entropy_rollback, rho_trial)
                    u_final = np.where(
                        floor_bad, u_entropy_rollback, u_trial)
                    v_final = np.where(
                        floor_bad, v_entropy_rollback, v_trial)
                    rho_final = np.where(
                        floor_bad & entropy_still_bad,
                        rho_entropy_rollback, rho_final)
                    u_final = np.where(
                        floor_bad & entropy_still_bad,
                        u_tangent_rollback, u_final)
                    v_final = np.where(
                        floor_bad & entropy_still_bad,
                        v_tangent_rollback, v_final)
                    rho_final = np.where(
                        floor_bad & entropy_still_bad & tangent_still_bad,
                        rho_anchor, rho_final)
                    u_final = np.where(
                        floor_bad & entropy_still_bad & tangent_still_bad,
                        u_anchor, u_final)
                    v_final = np.where(
                        floor_bad & entropy_still_bad & tangent_still_bad,
                        v_anchor, v_final)
                    rejected = (
                        (gate > 0.0)
                        & floor_bad
                        & (entropy_still_bad | tangent_still_bad))
                elif pair_spacing_on:
                    s_rollback = floor_bad | (
                        rough_trial > rough_anchor + roughness_cap)
                    full_rollback = (
                        (p_anchor < p_floor)
                        | (shock_off < 0.20)
                        | (patch_coherence < 0.50))
                    rho_final = np.where(s_rollback, rho_anchor, rho_trial)
                    u_s_rollback = un_anchor * nx + ut_trial * tx
                    v_s_rollback = un_anchor * ny + ut_trial * ty
                    u_final = np.where(s_rollback, u_s_rollback, u_trial)
                    v_final = np.where(s_rollback, v_s_rollback, v_trial)
                    rho_final = np.where(full_rollback, rho_anchor,
                                         rho_final)
                    u_final = np.where(full_rollback, u_anchor, u_final)
                    v_final = np.where(full_rollback, v_anchor, v_final)
                    rejected = (
                        (gate > 0.0) & (s_rollback | full_rollback))
                else:
                    denom = rho_anchor - rho_trial
                    theta_rho = np.where(
                        ((rho_trial < rho_floor) & (denom > _EPS)),
                        ((rho_anchor - rho_floor)
                         / np.maximum(denom, _EPS)),
                        1.0)
                    theta = np.clip(theta_rho, 0.0, 1.0)
                    theta = np.where(p_anchor < p_floor, 0.0, theta)
                    theta = np.where(hard_reject & ~floor_bad, 0.0, theta)
                    rejected = (
                        (gate > 0.0)
                        & ((theta < 0.999999) | floor_bad | hard_reject))
                    rho_final = rho_anchor + theta * (rho_trial - rho_anchor)
                    u_final = u_anchor + theta * (u_trial - u_anchor)
                    v_final = v_anchor + theta * (v_trial - v_anchor)
                _record_v40_patch_diagnostics(
                    side_name, gate, patch_coherence, rejected,
                    entropy_increment, tangential_increment)

                W_side[0, interior] = np.where(gate > 0.0, rho_final,
                                               W_side[0, interior])
                W_side[1, interior] = np.where(gate > 0.0, u_final,
                                               W_side[1, interior])
                W_side[2, interior] = np.where(gate > 0.0, v_final,
                                               W_side[2, interior])
                W_side[3, interior] = p_anchor

            if diagnostics_on:
                self._v40_patch_diag_latest_raw = None
            _apply_side(W_L_local, o_idx, dx_fo, 'owner')
            _apply_side(W_R_local, n_idx, dx_fn, 'neighbour')

        def _euler_pressure_face_jump_limiter(W_left, W_right):
            if not self.euler_pressure_face_jump_limiter_on:
                return
            if not W_left.size or not W_right.size:
                return
            if 3 not in active_vars:
                return
            p_l = np.maximum(W_left[3, interior], _EPS)
            p_r = np.maximum(W_right[3, interior], _EPS)
            p_o = np.maximum(W_recon_cell[3, o_idx], _EPS)
            p_n = np.maximum(W_recon_cell[3, n_idx], _EPS)
            base_jump = np.abs(p_n - p_o)
            pressure_scale = np.maximum(p_n + p_o, _EPS)
            growth_cap = max(
                float(self.euler_pressure_face_jump_limiter_growth_cap), 0.0)
            abs_floor = max(
                float(self.euler_pressure_face_jump_limiter_abs_floor), 0.0)
            allowed = base_jump * (1.0 + growth_cap) + abs_floor * pressure_scale
            face_jump = np.abs(p_r - p_l)
            active = face_jump > allowed
            if not np.any(active):
                return

            p_width = abs(float(
                self.euler_pressure_face_jump_limiter_p_jump_width))
            c_width = abs(float(
                self.euler_pressure_face_jump_limiter_compression_width))
            n_width = abs(float(
                self.euler_pressure_face_jump_limiter_normality_width))
            if p_width < 1.0e-30 or c_width < 1.0e-30 or n_width < 1.0e-30:
                return
            p_gate = np.clip(
                (pressure_jump
                 - float(self.euler_pressure_face_jump_limiter_p_jump_threshold))
                / p_width, 0.0, 1.0)
            c_gate = np.clip(
                (compression
                 - float(
                     self.euler_pressure_face_jump_limiter_compression_threshold))
                / c_width, 0.0, 1.0)
            n_gate = np.clip(
                (normality
                 - float(
                     self.euler_pressure_face_jump_limiter_normality_threshold))
                / n_width, 0.0, 1.0)
            shock_off = (1.0 - p_gate) * (1.0 - c_gate) * (1.0 - n_gate)
            clean_shear_contact = (
                np.clip(density_contact_weight, 0.0, 1.0)
                * np.clip(tangential_contact_weight, 0.0, 1.0)
                * shock_off)
            strength = np.clip(
                float(self.euler_pressure_face_jump_limiter_strength), 0.0, 1.0)
            w = np.where(active, strength * clean_shear_contact, 0.0)
            if not np.any(w > 0.0):
                return

            mid = 0.5 * (p_l + p_r)
            half_allowed = 0.5 * allowed
            sign_l = np.where(p_l >= p_r, 1.0, -1.0)
            target_l = np.maximum(mid + sign_l * half_allowed, _EPS)
            target_r = np.maximum(mid - sign_l * half_allowed, _EPS)
            W_left[3, interior] = p_l + w * (target_l - p_l)
            W_right[3, interior] = p_r + w * (target_r - p_r)

        fast_minmod_vertex_face_loop = (
            fast_vertex_face_loop_candidate
            and vertex_min_values is not None
            and vertex_max_values is not None
        )
        base_tvd_mode = (
            1 if self._tvd_name == 'mc'
            else 2 if self._tvd_name == 'van_leer'
            else 3 if self._tvd_name == 'umist'
            else 4 if self._tvd_name == 'superbee'
            else 5 if self._tvd_name in (
                'bounded_cd', 'central', 'cd', 'pure_downwind')
            else 6 if self._tvd_name == 'koren'
            else 16 if self._tvd_name in (
                'modified_superbee', 'superbee15')
            else 0)
        velocity_tvd_mode_fast = (
            1 if self._velocity_tvd_name == 'mc'
            else 2 if self._velocity_tvd_name == 'van_leer'
            else 3 if self._velocity_tvd_name == 'umist'
            else 4 if self._velocity_tvd_name == 'superbee'
            else 5 if self._velocity_tvd_name in (
                'bounded_cd', 'central', 'cd', 'pure_downwind')
            else 6 if self._velocity_tvd_name == 'koren'
            else 16 if self._velocity_tvd_name in (
                'modified_superbee', 'superbee15')
            else 0)
        density_tvd_mode_fast = (
            1 if self._density_tvd_name == 'mc'
            else 2 if self._density_tvd_name == 'van_leer'
            else 3 if self._density_tvd_name == 'umist'
            else 4 if self._density_tvd_name == 'superbee'
            else 5 if self._density_tvd_name in (
                'bounded_cd', 'central', 'cd', 'pure_downwind')
            else 6 if self._density_tvd_name == 'koren'
            else 7 if self._density_tvd_name == 'downwind'
            else 16 if self._density_tvd_name in (
                'modified_superbee', 'superbee15')
            else 0)
        face_gradient_shock_damping_name = str(
            self.face_gradient_shock_damping).strip().lower()
        face_gradient_fast_ok = (
            face_gradient_correction_name in ('jasak', 'beta')
            or (
                face_gradient_correction_name in (
                    'beta_shock_shear', 'shock_shear_beta',
                    'beta-shock-shear')
                and face_gradient_shock_damping_name in (
                    '', 'off', 'none', '0', 'false')
            )
        )
        weak_face_value_scaling_mode = str(
            self.euler_density_contact_weak_face_value_scaling_mode
        ).strip().lower()
        density_micro_restore_mode = (
            1 if weak_face_value_scaling_mode in (
                'clean_shear_micro_restore',
                'density_mlp_micro_restore')
            else 2 if weak_face_value_scaling_mode in (
                'coherent_shear_micro_restore',
                'density_mlp_coherent_restore')
            else 0)
        density_micro_restore_fast = (
            self.euler_density_contact_weak_face_value_scaling
            and os.environ.get(
                'TMLPU_ENABLE_FAST_WEAK_FACE_VALUE_SCALING', ''
            ).strip().lower() in ('1', 'true', 'yes', 'on')
            and density_micro_restore_mode > 0
            and not self.euler_density_contact_weak_face_stream_coherence_on
            and ctx.get('face_node_int_safe') is not None
            and ctx.get('face_node_int_valid') is not None
        )
        density_weak_face_mlp_fast = (
            self.euler_density_contact_weak_face_mlp
            and os.environ.get(
                'TMLPU_ENABLE_FAST_DENSITY_WEAK_FACE_MLP', ''
            ).strip().lower() in ('1', 'true', 'yes', 'on')
            and not self.weak_face_mlp
            and not self.euler_density_contact_weak_face_head_generic
            and not self.euler_density_contact_weak_face_disable_specialized_relax
            and not self.euler_density_contact_weak_face_legacy_order
            and not self.euler_density_contact_weak_face_admissibility_damp
            and not self.euler_density_contact_weak_face_entropy_accept
            and not self.euler_density_contact_weak_face_shock_gate
            and float(self.euler_density_contact_weak_face_root_blend) == 0.0
            and float(self.euler_density_contact_weak_face_swirl_extra) == 0.0
            and (
                not self.euler_density_contact_weak_face_value_scaling
                or density_micro_restore_fast
            )
            and ctx.get('face_node_int_safe') is not None
            and ctx.get('face_node_int_valid') is not None
        )
        density_weak_face_mlp_fast_blocked = (
            self.euler_density_contact_weak_face_mlp
            and not density_weak_face_mlp_fast
        )
        weak_face_value_scaling_fast_blocked = (
            self.euler_density_contact_weak_face_value_scaling
            and not density_micro_restore_fast
        )
        fast_jasak_vertex_face_loop = (
            _NUMBA_AVAILABLE
            and not os.environ.get('TMLPU_DISABLE_FAST_FACE_RECON')
            and self.mlp_bound
            and self.vertex_mlp
            and self.virtual_uu_gradient
            and self.face_skew_correction
            and face_gradient_fast_ok
            and face_increment_name in ('tmlpu', 'lsq')
            and r_form_name == 'far_upwind'
            and not self.cicsam_full
            and not self.weak_face_mlp
            and not density_weak_face_mlp_fast_blocked
            and not weak_face_value_scaling_fast_blocked
            and not use_extremum_relax
            and not self.vertex_mlp_augment
            and not self.physical_vertex_bounds
            and not self.physical_vertex_bounds_value_continuous_otsu
            and not self.phi_LL_unclipped
            and vertex_min_values is not None
            and vertex_max_values is not None
            and ctx.get('cell_node_arr') is not None
            and ctx.get('vertex_offsets') is not None
            and base_tvd_mode in (1, 4, 5, 6, 16)
            and velocity_tvd_mode_fast in (0, 1, 2, 3, 4, 5, 6, 16)
            and density_tvd_mode_fast in (0, 1, 2, 3, 4, 5, 6, 7, 16)
            and not self.euler_density_entropy_split
            and not self.euler_log_positive
        )
        if fast_jasak_vertex_face_loop:
            cell_node_safe_fast = np.where(ctx['cell_node_valid'],
                                           ctx['cell_node_arr'], 0)
            density_delta_fast = np.zeros((2, interior.size), dtype=np.float64)
            density_psi_tvd_fast = np.zeros((2, interior.size), dtype=np.float64)
            density_psi_base_fast = np.zeros((2, interior.size), dtype=np.float64)
            _tmlpu_jasak_vertex_faces_kernel(
                np.asarray(W_cell, dtype=np.float64),
                np.asarray(coeffs, dtype=np.float64),
                np.asarray(phi_min_cell, dtype=np.float64),
                np.asarray(phi_max_cell, dtype=np.float64),
                np.asarray(o_idx, dtype=np.int64),
                np.asarray(n_idx, dtype=np.int64),
                np.asarray(interior, dtype=np.int64),
                np.asarray(d_o_int, dtype=np.float64),
                np.asarray(tstar_o, dtype=np.float64),
                np.asarray(tstar_n, dtype=np.float64),
                np.asarray(dx_fo, dtype=np.float64),
                np.asarray(dx_fn, dtype=np.float64),
                np.asarray(d_len, dtype=np.float64),
                np.asarray(e_o, dtype=np.float64),
                np.asarray(e_n, dtype=np.float64),
                np.asarray(face_n_o, dtype=np.float64),
                1 if face_gradient_correction_name == 'jasak' else 0,
                float(theta_min),
                float(np.clip(self.zero_delta_psi, 0.0, 2.0)),
                bool(self.tmlpu_bound_tvd_separate),
                active_vars_arr,
                np.asarray(cell_node_safe_fast, dtype=np.int64),
                np.asarray(ctx['cell_node_valid'], dtype=np.bool_),
                np.asarray(ctx['vertex_offsets'], dtype=np.float64),
                np.asarray(vertex_min_values, dtype=np.float64),
                np.asarray(vertex_max_values, dtype=np.float64),
                np.asarray(shock_flatten, dtype=np.float64),
                np.asarray(pressure_flatten, dtype=np.float64),
                np.asarray(velocity_flatten, dtype=np.float64),
                np.asarray(density_flatten, dtype=np.float64),
                np.asarray(density_contact_weight, dtype=np.float64),
                np.asarray(tangential_contact_weight, dtype=np.float64),
                np.asarray(density_contact_hancock_scale, dtype=np.float64),
                np.asarray(one_minus_C_face, dtype=np.float64),
                np.asarray(face_hancock_courant, dtype=np.float64),
                float(tvb_eps),
                bool(self.euler_density_no_hancock),
                bool(self.euler_density_contact_wave_hancock),
                float(self.euler_density_contact_hancock_boost),
                float(self.euler_density_contact_hancock_boost_cap),
                bool(self.euler_density_first_order),
                bool(self.euler_pressure_first_order),
                bool(self.euler_velocity_no_hancock),
                bool(self.euler_velocity_shock_flatten),
                bool(self.euler_density_lsq_increment),
                bool(self.euler_density_full_lsq_increment),
                bool(self.euler_velocity_lsq_increment),
                bool(self.euler_pressure_shear_lsq_increment),
                bool(self.euler_pressure_nonshock_lsq_increment),
                1 if face_increment_name == 'lsq' else 0,
                int(base_tvd_mode),
                int(velocity_tvd_mode_fast),
                int(density_tvd_mode_fast),
                bool(self.euler_density_extrema_lmp),
                bool(self.euler_velocity_extrema_lmp),
                density_delta_fast,
                density_psi_tvd_fast,
                density_psi_base_fast,
                W_L,
                W_R,
            )
            if density_weak_face_mlp_fast or density_micro_restore_fast:
                _euler_density_micro_restore_kernel(
                    np.asarray(W_cell, dtype=np.float64),
                    np.asarray(coeffs, dtype=np.float64),
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(face_n_o, dtype=np.float64),
                    np.asarray(ctx['face_node_int_safe'], dtype=np.int64),
                    np.asarray(ctx['face_node_int_valid'], dtype=np.bool_),
                    np.asarray(vertex_min_values, dtype=np.float64),
                    np.asarray(vertex_max_values, dtype=np.float64),
                    density_delta_fast,
                    density_psi_tvd_fast,
                    density_psi_base_fast,
                    np.asarray(density_contact_weight, dtype=np.float64),
                    np.asarray(density_flatten, dtype=np.float64),
                    np.asarray(pressure_flatten, dtype=np.float64),
                    np.asarray(velocity_flatten, dtype=np.float64),
                    np.asarray(pressure_jump, dtype=np.float64),
                    np.asarray(compression, dtype=np.float64),
                    np.asarray(normality, dtype=np.float64),
                    float(getattr(eq, 'gamma', 1.4)),
                    float(tvb_eps),
                    float(np.clip(
                        self.euler_density_contact_weak_face_value_scaling_shear_blend_alpha,
                        0.0, 1.0)),
                    float(max(
                        self.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad,
                        0.0)),
                    float(max(
                        self.euler_density_contact_weak_face_density_increment_cap,
                        0.0)),
                    bool(
                        self.euler_density_contact_weak_face_value_scaling_require_coherent_shear),
                    bool(
                        self.euler_density_contact_weak_face_value_scaling_artifact_reject),
                    int(
                        (10 + density_micro_restore_mode)
                        if (density_weak_face_mlp_fast
                            and density_micro_restore_fast)
                        else (0 if density_weak_face_mlp_fast
                              else density_micro_restore_mode)),
                    float(max(
                        self.euler_density_contact_weak_face_mlp_cap, 0.0)),
                    float(max(
                        self.euler_density_contact_weak_face_shock_power, 1.0)),
                    W_L,
                    W_R,
                )
            if (_NUMBA_AVAILABLE
                    and self.euler_density_pressure_entropy
                    and nvar >= 4
                    and not euler_log_positive
                    and 0 in active_vars):
                _euler_density_pressure_entropy_kernel(
                    np.asarray(W_cell, dtype=np.float64),
                    np.asarray(coeffs, dtype=np.float64),
                    np.asarray(phi_min_cell, dtype=np.float64),
                    np.asarray(phi_max_cell, dtype=np.float64),
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(dx_fo, dtype=np.float64),
                    np.asarray(dx_fn, dtype=np.float64),
                    np.asarray(density_contact_weight, dtype=np.float64),
                    np.asarray(density_flatten, dtype=np.float64),
                    float(getattr(eq, 'gamma', 1.4)),
                    bool(euler_log_pressure_only),
                    bool(os.environ.get(
                        'TMLPU_EULER_DENSITY_PRESSURE_ENTROPY_ACCEPT', ''
                    ).strip().lower() in (
                        'entropy', 'entropy_residual',
                        'entropy-residual', 'residual')),
                    W_L,
                    W_R,
                )
            tangential_tvd_mode = (
                1 if self._tangential_velocity_tvd_name == 'mc'
                else 2 if self._tangential_velocity_tvd_name == 'van_leer'
                else 3 if self._tangential_velocity_tvd_name == 'umist'
                else 17 if self._tangential_velocity_tvd_name == 'koren'
                else 4 if self._tangential_velocity_tvd_name == 'contact_umist'
                else 5 if self._tangential_velocity_tvd_name == 'superbee'
                else 16 if self._tangential_velocity_tvd_name in (
                    'modified_superbee', 'superbee15')
                else 6 if self._tangential_velocity_tvd_name == 'contact_van_leer'
                else 7 if self._tangential_velocity_tvd_name == 'contact_van_leer_linear'
                else 8 if self._tangential_velocity_tvd_name == 'shock_van_leer'
                else 9 if self._tangential_velocity_tvd_name == 'shock_van_leer_strict'
                else 10 if self._tangential_velocity_tvd_name == 'contact_van_leer_root'
                else 11 if self._tangential_velocity_tvd_name == 'shock_van_leer_cubic'
                else 12 if self._tangential_velocity_tvd_name == 'contact_umist_shock'
                else 13 if self._tangential_velocity_tvd_name == 'contact_umist_shock_root'
                else 14 if self._tangential_velocity_tvd_name == 'contact_superbee'
                else 15 if self._tangential_velocity_tvd_name == 'contact_superbee_shock'
                else 18 if self._tangential_velocity_tvd_name == 'superbee_shock_blend'
                else 19 if self._tangential_velocity_tvd_name == 'shear_superbee_blend'
                else 20 if self._tangential_velocity_tvd_name == 'shear_superbee_root_blend'
                else 21 if self._tangential_velocity_tvd_name == 'shear_superbee_root_micro'
                else 22 if self._tangential_velocity_tvd_name == 'shear_superbee_root_mood'
                else 0)
            if (tangential_tvd_mode > 0
                    and _NUMBA_AVAILABLE
                    and nvar >= 4
                    and velocity_pair_active):
                tangential_weight = _ducros_tangential_weight(
                    tangential_contact_weight, coeffs)
                if (self.euler_velocity_no_hancock
                        or self.euler_tangential_velocity_no_hancock):
                    velocity_face_scale = (
                        1.0 - face_hancock_courant * shock_flatten)
                elif self.euler_tangential_contact_wave_hancock:
                    base_scale = one_minus_C_face
                    velocity_face_scale = (
                        base_scale
                        + tangential_weight
                        * (density_contact_hancock_scale - base_scale))
                else:
                    velocity_face_scale = one_minus_C_face
                if self.euler_velocity_shock_flatten:
                    tangential_flatten = tangential_shock_flatten
                    if self.euler_tangential_contact_relax_flatten:
                        tangential_flatten = tangential_flatten * (
                            1.0 - np.clip(tangential_weight, 0.0, 1.0))
                    velocity_face_scale = (
                        velocity_face_scale * (1.0 - tangential_flatten))
                _euler_tangential_velocity_mc_kernel(
                    np.asarray(W_cell, dtype=np.float64),
                    np.asarray(coeffs, dtype=np.float64),
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(d_o_int, dtype=np.float64),
                    np.asarray(dx_fo, dtype=np.float64),
                    np.asarray(dx_fn, dtype=np.float64),
                    np.asarray(alpha_o, dtype=np.float64),
                    np.asarray(alpha_n, dtype=np.float64),
                    np.asarray(face_n_o, dtype=np.float64),
                    np.asarray(velocity_face_scale, dtype=np.float64),
                    np.asarray(tangential_shock_flatten, dtype=np.float64),
                    np.asarray(tangential_weight, dtype=np.float64),
                    int(tangential_tvd_mode),
                    float(self.euler_tangential_shear_micro_blend),
                    float(self.euler_tangential_shear_micro_cap),
                    float(self.euler_tangential_mood_wavespeed_growth_cap),
                    float(self.euler_tangential_mood_jump_growth_cap),
                    bool(self.euler_tangential_velocity_lsq_increment),
                    W_L,
                    W_R,
                )
                _apply_tangential_pair_restore(W_L, W_R)
            if self.euler_pressure_contact_entropy_blend:
                _euler_pressure_contact_entropy_blend(W_L)
                _euler_pressure_contact_entropy_blend(W_R)
            if self.euler_pressure_face_jump_limiter_on:
                _euler_pressure_face_jump_limiter(W_L, W_R)
            _apply_contact_characteristic_postpass(W_L, W_R)
            _apply_patch_contact_shear_postpass(W_L, W_R)
            return _finish_faces(W_L, W_R)
        if fast_minmod_vertex_face_loop:
            cell_node_safe_fast = np.where(ctx['cell_node_valid'],
                                           ctx['cell_node_arr'], 0)
            velocity_tvd_mode = (
                1 if self._velocity_tvd_name == 'mc'
                else 2 if self._velocity_tvd_name == 'van_leer'
                else 3 if self._velocity_tvd_name == 'umist'
                else 4 if self._velocity_tvd_name == 'superbee'
                else 5 if self._velocity_tvd_name in (
                    'bounded_cd', 'central', 'cd', 'pure_downwind')
                else 6 if self._velocity_tvd_name == 'koren'
                else 16 if self._velocity_tvd_name in (
                    'modified_superbee', 'superbee15')
                else 0)
            density_tvd_mode = (
                1 if self._density_tvd_name == 'mc'
                else 2 if self._density_tvd_name == 'van_leer'
                else 3 if self._density_tvd_name == 'umist'
                else 4 if self._density_tvd_name == 'superbee'
                else 5 if self._density_tvd_name in (
                    'bounded_cd', 'central', 'cd', 'pure_downwind')
                else 6 if self._density_tvd_name == 'koren'
                else 16 if self._density_tvd_name in (
                    'modified_superbee', 'superbee15')
                else 0)
            _tmlpu_minmod_vertex_faces_kernel(
                np.asarray(W_cell, dtype=np.float64),
                np.asarray(coeffs, dtype=np.float64),
                np.asarray(phi_min_cell, dtype=np.float64),
                np.asarray(phi_max_cell, dtype=np.float64),
                np.asarray(o_idx, dtype=np.int64),
                np.asarray(n_idx, dtype=np.int64),
                np.asarray(interior, dtype=np.int64),
                np.asarray(d_o_int, dtype=np.float64),
                np.asarray(alpha_o, dtype=np.float64),
                np.asarray(alpha_n, dtype=np.float64),
                np.asarray(dx_fo, dtype=np.float64),
                np.asarray(dx_fn, dtype=np.float64),
                np.asarray(cell_node_safe_fast, dtype=np.int64),
                np.asarray(ctx['cell_node_valid'], dtype=np.bool_),
                np.asarray(ctx['vertex_offsets'], dtype=np.float64),
                np.asarray(vertex_min_values, dtype=np.float64),
                np.asarray(vertex_max_values, dtype=np.float64),
                np.asarray(shock_flatten, dtype=np.float64),
                np.asarray(pressure_flatten, dtype=np.float64),
                np.asarray(velocity_flatten, dtype=np.float64),
                np.asarray(density_flatten, dtype=np.float64),
                np.asarray(density_contact_weight, dtype=np.float64),
                np.asarray(density_contact_hancock_scale, dtype=np.float64),
                np.asarray(one_minus_C_face, dtype=np.float64),
                float(tvb_eps),
                bool(self.euler_shock_flatten),
                bool(self.euler_density_acoustic_flatten),
                bool(self.euler_density_lsq_increment),
                bool(self.euler_density_no_hancock),
                bool(self.euler_density_entropy_split),
                bool(self.euler_density_contact_wave_hancock),
                bool(self.euler_density_first_order),
                bool(self.euler_pressure_first_order),
                bool(self.euler_velocity_no_hancock),
                bool(self.euler_velocity_shock_flatten),
                bool(self.euler_velocity_lsq_increment),
                int(velocity_tvd_mode),
                int(density_tvd_mode),
                bool(self.euler_density_extrema_lmp),
                bool(self.euler_velocity_extrema_lmp),
                float(getattr(eq, 'gamma', 1.4)),
                W_L,
                W_R,
            )
            tangential_tvd_mode = (
                1 if self._tangential_velocity_tvd_name == 'mc'
                else 2 if self._tangential_velocity_tvd_name == 'van_leer'
                else 3 if self._tangential_velocity_tvd_name == 'umist'
                else 17 if self._tangential_velocity_tvd_name == 'koren'
                else 4 if self._tangential_velocity_tvd_name == 'contact_umist'
                else 5 if self._tangential_velocity_tvd_name == 'superbee'
                else 16 if self._tangential_velocity_tvd_name in (
                    'modified_superbee', 'superbee15')
                else 6 if self._tangential_velocity_tvd_name == 'contact_van_leer'
                else 7 if self._tangential_velocity_tvd_name == 'contact_van_leer_linear'
                else 8 if self._tangential_velocity_tvd_name == 'shock_van_leer'
                else 9 if self._tangential_velocity_tvd_name == 'shock_van_leer_strict'
                else 10 if self._tangential_velocity_tvd_name == 'contact_van_leer_root'
                else 11 if self._tangential_velocity_tvd_name == 'shock_van_leer_cubic'
                else 12 if self._tangential_velocity_tvd_name == 'contact_umist_shock'
                else 13 if self._tangential_velocity_tvd_name == 'contact_umist_shock_root'
                else 14 if self._tangential_velocity_tvd_name == 'contact_superbee'
                else 15 if self._tangential_velocity_tvd_name == 'contact_superbee_shock'
                else 18 if self._tangential_velocity_tvd_name == 'superbee_shock_blend'
                else 19 if self._tangential_velocity_tvd_name == 'shear_superbee_blend'
                else 20 if self._tangential_velocity_tvd_name == 'shear_superbee_root_blend'
                else 21 if self._tangential_velocity_tvd_name == 'shear_superbee_root_micro'
                else 22 if self._tangential_velocity_tvd_name == 'shear_superbee_root_mood'
                else 0)
            if (tangential_tvd_mode > 0 and nvar >= 4
                    and velocity_pair_active):
                tangential_weight = _ducros_tangential_weight(
                    tangential_contact_weight, coeffs)
                if (self.euler_velocity_no_hancock
                        or self.euler_tangential_velocity_no_hancock):
                    velocity_face_scale = (
                        1.0 - face_hancock_courant * shock_flatten)
                elif self.euler_tangential_contact_wave_hancock:
                    base_scale = one_minus_C_face
                    velocity_face_scale = (
                        base_scale
                        + tangential_weight
                        * (density_contact_hancock_scale - base_scale))
                else:
                    velocity_face_scale = one_minus_C_face
                _euler_tangential_velocity_mc_kernel(
                    np.asarray(W_cell, dtype=np.float64),
                    np.asarray(coeffs, dtype=np.float64),
                    np.asarray(o_idx, dtype=np.int64),
                    np.asarray(n_idx, dtype=np.int64),
                    np.asarray(interior, dtype=np.int64),
                    np.asarray(d_o_int, dtype=np.float64),
                    np.asarray(dx_fo, dtype=np.float64),
                    np.asarray(dx_fn, dtype=np.float64),
                    np.asarray(alpha_o, dtype=np.float64),
                    np.asarray(alpha_n, dtype=np.float64),
                    np.asarray(face_n_o, dtype=np.float64),
                    np.asarray(velocity_face_scale, dtype=np.float64),
                    np.asarray(tangential_shock_flatten, dtype=np.float64),
                    np.asarray(tangential_weight, dtype=np.float64),
                    int(tangential_tvd_mode),
                    float(self.euler_tangential_shear_micro_blend),
                    float(self.euler_tangential_shear_micro_cap),
                    float(self.euler_tangential_mood_wavespeed_growth_cap),
                    float(self.euler_tangential_mood_jump_growth_cap),
                    bool(self.euler_tangential_velocity_lsq_increment),
                    W_L,
                    W_R,
                )
                _apply_tangential_pair_restore(W_L, W_R)
            return _finish_faces(W_L, W_R)

        def _cicsam_face_value(grad_x, grad_y, d_vec, phi_self, delta_p,
                                phi_min_b, phi_max_b, smooth_mask):
            """One-side CICSAM (Ubbink) NVD blend.

            Inputs are aligned to "owner-as-U" or "neighbour-as-U" view of
            the face, with d_vec = x_D − x_U (for the local U).  Returns
            the limited face value (1-D array of length n_int).
            """
            gdotd = grad_x * d_vec[:, 0] + grad_y * d_vec[:, 1]
            denom = 2.0 * gdotd
            safe_denom = np.where(np.abs(denom) > _EPS,
                                   denom, np.copysign(_EPS, denom))
            delta_minus_v = -delta_p + 2.0 * gdotd
            phi_C_t = delta_minus_v / safe_denom
            # HC + UQ arms
            phi_f_HC = np.minimum(1.0, phi_C_t / max(Co_cic, 1e-10))
            phi_f_UQ = ((8.0 * Co_cic) * phi_C_t
                        + (1.0 - Co_cic) * (6.0 * phi_C_t + 3.0)) / 8.0
            phi_f_UQ = np.maximum(phi_f_UQ, phi_C_t)
            phi_f_UQ = np.minimum(phi_f_UQ, phi_f_HC)
            # γ_f = cos²(2θ) ; θ = ∠(∇φ, d_UD)
            grad_sq = grad_x * grad_x + grad_y * grad_y
            d_sq = d_vec[:, 0] ** 2 + d_vec[:, 1] ** 2
            cos2_th = (gdotd * gdotd) / np.maximum(grad_sq * d_sq, _EPS)
            cos_2th = 2.0 * cos2_th - 1.0
            gamma_f = np.minimum(cos_2th * cos_2th, 1.0)
            phi_f_t = gamma_f * phi_f_HC + (1.0 - gamma_f) * phi_f_UQ
            is_mono = (phi_C_t >= 0.0) & (phi_C_t <= 1.0)
            phi_f_t = np.where(is_mono, phi_f_t, phi_C_t)
            recon_cic = phi_self + (phi_f_t - phi_C_t) * denom
            if self.mlp_bound:
                clip_v = np.clip(recon_cic, phi_min_b, phi_max_b)
                if self.extremum_relax:
                    return np.where(smooth_mask, recon_cic, clip_v)
                return clip_v
            return recon_cic

        def _safe_ratio(num, den, zero_value=1.0):
            out = np.full_like(num, zero_value, dtype=float)
            np.divide(num, den, out=out, where=np.abs(den) > _EPS)
            large = np.where(num >= 0.0, 1e30, -1e30)
            return np.where((np.abs(den) <= _EPS) & (np.abs(num) > _EPS),
                            large, out)

        def _tmlpu_vertex_face_psi(var, L_idx, d_lr, phi_L, delta_plus,
                                   tstar, grad_corr_x, grad_corr_y,
                                   r_tmlpu, psi_tvd):
            """T-MLP-u vertex constraint from the paper formula.

            For every vertex V_i of the left/upwind cell L,

                Δφ_Vi = 0.5(φ_R-φ_L)
                        + \bar{∇φ} · (d_Vi - 0.5 d_LR)
                α_i = bound_i / (Δφ_Vi min(1,r))
                ψ_i = min(α_i r, α_i, ψ_TVD), with ψ_i=0 for r<=0

            The returned limiter is min_i ψ_i.  Invalid padded vertices are
            neutral and impose only the base TVD cap.
            """
            if (vertex_min_values is None or vertex_max_values is None
                    or ctx['cell_node_valid'] is None):
                return None

            def _psi_for_nodes(node_safe, node_valid, d_vi):
                enable_bounded_vertex_numba = os.environ.get(
                    'TMLPU_ENABLE_BOUNDED_VERTEX_NUMBA', ''
                ).strip().lower() in ('1', 'true', 'yes', 'on')
                if (_NUMBA_AVAILABLE and node_safe.shape[0] >= 256
                        and (enable_bounded_vertex_numba
                             or not self.tmlpu_bound_tvd_separate)):
                    zero_vertex_grad = vertex_min_values[var]
                    if (self.vertex_mlp_augment
                            and vertex_grad_x_values is not None
                            and vertex_grad_y_values is not None):
                        use_augment = True
                        vg_x = vertex_grad_x_values[var]
                        vg_y = vertex_grad_y_values[var]
                    else:
                        use_augment = False
                        vg_x = zero_vertex_grad
                        vg_y = zero_vertex_grad
                    return _tmlpu_vertex_psi_nodes_kernel(
                        np.asarray(phi_L, dtype=np.float64),
                        np.asarray(delta_plus, dtype=np.float64),
                        np.asarray(tstar, dtype=np.float64),
                        np.asarray(d_lr, dtype=np.float64),
                        np.asarray(grad_corr_x, dtype=np.float64),
                        np.asarray(grad_corr_y, dtype=np.float64),
                        np.asarray(r_tmlpu, dtype=np.float64),
                        np.asarray(psi_tvd, dtype=np.float64),
                        np.asarray(node_safe, dtype=np.int64),
                        np.asarray(node_valid, dtype=np.bool_),
                        np.asarray(d_vi, dtype=np.float64),
                        np.asarray(vertex_min_values[var], dtype=np.float64),
                        np.asarray(vertex_max_values[var], dtype=np.float64),
                        float(tvb_eps),
                        np.asarray(physical_vertex_cell[var, L_idx],
                                   dtype=np.bool_),
                        bool(self.physical_vertex_bounds),
                        bool(self.physical_vertex_bounds_value_continuous_otsu),
                        bool(use_augment),
                        np.asarray(vg_x, dtype=np.float64),
                        np.asarray(vg_y, dtype=np.float64),
                        bool(self._tvd_name in ('bounded_cd', 'central', 'cd',
                                                'pure_downwind')
                             or (self.euler_velocity_extrema_lmp
                                 and var in (1, 2))
                             or (self.euler_density_extrema_lmp
                                 and var == 0)))
                d_corr = d_vi - tstar[:, None, None] * d_lr[:, None, :]
                delta_vi = (tstar[:, None] * delta_plus[:, None]
                            + grad_corr_x[:, None] * d_corr[:, :, 0]
                            + grad_corr_y[:, None] * d_corr[:, :, 1])

                phys_mask = physical_vertex_cell[var, L_idx]
                if self.physical_vertex_bounds:
                    vmin = np.zeros_like(delta_vi)
                    vmax = np.ones_like(delta_vi)
                else:
                    vmin = vertex_min_values[var, node_safe] - tvb_eps
                    vmax = vertex_max_values[var, node_safe] + tvb_eps
                    if self.physical_vertex_bounds_value_continuous_otsu:
                        vmin = np.where(phys_mask[:, None], 0.0, vmin)
                        vmax = np.where(phys_mask[:, None], 1.0, vmax)
                allowed = np.where(delta_vi >= 0.0,
                                   vmax - phi_L[:, None],
                                   vmin - phi_L[:, None])

                r_pos = np.maximum(r_tmlpu, 0.0)
                monotone = r_tmlpu > _EPS
                min1r = np.minimum(1.0, r_pos)
                denom = delta_vi * min1r[:, None]
                alpha_i = np.full_like(delta_vi, np.inf)
                np.divide(allowed, denom, out=alpha_i,
                          where=np.abs(denom) > _EPS)
                alpha_i = np.where(alpha_i > 0.0, alpha_i, 0.0)

                base_i = np.full_like(delta_vi, np.inf)
                np.divide(allowed, delta_vi, out=base_i,
                          where=np.abs(delta_vi) > _EPS)
                base_i = np.where(base_i > 0.0, base_i, 0.0)
                # alpha*r = allowed*r/(Δφ_Vi*min(1,r)).  For r <= 1 this
                # reduces exactly to allowed/Δφ_Vi, avoiding inf*0 at r=0.
                with np.errstate(over='ignore', invalid='ignore'):
                    alpha_r_i = np.where(r_pos[:, None] <= 1.0,
                                         base_i,
                                         base_i * r_pos[:, None])
                psi_tvd_cap = np.nan_to_num(psi_tvd, nan=0.0,
                                            posinf=2.0, neginf=0.0)
                psi_tvd_cap = np.clip(psi_tvd_cap, 0.0, 2.0)
                bounded_vertex_lmp = (
                    self.tmlpu_bound_tvd_separate
                    or
                    self._tvd_name in ('bounded_cd', 'central', 'cd',
                                       'pure_downwind')
                    or (self.euler_velocity_extrema_lmp and var in (1, 2))
                    or (self.euler_density_extrema_lmp and var == 0))
                if bounded_vertex_lmp:
                    # These branches are bounded by the multidimensional
                    # vertex LMP; the 1-D Sweby r gate is directionally biased
                    # at rotating smooth extrema and thin gaps.
                    psi_i = np.minimum(base_i, psi_tvd_cap[:, None])
                else:
                    psi_i = np.minimum(alpha_r_i, alpha_i)
                    psi_i = np.minimum(psi_i, psi_tvd_cap[:, None])
                    psi_i = np.where(monotone[:, None], psi_i, 0.0)
                if (self.vertex_mlp_augment
                        and vertex_grad_x_values is not None
                        and vertex_grad_y_values is not None):
                    avg_proj = (
                        vertex_grad_x_values[var, node_safe] * d_vi[:, :, 0]
                        + vertex_grad_y_values[var, node_safe] * d_vi[:, :, 1])
                    augment_monotone = (
                        np.ones_like(monotone, dtype=bool)
                        if bounded_vertex_lmp
                        else monotone)
                    smooth_extremum = (
                        (avg_proj * delta_vi > 0.0)
                        & (np.abs(avg_proj) >= 0.5 * np.abs(delta_vi))
                        & node_valid
                        & augment_monotone[:, None]
                        & (np.abs(allowed)
                           <= _EPS * (1.0 + np.abs(phi_L[:, None])
                                      + np.abs(vmin) + np.abs(vmax))))
                    psi_i = np.where(smooth_extremum, 1.0, psi_i)
                psi_i = np.where(node_valid, psi_i, psi_tvd_cap[:, None])
                psi_i = np.nan_to_num(psi_i, nan=0.0,
                                      posinf=2.0, neginf=0.0)
                return np.min(psi_i, axis=1)

            cell_nodes = cell_node_safe[L_idx]
            cell_valid = cell_node_valid[L_idx]
            psi_cell = _psi_for_nodes(cell_nodes, cell_valid,
                                      vertex_offsets[L_idx])
            if not (self.vertex_mlp_face_local
                    and face_node_safe is not None
                    and face_node_valid is not None
                    and L_idx.shape[0] == interior.shape[0]):
                return psi_cell

            face_offsets = (mesh.nodes[face_node_safe]
                            - mesh.cell_centers[L_idx, None, :])
            psi_face = _psi_for_nodes(face_node_safe, face_node_valid,
                                      face_offsets)
            if not self.vertex_mlp_face_local_otsu:
                return psi_face

            mode = str(self.vertex_mlp_face_local_otsu_mode).lower()
            if mode == 'range':
                selector = phi_max_cell[var, L_idx] - phi_min_cell[var, L_idx]
            elif mode == 'value':
                selector = phi_max_cell[var, L_idx]
            else:
                raise ValueError(
                    "vertex_mlp_face_local_otsu_mode must be 'range' "
                    f"or 'value', got {self.vertex_mlp_face_local_otsu_mode!r}")
            finite = np.isfinite(selector)
            scale = max(float(np.max(np.abs(selector[finite]))), 1.0) if np.any(finite) else 1.0
            values = np.sort(selector[finite & (
                selector > 64.0 * np.finfo(float).eps * scale)])
            if values.size < 2 or values[0] == values[-1]:
                return psi_cell
            prefix = np.cumsum(values)
            total = prefix[-1]
            counts = np.arange(1, values.size)
            left_mean = prefix[:-1] / counts
            right_count = values.size - counts
            right_mean = (total - prefix[:-1]) / right_count
            between = counts * right_count * (left_mean - right_mean) ** 2
            between[values[:-1] == values[1:]] = -1.0
            idx = int(np.argmax(between))
            cutoff = 0.5 * (values[idx] + values[idx + 1])
            return np.where(selector >= cutoff, psi_face, psi_cell)

        def _weak_face_mlp_psi(var, face_ids, phi_L, delta_face, psi_tvd):
            if (vertex_min_values is None or vertex_max_values is None
                    or ctx.get('face_node_int_safe') is None
                    or ctx.get('face_node_int_valid') is None):
                return None
            local_pos = np.searchsorted(interior, face_ids)
            node_safe = ctx['face_node_int_safe'][local_pos]
            node_valid = ctx['face_node_int_valid'][local_pos]
            count = np.maximum(np.sum(node_valid, axis=1), 1)
            qmin_face = (
                np.sum(np.where(node_valid,
                                vertex_min_values[var, node_safe], 0.0),
                       axis=1) / count)
            qmax_face = (
                np.sum(np.where(node_valid,
                                vertex_max_values[var, node_safe], 0.0),
                       axis=1) / count)
            qmin_face = np.where(np.isfinite(qmin_face), qmin_face, phi_L)
            qmax_face = np.where(np.isfinite(qmax_face), qmax_face, phi_L)
            allowed = np.where(delta_face >= 0.0,
                               qmax_face - phi_L + tvb_eps,
                               qmin_face - phi_L - tvb_eps)
            eps = (64.0 * np.finfo(float).eps
                   * (1.0 + np.abs(phi_L) + np.abs(qmin_face)
                      + np.abs(qmax_face) + np.abs(delta_face)))
            psi_bound = np.ones_like(delta_face)
            active_pos = delta_face > eps
            active_neg = delta_face < -eps
            psi_bound = np.where(
                active_pos,
                allowed / np.maximum(delta_face, eps),
                psi_bound)
            psi_bound = np.where(
                active_neg,
                allowed / np.minimum(delta_face, -eps),
                psi_bound)
            psi_bound = np.where(psi_bound > 0.0, psi_bound, 0.0)
            psi_cap = np.nan_to_num(psi_tvd, nan=0.0,
                                    posinf=2.0, neginf=0.0)
            psi_cap = np.clip(psi_cap, 0.0, 2.0)
            return np.minimum(np.clip(psi_bound, 0.0, 2.0), psi_cap)

        def _density_lsq_contact_weight():
            w_density = np.clip(density_contact_weight, 0.0, 1.0)
            blend = float(self.euler_density_contact_lsq_root_blend)
            floor_strength = float(self.euler_density_contact_lsq_shear_floor)
            clean = None
            if blend > 0.0:
                clean = (
                    (1.0 - np.clip(pressure_flatten, 0.0, 1.0)) ** 2
                    * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)) ** 2
                    * (1.0 - np.clip(density_flatten, 0.0, 1.0)))
                w_root = np.sqrt(w_density)
                w_density = w_density + blend * clean * (
                    w_root - w_density)
                cap = float(self.euler_density_contact_lsq_root_blend_cap)
                if cap < 1.0:
                    w_density = np.minimum(w_density, cap)
            if floor_strength > 0.0:
                if clean is None:
                    clean = (
                        (1.0 - np.clip(pressure_flatten, 0.0, 1.0)) ** 2
                        * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)) ** 2
                        * (1.0 - np.clip(density_flatten, 0.0, 1.0)))
                shear = np.clip(tangential_contact_weight, 0.0, 1.0)
                shear_only = (
                    np.clip((shear - 0.55) / 0.25, 0.0, 1.0)
                    * np.clip((0.45 - w_density) / 0.25, 0.0, 1.0))
                rho_o = np.maximum(W_recon_cell[0, o_idx], _EPS)
                rho_n = np.maximum(W_recon_cell[0, n_idx], _EPS)
                rho_ratio = (
                    np.minimum(rho_o, rho_n) / np.maximum(rho_o, rho_n))
                p_o = np.maximum(W_recon_cell[3, o_idx], _EPS)
                p_n = np.maximum(W_recon_cell[3, n_idx], _EPS)
                p_ratio = (
                    np.minimum(p_o, p_n) / np.maximum(p_o, p_n))
                guard = (
                    np.clip((rho_ratio - 0.40) / 0.30, 0.0, 1.0)
                    * np.clip((p_ratio - 0.45) / 0.30, 0.0, 1.0))
                floor = floor_strength * shear_only * clean * guard
                w_density = np.maximum(w_density, floor)
                cap = float(self.euler_density_contact_lsq_shear_floor_cap)
                if cap < 1.0:
                    w_density = np.minimum(w_density, cap)
            return np.clip(w_density, 0.0, 1.0)

        def _density_contact_weak_relax_weight():
            shock_power = max(
                float(self.euler_density_contact_weak_face_shock_power),
                1.0)
            shock_gate = (
                (1.0 - np.clip(pressure_flatten, 0.0, 1.0))
                * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)))
            if shock_power != 1.0:
                shock_gate = shock_gate ** shock_power
            w0 = np.clip(density_contact_weight, 0.0, 1.0)
            blend = float(self.euler_density_contact_weak_face_root_blend)
            if blend > 0.0:
                shear = np.clip(tangential_contact_weight, 0.0, 1.0)
                clean = (
                    (1.0 - np.clip(pressure_flatten, 0.0, 1.0)) ** 2
                    * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)) ** 2
                    * (1.0 - np.clip(density_flatten, 0.0, 1.0)))
                rho_o = np.maximum(W_recon_cell[0, o_idx], _EPS)
                rho_n = np.maximum(W_recon_cell[0, n_idx], _EPS)
                rho_ratio = (
                    np.minimum(rho_o, rho_n) / np.maximum(rho_o, rho_n))
                p_o = np.maximum(W_recon_cell[3, o_idx], _EPS)
                p_n = np.maximum(W_recon_cell[3, n_idx], _EPS)
                p_ratio = (
                    np.minimum(p_o, p_n) / np.maximum(p_o, p_n))
                rho_guard = np.clip((rho_ratio - 0.30) / 0.35, 0.0, 1.0)
                p_guard = np.clip((p_ratio - 0.35) / 0.35, 0.0, 1.0)
                contact_core = np.clip((w0 * shear - 0.08) / 0.32,
                                       0.0, 1.0)
                w0 = w0 + (
                    blend * contact_core * clean * rho_guard * p_guard
                    * (np.sqrt(w0) - w0))
            relax_w = w0 * shock_gate
            swirl_extra = float(
                self.euler_density_contact_weak_face_swirl_extra)
            if swirl_extra > 0.0:
                du_dx = 0.5 * (coeffs[1, o_idx, 0] + coeffs[1, n_idx, 0])
                du_dy = 0.5 * (coeffs[1, o_idx, 1] + coeffs[1, n_idx, 1])
                dv_dx = 0.5 * (coeffs[2, o_idx, 0] + coeffs[2, n_idx, 0])
                dv_dy = 0.5 * (coeffs[2, o_idx, 1] + coeffs[2, n_idx, 1])
                omega12 = 0.5 * (dv_dx - du_dy)
                strain12 = 0.5 * (du_dy + dv_dx)
                omega_norm2 = 2.0 * omega12 * omega12
                strain_norm2 = (
                    du_dx * du_dx + dv_dy * dv_dy
                    + 2.0 * strain12 * strain12)
                q_crit = 0.5 * (omega_norm2 - strain_norm2)
                trace = du_dx + dv_dy
                det = du_dx * dv_dy - du_dy * dv_dx
                lambda_ci_like = np.sqrt(
                    np.maximum(4.0 * det - trace * trace, 0.0))
                grad_norm = np.sqrt(
                    du_dx * du_dx + du_dy * du_dy
                    + dv_dx * dv_dx + dv_dy * dv_dy) + _EPS
                q_gate = np.clip(
                    (q_crit / (grad_norm * grad_norm) - 0.03) / 0.17,
                    0.0, 1.0)
                ci_gate = np.clip(
                    (lambda_ci_like / grad_norm - 0.04) / 0.16,
                    0.0, 1.0)
                clean = (
                    (1.0 - np.clip(pressure_flatten, 0.0, 1.0)) ** 2
                    * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)) ** 2
                    * (1.0 - np.clip(density_flatten, 0.0, 1.0)))
                rho_o = np.maximum(W_recon_cell[0, o_idx], _EPS)
                rho_n = np.maximum(W_recon_cell[0, n_idx], _EPS)
                rho_ratio = (
                    np.minimum(rho_o, rho_n) / np.maximum(rho_o, rho_n))
                p_o = np.maximum(W_recon_cell[3, o_idx], _EPS)
                p_n = np.maximum(W_recon_cell[3, n_idx], _EPS)
                p_ratio = (
                    np.minimum(p_o, p_n) / np.maximum(p_o, p_n))
                safe_state = (
                    np.clip((rho_ratio - 0.45) / 0.30, 0.0, 1.0)
                    * np.clip((p_ratio - 0.50) / 0.30, 0.0, 1.0))
                low_contact_gap = np.clip((0.55 - relax_w) / 0.55,
                                          0.0, 1.0)
                relax_w = relax_w + (
                    swirl_extra * q_gate * ci_gate * clean * safe_state
                    * low_contact_gap)
            return np.minimum(
                relax_w,
                max(float(self.euler_density_contact_weak_face_mlp_cap),
                    0.0))

        def _density_contact_weak_legacy_relax_weight():
            relax_w = (
                np.clip(density_contact_weight, 0.0, 1.0)
                * (1.0 - np.clip(pressure_flatten, 0.0, 1.0))
                * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)))
            cap = max(
                float(self.euler_density_contact_weak_face_legacy_relax_cap),
                0.0)
            return np.clip(relax_w, 0.0, cap)

        def _density_contact_weak_legacy_final_safety(
                psi_relaxed, psi_tvd):
            psi_out = np.clip(psi_relaxed, 0.0, 2.0)
            if self.euler_density_contact_weak_face_legacy_tvd_after_weak:
                psi_cap = np.nan_to_num(psi_tvd, nan=0.0,
                                        posinf=2.0, neginf=0.0)
                psi_out = np.minimum(psi_out, np.clip(psi_cap, 0.0, 2.0))
            return psi_out

        def _density_contact_weak_admissibility_damp(
                relax_w, psi_base, psi_weak, delta, phi_u, p_face):
            if not self.euler_density_contact_weak_face_admissibility_damp:
                return relax_w
            strength = np.clip(
                float(
                    self.euler_density_contact_weak_face_admissibility_strength),
                0.0, 1.0)
            if strength <= 0.0:
                return relax_w
            rho_floor_factor = max(
                float(self.euler_density_contact_weak_face_rho_floor), 0.0)
            p_floor_factor = max(
                float(self.euler_density_contact_weak_face_p_floor), 0.0)
            if rho_floor_factor <= 0.0 and p_floor_factor <= 0.0:
                return relax_w

            rho_o = np.maximum(W_cell[0, o_idx], _EPS)
            rho_n = np.maximum(W_cell[0, n_idx], _EPS)
            p_o = np.maximum(W_cell[3, o_idx], _EPS)
            p_n = np.maximum(W_cell[3, n_idx], _EPS)
            rho_floor = rho_floor_factor * np.minimum(rho_o, rho_n)
            p_floor = p_floor_factor * np.minimum(p_o, p_n)

            psi_candidate = psi_base + relax_w * (psi_weak - psi_base)
            rho_base = phi_u + psi_base * delta
            rho_candidate = phi_u + psi_candidate * delta
            rho_guard = np.ones_like(relax_w)
            if rho_floor_factor > 0.0:
                worsening = rho_candidate < rho_base
                below = rho_candidate < rho_floor
                denom = rho_base - rho_candidate
                allowable = np.where(
                    np.abs(denom) > _EPS,
                    (rho_base - rho_floor) / denom,
                    0.0)
                rho_guard = np.where(
                    below & worsening,
                    np.clip(allowable, 0.0, 1.0),
                    1.0)

            p_guard = np.ones_like(relax_w)
            if p_floor_factor > 0.0:
                p_safe = np.maximum(p_floor, _EPS)
                p_guard = np.clip(
                    (np.maximum(p_face, 0.0) - 0.5 * p_safe)
                    / (0.5 * p_safe + _EPS),
                    0.0, 1.0)

            guard = np.minimum(rho_guard, p_guard)
            if self.euler_density_contact_weak_face_admissibility_shear_protect:
                clean = (
                    (1.0 - np.clip(pressure_flatten, 0.0, 1.0)) ** 2
                    * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)) ** 2
                    * (1.0 - np.clip(density_flatten, 0.0, 1.0)))
                shear_gate = np.clip(
                    (tangential_contact_weight - 0.60) / 0.22,
                    0.0, 1.0)
                contact_gate = np.clip(
                    (density_contact_weight - 0.20) / 0.35,
                    0.0, 1.0)
                protect = clean * shear_gate * contact_gate
                guard = guard + protect * (1.0 - guard)
            return relax_w * (1.0 - strength * (1.0 - guard))

        def _density_contact_weak_entropy_accept(
                relax_w, psi_base, psi_weak, delta, phi_u, p_face, side_idx):
            if not self.euler_density_contact_weak_face_entropy_accept:
                return relax_w
            eps_accept = max(
                float(
                    self.euler_density_contact_weak_face_entropy_accept_eps),
                0.0)
            reject_scale = np.clip(
                float(
                    self.euler_density_contact_weak_face_entropy_reject_scale),
                0.0, 1.0)
            if reject_scale >= 1.0:
                return relax_w
            gamma_entropy = float(getattr(eq, 'gamma', 1.4))
            rho_base = np.maximum(phi_u + psi_base * delta, _EPS)
            psi_candidate = psi_base + relax_w * (psi_weak - psi_base)
            rho_candidate = np.maximum(
                phi_u + psi_candidate * delta, _EPS)
            p_safe = np.maximum(p_face, _EPS)
            rho_ref = np.maximum(W_recon_cell[0, side_idx], _EPS)
            p_ref = np.maximum(W_recon_cell[3, side_idx], _EPS)
            s_ref = np.log(p_ref) - gamma_entropy * np.log(rho_ref)
            base_res = np.abs(
                np.log(p_safe) - gamma_entropy * np.log(rho_base) - s_ref)
            cand_res = np.abs(
                np.log(p_safe)
                - gamma_entropy * np.log(rho_candidate)
                - s_ref)
            reject = cand_res > base_res * (1.0 + eps_accept)
            return np.where(reject, relax_w * reject_scale, relax_w)

        def _density_contact_weak_shock_gate(relax_w):
            if not self.euler_density_contact_weak_face_shock_gate:
                return relax_w
            strength = np.clip(
                float(
                    self.euler_density_contact_weak_face_shock_gate_strength),
                0.0, 1.0)
            if strength <= 0.0:
                return relax_w
            floor = np.clip(
                float(
                    self.euler_density_contact_weak_face_shock_gate_floor),
                0.0, 1.0)
            mode = str(
                self.euler_density_contact_weak_face_shock_gate_mode).lower().strip()
            if mode == '':
                mode = 'wide'
            p_threshold = float(
                self.euler_density_contact_weak_face_shock_gate_p_threshold)
            p_width = float(
                self.euler_density_contact_weak_face_shock_gate_p_width)
            c_threshold = float(
                self.euler_density_contact_weak_face_shock_gate_compression_threshold)
            c_width = float(
                self.euler_density_contact_weak_face_shock_gate_compression_width)
            n_threshold = float(
                self.euler_density_contact_weak_face_shock_gate_normality_threshold)
            n_width = float(
                self.euler_density_contact_weak_face_shock_gate_normality_width)
            s_threshold = float(
                self.euler_density_contact_weak_face_shock_gate_shear_threshold)
            s_width = float(
                self.euler_density_contact_weak_face_shock_gate_shear_width)
            contact_threshold = float(
                self.euler_density_contact_weak_face_shock_gate_contact_threshold)
            contact_width = float(
                self.euler_density_contact_weak_face_shock_gate_contact_width)
            if mode not in ('wide', 'core'):
                mode = 'wide'
            p_width = abs(p_width)
            c_width = abs(c_width)
            n_width = abs(n_width)
            s_width = abs(s_width)
            contact_width = abs(contact_width)
            if (p_width < 1.0e-30 or c_width < 1.0e-30
                    or n_width < 1.0e-30 or s_width < 1.0e-30
                    or contact_width < 1.0e-30):
                return relax_w
            p_gate = np.clip((pressure_jump - p_threshold) / p_width, 0.0, 1.0)
            c_gate = np.clip((compression - c_threshold) / c_width, 0.0, 1.0)
            n_gate = np.clip(
                (normality - n_threshold) / n_width, 0.0, 1.0)
            shear_fraction = 1.0 - np.clip(normality, 0.0, 1.0)
            shear_gate = np.clip(
                (shear_fraction - s_threshold) / s_width, 0.0, 1.0)
            contact_gate = np.clip(
                (density_contact_weight - contact_threshold) / contact_width,
                0.0, 1.0)
            if mode == 'core':
                shock_core = p_gate * c_gate * n_gate
                pure_contact = shear_gate * contact_gate
                damp = shock_core * (1.0 - pure_contact)
            else:
                shock_like = np.maximum(p_gate * n_gate, c_gate * n_gate)
                pure_shear_contact = shear_gate * contact_gate * (1.0 - p_gate)
                damp = shock_like * (1.0 - pure_shear_contact)
            return relax_w * np.maximum(floor, 1.0 - strength * damp)

        def _euler_pressure_contact_entropy_blend(side_recon):
            if not self.euler_pressure_contact_entropy_blend:
                return
            if not side_recon.size:
                return
            if 3 not in active_vars:
                return
            p_face = np.maximum(side_recon[3, interior], _EPS)
            p_base = p_face.copy()
            rho_face = np.maximum(side_recon[0, interior], _EPS)
            gamma = float(getattr(eq, 'gamma', 1.4))
            rho_o = np.maximum(W_recon_cell[0, o_idx], _EPS)
            rho_n = np.maximum(W_recon_cell[0, n_idx], _EPS)
            p_o = np.maximum(W_recon_cell[3, o_idx], _EPS)
            p_n = np.maximum(W_recon_cell[3, n_idx], _EPS)
            s_o = np.log(p_o) - gamma * np.log(rho_o)
            s_n = np.log(p_n) - gamma * np.log(rho_n)
            s_bar = 0.5 * (s_o + s_n)
            p_entropy = np.exp(s_bar + gamma * np.log(rho_face))
            p_entropy = np.maximum(p_entropy, _EPS)
            contact = np.clip(density_contact_weight, 0.0, 1.0)
            shear = np.clip(tangential_contact_weight, 0.0, 1.0)
            p_threshold = float(
                self.euler_pressure_contact_entropy_p_jump_threshold)
            p_width = float(
                self.euler_pressure_contact_entropy_p_jump_width)
            c_threshold = float(
                self.euler_pressure_contact_entropy_compression_threshold)
            c_width = float(
                self.euler_pressure_contact_entropy_compression_width)
            n_threshold = float(
                self.euler_pressure_contact_entropy_normality_threshold)
            n_width = float(
                self.euler_pressure_contact_entropy_normality_width)
            if n_width <= 0.0:
                n_width = 1.0
            if p_width <= 0.0:
                p_width = 1.0
            if c_width <= 0.0:
                c_width = 1.0
            p_gate = np.clip((pressure_jump - p_threshold) / p_width, 0.0, 1.0)
            c_gate = np.clip(
                (compression - c_threshold) / c_width, 0.0, 1.0)
            n_gate = np.clip(
                (normality - n_threshold) / n_width, 0.0, 1.0)
            shock_off = (1.0 - p_gate) * (1.0 - c_gate) * (1.0 - n_gate)
            clean_contact = contact * shear * shock_off
            beta = np.clip(
                float(self.euler_pressure_contact_entropy_beta), 0.0, 1.0)
            cap = np.clip(
                float(self.euler_pressure_contact_entropy_cap), 0.0, 1.0)
            w = np.minimum(cap, beta * clean_contact)
            p_base = np.where(p_base <= _EPS, _EPS, p_base)
            downscale = np.clip(
                float(self.euler_pressure_contact_entropy_downscale), 0.0, 1.0)
            w = np.where(p_entropy < p_base, w * downscale, w)
            p_new = p_base + w * (p_entropy - p_base)
            p_new = np.maximum(p_new, _EPS)
            side_recon[3, interior] = p_new

        def _euler_pressure_face_jump_limiter(W_left, W_right):
            if not self.euler_pressure_face_jump_limiter_on:
                return
            if not W_left.size or not W_right.size:
                return
            if 3 not in active_vars:
                return
            p_l = np.maximum(W_left[3, interior], _EPS)
            p_r = np.maximum(W_right[3, interior], _EPS)
            p_o = np.maximum(W_recon_cell[3, o_idx], _EPS)
            p_n = np.maximum(W_recon_cell[3, n_idx], _EPS)
            base_jump = np.abs(p_n - p_o)
            pressure_scale = np.maximum(p_n + p_o, _EPS)
            growth_cap = max(
                float(self.euler_pressure_face_jump_limiter_growth_cap), 0.0)
            abs_floor = max(
                float(self.euler_pressure_face_jump_limiter_abs_floor), 0.0)
            allowed = base_jump * (1.0 + growth_cap) + abs_floor * pressure_scale
            face_jump = np.abs(p_r - p_l)
            active = face_jump > allowed
            if not np.any(active):
                return

            p_width = abs(float(
                self.euler_pressure_face_jump_limiter_p_jump_width))
            c_width = abs(float(
                self.euler_pressure_face_jump_limiter_compression_width))
            n_width = abs(float(
                self.euler_pressure_face_jump_limiter_normality_width))
            if p_width < 1.0e-30 or c_width < 1.0e-30 or n_width < 1.0e-30:
                return
            p_gate = np.clip(
                (pressure_jump
                 - float(self.euler_pressure_face_jump_limiter_p_jump_threshold))
                / p_width, 0.0, 1.0)
            c_gate = np.clip(
                (compression
                 - float(
                     self.euler_pressure_face_jump_limiter_compression_threshold))
                / c_width, 0.0, 1.0)
            n_gate = np.clip(
                (normality
                 - float(
                     self.euler_pressure_face_jump_limiter_normality_threshold))
                / n_width, 0.0, 1.0)
            shock_off = (1.0 - p_gate) * (1.0 - c_gate) * (1.0 - n_gate)
            clean_shear_contact = (
                np.clip(density_contact_weight, 0.0, 1.0)
                * np.clip(tangential_contact_weight, 0.0, 1.0)
                * shock_off)
            strength = np.clip(
                float(self.euler_pressure_face_jump_limiter_strength), 0.0, 1.0)
            w = np.where(active, strength * clean_shear_contact, 0.0)
            if not np.any(w > 0.0):
                return

            mid = 0.5 * (p_l + p_r)
            half_allowed = 0.5 * allowed
            sign_l = np.where(p_l >= p_r, 1.0, -1.0)
            target_l = np.maximum(mid + sign_l * half_allowed, _EPS)
            target_r = np.maximum(mid - sign_l * half_allowed, _EPS)
            W_left[3, interior] = p_l + w * (target_l - p_l)
            W_right[3, interior] = p_r + w * (target_r - p_r)

        def _density_contact_weak_face_value_scale(
                psi_base, psi_relaxed, delta, phi_u,
                p_face, side_idx):
            del side_idx
            if not self.euler_density_contact_weak_face_value_scaling:
                return psi_relaxed

            rho_floor_factor = max(
                float(self.euler_density_contact_weak_face_rho_floor_factor),
                0.0)
            p_floor_factor = max(
                float(
                    self.euler_density_contact_weak_face_value_scaling_p_floor_factor),
                0.0)
            hard_rho_floor_factor = max(
                float(self.euler_density_contact_weak_face_hard_rho_floor_factor),
                0.0)
            hard_p_floor_factor = max(
                float(self.euler_density_contact_weak_face_hard_p_floor_factor),
                0.0)
            mode = str(
                self.euler_density_contact_weak_face_value_scaling_mode).lower().strip()
            if mode == '':
                mode = 'global_floor'

            if mode == 'global_floor':
                if rho_floor_factor <= 0.0:
                    return psi_relaxed
                theta_floor = np.clip(
                    float(self.euler_density_contact_weak_face_theta_floor),
                    0.0, 1.0)
                rho_floor = rho_floor_factor * np.minimum(
                    np.maximum(W_cell[0, o_idx], _EPS),
                    np.maximum(W_cell[0, n_idx], _EPS))
                rho_base = phi_u + psi_base * delta
                rho_candidate = phi_u + psi_relaxed * delta
                needs_scale = (
                    (rho_candidate < rho_base)
                    & (rho_candidate < rho_floor))
                if not np.any(needs_scale):
                    return psi_relaxed

                denom = rho_base - rho_candidate
                denom = np.where(np.abs(denom) > _EPS, denom, np.copysign(_EPS, denom))
                theta = np.where(
                    needs_scale,
                    np.clip((rho_base - rho_floor) / denom, 0.0, 1.0),
                    1.0)
                theta = np.clip(theta, 0.0, 1.0)
                if theta_floor > 0.0:
                    theta = np.where(theta > theta_floor, theta_floor, theta)

                rho_final = rho_base + theta * (rho_candidate - rho_base)
                rho_final = np.where(
                    needs_scale,
                    np.maximum(rho_final, rho_floor),
                    rho_candidate)
                with np.errstate(divide='ignore', invalid='ignore'):
                    psi_scaled = np.where(
                        np.abs(delta) > _EPS,
                        (rho_final - phi_u) / delta,
                        psi_base)
                return np.where(
                    needs_scale,
                    np.maximum(psi_base, psi_scaled),
                    np.maximum(psi_base, psi_relaxed))

            if mode in ('shear_floor_blend', 'clean_shear_floor_blend'):
                if rho_floor_factor <= 0.0:
                    return psi_relaxed
                alpha = np.clip(
                    float(
                        self.euler_density_contact_weak_face_value_scaling_shear_blend_alpha),
                    0.0, 1.0)
                if alpha <= 0.0:
                    return psi_relaxed
                bound_pad = max(
                    float(
                        self.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad),
                    0.0)

                rho_o = np.maximum(W_cell[0, o_idx], _EPS)
                rho_n = np.maximum(W_cell[0, n_idx], _EPS)
                rho_floor = rho_floor_factor * np.minimum(rho_o, rho_n)
                rho_v13 = phi_u + psi_relaxed * delta
                rho_base = phi_u + psi_base * delta
                denom = rho_base - rho_v13
                needs_scale = (
                    (rho_v13 < rho_base)
                    & (rho_v13 < rho_floor))
                denom_safe = np.where(np.abs(denom) > _EPS, denom, 1.0)
                theta = np.where(
                    needs_scale,
                    np.clip((rho_base - rho_floor) / denom_safe, 0.0, 1.0),
                    1.0)
                rho_floor_candidate = np.where(
                    needs_scale,
                    np.maximum(
                        rho_base + theta * (rho_v13 - rho_base),
                        rho_floor),
                    rho_v13)

                rho_avg = 0.5 * (rho_o + rho_n)
                rho_span = np.maximum(np.maximum(rho_o, rho_n), 1.0)
                lo = np.minimum(rho_o, rho_n) - bound_pad * rho_span
                hi = np.maximum(rho_o, rho_n) + bound_pad * rho_span
                contrast_v13 = rho_v13 - rho_avg
                contrast_floor = rho_floor_candidate - rho_avg
                useful = (
                    (np.abs(contrast_floor) > np.abs(contrast_v13))
                    & (contrast_floor * contrast_v13 >= 0.0)
                    & (rho_floor_candidate >= lo)
                    & (rho_floor_candidate <= hi))

                shear_fraction = 1.0 - np.clip(normality, 0.0, 1.0)
                contact_gate = np.clip(
                    (density_contact_weight - 0.25) / 0.30, 0.0, 1.0)
                shear_gate = np.clip(
                    (shear_fraction - 0.68) / 0.20, 0.0, 1.0)
                p_gate = np.clip(
                    (pressure_jump - 0.035) / 0.050, 0.0, 1.0)
                c_gate = np.clip(
                    (compression - 0.010) / 0.035, 0.0, 1.0)
                n_gate = np.clip(
                    (normality - 0.45) / 0.25, 0.0, 1.0)
                clean = (
                    contact_gate * shear_gate
                    * (1.0 - p_gate) * (1.0 - c_gate) * (1.0 - n_gate))
                w = alpha * clean * useful
                rho_final = rho_v13 + w * (rho_floor_candidate - rho_v13)
                with np.errstate(divide='ignore', invalid='ignore'):
                    psi_blend = np.where(
                        np.abs(delta) > _EPS,
                        (rho_final - phi_u) / delta,
                        psi_relaxed)
                return np.where(w > 0.0, psi_blend, psi_relaxed)

            if mode in (
                'clean_shear_micro_restore',
                'density_mlp_micro_restore',
                'coherent_shear_micro_restore',
                'density_mlp_coherent_restore',
                'contour_continuity_micro_restore',
                'density_contour_continuity_micro_restore'
            ):
                alpha = np.clip(
                    float(
                        self.euler_density_contact_weak_face_value_scaling_shear_blend_alpha),
                    0.0, 1.0)
                if alpha <= 0.0:
                    return psi_relaxed
                bound_pad = max(
                    float(
                        self.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad),
                    0.0)

                rho_o = np.maximum(W_cell[0, o_idx], _EPS)
                rho_n = np.maximum(W_cell[0, n_idx], _EPS)
                rho_off = phi_u + psi_base * delta
                rho_on = phi_u + psi_relaxed * delta

                def _smoothstep(lo, hi, x):
                    width = max(float(hi - lo), _EPS)
                    t = np.clip((x - lo) / width, 0.0, 1.0)
                    return t * t * (3.0 - 2.0 * t)

                shear_fraction = 1.0 - np.clip(normality, 0.0, 1.0)
                if mode in ('clean_shear_micro_restore',
                            'density_mlp_micro_restore'):
                    contact_gate = _smoothstep(
                        0.25, 0.55, np.clip(density_contact_weight, 0.0, 1.0))
                    shear_gate = _smoothstep(0.65, 0.85, shear_fraction)
                    p_gate = _smoothstep(0.035, 0.085, pressure_jump)
                    c_gate = _smoothstep(0.010, 0.050, compression)
                    n_gate = _smoothstep(0.45, 0.70, normality)
                    gate = (
                        contact_gate * shear_gate
                        * (1.0 - p_gate) * (1.0 - c_gate) * (1.0 - n_gate))
                elif mode in (
                    'contour_continuity_micro_restore',
                    'density_contour_continuity_micro_restore'
                ):
                    contact_gate = _smoothstep(
                        0.30, 0.60, np.clip(density_contact_weight, 0.0, 1.0))
                    shear_gate = _smoothstep(0.70, 0.90, shear_fraction)
                    p_gate = _smoothstep(0.030, 0.075, pressure_jump)
                    c_gate = _smoothstep(0.008, 0.040, compression)
                    n_gate = _smoothstep(0.40, 0.65, normality)
                    shock_off = (
                        (1.0 - p_gate) * (1.0 - c_gate) * (1.0 - n_gate))
                    contour_on = bool(
                        self.euler_density_contact_weak_face_contour_continuity_on)
                    if contour_on:
                        nx = face_n_o[:, 0]
                        ny = face_n_o[:, 1]
                        tx = -ny
                        ty = nx
                        grad_rho = 0.5 * (
                            coeffs[0, o_idx, :2] + coeffs[0, n_idx, :2]
                        )
                        contour_x = -grad_rho[:, 1]
                        contour_y = grad_rho[:, 0]
                        contour_norm = np.sqrt(
                            contour_x * contour_x + contour_y * contour_y)
                        tangent_norm = np.sqrt(tx * tx + ty * ty)
                        contour_alignment = np.where(
                            contour_norm > _EPS,
                            np.abs(tx * contour_x + ty * contour_y)
                            / np.maximum(tangent_norm * contour_norm, _EPS),
                            0.0)
                        cmin = float(
                            self.euler_density_contact_weak_face_contour_continuity_min)
                        cfull = float(
                            self.euler_density_contact_weak_face_contour_continuity_full)
                        contour_gate = _smoothstep(
                            cmin, cfull, contour_alignment)
                        grad_normal = grad_rho[:, 0] * nx + grad_rho[:, 1] * ny
                        density_delta = rho_on - rho_off
                        sign_consistent = np.where(
                            (np.abs(density_delta) > _EPS)
                            & (np.abs(grad_normal) > _EPS)
                            & (np.sign(density_delta)
                               == np.sign(grad_normal)),
                            1.0, 0.0)
                    else:
                        contour_gate = 1.0
                        sign_consistent = 1.0
                    gate = (
                        contact_gate * shear_gate * shock_off
                        * contour_gate * sign_consistent)
                else:
                    contact_gate = _smoothstep(
                        0.30, 0.60, np.clip(density_contact_weight, 0.0, 1.0))
                    shear_gate = _smoothstep(0.70, 0.90, shear_fraction)
                    p_gate = _smoothstep(0.030, 0.075, pressure_jump)
                    c_gate = _smoothstep(0.008, 0.040, compression)
                    n_gate = _smoothstep(0.40, 0.65, normality)
                    coherent_shear_support = (
                        _smoothstep(0.03, 0.10, shear_fraction)
                        * np.where(
                            (np.sign(rho_n - rho_o) == 0.0)
                            | (np.sign(rho_on - rho_off) == 0.0)
                            | (np.sign(rho_n - rho_o) * np.sign(rho_on - rho_off) > 0.0),
                            1.0, 0.0)
                    )
                    if not bool(
                        self.euler_density_contact_weak_face_value_scaling_require_coherent_shear
                    ):
                        coherent_shear_support = 1.0
                    shock_off = (
                        (1.0 - p_gate) * (1.0 - c_gate) * (1.0 - n_gate))
                    local_roughness = np.maximum(
                        np.maximum(rho_on - np.maximum(rho_o, rho_n), _EPS),
                        np.maximum(np.minimum(rho_o, rho_n) - rho_on, _EPS))
                    local_span = np.maximum(np.abs(rho_n - rho_o), _EPS)
                    local_density_roughness = np.clip(
                        local_roughness / local_span, 0.0, 1.0)
                    artifact_gate = 1.0 - _smoothstep(
                        0.06, 0.14, local_density_roughness)
                    if not bool(
                        self.euler_density_contact_weak_face_value_scaling_artifact_reject
                    ):
                        artifact_gate = 1.0
                    gate = (
                        contact_gate * shear_gate * shock_off
                        * coherent_shear_support * artifact_gate)

                gate = np.clip(gate, 0.0, 1.0)

                rho_avg = 0.5 * (rho_o + rho_n)
                p_o = np.maximum(W_cell[3, o_idx], _EPS)
                p_n = np.maximum(W_cell[3, n_idx], _EPS)
                c_o = np.sqrt(
                    np.maximum(float(getattr(eq, 'gamma', 1.4))
                               * p_o / rho_o, _EPS))
                c_n = np.sqrt(
                    np.maximum(float(getattr(eq, 'gamma', 1.4))
                               * p_n / rho_n, _EPS))
                c_sum = np.maximum(c_o + c_n, _EPS)
                pad = bound_pad * np.maximum(1.0, rho_avg)
                lo = np.minimum(rho_o, rho_n) - pad
                hi = np.maximum(rho_o, rho_n) + pad
                density_delta = rho_on - rho_off
                increment_cap = max(
                    float(
                        self.euler_density_contact_weak_face_density_increment_cap),
                    0.0)
                if increment_cap > 0.0:
                    density_delta = np.clip(
                        density_delta,
                        -increment_cap * np.maximum(1.0, rho_avg),
                        increment_cap * np.maximum(1.0, rho_avg))
                rho_micro = rho_off + alpha * gate * density_delta
                rho_final = np.clip(rho_micro, lo, hi)
                rho_final = np.maximum(rho_final, _EPS)

                stream_on = bool(
                    self.euler_density_contact_weak_face_stream_coherence_on)
                if stream_on:
                    u_o = W_cell[1, o_idx]
                    v_o = W_cell[2, o_idx]
                    u_n = W_cell[1, n_idx]
                    v_n = W_cell[2, n_idx]
                    u_avg = 0.5 * (u_o + u_n)
                    v_avg = 0.5 * (v_o + v_n)
                    vel_mag = np.sqrt(u_avg * u_avg + v_avg * v_avg)
                    vel_mag_safe = np.where(vel_mag > _EPS, vel_mag, 1.0)
                    tx = -face_n_o[:, 1]
                    ty = face_n_o[:, 0]
                    e_sx = np.where(
                        vel_mag > _EPS, u_avg / vel_mag_safe, tx)
                    e_sy = np.where(
                        vel_mag > _EPS, v_avg / vel_mag_safe, ty)
                    grad_rho = 0.5 * (
                        coeffs[0, o_idx, :2] + coeffs[0, n_idx, :2]
                    )
                    grad_parallel = (
                        grad_rho[:, 0] * e_sx + grad_rho[:, 1] * e_sy)
                    grad_norm = (
                        np.abs(grad_rho[:, 0]) + np.abs(grad_rho[:, 1]))
                    stream_frac = np.where(
                        grad_norm > _EPS,
                        np.abs(grad_parallel) / np.maximum(grad_norm, _EPS),
                        0.0)
                    stream_min = float(
                        self.euler_density_contact_weak_face_stream_coherence_min)
                    stream_full = float(
                        self.euler_density_contact_weak_face_stream_coherence_full)
                    width = max(stream_full - stream_min, _EPS)
                    t = np.clip((stream_frac - stream_min) / width, 0.0, 1.0)
                    stream_gate = t * t * (3.0 - 2.0 * t)
                    downstream_gate = gate * stream_gate
                    downstream_beta = np.clip(
                        float(
                            self.euler_density_contact_weak_face_downstream_rho_beta),
                        0.0, 1.0)
                    downstream_cap = max(
                        float(
                            self.euler_density_contact_weak_face_downstream_rho_cap),
                        0.0)
                    downstream_wave_cap = max(
                        float(
                            self.euler_density_contact_weak_face_downstream_rho_wave_cap),
                        0.0) * c_sum
                    delta_cap = np.maximum(downstream_cap, downstream_wave_cap)
                    if downstream_beta > 0.0:
                        d_rho = (
                            downstream_beta * downstream_gate
                            * (rho_on - rho_final))
                        d_rho = np.where(
                            delta_cap > _EPS,
                            np.clip(d_rho, -delta_cap, delta_cap),
                            0.0)
                        rho_final = np.clip(rho_final + d_rho, lo, hi)
                        rho_final = np.maximum(rho_final, _EPS)

                with np.errstate(divide='ignore', invalid='ignore'):
                    psi_micro = np.where(
                        np.abs(delta) > _EPS,
                        (rho_final - phi_u) / delta,
                        psi_base)
                return np.where(gate > 0.0, psi_micro, psi_base)

            risk_width = abs(
                float(self.euler_density_contact_weak_face_value_scaling_risk_width))
            if risk_width <= 0.0:
                risk_width = _EPS

            rho_base = phi_u + psi_base * delta
            rho_candidate = phi_u + psi_relaxed * delta
            denom = rho_base - rho_candidate
            denom_safe = np.where(np.abs(denom) > _EPS, denom, 1.0)
            needs_scale = (rho_candidate < rho_base)
            rho_o = np.maximum(W_cell[0, o_idx], _EPS)
            rho_n = np.maximum(W_cell[0, n_idx], _EPS)
            p_o = np.maximum(W_cell[3, o_idx], _EPS)
            p_n = np.maximum(W_cell[3, n_idx], _EPS)
            rho_floor = rho_floor_factor * np.minimum(rho_o, rho_n)
            p_floor = p_floor_factor * np.minimum(p_o, p_n)
            hard_rho_floor = hard_rho_floor_factor * np.minimum(rho_o, rho_n)
            hard_p_floor = hard_p_floor_factor * np.minimum(p_o, p_n)
            p_safe = (
                np.maximum(np.asarray(p_face), _EPS) if p_face is not None else None)

            # Quality gate (existing v19 behavior).
            rho_def = np.clip(
                (rho_floor - np.minimum(rho_candidate, rho_floor))
                / np.maximum(rho_floor, _EPS),
                0.0, np.inf) / risk_width
            rho_def = np.clip(rho_def, 0.0, 1.0)

            if p_face is None or p_floor_factor <= 0.0:
                p_def = np.zeros_like(rho_def)
            else:
                p_floor_safe = np.maximum(p_floor, _EPS)
                p_candidate = p_safe
                p_def = np.clip(
                    (p_floor_safe - np.minimum(p_candidate, p_floor_safe))
                    / p_floor_safe,
                    0.0, np.inf) / risk_width
                p_def = np.clip(p_def, 0.0, 1.0)

            p_threshold = np.clip(
                float(self.euler_density_contact_weak_face_value_scaling_p_threshold),
                0.0, 1.0)
            p_width = abs(
                float(self.euler_density_contact_weak_face_value_scaling_p_width))
            c_threshold = np.clip(
                float(
                    self.euler_density_contact_weak_face_value_scaling_compression_threshold),
                0.0, 1.0)
            c_width = abs(
                float(
                    self.euler_density_contact_weak_face_value_scaling_compression_width))
            n_threshold = np.clip(
                float(
                    self.euler_density_contact_weak_face_value_scaling_normality_threshold),
                0.0, 1.0)
            n_width = abs(
                float(
                    self.euler_density_contact_weak_face_value_scaling_normality_width))
            contact_threshold = np.clip(
                float(
                    self.euler_density_contact_weak_face_value_scaling_contact_threshold),
                0.0, 1.0)
            contact_width = abs(
                float(
                    self.euler_density_contact_weak_face_value_scaling_contact_width))
            shear_threshold = np.clip(
                float(
                    self.euler_density_contact_weak_face_value_scaling_shear_threshold),
                0.0, 1.0)
            shear_width = abs(
                float(
                    self.euler_density_contact_weak_face_value_scaling_shear_width))
            pclean_threshold = np.clip(
                float(
                    self.euler_density_contact_weak_face_value_scaling_pressure_clean_threshold),
                0.0, 1.0)
            pclean_width = abs(
                float(
                    self.euler_density_contact_weak_face_value_scaling_pressure_clean_width))
            protect_cutoff = np.clip(
                float(
                    self.euler_density_contact_weak_face_value_scaling_hard_protect_cutoff),
                0.0, 1.0)
            if p_width < _EPS:
                p_width = 1.0
            if c_width < _EPS:
                c_width = 1.0
            if n_width < _EPS:
                n_width = 1.0
            if contact_width < _EPS:
                contact_width = 1.0
            if shear_width < _EPS:
                shear_width = 1.0
            if pclean_width < _EPS:
                pclean_width = 1.0

            p_gate = np.clip((pressure_jump - p_threshold) / p_width, 0.0, 1.0)
            c_gate = np.clip((compression - c_threshold) / c_width, 0.0, 1.0)
            n_gate = np.clip((normality - n_threshold) / n_width, 0.0, 1.0)
            shock_gate = np.maximum(p_gate * n_gate, c_gate * n_gate)
            shear_frac = 1.0 - np.clip(normality, 0.0, 1.0)
            contact_gate = np.clip((density_contact_weight - contact_threshold)
                                  / contact_width, 0.0, 1.0)
            shear_gate = np.clip(
                (shear_frac - shear_threshold) / shear_width, 0.0, 1.0)
            p_clean_gate = np.clip(
                (pressure_jump - pclean_threshold) / pclean_width,
                0.0, 1.0)
            protect = contact_gate * shear_gate * (1.0 - p_clean_gate)
            gate = np.maximum(rho_def, p_def) * shock_gate * (1.0 - protect)
            gate = np.where(protect >= protect_cutoff, 0.0, gate)
            gate = np.clip(gate, 0.0, 1.0)

            theta_floor = np.clip(
                float(self.euler_density_contact_weak_face_theta_floor),
                0.0, 1.0)
            need_qual = needs_scale & (rho_candidate < rho_floor) & (rho_floor > _EPS)
            theta_safe = np.where(
                need_qual,
                np.clip((rho_base - rho_floor) / denom_safe, 0.0, 1.0),
                1.0)
            if theta_floor > 0.0:
                theta_safe = np.where(
                    theta_safe > theta_floor, theta_floor, theta_safe)

            theta_quality = 1.0 - gate * (1.0 - theta_safe)
            theta_quality = np.clip(theta_quality, 0.0, 1.0)

            # Optional hard floor guard (full strength).
            use_hard = (
                (hard_rho_floor_factor > 0.0)
                or (hard_p_floor_factor > 0.0))
            if not use_hard:
                if not np.any(need_qual):
                    return psi_relaxed
                strength = np.clip(
                    float(
                        self.euler_density_contact_weak_face_value_scaling_strength),
                    0.0, 1.0)
                rho_quality = rho_base + theta_quality * (rho_candidate - rho_base)
                rho_final = rho_candidate + strength * (rho_quality - rho_candidate)
                rho_final = np.where(
                    need_qual,
                    np.maximum(rho_final, rho_floor),
                    rho_candidate)
            else:
                hard_rho_num = rho_base - np.minimum(hard_rho_floor, rho_base)
                hard_theta_rho = np.where(
                    need_qual & (hard_rho_floor_factor > 0.0),
                    np.clip(
                        np.divide(
                            hard_rho_num,
                            denom_safe,
                            out=np.zeros_like(denom),
                            where=np.abs(denom) > _EPS),
                        0.0, 1.0),
                    1.0)
                hard_theta_p = np.where(
                    (hard_p_floor_factor > 0.0) & (p_face is not None),
                    np.where(p_safe >= hard_p_floor, 1.0, 0.0),
                    1.0)
                theta_hard = np.minimum(hard_theta_rho, hard_theta_p)
                strength = np.clip(
                    float(
                        self.euler_density_contact_weak_face_value_scaling_strength),
                    0.0, 1.0)
                theta = theta_hard - strength * gate * (theta_hard - theta_quality)
                theta = np.where(
                    need_qual, np.clip(theta, 0.0, 1.0), 1.0)
                rho_final = rho_base + theta * (rho_candidate - rho_base)
                rho_final = np.where(
                    need_qual,
                    np.maximum(hard_rho_floor, rho_final),
                    rho_candidate)

            if not np.any(need_qual):
                return psi_relaxed
            rho_final = np.maximum(rho_final, _EPS)
            with np.errstate(divide='ignore', invalid='ignore'):
                psi_scaled = np.where(
                    np.abs(delta) > _EPS,
                    (rho_final - phi_u) / delta,
                    psi_base)
            return np.where(need_qual, psi_scaled, psi_relaxed)

        def _density_contact_weak_mlp_downstream_bridge(
                psi_base, psi_relaxed, psi_weak, delta, phi_u, side_idx):
            if not self.euler_density_contact_weak_face_stream_coherence_on:
                return psi_relaxed
            beta = np.clip(
                float(self.euler_density_contact_weak_face_downstream_rho_beta),
                0.0, 1.0)
            if beta <= 0.0:
                return psi_relaxed
            cap = max(
                float(self.euler_density_contact_weak_face_downstream_rho_cap),
                0.0)
            wave_cap = max(
                float(self.euler_density_contact_weak_face_downstream_rho_wave_cap),
                0.0)

            u_o = W_cell[1, o_idx]
            v_o = W_cell[2, o_idx]
            u_n = W_cell[1, n_idx]
            v_n = W_cell[2, n_idx]
            u_avg = 0.5 * (u_o + u_n)
            v_avg = 0.5 * (v_o + v_n)
            vel_mag = np.sqrt(u_avg * u_avg + v_avg * v_avg)
            vel_mag_safe = np.where(vel_mag > _EPS, vel_mag, 1.0)
            tx = -face_n_o[:, 1]
            ty = face_n_o[:, 0]
            e_sx = np.where(vel_mag > _EPS, u_avg / vel_mag_safe, tx)
            e_sy = np.where(vel_mag > _EPS, v_avg / vel_mag_safe, ty)

            grad_rho = 0.5 * (coeffs[0, o_idx, :2] + coeffs[0, n_idx, :2])
            grad_parallel = grad_rho[:, 0] * e_sx + grad_rho[:, 1] * e_sy
            grad_norm = np.abs(grad_rho[:, 0]) + np.abs(grad_rho[:, 1])
            stream_frac = np.where(
                grad_norm > _EPS,
                np.abs(grad_parallel) / np.maximum(grad_norm, _EPS),
                0.0)
            stream_min = float(
                self.euler_density_contact_weak_face_stream_coherence_min)
            stream_full = float(
                self.euler_density_contact_weak_face_stream_coherence_full)
            width = max(stream_full - stream_min, _EPS)
            t = np.clip((stream_frac - stream_min) / width, 0.0, 1.0)
            stream_gate = t * t * (3.0 - 2.0 * t)
            clean = (
                (1.0 - np.clip(pressure_flatten, 0.0, 1.0))
                * (1.0 - np.clip(velocity_flatten, 0.0, 1.0))
                * (1.0 - np.clip(density_flatten, 0.0, 1.0)))
            contact_gate = np.clip(density_contact_weight, 0.0, 1.0)
            gate = stream_gate * clean * contact_gate
            if not np.any(gate > 0.0):
                return psi_relaxed

            rho_base = phi_u + psi_base * delta
            rho_current = phi_u + psi_relaxed * delta
            rho_weak = phi_u + psi_weak * delta
            d_rho = beta * gate * (rho_weak - rho_current)

            rho_o = np.maximum(W_cell[0, o_idx], _EPS)
            rho_n = np.maximum(W_cell[0, n_idx], _EPS)
            p_o = np.maximum(W_cell[3, o_idx], _EPS)
            p_n = np.maximum(W_cell[3, n_idx], _EPS)
            gamma = float(getattr(eq, 'gamma', 1.4))
            c_o = np.sqrt(np.maximum(gamma * p_o / rho_o, _EPS))
            c_n = np.sqrt(np.maximum(gamma * p_n / rho_n, _EPS))
            delta_cap = np.maximum(cap, wave_cap * (c_o + c_n))
            d_rho = np.where(
                delta_cap > _EPS,
                np.clip(d_rho, -delta_cap, delta_cap),
                0.0)

            lo = np.maximum(phi_min_cell[0, side_idx], _EPS)
            hi = np.maximum(phi_max_cell[0, side_idx], lo)
            rho_next = np.clip(rho_current + d_rho, lo, hi)
            rho_next = np.where(gate > 0.0, rho_next, rho_current)
            rho_next = np.maximum(rho_next, _EPS)
            with np.errstate(divide='ignore', invalid='ignore'):
                psi_next = np.where(
                    np.abs(delta) > _EPS,
                    (rho_next - phi_u) / delta,
                    psi_base)
            return np.where(gate > 0.0, psi_next, psi_relaxed)

        for v in active_vars:
            # ---------- Owner side ----------
            phi_U  = W_recon_cell[v, o_idx]
            phi_D  = W_recon_cell[v, n_idx]
            phi_UU = W_recon_cell[v, UU_o_safe]

            delta_plus = phi_D - phi_U
            grad_x_U = coeffs[v, o_idx, 0]
            grad_y_U = coeffs[v, o_idx, 1]
            if self.face_skew_correction:
                tstar = tstar_o
                n_local = face_n_o
                grad_bar_x = ((1.0 - tstar) * coeffs[v, o_idx, 0]
                              + tstar * coeffs[v, n_idx, 0])
                grad_bar_y = ((1.0 - tstar) * coeffs[v, o_idx, 1]
                              + tstar * coeffs[v, n_idx, 1])
                cell_slope = delta_plus / d_len
                cos_no = e_o[:, 0] * n_local[:, 0] + e_o[:, 1] * n_local[:, 1]
                if face_gradient_correction_name == 'jasak':
                    cos_safe = np.where(
                        np.abs(cos_no) > _EPS,
                        cos_no,
                        np.where(cos_no >= 0.0, _EPS, -_EPS))
                    grad_n = (grad_bar_x * n_local[:, 0]
                              + grad_bar_y * n_local[:, 1])
                    corr = cell_slope / cos_safe - grad_n
                    grad_corr_x = grad_bar_x + corr * n_local[:, 0]
                    grad_corr_y = grad_bar_y + corr * n_local[:, 1]
                else:
                    grad_proj = grad_bar_x * e_o[:, 0] + grad_bar_y * e_o[:, 1]
                    beta = np.minimum(
                        1.0,
                        np.maximum(cos_no, 0.0) / theta_min)
                    corr = beta * (grad_proj - cell_slope)
                    if face_gradient_correction_name in (
                            'beta_shock_shear', 'shock_shear_beta',
                            'beta-shock-shear'):
                        damping_mode = str(
                            self.face_gradient_shock_damping).lower()
                        if damping_mode in ('', 'off', 'none', '0', 'false'):
                            shock_damp = 0.0
                        elif damping_mode in (
                                'euler', 'pressure_density',
                                'density_pressure', 'strong'):
                            shock_sensor = np.maximum(
                                velocity_flatten,
                                np.maximum(pressure_flatten, density_flatten))
                            shock_damp = np.clip(
                                shock_sensor
                                * (1.0 - np.clip(tangential_contact_weight,
                                                 0.0, 1.0)),
                                0.0, 1.0)
                        elif damping_mode in (
                                'density_strong', 'density-strong') and v == 0:
                            shock_sensor = np.maximum(
                                velocity_flatten,
                                np.maximum(pressure_flatten, density_flatten))
                            shock_damp = np.clip(
                                shock_sensor
                                * (1.0 - np.clip(tangential_contact_weight,
                                                 0.0, 1.0)),
                                0.0, 1.0)
                        else:
                            shock_sensor = velocity_flatten
                            shock_damp = np.clip(
                                shock_sensor
                                * (1.0 - np.clip(tangential_contact_weight,
                                                 0.0, 1.0)),
                                0.0, 1.0)
                        corr = corr * (1.0 - shock_damp)
                    grad_corr_x = grad_bar_x - corr * e_o[:, 0]
                    grad_corr_y = grad_bar_y - corr * e_o[:, 1]
                d_face_corr = dx_fo - tstar[:, None] * d_o_int
                delta_face = (tstar * delta_plus
                              + grad_corr_x * d_face_corr[:, 0]
                              + grad_corr_y * d_face_corr[:, 1])
                if face_increment_name == 'lsq':
                    delta_face = (grad_x_U * dx_fo[:, 0]
                                  + grad_y_U * dx_fo[:, 1])
                elif self.euler_density_full_lsq_increment and v == 0:
                    delta_face = (grad_x_U * dx_fo[:, 0]
                                  + grad_y_U * dx_fo[:, 1])
                elif self.euler_density_lsq_increment and v == 0:
                    delta_lsq = (grad_x_U * dx_fo[:, 0]
                                 + grad_y_U * dx_fo[:, 1])
                    w_density = _density_lsq_contact_weight()
                    delta_face = (
                        delta_face
                        + w_density * (delta_lsq - delta_face))
                elif self.euler_velocity_lsq_increment and (v == 1 or v == 2):
                    delta_face = (grad_x_U * dx_fo[:, 0]
                                  + grad_y_U * dx_fo[:, 1])
                elif self.euler_pressure_nonshock_lsq_increment and v == 3:
                    delta_lsq = (grad_x_U * dx_fo[:, 0]
                                 + grad_y_U * dx_fo[:, 1])
                    w_pressure = 1.0 - np.clip(pressure_flatten, 0.0, 1.0)
                    delta_face = (
                        delta_face
                        + w_pressure * (delta_lsq - delta_face))
                elif self.euler_pressure_shear_lsq_increment and v == 3:
                    delta_lsq = (grad_x_U * dx_fo[:, 0]
                                 + grad_y_U * dx_fo[:, 1])
                    w_pressure = np.clip(tangential_contact_weight, 0.0, 1.0)
                    delta_face = (
                        delta_face
                        + w_pressure * (delta_lsq - delta_face))
                if v == 0 and self.euler_density_no_hancock:
                    scale = 1.0 - face_hancock_courant * density_flatten
                elif v == 0 and self.euler_density_contact_wave_hancock:
                    w_density_scale = np.clip(
                        density_contact_weight, 0.0, 1.0)
                    scale = (one_minus_C_face
                             + w_density_scale
                             * (density_contact_hancock_scale
                                - one_minus_C_face))
                    if self.euler_density_contact_hancock_boost > 0.0:
                        clean_shear = (
                            w_density_scale
                            * np.clip(tangential_contact_weight, 0.0, 1.0)
                            * (1.0 - np.clip(pressure_flatten, 0.0, 1.0)) ** 2
                            * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)) ** 2)
                        scale = scale + (
                            float(self.euler_density_contact_hancock_boost)
                            * clean_shear * (1.0 - scale))
                        scale = np.minimum(
                            scale,
                            float(self.euler_density_contact_hancock_boost_cap))
                else:
                    scale = (velocity_scale if (v == 1 or v == 2)
                             else one_minus_C_face)
                delta = scale * delta_face
                delta_plus_tvd = delta_plus
            else:
                if v == 0 and self.euler_density_no_hancock:
                    scale = 1.0 - face_hancock_courant * density_flatten
                elif v == 0 and self.euler_density_contact_wave_hancock:
                    w_density_scale = np.clip(
                        density_contact_weight, 0.0, 1.0)
                    scale = (one_minus_C_face
                             + w_density_scale
                             * (density_contact_hancock_scale
                                - one_minus_C_face))
                    if self.euler_density_contact_hancock_boost > 0.0:
                        clean_shear = (
                            w_density_scale
                            * np.clip(tangential_contact_weight, 0.0, 1.0)
                            * (1.0 - np.clip(pressure_flatten, 0.0, 1.0)) ** 2
                            * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)) ** 2)
                        scale = scale + (
                            float(self.euler_density_contact_hancock_boost)
                            * clean_shear * (1.0 - scale))
                        scale = np.minimum(
                            scale,
                            float(self.euler_density_contact_hancock_boost_cap))
                else:
                    scale = (velocity_scale if (v == 1 or v == 2)
                             else one_minus_C_face)
                delta = scale * alpha_o * delta_plus
                if self.euler_density_full_lsq_increment and v == 0:
                    delta = scale * (
                        grad_x_U * dx_fo[:, 0] + grad_y_U * dx_fo[:, 1])
                elif self.euler_density_lsq_increment and v == 0:
                    delta_line = alpha_o * delta_plus
                    delta_lsq = (grad_x_U * dx_fo[:, 0]
                                 + grad_y_U * dx_fo[:, 1])
                    w_density = _density_lsq_contact_weight()
                    delta = scale * (
                        delta_line
                        + w_density * (delta_lsq - delta_line))
                if self.euler_velocity_lsq_increment and (v == 1 or v == 2):
                    delta = scale * (
                        grad_x_U * dx_fo[:, 0] + grad_y_U * dx_fo[:, 1])
                elif self.euler_pressure_nonshock_lsq_increment and v == 3:
                    delta_line = alpha_o * delta_plus
                    delta_lsq = (grad_x_U * dx_fo[:, 0]
                                 + grad_y_U * dx_fo[:, 1])
                    w_pressure = 1.0 - np.clip(pressure_flatten, 0.0, 1.0)
                    delta = scale * (
                        delta_line
                        + w_pressure * (delta_lsq - delta_line))
                elif self.euler_pressure_shear_lsq_increment and v == 3:
                    delta_line = alpha_o * delta_plus
                    delta_lsq = (grad_x_U * dx_fo[:, 0]
                                 + grad_y_U * dx_fo[:, 1])
                    w_pressure = np.clip(tangential_contact_weight, 0.0, 1.0)
                    delta = scale * (
                        delta_line
                        + w_pressure * (delta_lsq - delta_line))
                tstar = alpha_o
                grad_corr_x = 0.5 * (coeffs[v, o_idx, 0]
                                     + coeffs[v, n_idx, 0])
                grad_corr_y = 0.5 * (coeffs[v, o_idx, 1]
                                     + coeffs[v, n_idx, 1])
                delta_plus_tvd = delta_plus
            if self.euler_density_first_order and v == 0:
                delta = np.zeros_like(delta)
            elif self.euler_pressure_first_order and v == 3:
                delta = np.zeros_like(delta)
            abs_dp = np.abs(delta_plus_tvd)
            is_zero_dp = abs_dp <= _EPS
            safe_dp = np.where(is_zero_dp,
                               np.copysign(_EPS, delta_plus_tvd),
                               delta_plus_tvd)

            # ─── Full CICSAM path (bypasses ψ_TVD machinery) ───────────
            if self.cicsam_full:
                phi_min_b = phi_min_cell[v, o_idx] - tvb_eps
                phi_max_b = phi_max_cell[v, o_idx] + tvb_eps
                recon_o = _cicsam_face_value(
                    coeffs[v, o_idx, 0], coeffs[v, o_idx, 1],
                    d_o_int, phi_U, delta_plus,
                    phi_min_b, phi_max_b,
                    is_smooth_cell[v, o_idx] if use_extremum_relax else None,
                )
                if all_valid_when_virt:
                    W_L[v, interior] = recon_o
                else:
                    W_L[v, interior] = np.where(valid_o, recon_o, phi_U)
                # ----- Neighbour side ---------------------------------
                phi_Un = W_recon_cell[v, n_idx]
                phi_Dn = W_recon_cell[v, o_idx]
                delta_p_n = phi_Dn - phi_Un
                phi_min_bn = phi_min_cell[v, n_idx] - tvb_eps
                phi_max_bn = phi_max_cell[v, n_idx] + tvb_eps
                # Neighbour's d_UD = x_o − x_n = −d_o_int
                d_n_int = -d_o_int
                recon_n = _cicsam_face_value(
                    coeffs[v, n_idx, 0], coeffs[v, n_idx, 1],
                    d_n_int, phi_Un, delta_p_n,
                    phi_min_bn, phi_max_bn,
                    is_smooth_cell[v, n_idx] if use_extremum_relax else None,
                )
                if all_valid_when_virt:
                    W_R[v, interior] = recon_n
                else:
                    W_R[v, interior] = np.where(valid_n, recon_n, phi_Un)
                continue   # next variable — skip standard ψ path

            if self.virtual_uu_gradient:
                gdotd = grad_x_U * d_o_int[:, 0] + grad_y_U * d_o_int[:, 1]
                phi_LL_raw = phi_U - gdotd
            else:
                phi_LL_raw = np.where(valid_o, phi_UU, phi_U)
            if self.phi_LL_unclipped:
                phi_LL = phi_LL_raw
            else:
                phi_LL = np.clip(
                    phi_LL_raw,
                    phi_min_cell[v, o_idx] - tvb_eps,
                    phi_max_cell[v, o_idx] + tvb_eps,
                )
            delta_minus = phi_U - phi_LL
            if r_form_name == 'nvf':
                gdotd = grad_x_U * d_o_int[:, 0] + grad_y_U * d_o_int[:, 1]
                r_tilde = 2.0 * _safe_ratio(gdotd, delta_plus_tvd) - 1.0
                r = _safe_ratio(1.0 + r_tilde, 1.0 - r_tilde)
                r = np.nan_to_num(r, nan=0.0, posinf=1.0e30,
                                  neginf=-1.0e30)
            else:
                den_floor = (
                    64.0 * np.finfo(float).eps
                    * (1.0 + np.abs(phi_U) + np.abs(phi_LL)
                       + np.abs(delta_plus_tvd)))
                den_safe = np.where(
                    np.abs(delta_minus) > den_floor,
                    delta_minus,
                    np.where(delta_minus >= 0.0, den_floor, -den_floor))
                r = delta_plus_tvd / den_safe
            psi_tvd = self._psi_tvd(r)
            if self._psi_tvd_density is not None and v == 0:
                psi_density = self._psi_tvd_density(r)
                if self._density_tvd_name in ('van_leer', 'umist'):
                    psi_tvd = psi_tvd + density_contact_weight * (
                        psi_density - psi_tvd)
                else:
                    psi_tvd = psi_density
            if self._psi_tvd_velocity is not None and (v == 1 or v == 2):
                if self._velocity_tvd_name in (
                        'shock_downwind_modified_superbee',
                        'shock-aware-downwind-modified-superbee',
                        'shock_downwind_superbee15',
                        'shock_downwind_superbee',
                        'shock-aware-downwind-superbee'):
                    psi_downwind = TVD_LIMITERS['downwind'](r)
                    if self._velocity_tvd_name in (
                            'shock_downwind_superbee',
                            'shock-aware-downwind-superbee'):
                        psi_shock = TVD_LIMITERS['superbee'](r)
                    else:
                        psi_shock = TVD_LIMITERS['modified_superbee'](r)
                    w_shock = np.clip(velocity_flatten, 0.0, 1.0)
                    psi_tvd = psi_downwind + w_shock * (
                        psi_shock - psi_downwind)
                else:
                    psi_tvd = self._psi_tvd_velocity(r)
            zero_delta_cap = float(np.clip(self.zero_delta_psi, 0.0, 2.0))
            psi_tvd = np.where(is_zero_dp, zero_delta_cap, psi_tvd)
            if self.euler_velocity_extrema_lmp and (v == 1 or v == 2):
                psi_tvd = np.where(r <= _EPS, np.maximum(psi_tvd, 1.0),
                                   psi_tvd)
            if self.euler_density_extrema_lmp and v == 0:
                psi_tvd = np.where(r <= _EPS, np.maximum(psi_tvd, 1.0),
                                   psi_tvd)

            if self.mlp_bound:
                if psi_vertex_cell is not None:
                    psi_lmp = _tmlpu_vertex_face_psi(
                        v, o_idx, d_o_int, phi_U, delta_plus,
                        tstar, grad_corr_x, grad_corr_y, r, psi_tvd)
                    if psi_lmp is None:
                        psi_lmp = psi_tvd
                else:
                    phi_min = phi_min_cell[v, o_idx] - tvb_eps
                    phi_max = phi_max_cell[v, o_idx] + tvb_eps
                    # Single-pass NVD bound: clip δ away from 0 (sign-aware).
                    delta_clip_pos = np.maximum(delta,  _EPS)
                    delta_clip_neg = np.minimum(delta, -_EPS)
                    psi_mlp_pos = (phi_max - phi_U) / delta_clip_pos
                    psi_mlp_neg = (phi_min - phi_U) / delta_clip_neg
                    psi_mlp = np.where(delta >  _EPS, psi_mlp_pos,
                              np.where(delta < -_EPS, psi_mlp_neg, 2.0))
                    psi_lmp = np.minimum(psi_tvd, psi_mlp)
                # Final clip [0, 2] outside the inner branches.
                np.clip(psi_lmp, 0.0, 2.0, out=psi_lmp)
                density_contact_weak = (
                    self.euler_density_contact_weak_face_mlp and v == 0)
                density_micro_restore = (
                    v == 0
                    and self.euler_density_contact_weak_face_value_scaling
                    and str(
                        self.euler_density_contact_weak_face_value_scaling_mode
                    ).lower().strip() in (
                        'clean_shear_micro_restore',
                        'density_mlp_micro_restore',
                        'coherent_shear_micro_restore',
                        'density_mlp_coherent_restore',
                        'contour_continuity_micro_restore',
                        'density_contour_continuity_micro_restore'))
                if self.weak_face_mlp or density_contact_weak or density_micro_restore:
                    psi_weak = _weak_face_mlp_psi(
                        v, interior, phi_U, delta, psi_tvd)
                    if psi_weak is not None:
                        if (density_micro_restore and not density_contact_weak
                                and not self.weak_face_mlp):
                            psi_lmp = _density_contact_weak_face_value_scale(
                                psi_lmp, psi_weak, delta, phi_U,
                                W_L[3, interior], o_idx)
                        elif density_contact_weak and not self.weak_face_mlp:
                            if (
                                    self
                                    .euler_density_contact_weak_face_head_generic
                                    or self
                                    .euler_density_contact_weak_face_disable_specialized_relax):
                                cap = np.clip(
                                    float(
                                        self
                                        .euler_density_contact_weak_face_head_generic_blend_cap),
                                    0.0, 1.0)
                                psi_lmp = (
                                    psi_lmp + cap * (psi_weak - psi_lmp))
                            elif (
                                    self
                                    .euler_density_contact_weak_face_legacy_order):
                                if (
                                        self
                                        .euler_density_contact_weak_face_legacy_relax):
                                    relax_w = (
                                        _density_contact_weak_legacy_relax_weight())
                                else:
                                    relax_w = (
                                        _density_contact_weak_relax_weight())
                                psi_relaxed = (
                                    psi_lmp + relax_w
                                    * (psi_weak - psi_lmp))
                                psi_lmp = (
                                    _density_contact_weak_legacy_final_safety(
                                        psi_relaxed, psi_tvd))
                            else:
                                relax_w = _density_contact_weak_relax_weight()
                                relax_w = _density_contact_weak_entropy_accept(
                                    relax_w, psi_lmp, psi_weak, delta, phi_U,
                                    W_L[3, interior], o_idx)
                                relax_w = _density_contact_weak_shock_gate(
                                    relax_w)
                                relax_w = (
                                    _density_contact_weak_admissibility_damp(
                                        relax_w, psi_lmp, psi_weak, delta,
                                        phi_U, W_L[3, interior]))
                                psi_relaxed = (
                                    psi_lmp + relax_w
                                    * (psi_weak - psi_lmp))
                                psi_relaxed = (
                                    _density_contact_weak_mlp_downstream_bridge(
                                        psi_lmp, psi_relaxed, psi_weak,
                                        delta, phi_U, o_idx))
                                psi_lmp = (
                                    _density_contact_weak_face_value_scale(
                                        psi_lmp, psi_relaxed, delta, phi_U,
                                        W_L[3, interior], o_idx))
                        elif (self.weak_face_mlp_smooth_otsu
                                or self.weak_face_mlp_range_otsu
                                or self.weak_face_mlp_high_range_otsu
                                or self.weak_face_mlp_value_otsu
                                or self.weak_face_mlp_value_upper_otsu
                                or self.weak_face_mlp_curved_value_otsu
                                or self.weak_face_mlp_value_continuous_otsu):
                            use_weak = weak_face_smooth_cell[v, o_idx]
                            psi_lmp = np.where(use_weak, psi_weak, psi_lmp)
                        else:
                            psi_lmp = psi_weak
                if use_extremum_relax:
                    if self._psi_tvd_smooth is not None:
                        psi_smooth = np.clip(self._psi_tvd_smooth(r),
                                             0.0, 2.0)
                    else:
                        psi_smooth = np.clip(psi_tvd, 0.0, 2.0)
                    # Legacy explicit dispatch.  No continuous threshold
                    # blend is applied: each face side uses one ψ_TVD arm.
                    if self._psi_tvd_smooth2 is not None:
                        psi_smooth2 = np.clip(self._psi_tvd_smooth2(r),
                                              0.0, 2.0)
                        psi_final = np.where(
                            is_very_smooth_cell[v, o_idx], psi_smooth2,
                            np.where(is_smooth_cell[v, o_idx],
                                     psi_smooth, psi_lmp))
                    else:
                        psi_final = np.where(is_smooth_cell[v, o_idx],
                                             psi_smooth, psi_lmp)
                else:
                    psi_final = psi_lmp
            else:
                psi_final = np.clip(psi_tvd, 0.0, 2.0)
            if self.euler_density_acoustic_flatten and v == 0:
                psi_final = psi_final * (1.0 - density_flatten)
            elif self.euler_shock_flatten and v == 3:
                psi_final = psi_final * (1.0 - pressure_flatten)
            if self.euler_velocity_shock_flatten and (v == 1 or v == 2):
                psi_final = psi_final * (1.0 - velocity_flatten)
            recon = phi_U + psi_final * delta
            if all_valid_when_virt:
                W_L[v, interior] = recon       # virt-UU: every face valid
            else:
                W_L[v, interior] = np.where(valid_o, recon, phi_U)

            # ---------- Neighbour side ----------
            phi_U  = W_recon_cell[v, n_idx]
            phi_D  = W_recon_cell[v, o_idx]
            phi_UU = W_recon_cell[v, UU_n_safe]

            delta_plus = phi_D - phi_U
            d_n_int = -d_o_int
            grad_x_U = coeffs[v, n_idx, 0]
            grad_y_U = coeffs[v, n_idx, 1]
            if self.face_skew_correction:
                tstar = tstar_n
                n_local = -face_n_o
                grad_bar_x = ((1.0 - tstar) * coeffs[v, n_idx, 0]
                              + tstar * coeffs[v, o_idx, 0])
                grad_bar_y = ((1.0 - tstar) * coeffs[v, n_idx, 1]
                              + tstar * coeffs[v, o_idx, 1])
                cell_slope = delta_plus / d_len
                cos_no = e_n[:, 0] * n_local[:, 0] + e_n[:, 1] * n_local[:, 1]
                if face_gradient_correction_name == 'jasak':
                    cos_safe = np.where(
                        np.abs(cos_no) > _EPS,
                        cos_no,
                        np.where(cos_no >= 0.0, _EPS, -_EPS))
                    grad_n = (grad_bar_x * n_local[:, 0]
                              + grad_bar_y * n_local[:, 1])
                    corr = cell_slope / cos_safe - grad_n
                    grad_corr_x = grad_bar_x + corr * n_local[:, 0]
                    grad_corr_y = grad_bar_y + corr * n_local[:, 1]
                else:
                    grad_proj = grad_bar_x * e_n[:, 0] + grad_bar_y * e_n[:, 1]
                    beta = np.minimum(
                        1.0,
                        np.maximum(cos_no, 0.0) / theta_min)
                    corr = beta * (grad_proj - cell_slope)
                    if face_gradient_correction_name in (
                            'beta_shock_shear', 'shock_shear_beta',
                            'beta-shock-shear'):
                        damping_mode = str(
                            self.face_gradient_shock_damping).lower()
                        if damping_mode in ('', 'off', 'none', '0', 'false'):
                            shock_damp = 0.0
                        elif damping_mode in (
                                'euler', 'pressure_density',
                                'density_pressure', 'strong'):
                            shock_sensor = np.maximum(
                                velocity_flatten,
                                np.maximum(pressure_flatten, density_flatten))
                            shock_damp = np.clip(
                                shock_sensor
                                * (1.0 - np.clip(tangential_contact_weight,
                                                 0.0, 1.0)),
                                0.0, 1.0)
                        elif damping_mode in (
                                'density_strong', 'density-strong') and v == 0:
                            shock_sensor = np.maximum(
                                velocity_flatten,
                                np.maximum(pressure_flatten, density_flatten))
                            shock_damp = np.clip(
                                shock_sensor
                                * (1.0 - np.clip(tangential_contact_weight,
                                                 0.0, 1.0)),
                                0.0, 1.0)
                        else:
                            shock_sensor = velocity_flatten
                            shock_damp = np.clip(
                                shock_sensor
                                * (1.0 - np.clip(tangential_contact_weight,
                                                 0.0, 1.0)),
                                0.0, 1.0)
                        corr = corr * (1.0 - shock_damp)
                    grad_corr_x = grad_bar_x - corr * e_n[:, 0]
                    grad_corr_y = grad_bar_y - corr * e_n[:, 1]
                d_face_corr = dx_fn - tstar[:, None] * d_n_int
                delta_face = (tstar * delta_plus
                              + grad_corr_x * d_face_corr[:, 0]
                              + grad_corr_y * d_face_corr[:, 1])
                if face_increment_name == 'lsq':
                    delta_face = (grad_x_U * dx_fn[:, 0]
                                  + grad_y_U * dx_fn[:, 1])
                elif self.euler_density_full_lsq_increment and v == 0:
                    delta_face = (grad_x_U * dx_fn[:, 0]
                                  + grad_y_U * dx_fn[:, 1])
                elif self.euler_density_lsq_increment and v == 0:
                    delta_lsq = (grad_x_U * dx_fn[:, 0]
                                 + grad_y_U * dx_fn[:, 1])
                    w_density = _density_lsq_contact_weight()
                    delta_face = (
                        delta_face
                        + w_density * (delta_lsq - delta_face))
                elif self.euler_velocity_lsq_increment and (v == 1 or v == 2):
                    delta_face = (grad_x_U * dx_fn[:, 0]
                                  + grad_y_U * dx_fn[:, 1])
                elif self.euler_pressure_nonshock_lsq_increment and v == 3:
                    delta_lsq = (grad_x_U * dx_fn[:, 0]
                                 + grad_y_U * dx_fn[:, 1])
                    w_pressure = 1.0 - np.clip(pressure_flatten, 0.0, 1.0)
                    delta_face = (
                        delta_face
                        + w_pressure * (delta_lsq - delta_face))
                elif self.euler_pressure_shear_lsq_increment and v == 3:
                    delta_lsq = (grad_x_U * dx_fn[:, 0]
                                 + grad_y_U * dx_fn[:, 1])
                    w_pressure = np.clip(tangential_contact_weight, 0.0, 1.0)
                    delta_face = (
                        delta_face
                        + w_pressure * (delta_lsq - delta_face))
                if v == 0 and self.euler_density_no_hancock:
                    scale = 1.0 - face_hancock_courant * density_flatten
                elif v == 0 and self.euler_density_contact_wave_hancock:
                    w_density_scale = np.clip(
                        density_contact_weight, 0.0, 1.0)
                    scale = (one_minus_C_face
                             + w_density_scale
                             * (density_contact_hancock_scale
                                - one_minus_C_face))
                    if self.euler_density_contact_hancock_boost > 0.0:
                        clean_shear = (
                            w_density_scale
                            * np.clip(tangential_contact_weight, 0.0, 1.0)
                            * (1.0 - np.clip(pressure_flatten, 0.0, 1.0)) ** 2
                            * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)) ** 2)
                        scale = scale + (
                            float(self.euler_density_contact_hancock_boost)
                            * clean_shear * (1.0 - scale))
                        scale = np.minimum(
                            scale,
                            float(self.euler_density_contact_hancock_boost_cap))
                else:
                    scale = (velocity_scale if (v == 1 or v == 2)
                             else one_minus_C_face)
                delta = scale * delta_face
                delta_plus_tvd = delta_plus
            else:
                if v == 0 and self.euler_density_no_hancock:
                    scale = 1.0 - face_hancock_courant * density_flatten
                elif v == 0 and self.euler_density_contact_wave_hancock:
                    w_density_scale = np.clip(
                        density_contact_weight, 0.0, 1.0)
                    scale = (one_minus_C_face
                             + w_density_scale
                             * (density_contact_hancock_scale
                                - one_minus_C_face))
                    if self.euler_density_contact_hancock_boost > 0.0:
                        clean_shear = (
                            w_density_scale
                            * np.clip(tangential_contact_weight, 0.0, 1.0)
                            * (1.0 - np.clip(pressure_flatten, 0.0, 1.0)) ** 2
                            * (1.0 - np.clip(velocity_flatten, 0.0, 1.0)) ** 2)
                        scale = scale + (
                            float(self.euler_density_contact_hancock_boost)
                            * clean_shear * (1.0 - scale))
                        scale = np.minimum(
                            scale,
                            float(self.euler_density_contact_hancock_boost_cap))
                else:
                    scale = (velocity_scale if (v == 1 or v == 2)
                             else one_minus_C_face)
                delta = scale * alpha_n * delta_plus
                if self.euler_density_full_lsq_increment and v == 0:
                    delta = scale * (
                        grad_x_U * dx_fn[:, 0] + grad_y_U * dx_fn[:, 1])
                elif self.euler_density_lsq_increment and v == 0:
                    delta_line = alpha_n * delta_plus
                    delta_lsq = (grad_x_U * dx_fn[:, 0]
                                 + grad_y_U * dx_fn[:, 1])
                    w_density = _density_lsq_contact_weight()
                    delta = scale * (
                        delta_line
                        + w_density * (delta_lsq - delta_line))
                if self.euler_velocity_lsq_increment and (v == 1 or v == 2):
                    delta = scale * (
                        grad_x_U * dx_fn[:, 0] + grad_y_U * dx_fn[:, 1])
                elif self.euler_pressure_nonshock_lsq_increment and v == 3:
                    delta_line = alpha_n * delta_plus
                    delta_lsq = (grad_x_U * dx_fn[:, 0]
                                 + grad_y_U * dx_fn[:, 1])
                    w_pressure = 1.0 - np.clip(pressure_flatten, 0.0, 1.0)
                    delta = scale * (
                        delta_line
                        + w_pressure * (delta_lsq - delta_line))
                elif self.euler_pressure_shear_lsq_increment and v == 3:
                    delta_line = alpha_n * delta_plus
                    delta_lsq = (grad_x_U * dx_fn[:, 0]
                                 + grad_y_U * dx_fn[:, 1])
                    w_pressure = np.clip(tangential_contact_weight, 0.0, 1.0)
                    delta = scale * (
                        delta_line
                        + w_pressure * (delta_lsq - delta_line))
                tstar = alpha_n
                grad_corr_x = 0.5 * (coeffs[v, n_idx, 0]
                                     + coeffs[v, o_idx, 0])
                grad_corr_y = 0.5 * (coeffs[v, n_idx, 1]
                                     + coeffs[v, o_idx, 1])
                delta_plus_tvd = delta_plus
            if self.euler_density_first_order and v == 0:
                delta = np.zeros_like(delta)
            elif self.euler_pressure_first_order and v == 3:
                delta = np.zeros_like(delta)
            abs_dp = np.abs(delta_plus_tvd)
            is_zero_dp = abs_dp <= _EPS
            safe_dp = np.where(is_zero_dp,
                               np.copysign(_EPS, delta_plus_tvd),
                               delta_plus_tvd)
            if self.virtual_uu_gradient:
                gdotd = grad_x_U * d_n_int[:, 0] + grad_y_U * d_n_int[:, 1]
                phi_LL_raw = phi_U - gdotd
            else:
                phi_LL_raw = np.where(valid_n, phi_UU, phi_U)
            if self.phi_LL_unclipped:
                phi_LL = phi_LL_raw
            else:
                phi_LL = np.clip(
                    phi_LL_raw,
                    phi_min_cell[v, n_idx] - tvb_eps,
                    phi_max_cell[v, n_idx] + tvb_eps,
                )
            delta_minus = phi_U - phi_LL
            if r_form_name == 'nvf':
                gdotd = grad_x_U * d_n_int[:, 0] + grad_y_U * d_n_int[:, 1]
                r_tilde = 2.0 * _safe_ratio(gdotd, delta_plus_tvd) - 1.0
                r = _safe_ratio(1.0 + r_tilde, 1.0 - r_tilde)
                r = np.nan_to_num(r, nan=0.0, posinf=1.0e30,
                                  neginf=-1.0e30)
            else:
                den_floor = (
                    64.0 * np.finfo(float).eps
                    * (1.0 + np.abs(phi_U) + np.abs(phi_LL)
                       + np.abs(delta_plus_tvd)))
                den_safe = np.where(
                    np.abs(delta_minus) > den_floor,
                    delta_minus,
                    np.where(delta_minus >= 0.0, den_floor, -den_floor))
                r = delta_plus_tvd / den_safe
            psi_tvd = self._psi_tvd(r)
            if self._psi_tvd_density is not None and v == 0:
                psi_density = self._psi_tvd_density(r)
                if self._density_tvd_name in ('van_leer', 'umist'):
                    psi_tvd = psi_tvd + density_contact_weight * (
                        psi_density - psi_tvd)
                else:
                    psi_tvd = psi_density
            if self._psi_tvd_velocity is not None and (v == 1 or v == 2):
                if self._velocity_tvd_name in (
                        'shock_downwind_modified_superbee',
                        'shock-aware-downwind-modified-superbee',
                        'shock_downwind_superbee15',
                        'shock_downwind_superbee',
                        'shock-aware-downwind-superbee'):
                    psi_downwind = TVD_LIMITERS['downwind'](r)
                    if self._velocity_tvd_name in (
                            'shock_downwind_superbee',
                            'shock-aware-downwind-superbee'):
                        psi_shock = TVD_LIMITERS['superbee'](r)
                    else:
                        psi_shock = TVD_LIMITERS['modified_superbee'](r)
                    w_shock = np.clip(velocity_flatten, 0.0, 1.0)
                    psi_tvd = psi_downwind + w_shock * (
                        psi_shock - psi_downwind)
                else:
                    psi_tvd = self._psi_tvd_velocity(r)
            zero_delta_cap = float(np.clip(self.zero_delta_psi, 0.0, 2.0))
            psi_tvd = np.where(is_zero_dp, zero_delta_cap, psi_tvd)
            if self.euler_velocity_extrema_lmp and (v == 1 or v == 2):
                psi_tvd = np.where(r <= _EPS, np.maximum(psi_tvd, 1.0),
                                   psi_tvd)
            if self.euler_density_extrema_lmp and v == 0:
                psi_tvd = np.where(r <= _EPS, np.maximum(psi_tvd, 1.0),
                                   psi_tvd)

            if self.mlp_bound:
                if psi_vertex_cell is not None:
                    psi_lmp = _tmlpu_vertex_face_psi(
                        v, n_idx, d_n_int, phi_U, delta_plus,
                        tstar, grad_corr_x, grad_corr_y, r, psi_tvd)
                    if psi_lmp is None:
                        psi_lmp = psi_tvd
                else:
                    phi_min = phi_min_cell[v, n_idx] - tvb_eps
                    phi_max = phi_max_cell[v, n_idx] + tvb_eps
                    delta_clip_pos = np.maximum(delta,  _EPS)
                    delta_clip_neg = np.minimum(delta, -_EPS)
                    psi_mlp_pos = (phi_max - phi_U) / delta_clip_pos
                    psi_mlp_neg = (phi_min - phi_U) / delta_clip_neg
                    psi_mlp = np.where(delta >  _EPS, psi_mlp_pos,
                              np.where(delta < -_EPS, psi_mlp_neg, 2.0))
                    psi_lmp = np.minimum(psi_tvd, psi_mlp)
                np.clip(psi_lmp, 0.0, 2.0, out=psi_lmp)
                density_contact_weak = (
                    self.euler_density_contact_weak_face_mlp and v == 0)
                density_micro_restore = (
                    v == 0
                    and self.euler_density_contact_weak_face_value_scaling
                    and str(
                        self.euler_density_contact_weak_face_value_scaling_mode
                    ).lower().strip() in (
                        'clean_shear_micro_restore',
                        'density_mlp_micro_restore',
                        'coherent_shear_micro_restore',
                        'density_mlp_coherent_restore',
                        'contour_continuity_micro_restore',
                        'density_contour_continuity_micro_restore'))
                if self.weak_face_mlp or density_contact_weak or density_micro_restore:
                    psi_weak = _weak_face_mlp_psi(
                        v, interior, phi_U, delta, psi_tvd)
                    if psi_weak is not None:
                        if (density_micro_restore and not density_contact_weak
                                and not self.weak_face_mlp):
                            psi_lmp = _density_contact_weak_face_value_scale(
                                psi_lmp, psi_weak, delta, phi_U,
                                W_R[3, interior], n_idx)
                        elif density_contact_weak and not self.weak_face_mlp:
                            if (
                                    self
                                    .euler_density_contact_weak_face_head_generic
                                    or self
                                    .euler_density_contact_weak_face_disable_specialized_relax):
                                cap = np.clip(
                                    float(
                                        self
                                        .euler_density_contact_weak_face_head_generic_blend_cap),
                                    0.0, 1.0)
                                psi_lmp = (
                                    psi_lmp + cap * (psi_weak - psi_lmp))
                            elif (
                                    self
                                    .euler_density_contact_weak_face_legacy_order):
                                if (
                                        self
                                        .euler_density_contact_weak_face_legacy_relax):
                                    relax_w = (
                                        _density_contact_weak_legacy_relax_weight())
                                else:
                                    relax_w = (
                                        _density_contact_weak_relax_weight())
                                psi_relaxed = (
                                    psi_lmp + relax_w
                                    * (psi_weak - psi_lmp))
                                psi_lmp = (
                                    _density_contact_weak_legacy_final_safety(
                                        psi_relaxed, psi_tvd))
                            else:
                                relax_w = _density_contact_weak_relax_weight()
                                relax_w = _density_contact_weak_entropy_accept(
                                    relax_w, psi_lmp, psi_weak, delta, phi_U,
                                    W_R[3, interior], n_idx)
                                relax_w = _density_contact_weak_shock_gate(
                                    relax_w)
                                relax_w = (
                                    _density_contact_weak_admissibility_damp(
                                        relax_w, psi_lmp, psi_weak, delta,
                                        phi_U, W_R[3, interior]))
                                psi_relaxed = (
                                    psi_lmp + relax_w
                                    * (psi_weak - psi_lmp))
                                psi_relaxed = (
                                    _density_contact_weak_mlp_downstream_bridge(
                                        psi_lmp, psi_relaxed, psi_weak,
                                        delta, phi_U, n_idx))
                                psi_lmp = (
                                    _density_contact_weak_face_value_scale(
                                        psi_lmp, psi_relaxed, delta, phi_U,
                                        W_R[3, interior], n_idx))
                        elif (self.weak_face_mlp_smooth_otsu
                                or self.weak_face_mlp_range_otsu
                                or self.weak_face_mlp_high_range_otsu
                                or self.weak_face_mlp_value_otsu
                                or self.weak_face_mlp_value_upper_otsu
                                or self.weak_face_mlp_curved_value_otsu
                                or self.weak_face_mlp_value_continuous_otsu):
                            use_weak = weak_face_smooth_cell[v, n_idx]
                            psi_lmp = np.where(use_weak, psi_weak, psi_lmp)
                        else:
                            psi_lmp = psi_weak
                if use_extremum_relax:
                    if self._psi_tvd_smooth is not None:
                        psi_smooth = np.clip(self._psi_tvd_smooth(r),
                                             0.0, 2.0)
                    else:
                        psi_smooth = np.clip(psi_tvd, 0.0, 2.0)
                    if self._psi_tvd_smooth2 is not None:
                        psi_smooth2 = np.clip(self._psi_tvd_smooth2(r),
                                              0.0, 2.0)
                        psi_final = np.where(
                            is_very_smooth_cell[v, n_idx], psi_smooth2,
                            np.where(is_smooth_cell[v, n_idx],
                                     psi_smooth, psi_lmp))
                    else:
                        psi_final = np.where(is_smooth_cell[v, n_idx],
                                             psi_smooth, psi_lmp)
                else:
                    psi_final = psi_lmp
            else:
                psi_final = np.clip(psi_tvd, 0.0, 2.0)
            if self.euler_density_acoustic_flatten and v == 0:
                psi_final = psi_final * (1.0 - density_flatten)
            elif self.euler_shock_flatten and v == 3:
                psi_final = psi_final * (1.0 - pressure_flatten)
            if self.euler_velocity_shock_flatten and (v == 1 or v == 2):
                psi_final = psi_final * (1.0 - velocity_flatten)
            recon = phi_U + psi_final * delta
            if all_valid_when_virt:
                W_R[v, interior] = recon
            else:
                W_R[v, interior] = np.where(valid_n, recon, phi_U)

        if (_NUMBA_AVAILABLE
                and self.euler_density_pressure_entropy
                and nvar >= 4
                and not euler_log_positive
                and 0 in active_vars):
            _euler_density_pressure_entropy_kernel(
                np.asarray(W_cell, dtype=np.float64),
                np.asarray(coeffs, dtype=np.float64),
                np.asarray(phi_min_cell, dtype=np.float64),
                np.asarray(phi_max_cell, dtype=np.float64),
                np.asarray(o_idx, dtype=np.int64),
                np.asarray(n_idx, dtype=np.int64),
                np.asarray(interior, dtype=np.int64),
                np.asarray(dx_fo, dtype=np.float64),
                np.asarray(dx_fn, dtype=np.float64),
                np.asarray(density_contact_weight, dtype=np.float64),
                np.asarray(density_flatten, dtype=np.float64),
                float(getattr(eq, 'gamma', 1.4)),
                bool(euler_log_pressure_only),
                bool(os.environ.get(
                    'TMLPU_EULER_DENSITY_PRESSURE_ENTROPY_ACCEPT', ''
                ).strip().lower() in (
                    'entropy', 'entropy_residual',
                    'entropy-residual', 'residual')),
                W_L,
                W_R,
            )

        if (self.euler_density_contact_bvd
                and nvar >= 4
                and not euler_log_positive
                and 0 in active_vars):
            _euler_density_contact_bvd_kernel(
                np.asarray(W_cell, dtype=np.float64),
                np.asarray(phi_min_cell, dtype=np.float64),
                np.asarray(phi_max_cell, dtype=np.float64),
                np.asarray(o_idx, dtype=np.int64),
                np.asarray(n_idx, dtype=np.int64),
                np.asarray(interior, dtype=np.int64),
                np.asarray(density_contact_weight, dtype=np.float64),
                np.asarray(density_contact_hancock_scale, dtype=np.float64),
                np.asarray(shock_flatten, dtype=np.float64),
                float(1.0 - self.hancock_courant),
                float(self.euler_density_contact_bvd_cap),
                W_L,
                W_R,
            )
        if (self.euler_density_contact_cell_bvd
                and nvar >= 4
                and not euler_log_positive
                and 0 in active_vars):
            _euler_density_contact_cell_bvd_kernel(
                np.asarray(W_cell, dtype=np.float64),
                np.asarray(phi_min_cell, dtype=np.float64),
                np.asarray(phi_max_cell, dtype=np.float64),
                np.asarray(o_idx, dtype=np.int64),
                np.asarray(n_idx, dtype=np.int64),
                np.asarray(interior, dtype=np.int64),
                np.asarray(density_contact_weight, dtype=np.float64),
                np.asarray(density_contact_hancock_scale, dtype=np.float64),
                float(1.0 - self.hancock_courant),
                W_L,
                W_R,
            )

        tangential_tvd_mode = (
            1 if self._tangential_velocity_tvd_name == 'mc'
            else 2 if self._tangential_velocity_tvd_name == 'van_leer'
            else 3 if self._tangential_velocity_tvd_name == 'umist'
            else 17 if self._tangential_velocity_tvd_name == 'koren'
            else 4 if self._tangential_velocity_tvd_name == 'contact_umist'
            else 5 if self._tangential_velocity_tvd_name == 'superbee'
            else 16 if self._tangential_velocity_tvd_name in (
                'modified_superbee', 'superbee15')
            else 6 if self._tangential_velocity_tvd_name == 'contact_van_leer'
            else 7 if self._tangential_velocity_tvd_name == 'contact_van_leer_linear'
            else 8 if self._tangential_velocity_tvd_name == 'shock_van_leer'
            else 9 if self._tangential_velocity_tvd_name == 'shock_van_leer_strict'
            else 10 if self._tangential_velocity_tvd_name == 'contact_van_leer_root'
            else 11 if self._tangential_velocity_tvd_name == 'shock_van_leer_cubic'
            else 12 if self._tangential_velocity_tvd_name == 'contact_umist_shock'
            else 13 if self._tangential_velocity_tvd_name == 'contact_umist_shock_root'
            else 14 if self._tangential_velocity_tvd_name == 'contact_superbee'
            else 15 if self._tangential_velocity_tvd_name == 'contact_superbee_shock'
            else 18 if self._tangential_velocity_tvd_name == 'superbee_shock_blend'
            else 19 if self._tangential_velocity_tvd_name == 'shear_superbee_blend'
            else 20 if self._tangential_velocity_tvd_name == 'shear_superbee_root_blend'
            else 21 if self._tangential_velocity_tvd_name == 'shear_superbee_root_micro'
            else 22 if self._tangential_velocity_tvd_name == 'shear_superbee_root_mood'
            else 0)
        if (tangential_tvd_mode > 0
                and _NUMBA_AVAILABLE
                and nvar >= 4
                and velocity_pair_active):
            tangential_weight = _ducros_tangential_weight(
                tangential_contact_weight, coeffs)
            if (self.euler_velocity_no_hancock
                    or self.euler_tangential_velocity_no_hancock):
                velocity_face_scale = 1.0 - face_hancock_courant * shock_flatten
            elif self.euler_tangential_contact_wave_hancock:
                base_scale = one_minus_C_face
                velocity_face_scale = (
                    base_scale
                    + tangential_weight
                    * (density_contact_hancock_scale - base_scale))
            else:
                velocity_face_scale = one_minus_C_face
            if self.euler_velocity_shock_flatten:
                tangential_flatten = tangential_shock_flatten
                if self.euler_tangential_contact_relax_flatten:
                    tangential_flatten = tangential_flatten * (
                        1.0 - np.clip(tangential_weight, 0.0, 1.0))
                velocity_face_scale = (
                    velocity_face_scale * (1.0 - tangential_flatten))
            _euler_tangential_velocity_mc_kernel(
                np.asarray(W_cell, dtype=np.float64),
                np.asarray(coeffs, dtype=np.float64),
                np.asarray(o_idx, dtype=np.int64),
                np.asarray(n_idx, dtype=np.int64),
                np.asarray(interior, dtype=np.int64),
                np.asarray(d_o_int, dtype=np.float64),
                np.asarray(dx_fo, dtype=np.float64),
                np.asarray(dx_fn, dtype=np.float64),
                np.asarray(alpha_o, dtype=np.float64),
                np.asarray(alpha_n, dtype=np.float64),
                np.asarray(face_n_o, dtype=np.float64),
                np.asarray(velocity_face_scale, dtype=np.float64),
                np.asarray(tangential_shock_flatten, dtype=np.float64),
                np.asarray(tangential_weight, dtype=np.float64),
                int(tangential_tvd_mode),
                float(self.euler_tangential_shear_micro_blend),
                float(self.euler_tangential_shear_micro_cap),
                float(self.euler_tangential_mood_wavespeed_growth_cap),
                float(self.euler_tangential_mood_jump_growth_cap),
                bool(self.euler_tangential_velocity_lsq_increment),
                W_L,
                W_R,
            )
            _apply_tangential_pair_restore(W_L, W_R)

        if euler_log_positive:
            if 0 in active_vars:
                W_L[0, interior] = np.exp(W_L[0, interior])
                W_R[0, interior] = np.exp(W_R[0, interior])
            if 3 in active_vars:
                W_L[3, interior] = np.exp(W_L[3, interior])
                W_R[3, interior] = np.exp(W_R[3, interior])
        elif euler_log_pressure_only:
            if 3 in active_vars:
                W_L[3, interior] = np.exp(W_L[3, interior])
                W_R[3, interior] = np.exp(W_R[3, interior])
        elif euler_density_entropy_variable and 0 in active_vars:
            gamma = float(getattr(eq, 'gamma', 1.4))
            W_L[0, interior] = np.exp(
                W_L[0, interior]
                + np.log(np.maximum(W_L[3, interior], 1.0e-300)) / gamma)
            W_R[0, interior] = np.exp(
                W_R[0, interior]
                + np.log(np.maximum(W_R[3, interior], 1.0e-300)) / gamma)
        if self.euler_pressure_contact_entropy_blend:
            _euler_pressure_contact_entropy_blend(W_L)
            _euler_pressure_contact_entropy_blend(W_R)
        if self.euler_pressure_face_jump_limiter_on:
            _euler_pressure_face_jump_limiter(W_L, W_R)
        _apply_contact_characteristic_postpass(W_L, W_R)
        _apply_patch_contact_shear_postpass(W_L, W_R)

        return _finish_faces(W_L, W_R)

    # --- Mesh-dependent cache for the unstructured path ---------------------
    def _unstructured_cache(self, mesh):
        """Cache LSQ operator, LMP stencil, UU lookup, face offsets.
        Same dict is returned on every call (mesh-keyed)."""
        cache_key = f'_tmlpu_cache_{id(self)}'
        if hasattr(mesh, cache_key):
            return getattr(mesh, cache_key)

        N = mesh.n_cells
        n_faces = mesh.n_faces
        n_centers = mesh.cell_centers
        f_centers = mesh.face_centers
        owner = mesh.face_owner
        nei = mesh.face_neighbour

        # ---- 1) Choose stencil for LSQ + LMP bound -----------------------
        if self.stencil in ('vertex', 'vertex2'):
            n_rings = 1 if self.stencil == 'vertex' else 2
            nb_lists = _build_vertex_neighbours(mesh, n_rings=n_rings)
            if nb_lists is None:
                raise ValueError(
                    f"stencil='{self.stencil}' requires mesh.cell_nodes; "
                    "use the unstructured constructor or `criss_cross_box`.")
        else:
            nb_lists = mesh.cell_neighbours

        # Always keep the *face* neighbour list separately — UU pick must
        # use it (the TVD ratio is defined on the face-cell direction).
        face_nb_lists = mesh.cell_neighbours

        # ---- 2) Padded neighbour table (LSQ + LMP) -----------------------
        max_nb = max((len(nbs) for nbs in nb_lists if nbs), default=1)
        max_nb = max(max_nb, 1)
        nb_padded = np.full((N, max_nb), -1, dtype=int)
        for c in range(N):
            nbs = [int(nb) for nb in nb_lists[c] if int(nb) >= 0]
            nb_padded[c, :len(nbs)] = nbs
        valid_nb = nb_padded >= 0
        nb_safe = np.where(valid_nb, nb_padded, 0)
        d_full = n_centers[nb_safe] - n_centers[:, None, :]  # (N, max_nb, 2)
        d_full = d_full * valid_nb[:, :, None]

        # ---- 3) LSQ basis matrix A and (Aᵀ A)⁻¹ --------------------------
        # order=1: A = [dx, dy]                            (nbasis = 2)
        # order=2: A = [dx, dy, ½dx², dx·dy, ½dy²]        (nbasis = 5)
        dx = d_full[:, :, 0]
        dy = d_full[:, :, 1]
        if self.order == 1:
            A = d_full                                         # (N, max_nb, 2)
            nbasis = 2
        elif self.order == 2:
            A = np.stack([dx, dy,
                          0.5 * dx * dx,
                          dx * dy,
                          0.5 * dy * dy], axis=-1)             # (N, max_nb, 5)
            nbasis = 5
        else:  # order == 3
            A = np.stack([dx, dy,
                          0.5 * dx * dx, dx * dy, 0.5 * dy * dy,
                          dx * dx * dx / 6.0,
                          0.5 * dx * dx * dy,
                          0.5 * dx * dy * dy,
                          dy * dy * dy / 6.0], axis=-1)        # (N, max_nb, 9)
            nbasis = 9
        A = A * valid_nb[:, :, None]
        # Inverse-distance LSQ weighting — emphasises closer cells.
        # weight = 1/d^p,  sqrt_w = (1/dist_sq)^(p/4)
        dist_sq = dx * dx + dy * dy + 1e-30
        sqrt_w = (1.0 / dist_sq) ** (self.idw_p / 4.0) * valid_nb
        A = A * sqrt_w[:, :, None]                              # A → √W · A
        ATA = np.einsum('cki,ckj->cij', A, A)                  # (N, nbasis, nbasis)

        if nbasis == 2:
            det = ATA[:, 0, 0] * ATA[:, 1, 1] - ATA[:, 0, 1] * ATA[:, 1, 0]
            ok = np.abs(det) > 1e-30
            det_safe = np.where(ok, det, 1.0)
            ATA_inv = np.empty_like(ATA)
            ATA_inv[:, 0, 0] = ATA[:, 1, 1] / det_safe
            ATA_inv[:, 1, 1] = ATA[:, 0, 0] / det_safe
            ATA_inv[:, 0, 1] = -ATA[:, 0, 1] / det_safe
            ATA_inv[:, 1, 0] = -ATA[:, 1, 0] / det_safe
            ATA_inv = np.where(ok[:, None, None], ATA_inv, 0.0)
        else:
            # 5×5 batched inverse — fall back to per-cell pinv for singular
            # cells (typical: cells with too few valid neighbours).
            ATA_inv = np.zeros_like(ATA)
            sign, logdet = np.linalg.slogdet(ATA)
            ok = np.isfinite(logdet) & (sign != 0)
            ok_idx = np.where(ok)[0]
            if ok_idx.size > 0:
                ATA_inv[ok_idx] = np.linalg.inv(ATA[ok_idx])
            for c in np.where(~ok)[0]:
                try:
                    ATA_inv[c] = np.linalg.pinv(ATA[c])
                except np.linalg.LinAlgError:
                    pass
        lsq_op = np.einsum('cij,ckj->cik', ATA_inv, A, optimize=True)

        grad_nb_padded = nb_padded
        grad_valid_nb = valid_nb
        grad_nb_safe = nb_safe
        grad_A = A
        grad_sqrt_w = sqrt_w
        grad_lsq_op = lsq_op
        rank_aware_cells = np.zeros(N, dtype=bool)
        if self.order >= 2 and self.stencil == 'vertex':
            # A quadratic LSQ fit needs a full-rank 5-column basis.  Boundary
            # and corner cells can have geometrically one-sided vertex stencils;
            # when the normal equations become rank-deficient or too ill
            # conditioned, use a wider first-degree gradient operator for that
            # cell only.  The threshold is the standard linear-algebra limit
            # κ(AᵀA) ≲ 1/sqrt(eps), not a case-specific CFD coefficient.
            ranks = np.linalg.matrix_rank(ATA, tol=1.0e-12)
            try:
                cond = np.linalg.cond(ATA)
            except np.linalg.LinAlgError:
                cond = np.empty(N, dtype=float)
                for c in range(N):
                    try:
                        cond[c] = np.linalg.cond(ATA[c])
                    except np.linalg.LinAlgError:
                        cond[c] = np.inf
            cond_limit = 1.0 / np.sqrt(np.finfo(float).eps)
            rank_aware_cells = (ranks < nbasis) | (~np.isfinite(cond)) | (cond > cond_limit)
            if np.any(rank_aware_cells):
                wide_lists = _build_vertex_neighbours(mesh, n_rings=2)
                if wide_lists is not None:
                    max_grad_nb = max((len(nbs) for nbs in wide_lists if nbs),
                                      default=1)
                    max_grad_nb = max(max_grad_nb, max_nb, 1)
                    grad_nb_padded = np.full((N, max_grad_nb), -1, dtype=int)
                    grad_nb_padded[:, :max_nb] = nb_padded
                    for c in np.where(rank_aware_cells)[0]:
                        nbs = [int(nb) for nb in wide_lists[c] if int(nb) >= 0]
                        grad_nb_padded[c, :] = -1
                        grad_nb_padded[c, :len(nbs)] = nbs
                    grad_valid_nb = grad_nb_padded >= 0
                    grad_nb_safe = np.where(grad_valid_nb, grad_nb_padded, 0)
                    grad_d = n_centers[grad_nb_safe] - n_centers[:, None, :]
                    grad_d = grad_d * grad_valid_nb[:, :, None]
                    gdx = grad_d[:, :, 0]
                    gdy = grad_d[:, :, 1]
                    grad_A = np.zeros((N, max_grad_nb, nbasis), dtype=float)
                    grad_A[:, :max_nb, :] = A
                    dist_sq_g = gdx * gdx + gdy * gdy + 1e-30
                    grad_sqrt_w = np.zeros((N, max_grad_nb), dtype=float)
                    grad_sqrt_w[:, :max_nb] = sqrt_w
                    wide_sqrt_w = ((1.0 / dist_sq_g)
                                   ** (self.idw_p / 4.0) * grad_valid_nb)
                    grad_lsq_op = np.zeros((N, nbasis, max_grad_nb),
                                           dtype=float)
                    grad_lsq_op[:, :, :max_nb] = lsq_op
                    bad_idx = np.where(rank_aware_cells)[0]
                    if bad_idx.size > 0:
                        grad_sqrt_w[bad_idx] = wide_sqrt_w[bad_idx]
                        A1 = grad_d[bad_idx] * wide_sqrt_w[bad_idx, :, None]
                        ATA1 = np.einsum('cki,ckj->cij', A1, A1)
                        det = (ATA1[:, 0, 0] * ATA1[:, 1, 1]
                               - ATA1[:, 0, 1] * ATA1[:, 1, 0])
                        ok1 = np.abs(det) > 1e-30
                        det_safe = np.where(ok1, det, 1.0)
                        inv1 = np.zeros_like(ATA1)
                        inv1[:, 0, 0] = ATA1[:, 1, 1] / det_safe
                        inv1[:, 1, 1] = ATA1[:, 0, 0] / det_safe
                        inv1[:, 0, 1] = -ATA1[:, 0, 1] / det_safe
                        inv1[:, 1, 0] = -ATA1[:, 1, 0] / det_safe
                        inv1 = np.where(ok1[:, None, None], inv1, 0.0)
                        op1 = np.einsum('cij,ckj->cik', inv1, A1,
                                        optimize=True)
                        grad_A[bad_idx] = 0.0
                        grad_A[bad_idx, :, 0:2] = A1
                        grad_lsq_op[bad_idx] = 0.0
                        grad_lsq_op[bad_idx, 0:2, :] = op1

        # ---- 4) Per-face UU pick (uses FACE-neighbour set) ----------------
        face_max_nb = max((len(nbs) for nbs in face_nb_lists), default=1)
        face_max_nb = max(face_max_nb, 1)
        face_nb_padded = np.full((N, face_max_nb), -1, dtype=int)
        for c in range(N):
            nbs = [int(nb) for nb in face_nb_lists[c] if int(nb) >= 0]
            face_nb_padded[c, :len(nbs)] = nbs
        face_valid = face_nb_padded >= 0
        face_safe = np.where(face_valid, face_nb_padded, 0)
        d_face = (n_centers[face_safe] - n_centers[:, None, :]) * face_valid[:, :, None]

        interior = np.where(nei >= 0)[0]
        if interior.size > 0:
            o_idx = owner[interior]
            n_idx = nei[interior]
            d_o = n_centers[n_idx] - n_centers[o_idx]                # (Nint, 2)
            score_o = -np.einsum('fki,fi->fk', d_face[o_idx], d_o)
            score_o = np.where(face_valid[o_idx], score_o, -np.inf)
            best_k_o = np.argmax(score_o, axis=1)
            best_score_o = score_o[np.arange(interior.size), best_k_o]
            UU_o_int = face_nb_padded[o_idx, best_k_o]
            UU_o_int = np.where(best_score_o > 0.0, UU_o_int, -1)

            d_n = n_centers[o_idx] - n_centers[n_idx]
            score_n = -np.einsum('fki,fi->fk', d_face[n_idx], d_n)
            score_n = np.where(face_valid[n_idx], score_n, -np.inf)
            best_k_n = np.argmax(score_n, axis=1)
            best_score_n = score_n[np.arange(interior.size), best_k_n]
            UU_n_int = face_nb_padded[n_idx, best_k_n]
            UU_n_int = np.where(best_score_n > 0.0, UU_n_int, -1)

            dx_fo = f_centers[interior] - n_centers[o_idx]
            dx_fn = f_centers[interior] - n_centers[n_idx]
            # Owner→neighbour displacement, used by the gradient-based
            # virtual-UU formula (Darwish-Moukalled).  d_n_int = −d_o_int.
            d_o_int = d_o
        else:
            UU_o_int = np.zeros(0, dtype=int)
            UU_n_int = np.zeros(0, dtype=int)
            dx_fo = np.zeros((0, 2), dtype=float)
            dx_fn = np.zeros((0, 2), dtype=float)
            d_o_int = np.zeros((0, 2), dtype=float)

        # ---- 5) Vertex-MLP supporting structures ------------------------
        # Used only when self.vertex_mlp is True; cheap to build always.
        cn = getattr(mesh, 'cell_nodes', None)
        if cn:
            n_v_per_cell = max(len(c) for c in cn)
            cell_node_arr = np.full((N, n_v_per_cell), -1, dtype=int)
            for c, vs in enumerate(cn):
                cell_node_arr[c, :len(vs)] = vs
            cell_node_valid = cell_node_arr >= 0
            cell_node_safe = np.where(cell_node_valid, cell_node_arr, 0)
            # Vertex coordinates (broadcast vs cell centre).
            vertex_xy = mesh.nodes[cell_node_safe]            # (N, V, 2)
            vertex_offsets = vertex_xy - mesh.cell_centers[:, None, :]
            vertex_offsets = vertex_offsets * cell_node_valid[:, :, None]
            # Inverse map vertex → cells.
            n_nodes = mesh.nodes.shape[0]
            v2c_lists = [[] for _ in range(n_nodes)]
            for c, vs in enumerate(cn):
                for v in vs:
                    v2c_lists[int(v)].append(c)
            v2c_max = max(len(L) for L in v2c_lists) if v2c_lists else 1
            v2c_max = max(v2c_max, 1)
            v2c_padded = np.full((n_nodes, v2c_max), -1, dtype=int)
            for vi, cs in enumerate(v2c_lists):
                v2c_padded[vi, :len(cs)] = cs
            v2c_valid = v2c_padded >= 0
            v2c_safe = np.where(v2c_valid, v2c_padded, 0)
            fn = getattr(mesh, 'face_nodes', None)
            if fn and interior.size > 0:
                n_v_per_face = max(len(vs) for vs in fn)
                face_node_arr = np.full((mesh.n_faces, n_v_per_face), -1,
                                        dtype=int)
                for f, vs in enumerate(fn):
                    face_node_arr[f, :len(vs)] = vs
                face_node_int = face_node_arr[interior]
                face_node_int_valid = face_node_int >= 0
                face_node_int_safe = np.where(face_node_int_valid,
                                              face_node_int, 0)
            else:
                face_node_int_safe = None
                face_node_int_valid = None
        else:
            cell_node_arr = None
            cell_node_valid = None
            vertex_offsets = None
            v2c_padded = v2c_safe = v2c_valid = None
            face_node_int_safe = None
            face_node_int_valid = None

        if interior.size > 0:
            face_n_o = mesh.face_normals[interior]
            d_sq = np.sum(d_o_int * d_o_int, axis=1)
            safe_d_sq = np.maximum(d_sq, 1e-30)
            alpha_o = np.sum(dx_fo * d_o_int, axis=1) / safe_d_sq
            alpha_n = np.sum(dx_fn * (-d_o_int), axis=1) / safe_d_sq
            alpha_o = np.clip(alpha_o, 0.0, 1.0)
            alpha_n = np.clip(alpha_n, 0.0, 1.0)
            norm_den = np.sum(d_o_int * face_n_o, axis=1)
            norm_den_safe = np.where(np.abs(norm_den) > 1e-30,
                                     norm_den, np.copysign(1e-30, norm_den))
            tstar_o = np.sum(dx_fo * face_n_o, axis=1) / norm_den_safe
            tstar_n = np.sum(dx_fn * (-face_n_o), axis=1) / norm_den_safe
            tstar_o = np.where(np.abs(norm_den) > 1e-30, tstar_o, alpha_o)
            tstar_n = np.where(np.abs(norm_den) > 1e-30, tstar_n, alpha_n)
            if self.clamp_tstar:
                tstar_o = np.clip(tstar_o, 0.0, 1.0)
                tstar_n = np.clip(tstar_n, 0.0, 1.0)
            d_len = np.sqrt(np.maximum(d_sq, 1e-30))
            e_o = d_o_int / d_len[:, None]
            e_n = -e_o
        else:
            face_n_o = np.zeros((0, mesh.dim), dtype=float)
            d_sq = np.zeros(0, dtype=float)
            alpha_o = np.zeros(0, dtype=float)
            alpha_n = np.zeros(0, dtype=float)
            tstar_o = np.zeros(0, dtype=float)
            tstar_n = np.zeros(0, dtype=float)
            d_len = np.zeros(0, dtype=float)
            e_o = np.zeros_like(d_o_int)
            e_n = np.zeros_like(d_o_int)

        ctx = dict(
            nb_padded=nb_padded, nb_safe=nb_safe, valid_nb=valid_nb,
            A=A, ATA_inv=ATA_inv, lsq_op=lsq_op, nbasis=nbasis,
            sqrt_w=sqrt_w,
            grad_nb_padded=grad_nb_padded,
            grad_nb_safe=grad_nb_safe,
            grad_valid_nb=grad_valid_nb,
            grad_A=grad_A,
            grad_sqrt_w=grad_sqrt_w,
            grad_lsq_op=grad_lsq_op,
            rank_aware_lsq_cells=rank_aware_cells,
            interior=interior,
            UU_o_int=UU_o_int, UU_n_int=UU_n_int,
            d_o_int=d_o_int,
            dx_fo=dx_fo, dx_fn=dx_fn,
            d_sq=d_sq, alpha_o=alpha_o, alpha_n=alpha_n,
            face_n_o=face_n_o, tstar_o=tstar_o, tstar_n=tstar_n,
            d_len=d_len, e_o=e_o, e_n=e_n,
            order=self.order, stencil=self.stencil,
            cell_node_arr=cell_node_arr,
            cell_node_valid=cell_node_valid,
            vertex_offsets=vertex_offsets,
            v2c_safe=v2c_safe, v2c_valid=v2c_valid,
            face_node_int_safe=face_node_int_safe,
            face_node_int_valid=face_node_int_valid,
        )
        setattr(mesh, cache_key, ctx)
        return ctx


@dataclass
class TMLPUBVD(Reconstruction):
    """BVD selection between MLP-u1 and a T-MLP-u candidate.

    The smooth candidate is canonical one-ring MLP-u1.  The compressive
    candidate is T-MLP-u with one supplied ψ_TVD arm.  For each cell and
    variable, the candidate with smaller total boundary variation over the
    cell's incident faces is selected.  This is a no-threshold BVD choice:
    smooth regions keep the lower-variation MLP-u1 state, while sharp
    interfaces may select the T-MLP-u state when it reduces boundary
    variation.
    """
    name: str = 't_mlp_u_bvd'
    tvd: object = 'tmlpu_shape'
    stencil: str = 'vertex'
    order: int = 1
    idw_p: float = 2.0
    vertex_mlp_cap: float = 2.0
    vertex_mlp_face_local_branch: bool = False
    face_skew_correction: bool = True
    face_gradient_correction: str = 'beta'
    vertex_mlp_augment: bool = False
    r_form: str = 'far_upwind'
    clamp_tstar: bool = True
    hancock_courant: float = 0.0
    moment_bvd: bool = False
    moment_bvd_mode: str = 'and'
    moment_bvd_normalize_length: bool = False
    interface_force_tmlpu: bool = False
    interface_force_range: float = 0.75
    interface_force_only: bool = False

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        base = MLPU1()
        tmlpu = TMLPU(
            tvd=self.tvd,
            mlp_bound=True,
            extremum_relax=False,
            tvb_M=0.0,
            virtual_uu_gradient=True,
            stencil=self.stencil,
            order=self.order,
            idw_p=self.idw_p,
            hancock_courant=self.hancock_courant,
            vertex_mlp=True,
            vertex_mlp_cap=self.vertex_mlp_cap,
            vertex_mlp_face_local=self.vertex_mlp_face_local_branch,
            vertex_mlp_augment=self.vertex_mlp_augment,
            face_skew_correction=self.face_skew_correction,
            face_gradient_correction=self.face_gradient_correction,
            r_form=self.r_form,
            clamp_tstar=self.clamp_tstar,
        )
        A_L, A_R = base.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        B_L, B_R = tmlpu.reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)

        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return A_L, A_R

        nvar, n_cells = W_cell.shape
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

        def _bvd_scores(left, right):
            tbv = np.zeros((nvar, n_cells), dtype=float)
            jump = np.abs(left[:, interior] - right[:, interior])
            jump_w = jump * bvd_weight[None, :]
            if nvar == 1:
                j = jump_w[0]
                tbv[0] = (
                    np.bincount(o_idx, weights=j, minlength=n_cells)
                    + np.bincount(n_idx, weights=j, minlength=n_cells))
                if not self.moment_bvd:
                    return tbv, None
                face_c = score_points
                d_o = face_c - mesh.cell_centers[o_idx]
                d_n = face_c - mesh.cell_centers[n_idx]
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
                mbv = np.sqrt(mx * mx + my * my)[None, :]
                if self.moment_bvd_normalize_length:
                    h = np.sqrt(np.maximum(
                        np.asarray(mesh.cell_volumes, dtype=float),
                        np.finfo(float).tiny))
                    mbv = mbv / h[None, :]
                return tbv, mbv
            if not self.moment_bvd:
                for vv in range(nvar):
                    np.add.at(tbv[vv], o_idx, jump_w[vv])
                    np.add.at(tbv[vv], n_idx, jump_w[vv])
                return tbv, None
            mx = np.zeros((nvar, n_cells), dtype=float)
            my = np.zeros((nvar, n_cells), dtype=float)
            face_c = score_points
            d_o = face_c - mesh.cell_centers[o_idx]
            d_n = face_c - mesh.cell_centers[n_idx]
            for vv in range(nvar):
                j = jump_w[vv]
                np.add.at(tbv[vv], o_idx, j)
                np.add.at(tbv[vv], n_idx, j)
                np.add.at(mx[vv], o_idx, j * d_o[:, 0])
                np.add.at(my[vv], o_idx, j * d_o[:, 1])
                np.add.at(mx[vv], n_idx, j * d_n[:, 0])
                np.add.at(my[vv], n_idx, j * d_n[:, 1])
            mbv = np.sqrt(mx * mx + my * my)
            if self.moment_bvd_normalize_length:
                h = np.sqrt(np.maximum(
                    np.asarray(mesh.cell_volumes, dtype=float),
                    np.finfo(float).tiny))
                mbv = mbv / h[None, :]
            return tbv, mbv

        interface_cells = None
        if self.interface_force_tmlpu:
            local_min = np.array(W_cell, copy=True)
            local_max = np.array(W_cell, copy=True)
            for vv in range(nvar):
                np.minimum.at(local_min[vv], o_idx, W_cell[vv, n_idx])
                np.minimum.at(local_min[vv], n_idx, W_cell[vv, o_idx])
                np.maximum.at(local_max[vv], o_idx, W_cell[vv, n_idx])
                np.maximum.at(local_max[vv], n_idx, W_cell[vv, o_idx])
            interface_cells = (
                local_max - local_min
                >= max(0.0, float(self.interface_force_range)))
        if self.interface_force_only and interface_cells is not None:
            use_B = interface_cells
        else:
            tbv_A, mbv_A = _bvd_scores(A_L, A_R)
            tbv_B, mbv_B = _bvd_scores(B_L, B_R)
            if self.moment_bvd:
                mode = str(self.moment_bvd_mode).lower()
                if mode == 'moment':
                    use_B = mbv_B < mbv_A
                elif mode == 'product':
                    use_B = (tbv_B * mbv_B) < (tbv_A * mbv_A)
                elif mode == 'combined':
                    use_B = (tbv_B * tbv_B + mbv_B * mbv_B
                             < tbv_A * tbv_A + mbv_A * mbv_A)
                else:
                    use_B = (tbv_B < tbv_A) & (mbv_B < mbv_A)
            else:
                use_B = tbv_B < tbv_A
        if interface_cells is not None and not self.interface_force_only:
            use_B = use_B | interface_cells
        W_L = A_L.copy()
        W_R = A_R.copy()
        for v in range(nvar):
            mask_L = use_B[v, o_idx]
            if np.any(mask_L):
                W_L[v, interior[mask_L]] = B_L[v, interior[mask_L]]
            mask_R = use_B[v, n_idx]
            if np.any(mask_R):
                W_R[v, interior[mask_R]] = B_R[v, interior[mask_R]]
        return W_L, W_R


@dataclass
class TMLPUSmoothSharpBVD(Reconstruction):
    """BVD selection between smooth and sharp all-TMLP-u candidates.

    The default branches are both all-TMLP-u candidates: a smooth bounded-CD
    branch using the least-squares face increment and a sharp TMLP-u preserve
    branch using the skew-corrected TMLP-u face increment.  BVD is evaluated by
    total jump variation times a centroid-moment variation, then the sharp arm
    is admitted more conservatively in locally linear cells as identified by a
    scale-free WLSQ residual indicator.  A Zalesak-style flux-correction bound
    uses the actual SSP-RK substep/quadrature contribution supplied by the
    solver.  No MLP-u1 branch is used.
    """
    name: str = 't_mlp_u_smooth_sharp_bvd'
    smooth_mode: str = 'tmlpu'
    smooth_tvd: object = 'bounded_cd'
    smooth_face_increment: str = 'lsq'
    sharp_tvd: object = 'tmlpu_preserve'
    sharp_face_increment: str = 'tmlpu'
    interface_tvd: object | None = 'mstacs'
    stencil: str = 'vertex'
    order: int = 1
    smooth_stencil: str | None = None
    sharp_stencil: str | None = None
    smooth_order: int | None = None
    sharp_order: int | None = None
    idw_p: float = 1.0
    smooth_idw_p: float | None = None
    smooth_physical_vertex_bounds: bool = False
    smooth_physical_vertex_bounds_value_continuous_otsu: bool = False
    smooth_physical_vertex_bounds_value_upper_otsu: bool = False
    smooth_weak_face_mlp: bool = False
    smooth_weak_face_mlp_smooth_otsu: bool = False
    smooth_weak_face_mlp_range_otsu: bool = False
    smooth_weak_face_mlp_high_range_otsu: bool = False
    smooth_weak_face_mlp_dominant_gap_or_high_range: bool = False
    smooth_weak_face_mlp_value_otsu: bool = False
    smooth_weak_face_mlp_value_upper_otsu: bool = False
    smooth_weak_face_mlp_curved_value_otsu: bool = False
    smooth_weak_face_mlp_value_continuous_otsu: bool = False
    smooth_weak_face_bvd: bool = False
    smooth_weak_face_bvd_linear_otsu: bool = False
    smooth_weak_face_bvd_high_value_otsu: bool = False
    smooth_weak_face_bvd_high_value_or_range_otsu: bool = False
    smooth_curved_value_alt_bvd: bool = False
    smooth_physical_value_alt_bvd: bool = False
    smooth_linear_curved_split_otsu: bool = False
    smooth_cell_vertex_mlp: bool = False
    smooth_extremum_relax: bool = False
    smooth_extremum_relax_curved_otsu: bool = False
    sharp_idw_p: float | None = None
    vertex_mlp_cap: float = 2.0
    vertex_mlp_face_local_branch: bool = False
    vertex_mlp_face_local_otsu: bool = False
    vertex_mlp_face_local_otsu_mode: str = 'range'
    sharp_vertex_mlp_face_local_otsu: bool | None = None
    sharp_vertex_mlp_face_local_otsu_mode: str | None = None
    face_skew_correction: bool = True
    face_gradient_correction: str = 'jasak'
    vertex_mlp_augment: bool = True
    r_form: str = 'far_upwind'
    clamp_tstar: bool = True
    zero_delta_psi: float = 2.0
    moment_bvd: bool = True
    moment_bvd_mode: str = 'product'
    moment_bvd_normalize_length: bool = False
    face_consistent_bvd: bool = False
    sharp_bvd_factor: float = 0.72
    sharp_bvd_factor_mode: str = 'linear_residual'
    bvd_smoothness_gate: bool = False
    bvd_linear_gate: bool = False
    bvd_range_gate: bool = False
    bvd_range_median_gate: bool = False
    interface_range_gate: bool = False
    interface_smoothness_gate: bool = False
    interface_nonlinear_gate: bool = False
    interface_smooth_extrema_guard: bool = False
    interface_face_consistent: bool = False
    interface_face_consistent_except_dominant_gap: bool = False
    interface_face_consistent_except_separable_gap: bool = False
    interface_pair_bound: bool = False
    interface_residual_split_bound: bool = False
    interface_residual_split_mode: str = 'otsu'
    interface_bvd_gate: bool = False
    interface_bvd_gate_otsu_exempt: bool = False
    interface_update_bvd_gate: bool = False
    interface_update_bvd_dominant_exempt: bool = False
    interface_range_median_gate: bool = False
    interface_range_otsu_gate: bool = False
    interface_jump_otsu_gate: bool = False
    interface_face_jump_dominant_gate: bool = False
    interface_value_gap_gate: bool = False
    interface_value_gap_relax_consistency: bool = False
    interface_value_separability_gate: bool = False
    interface_thin_gap_boost: bool = False
    interface_thin_gap_dominant_gate: bool = False
    interface_thin_gap_face_only: bool = False
    interface_thin_gap_bvd_exempt: bool = False
    interface_thin_gap_update_exempt: bool = False
    interface_range_fraction: float = 0.5
    interface_thinc: bool = False
    interface_thinc_beta: float = 1.6
    local_face_bound: bool = False
    global_face_bound: bool = False
    unit_interval_face_bound: bool = True
    update_bound_cfl: float = 0.0
    update_bound_mode: str = 'zalesak_actual'
    # Euler-only shock/positivity controls propagated to both BVD branch TMLPUs.
    euler_shock_flatten: bool = False
    euler_density_acoustic_flatten: bool = False
    euler_pressure_face_jump_limiter_on: bool = False
    euler_pressure_face_jump_limiter_strength: float = 0.0
    euler_face_positivity_limiter: bool = False
    _timestep_dt: float | None = field(default=None, init=False, repr=False)
    _timestep_total_dt: float | None = field(default=None, init=False, repr=False)
    _quad_weight: float = field(default=1.0, init=False, repr=False)
    _quad_points: object | None = field(default=None, init=False, repr=False)
    _quad_weights: object | None = field(default=None, init=False, repr=False)

    def set_timestep_context(self, dt: float | None, *, total_dt=None,
                             quad_weight=1.0, quad_points=None,
                             quad_weights=None) -> None:
        self._timestep_dt = None if dt is None else float(dt)
        self._timestep_total_dt = (
            self._timestep_dt if total_dt is None else float(total_dt)
        )
        self._quad_weight = float(quad_weight)
        self._quad_points = quad_points
        self._quad_weights = quad_weights

    def _apply_scalar_update_bound(self, mesh, W_cell, eq, W_L, W_R,
                                   eval_points=None):
        """CFL-aware bound-preserving outgoing-state limiter for scalar advection.

        This is a reconstruction-side finite-volume maximum-principle guard:
        outgoing reconstructed states are limited only as much as needed so a
        forward-Euler upwind update with the configured CFL cannot drain more
        than the local [0,1] budget.  SSP-RK stages inherit the same convex
        bound when the reconstruction is recomputed at each stage.
        """
        mode = str(self.update_bound_mode).lower()
        actual_dt = mode.endswith('_actual')
        if actual_dt:
            mode = mode[:-7]
        mono_dt = mode.endswith('_mono')
        if mono_dt:
            mode = mode[:-5]
        if (self.update_bound_cfl <= 0.0 and not mono_dt and not actual_dt
                or getattr(eq, 'nvar', 0) != 1):
            return W_L, W_R
        if not hasattr(eq, 'velocity_at'):
            return W_L, W_R
        pts = mesh.face_centers if eval_points is None else eval_points
        try:
            vel = eq.velocity_at(pts)
            a_dot_n = np.einsum('fi,fi->f', vel, mesh.face_normals,
                                optimize=True)
        except Exception:
            return W_L, W_R
        wmax = float(np.max(np.abs(a_dot_n))) if a_dot_n.size else 0.0
        if not np.isfinite(wmax) or wmax <= 0.0:
            return W_L, W_R
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        area = mesh.face_areas
        vol = mesh.cell_volumes
        if actual_dt:
            if self._timestep_dt is None or self._timestep_dt <= 0.0:
                return W_L, W_R
            dt = float(self._timestep_dt)
        elif mono_dt:
            sigma = np.zeros(mesh.n_cells, dtype=float)
            out_owner = a_dot_n > 0.0
            np.add.at(sigma, owner[out_owner], area[out_owner] * a_dot_n[out_owner])
            intf_mono = nei >= 0
            out_nei = intf_mono & (a_dot_n < 0.0)
            np.add.at(sigma, nei[out_nei], area[out_nei] * (-a_dot_n[out_nei]))
            active = sigma > 0.0
            if not np.any(active):
                return W_L, W_R
            dt = float(np.min(vol[active] / sigma[active]))
        else:
            if not hasattr(mesh, '_cell_length_scale_cache'):
                cf = mesh.cell_faces
                max_f = max((len(faces) for faces in cf), default=1)
                cf_pad = np.full((mesh.n_cells, max_f), -1, dtype=int)
                for c, faces in enumerate(cf):
                    cf_pad[c, :len(faces)] = faces
                valid = cf_pad >= 0
                fa = mesh.face_areas[np.where(valid, cf_pad, 0)]
                fa = np.where(valid, fa, -np.inf)
                mesh._cell_length_scale_cache = mesh.cell_volumes / np.maximum(
                    np.max(fa, axis=1), 1.0e-300)
            dt = float(self.update_bound_cfl) * float(
                np.min(mesh._cell_length_scale_cache)) / wmax
        out_o = a_dot_n > 0.0
        coeff_o = np.where(out_o, dt * area * a_dot_n / vol[owner], 0.0)
        intf = nei >= 0
        out_n = intf & (a_dot_n < 0.0)
        coeff_n = np.zeros_like(a_dot_n)
        coeff_n[out_n] = dt * area[out_n] * (-a_dot_n[out_n]) / vol[nei[out_n]]

        WL = W_L.copy()
        WR = W_R.copy()
        phi = np.clip(W_cell[0], 0.0, 1.0)
        if mode == 'zalesak_quad':
            return WL, WR
        def _total_low_update():
            if (self._timestep_total_dt is None or self._timestep_total_dt <= 0.0
                    or self._quad_points is None or self._quad_weights is None):
                fl_local = dt * area * a_dot_n * np.where(
                    a_dot_n >= 0.0, phi[owner],
                    np.where(intf, phi[np.where(intf, nei, owner)], 0.0))
                du_local = np.zeros(mesh.n_cells, dtype=float)
                np.add.at(du_local, owner, -fl_local / vol[owner])
                np.add.at(du_local, nei[intf], fl_local[intf] / vol[nei[intf]])
                return phi + du_local

            qpts = np.asarray(self._quad_points)
            qw = np.asarray(self._quad_weights, dtype=float)
            du = np.zeros(mesh.n_cells, dtype=float)
            for q, weight in enumerate(qw):
                pts_q = qpts[:, q, :]
                vel_q = eq.velocity_at(pts_q)
                adn_q = np.einsum('fi,fi->f', vel_q, mesh.face_normals,
                                  optimize=True)
                low_R_q = np.zeros_like(adn_q)
                low_R_q[intf] = phi[nei[intf]]
                low_up_q = np.where(adn_q >= 0.0, phi[owner], low_R_q)
                fl_q = (float(self._timestep_total_dt) * float(weight)
                        * area * adn_q * low_up_q)
                np.add.at(du, owner, -fl_q / vol[owner])
                np.add.at(du, nei[intf], fl_q[intf] / vol[nei[intf]])
            return phi + du

        if mode == 'zalesak_budget':
            high_up = np.where(a_dot_n >= 0.0, WL[0], WR[0])
            low_R = np.zeros_like(a_dot_n)
            intf = nei >= 0
            low_R[intf] = phi[nei[intf]]
            low_up = np.where(a_dot_n >= 0.0, phi[owner], low_R)
            fc = dt * area * a_dot_n * (high_up - low_up)
            u_low = _total_low_update()
            a_owner = -fc / vol[owner]
            a_nei = np.zeros_like(fc)
            a_nei[intf] = fc[intf] / vol[nei[intf]]
            p_pos = np.zeros(mesh.n_cells, dtype=float)
            p_neg = np.zeros(mesh.n_cells, dtype=float)
            np.add.at(p_pos, owner, np.maximum(a_owner, 0.0))
            np.add.at(p_neg, owner, np.minimum(a_owner, 0.0))
            np.add.at(p_pos, nei[intf], np.maximum(a_nei[intf], 0.0))
            np.add.at(p_neg, nei[intf], np.minimum(a_nei[intf], 0.0))
            budget = max(0.0, min(1.0, float(self._quad_weight)))
            q_pos = budget * np.maximum(0.0, 1.0 - u_low)
            q_neg = budget * np.minimum(0.0, 0.0 - u_low)
            r_pos = np.ones(mesh.n_cells, dtype=float)
            m = p_pos > 0.0
            r_pos[m] = np.minimum(1.0, np.maximum(0.0, q_pos[m] / p_pos[m]))
            r_neg = np.ones(mesh.n_cells, dtype=float)
            m = p_neg < 0.0
            r_neg[m] = np.minimum(1.0, np.maximum(0.0, q_neg[m] / p_neg[m]))
            theta_o = np.where(a_owner >= 0.0, r_pos[owner], r_neg[owner])
            theta_f = theta_o.copy()
            safe_nei = np.where(intf, nei, owner)
            theta_n = np.where(a_nei >= 0.0, r_pos[safe_nei], r_neg[safe_nei])
            theta_f[intf] = np.minimum(theta_f[intf], theta_n[intf])
            blend = low_up + theta_f * (high_up - low_up)
            left_up = a_dot_n >= 0.0
            right_up = ~left_up
            WL[0, left_up] = blend[left_up]
            WR[0, right_up] = blend[right_up]
        elif mode == 'zalesak':
            high_up = np.where(a_dot_n >= 0.0, WL[0], WR[0])
            low_R = np.zeros_like(a_dot_n)
            intf = nei >= 0
            low_R[intf] = phi[nei[intf]]
            low_up = np.where(a_dot_n >= 0.0, phi[owner], low_R)
            fl = dt * area * a_dot_n * low_up
            fc = dt * area * a_dot_n * (high_up - low_up)
            du_low = np.zeros(mesh.n_cells, dtype=float)
            np.add.at(du_low, owner, -fl / vol[owner])
            np.add.at(du_low, nei[intf], fl[intf] / vol[nei[intf]])
            u_low = phi + du_low
            a_owner = -fc / vol[owner]
            a_nei = np.zeros_like(fc)
            a_nei[intf] = fc[intf] / vol[nei[intf]]
            p_pos = np.zeros(mesh.n_cells, dtype=float)
            p_neg = np.zeros(mesh.n_cells, dtype=float)
            np.add.at(p_pos, owner, np.maximum(a_owner, 0.0))
            np.add.at(p_neg, owner, np.minimum(a_owner, 0.0))
            np.add.at(p_pos, nei[intf], np.maximum(a_nei[intf], 0.0))
            np.add.at(p_neg, nei[intf], np.minimum(a_nei[intf], 0.0))
            r_pos = np.ones(mesh.n_cells, dtype=float)
            m = p_pos > 0.0
            r_pos[m] = np.minimum(1.0, np.maximum(0.0, (1.0 - u_low[m]) / p_pos[m]))
            r_neg = np.ones(mesh.n_cells, dtype=float)
            m = p_neg < 0.0
            r_neg[m] = np.minimum(1.0, np.maximum(0.0, (0.0 - u_low[m]) / p_neg[m]))
            theta_o = np.where(a_owner >= 0.0, r_pos[owner], r_neg[owner])
            theta_f = theta_o.copy()
            theta_n = np.where(a_nei >= 0.0, r_pos[np.where(intf, nei, owner)], r_neg[np.where(intf, nei, owner)])
            theta_f[intf] = np.minimum(theta_f[intf], theta_n[intf])
            blend = low_up + theta_f * (high_up - low_up)
            left_up = a_dot_n >= 0.0
            right_up = ~left_up
            WL[0, left_up] = blend[left_up]
            WR[0, right_up] = blend[right_up]
        elif mode == 'fct':
            high_up = np.where(a_dot_n >= 0.0, WL[0], WR[0])
            low_R = np.zeros_like(a_dot_n)
            intf = nei >= 0
            low_R[intf] = phi[nei[intf]]
            low_up = np.where(a_dot_n >= 0.0, phi[owner], low_R)
            flux_low = a_dot_n * low_up
            flux_high = a_dot_n * high_up
            du_low = np.zeros(mesh.n_cells, dtype=float)
            du_corr = np.zeros(mesh.n_cells, dtype=float)
            fl = dt * area * flux_low
            fc = dt * area * (flux_high - flux_low)
            np.add.at(du_low, owner, -fl / vol[owner])
            np.add.at(du_corr, owner, -fc / vol[owner])
            np.add.at(du_low, nei[intf], fl[intf] / vol[nei[intf]])
            np.add.at(du_corr, nei[intf], fc[intf] / vol[nei[intf]])
            u_low = phi + du_low
            theta_cell = np.ones(mesh.n_cells, dtype=float)
            pos = du_corr > 0.0
            theta_cell[pos] = np.minimum(1.0, np.maximum(0.0, (1.0 - u_low[pos]) / du_corr[pos]))
            neg = du_corr < 0.0
            theta_cell[neg] = np.minimum(1.0, np.maximum(0.0, (0.0 - u_low[neg]) / du_corr[neg]))
            theta_face = theta_cell[owner].copy()
            theta_face[intf] = np.minimum(theta_face[intf], theta_cell[nei[intf]])
            blend = low_up + theta_face * (high_up - low_up)
            left_up = a_dot_n >= 0.0
            right_up = ~left_up
            WL[0, left_up] = blend[left_up]
            WR[0, right_up] = blend[right_up]
        elif mode == 'affine':
            sumc = np.zeros(mesh.n_cells, dtype=float)
            num = np.zeros(mesh.n_cells, dtype=float)
            np.add.at(sumc, owner[out_o], coeff_o[out_o])
            np.add.at(sumc, nei[out_n], coeff_n[out_n])
            np.add.at(num, owner[out_o], coeff_o[out_o] * (WL[0, out_o] - phi[owner[out_o]]))
            np.add.at(num, nei[out_n], coeff_n[out_n] * (WR[0, out_n] - phi[nei[out_n]]))
            theta = np.ones(mesh.n_cells, dtype=float)
            rhs = phi * (1.0 - sumc)
            mask = (num > 0.0) & (rhs < num)
            theta[mask] = np.maximum(0.0, rhs[mask] / num[mask])
            WL[0, out_o] = phi[owner[out_o]] + theta[owner[out_o]] * (WL[0, out_o] - phi[owner[out_o]])
            WR[0, out_n] = phi[nei[out_n]] + theta[nei[out_n]] * (WR[0, out_n] - phi[nei[out_n]])
            q = 1.0 - phi
            numq = np.zeros(mesh.n_cells, dtype=float)
            np.add.at(numq, owner[out_o], coeff_o[out_o] * ((1.0 - WL[0, out_o]) - q[owner[out_o]]))
            np.add.at(numq, nei[out_n], coeff_n[out_n] * ((1.0 - WR[0, out_n]) - q[nei[out_n]]))
            thetaq = np.ones(mesh.n_cells, dtype=float)
            rhsq = q * (1.0 - sumc)
            mask = (numq > 0.0) & (rhsq < numq)
            thetaq[mask] = np.maximum(0.0, rhsq[mask] / numq[mask])
            WL[0, out_o] = 1.0 - (q[owner[out_o]] + thetaq[owner[out_o]] * ((1.0 - WL[0, out_o]) - q[owner[out_o]]))
            WR[0, out_n] = 1.0 - (q[nei[out_n]] + thetaq[nei[out_n]] * ((1.0 - WR[0, out_n]) - q[nei[out_n]]))
        else:
            # Lower-bound drain: sum c_f phi_f <= phi_cell.
            drain = np.zeros(mesh.n_cells, dtype=float)
            np.add.at(drain, owner[out_o], coeff_o[out_o] * WL[0, out_o])
            np.add.at(drain, nei[out_n], coeff_n[out_n] * WR[0, out_n])
            scale = np.ones(mesh.n_cells, dtype=float)
            mask = drain > np.maximum(phi, 0.0) + 64.0 * np.finfo(float).eps
            scale[mask] = np.divide(phi[mask], drain[mask],
                                    out=np.zeros_like(phi[mask]),
                                    where=drain[mask] > 0.0)
            WL[0, out_o] *= scale[owner[out_o]]
            WR[0, out_n] *= scale[nei[out_n]]
            # Upper-bound drain applied to q=1-phi after lower limiting.
            q = 1.0 - phi
            drain_q = np.zeros(mesh.n_cells, dtype=float)
            np.add.at(drain_q, owner[out_o], coeff_o[out_o] * (1.0 - WL[0, out_o]))
            np.add.at(drain_q, nei[out_n], coeff_n[out_n] * (1.0 - WR[0, out_n]))
            scale_q = np.ones(mesh.n_cells, dtype=float)
            mask = drain_q > np.maximum(q, 0.0) + 64.0 * np.finfo(float).eps
            scale_q[mask] = np.divide(q[mask], drain_q[mask],
                                      out=np.zeros_like(q[mask]),
                                      where=drain_q[mask] > 0.0)
            WL[0, out_o] = 1.0 - scale_q[owner[out_o]] * (1.0 - WL[0, out_o])
            WR[0, out_n] = 1.0 - scale_q[nei[out_n]] * (1.0 - WR[0, out_n])
        return np.minimum(np.maximum(WL, 0.0), 1.0), np.minimum(np.maximum(WR, 0.0), 1.0)

    def apply_quadrature_update_bound(self, mesh, W_cell, eq, W_L_quad,
                                      W_R_quad, eval_points_quad,
                                      quad_weights):
        """Apply one MPP flux-correction limiter over all face quadrature points."""
        mode = str(self.update_bound_mode).lower()
        actual_dt = mode.endswith('_actual')
        if actual_dt:
            mode = mode[:-7]
        if mode != 'zalesak_quad' or getattr(eq, 'nvar', 0) != 1:
            return W_L_quad, W_R_quad
        if not hasattr(eq, 'velocity_at'):
            return W_L_quad, W_R_quad
        if self._timestep_total_dt is None or self._timestep_total_dt <= 0.0:
            return W_L_quad, W_R_quad

        WL = np.asarray(W_L_quad, dtype=float).copy()
        WR = np.asarray(W_R_quad, dtype=float).copy()
        pts = np.asarray(eval_points_quad, dtype=float)
        weights = np.asarray(quad_weights, dtype=float)
        nq = int(WL.shape[0])
        if nq == 0:
            return W_L_quad, W_R_quad

        owner = mesh.face_owner
        nei = mesh.face_neighbour
        intf = nei >= 0
        area = mesh.face_areas
        vol = mesh.cell_volumes
        phi = np.clip(W_cell[0], 0.0, 1.0)
        dt = float(self._timestep_total_dt)

        u_low = phi.copy()
        corr_owner = []
        corr_nei = []
        high_up_all = []
        low_up_all = []
        adn_all = []
        for q in range(nq):
            vel_q = eq.velocity_at(pts[:, q, :])
            adn_q = np.einsum('fi,fi->f', vel_q, mesh.face_normals,
                              optimize=True)
            low_R = np.zeros_like(adn_q)
            low_R[intf] = phi[nei[intf]]
            low_up = np.where(adn_q >= 0.0, phi[owner], low_R)
            high_up = np.where(adn_q >= 0.0, WL[q, 0], WR[q, 0])
            fl = dt * float(weights[q]) * area * adn_q * low_up
            fc = dt * float(weights[q]) * area * adn_q * (high_up - low_up)
            np.add.at(u_low, owner, -fl / vol[owner])
            np.add.at(u_low, nei[intf], fl[intf] / vol[nei[intf]])
            a_owner = -fc / vol[owner]
            a_nei = np.zeros_like(fc)
            a_nei[intf] = fc[intf] / vol[nei[intf]]
            corr_owner.append(a_owner)
            corr_nei.append(a_nei)
            high_up_all.append(high_up)
            low_up_all.append(low_up)
            adn_all.append(adn_q)

        p_pos = np.zeros(mesh.n_cells, dtype=float)
        p_neg = np.zeros(mesh.n_cells, dtype=float)
        for q in range(nq):
            a_owner = corr_owner[q]
            a_nei = corr_nei[q]
            np.add.at(p_pos, owner, np.maximum(a_owner, 0.0))
            np.add.at(p_neg, owner, np.minimum(a_owner, 0.0))
            np.add.at(p_pos, nei[intf], np.maximum(a_nei[intf], 0.0))
            np.add.at(p_neg, nei[intf], np.minimum(a_nei[intf], 0.0))

        r_pos = np.ones(mesh.n_cells, dtype=float)
        m = p_pos > 0.0
        r_pos[m] = np.minimum(1.0, np.maximum(0.0, (1.0 - u_low[m]) / p_pos[m]))
        r_neg = np.ones(mesh.n_cells, dtype=float)
        m = p_neg < 0.0
        r_neg[m] = np.minimum(1.0, np.maximum(0.0, (0.0 - u_low[m]) / p_neg[m]))

        safe_nei = np.where(intf, nei, owner)
        for q in range(nq):
            a_owner = corr_owner[q]
            a_nei = corr_nei[q]
            theta_o = np.where(a_owner >= 0.0, r_pos[owner], r_neg[owner])
            theta_f = theta_o.copy()
            theta_n = np.where(a_nei >= 0.0, r_pos[safe_nei], r_neg[safe_nei])
            theta_f[intf] = np.minimum(theta_f[intf], theta_n[intf])
            blend = low_up_all[q] + theta_f * (high_up_all[q] - low_up_all[q])
            left_up = adn_all[q] >= 0.0
            right_up = ~left_up
            WL[q, 0, left_up] = blend[left_up]
            WR[q, 0, right_up] = blend[right_up]

        WL = np.minimum(np.maximum(WL, 0.0), 1.0)
        WR = np.minimum(np.maximum(WR, 0.0), 1.0)
        return [WL[q] for q in range(nq)], [WR[q] for q in range(nq)]

    def _tmlpu_candidate(self, tvd, face_increment, idw_p=None,
                         stencil=None, order=None,
                         physical_vertex_bounds=False,
                         weak_face_mlp=False,
                         extremum_relax=False,
                         extremum_relax_curved_otsu=False,
                         weak_face_mlp_curved_value_otsu=None,
                         weak_face_mlp_value_upper_otsu=None,
                         physical_vertex_bounds_value_continuous_otsu=None,
                         vertex_mlp_face_local=None,
                         vertex_mlp_face_local_otsu=None,
                         vertex_mlp_face_local_otsu_mode=None):
        face_local = (self.vertex_mlp_face_local_branch
                      if vertex_mlp_face_local is None
                      else bool(vertex_mlp_face_local))
        face_local_otsu = (self.vertex_mlp_face_local_otsu
                           if vertex_mlp_face_local_otsu is None
                           else bool(vertex_mlp_face_local_otsu))
        face_local_otsu_mode = (
            self.vertex_mlp_face_local_otsu_mode
            if vertex_mlp_face_local_otsu_mode is None
            else str(vertex_mlp_face_local_otsu_mode))
        curved_value_otsu = (
            self.smooth_weak_face_mlp_curved_value_otsu
            if weak_face_mlp_curved_value_otsu is None
            else bool(weak_face_mlp_curved_value_otsu))
        value_upper_otsu = (
            self.smooth_weak_face_mlp_value_upper_otsu
            if weak_face_mlp_value_upper_otsu is None
            else bool(weak_face_mlp_value_upper_otsu))
        physical_continuous_otsu = (
            self.smooth_physical_vertex_bounds_value_continuous_otsu
            if physical_vertex_bounds_value_continuous_otsu is None
            else bool(physical_vertex_bounds_value_continuous_otsu))
        return TMLPU(
            tvd=tvd,
            extremum_relax=extremum_relax,
            extremum_relax_curved_otsu=extremum_relax_curved_otsu,
            tvb_M=0.0,
            virtual_uu_gradient=True,
            stencil=self.stencil if stencil is None else stencil,
            order=self.order if order is None else order,
            idw_p=self.idw_p if idw_p is None else idw_p,
            vertex_mlp=True,
            vertex_mlp_cap=self.vertex_mlp_cap,
            vertex_mlp_face_local=face_local,
            vertex_mlp_face_local_otsu=face_local_otsu,
            vertex_mlp_face_local_otsu_mode=face_local_otsu_mode,
            vertex_mlp_augment=self.vertex_mlp_augment,
            face_skew_correction=self.face_skew_correction,
            face_gradient_correction=self.face_gradient_correction,
            face_increment=face_increment,
            r_form=self.r_form,
            physical_vertex_bounds=physical_vertex_bounds,
            physical_vertex_bounds_value_continuous_otsu=(
                physical_continuous_otsu),
            physical_vertex_bounds_value_upper_otsu=(
                self.smooth_physical_vertex_bounds_value_upper_otsu),
            weak_face_mlp=weak_face_mlp,
            weak_face_mlp_smooth_otsu=self.smooth_weak_face_mlp_smooth_otsu,
            weak_face_mlp_range_otsu=self.smooth_weak_face_mlp_range_otsu,
            weak_face_mlp_high_range_otsu=(
                self.smooth_weak_face_mlp_high_range_otsu),
            weak_face_mlp_dominant_gap_or_high_range=(
                self.smooth_weak_face_mlp_dominant_gap_or_high_range),
            weak_face_mlp_value_otsu=self.smooth_weak_face_mlp_value_otsu,
            weak_face_mlp_value_upper_otsu=value_upper_otsu,
            weak_face_mlp_curved_value_otsu=(
                curved_value_otsu),
            weak_face_mlp_value_continuous_otsu=(
                self.smooth_weak_face_mlp_value_continuous_otsu),
            clamp_tstar=self.clamp_tstar,
            zero_delta_psi=self.zero_delta_psi,
            euler_shock_flatten=self.euler_shock_flatten,
            euler_density_acoustic_flatten=self.euler_density_acoustic_flatten,
            euler_pressure_face_jump_limiter_on=(
                self.euler_pressure_face_jump_limiter_on),
            euler_pressure_face_jump_limiter_strength=(
                self.euler_pressure_face_jump_limiter_strength),
            euler_face_positivity_limiter=self.euler_face_positivity_limiter,
        )

    def _smooth_candidate(self):
        if str(self.smooth_mode).lower() == 'mlp_u1_tmlpu':
            raise ValueError(
                "TMLPUSmoothSharpBVD smooth branch must be TMLP-u based; "
                "MLP-u1 is reserved for external baselines only.")
        return self._tmlpu_candidate(self.smooth_tvd,
                                     self.smooth_face_increment,
                                     self.smooth_idw_p,
                                     self.smooth_stencil,
                                     self.smooth_order,
                                     self.smooth_physical_vertex_bounds,
                                     self.smooth_weak_face_mlp,
                                     self.smooth_extremum_relax,
                                     self.smooth_extremum_relax_curved_otsu,
                                     vertex_mlp_face_local=(
                                         False if self.smooth_cell_vertex_mlp
                                         else None))

    def _linear_residual_indicator(self, mesh, W_cell):
        """Dimensionless cell smoothness indicator for BVD sharp-arm gating.

        The indicator is the RMS residual of a local linear WLSQ fit,
        normalized by the local neighbour range.  It is zero for locally
        linear smooth data and grows near discontinuities or under-resolved
        nonlinear features.  Geometry-only matrices are cached on the mesh.
        """
        if mesh.dim != 2 or mesh.kind != 'unstructured_2d':
            return None
        cache_key = '_tmlpu_bvd_linear_residual_cache'
        ctx = getattr(mesh, cache_key, None)
        if ctx is None:
            nb_lists = mesh.cell_neighbours
            max_nb = max((len(nbs) for nbs in nb_lists), default=1)
            max_nb = max(max_nb, 1)
            nb = -np.ones((mesh.n_cells, max_nb), dtype=int)
            valid = np.zeros((mesh.n_cells, max_nb), dtype=bool)
            for c, nbs in enumerate(nb_lists):
                nn = [int(n) for n in nbs if n >= 0]
                if nn:
                    nb[c, :len(nn)] = nn
                    valid[c, :len(nn)] = True
            nb_safe = np.where(valid, nb, np.arange(mesh.n_cells)[:, None])
            dx = mesh.cell_centers[nb_safe] - mesh.cell_centers[:, None, :]
            w = np.where(valid, 1.0 / np.maximum(
                np.linalg.norm(dx, axis=2), 1.0e-14) ** 2, 0.0)
            A = dx * np.sqrt(w)[:, :, None]
            ATA = np.einsum('cni,cnj->cij', A, A, optimize=True)
            inv = np.empty_like(ATA)
            eye = np.eye(2)
            for c in range(mesh.n_cells):
                inv[c] = np.linalg.pinv(ATA[c] + 1.0e-14 * eye)
            ctx = {'nb_safe': nb_safe, 'valid': valid, 'A': A,
                   'sqrt_w': np.sqrt(w), 'inv': inv}
            setattr(mesh, cache_key, ctx)
        nb_safe = ctx['nb_safe']
        valid = ctx['valid']
        A = ctx['A']
        sqrt_w = ctx['sqrt_w']
        inv = ctx['inv']
        nvar, n_cells = W_cell.shape
        indicator = np.zeros((nvar, n_cells), dtype=float)
        for v in range(nvar):
            dphi = (W_cell[v, nb_safe] - W_cell[v, :, None]) * sqrt_w
            rhs = np.einsum('cni,cn->ci', A, dphi, optimize=True)
            coeff = np.einsum('cij,cj->ci', inv, rhs, optimize=True)
            pred = np.einsum('cni,ci->cn', A, coeff, optimize=True)
            res = np.where(valid, dphi - pred, 0.0)
            rms = np.sqrt(np.sum(res * res, axis=1)
                          / np.maximum(np.sum(valid, axis=1), 1))
            vals = np.where(valid, W_cell[v, nb_safe], W_cell[v, :, None])
            local_range = np.max(vals, axis=1) - np.min(vals, axis=1)
            indicator[v] = rms / np.maximum(local_range, 1.0e-14)
        return indicator

    @staticmethod
    def _otsu_range_threshold(local_range):
        """Exact one-dimensional Otsu split for each variable's local range."""
        thresholds = np.zeros((local_range.shape[0], 1), dtype=float)
        for v in range(local_range.shape[0]):
            values = np.sort(np.asarray(local_range[v], dtype=float))
            values = values[np.isfinite(values)]
            if values.size < 2 or values[0] == values[-1]:
                thresholds[v, 0] = values[-1] if values.size else 0.0
                continue
            prefix = np.cumsum(values)
            total = prefix[-1]
            counts = np.arange(1, values.size)
            left_mean = prefix[:-1] / counts
            right_count = values.size - counts
            right_mean = (total - prefix[:-1]) / right_count
            between = counts * right_count * (left_mean - right_mean) ** 2
            between[values[:-1] == values[1:]] = -1.0
            idx = int(np.argmax(between))
            thresholds[v, 0] = 0.5 * (values[idx] + values[idx + 1])
        return thresholds

    @staticmethod
    def _otsu_values_threshold(values, default=1.0):
        values = np.sort(np.asarray(values, dtype=float))
        values = values[np.isfinite(values)]
        if values.size < 2 or values[0] == values[-1]:
            return float(values[-1]) if values.size else float(default)
        prefix = np.cumsum(values)
        total = prefix[-1]
        counts = np.arange(1, values.size)
        left_mean = prefix[:-1] / counts
        right_count = values.size - counts
        right_mean = (total - prefix[:-1]) / right_count
        between = counts * right_count * (left_mean - right_mean) ** 2
        between[values[:-1] == values[1:]] = -1.0
        idx = int(np.argmax(between))
        return float(0.5 * (values[idx] + values[idx + 1]))

    @staticmethod
    def _dominant_gap_mask(W_stencil, local_range):
        """Detect locally two-state data without a tuned threshold.

        A sharp scalar interface has one gap in the sorted local stencil that
        dominates the spread inside both value clusters. A smooth ramp, even
        with large total range, has no such dominant gap.
        """
        nvar, n_cells, n_stencil = W_stencil.shape
        if n_stencil < 2:
            return np.zeros((nvar, n_cells), dtype=bool)
        vals = np.sort(np.asarray(W_stencil, dtype=float), axis=2)
        gaps = np.diff(vals, axis=2)
        imax = np.argmax(gaps, axis=2)
        max_gap = np.take_along_axis(
            gaps, imax[:, :, None], axis=2)[:, :, 0]
        left_hi = np.take_along_axis(
            vals, imax[:, :, None], axis=2)[:, :, 0]
        right_lo = np.take_along_axis(
            vals, (imax + 1)[:, :, None], axis=2)[:, :, 0]
        left_range = left_hi - vals[:, :, 0]
        right_range = vals[:, :, -1] - right_lo
        eps = (64.0 * np.finfo(float).eps
               * (1.0 + np.abs(vals[:, :, 0]) + np.abs(vals[:, :, -1])))
        return (
            (local_range > eps)
            & (max_gap > np.maximum(left_range, right_range) + eps)
        )

    @staticmethod
    def _otsu_separability_scores(W_stencil):
        """Return the best two-class Otsu separability for each cell stencil."""
        vals = np.sort(np.asarray(W_stencil, dtype=float), axis=2)
        nvar, n_cells, n_stencil = vals.shape
        if n_stencil < 2:
            return np.zeros((nvar, n_cells), dtype=float)
        total = np.sum(vals, axis=2)
        total_sq = np.sum(vals * vals, axis=2)
        total_var = total_sq - total * total / float(n_stencil)
        best = np.zeros((nvar, n_cells), dtype=float)
        prefix = np.cumsum(vals, axis=2)
        for k in range(1, n_stencil):
            left_sum = prefix[:, :, k - 1]
            right_sum = total - left_sum
            left_n = float(k)
            right_n = float(n_stencil - k)
            left_mean = left_sum / left_n
            right_mean = right_sum / right_n
            between = left_n * right_n * (left_mean - right_mean) ** 2
            best = np.maximum(best, between)
        denom = np.maximum(float(n_stencil) * total_var,
                           64.0 * np.finfo(float).eps)
        return np.minimum(np.maximum(best / denom, 0.0), 1.0)

    def _scalar_update_bvd_scores(self, mesh, W_cell, eq, W_L, W_R,
                                  eval_points=None):
        if getattr(eq, 'nvar', 0) != 1 or not hasattr(eq, 'velocity_at'):
            return None
        pts = mesh.face_centers if eval_points is None else eval_points
        try:
            vel = eq.velocity_at(pts)
            a_dot_n = np.einsum('fi,fi->f', vel, mesh.face_normals,
                                optimize=True)
        except Exception:
            return None
        wmax = float(np.max(np.abs(a_dot_n))) if a_dot_n.size else 0.0
        if not np.isfinite(wmax) or wmax <= 0.0:
            return None
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        intf = nei >= 0
        mode = str(self.update_bound_mode).lower()
        if mode.endswith('_actual'):
            if self._timestep_dt is None or self._timestep_dt <= 0.0:
                return None
            dt = float(self._timestep_dt)
        elif mode.endswith('_mono'):
            sigma = np.zeros(mesh.n_cells, dtype=float)
            out_owner = a_dot_n > 0.0
            np.add.at(sigma, owner[out_owner],
                      mesh.face_areas[out_owner] * a_dot_n[out_owner])
            out_nei = intf & (a_dot_n < 0.0)
            np.add.at(sigma, nei[out_nei],
                      mesh.face_areas[out_nei] * (-a_dot_n[out_nei]))
            active = sigma > 0.0
            if not np.any(active):
                return None
            dt = float(np.min(mesh.cell_volumes[active] / sigma[active]))
        else:
            cfl = self.update_bound_cfl if self.update_bound_cfl > 0.0 else 1.0
            if not hasattr(mesh, '_cell_length_scale_cache'):
                cf = mesh.cell_faces
                max_f = max((len(faces) for faces in cf), default=1)
                cf_pad = np.full((mesh.n_cells, max_f), -1, dtype=int)
                for c, faces in enumerate(cf):
                    cf_pad[c, :len(faces)] = faces
                valid = cf_pad >= 0
                fa = mesh.face_areas[np.where(valid, cf_pad, 0)]
                fa = np.where(valid, fa, -np.inf)
                mesh._cell_length_scale_cache = mesh.cell_volumes / np.maximum(
                    np.max(fa, axis=1), 1.0e-300)
            dt = float(cfl) * float(np.min(mesh._cell_length_scale_cache)) / wmax
        up = np.where(a_dot_n >= 0.0, W_L[0], W_R[0])
        flux = dt * mesh.face_areas * a_dot_n * up
        u = W_cell[0].copy()
        np.add.at(u, owner, -flux / mesh.cell_volumes[owner])
        np.add.at(u, nei[intf], flux[intf] / mesh.cell_volumes[nei[intf]])
        tbv = np.zeros(mesh.n_cells, dtype=float)
        jump = np.abs(u[owner[intf]] - u[nei[intf]])
        weight = mesh.face_areas[intf] * np.abs(a_dot_n[intf])
        contrib = jump * weight
        np.add.at(tbv, owner[intf], contrib)
        np.add.at(tbv, nei[intf], contrib)
        return tbv[None, :]

    def _thinc_candidate(self, mesh, W_cell, eval_points=None):
        """Bounded multidimensional THINC-like reconstruction candidate.

        This supplies only a reconstruction branch for BVD selection.  The
        hyperbolic tangent steepness uses the common MUSCL-THINC-BVD beta
        value from the published THINC-BVD family; all face values are kept
        inside the local self-plus-neighbour bounds before the existing
        update-bound limiter is applied.
        """
        if mesh.dim != 2 or mesh.kind != 'unstructured_2d':
            return self._tmlpu_candidate(
                self.interface_tvd, self.sharp_face_increment,
                self.sharp_idw_p, self.sharp_stencil,
                self.sharp_order).reconstruct(
                    mesh, W_cell, None, eval_points=eval_points)
        ctx = self._tmlpu_candidate(
            self.smooth_tvd, self.smooth_face_increment,
            self.smooth_idw_p, self.smooth_stencil,
            self.smooth_order)._unstructured_cache(mesh)
        if eval_points is None:
            eval_points = mesh.face_centers
        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = ctx['interior']
        nvar, n_cells = W_cell.shape
        W_L = np.empty((nvar, mesh.n_faces), dtype=float)
        W_R = np.empty((nvar, mesh.n_faces), dtype=float)
        n_idx_def = np.maximum(nei, 0)
        for v in range(nvar):
            W_L[v] = W_cell[v, owner]
            W_R[v] = np.where(nei >= 0, W_cell[v, n_idx_def], W_cell[v, owner])
        if interior.size == 0:
            return W_L, W_R
        nb_safe = ctx['nb_safe']
        valid_nb = ctx['valid_nb']
        A = ctx['A']
        ATA_inv = ctx['ATA_inv']
        sqrt_w = ctx['sqrt_w']
        W_self = W_cell[:, :, None]
        W_nb = np.where(valid_nb[None, :, :], W_cell[:, nb_safe], W_self)
        W_stencil = np.concatenate([W_self, W_nb], axis=2)
        local_min = W_stencil.min(axis=2)
        local_max = W_stencil.max(axis=2)
        local_range = local_max - local_min
        coeffs = np.empty((nvar, n_cells, ctx['nbasis']), dtype=float)
        for v in range(nvar):
            delta_w = (W_cell[v, nb_safe] - W_cell[v, :, None]) * valid_nb
            delta_w = delta_w * sqrt_w
            coeffs[v] = np.einsum('cij,ckj,ck->ci',
                                  ATA_inv, A, delta_w, optimize=True)
        node_valid = ctx.get('cell_node_valid')
        vertex_offsets = ctx.get('vertex_offsets')
        beta = float(self.interface_thinc_beta)
        eps = 64.0 * np.finfo(float).eps

        def _side(var, cells, dx_face):
            phi = W_cell[var, cells]
            lo = local_min[var, cells]
            hi = local_max[var, cells]
            rng = hi - lo
            out = phi.copy()
            active = rng > eps * (1.0 + np.abs(lo) + np.abs(hi))
            if not np.any(active):
                return out
            grad = coeffs[var, cells, :2]
            gnorm = np.linalg.norm(grad, axis=1)
            active &= gnorm > eps
            if not np.any(active):
                return out
            q = np.divide(phi - lo, rng, out=np.full_like(phi, 0.5),
                          where=rng > 0.0)
            q = np.clip(q, eps, 1.0 - eps)
            gamma = np.where(q >= 0.5, 1.0, -1.0)
            nvec = grad / np.maximum(gnorm, eps)[:, None]
            nvec = nvec * gamma[:, None]
            if vertex_offsets is not None and node_valid is not None:
                proj = np.abs(np.einsum('cvi,ci->cv',
                                        vertex_offsets[cells], nvec,
                                        optimize=True))
                proj = np.where(node_valid[cells], proj, 0.0)
                h = np.max(proj, axis=1)
            else:
                h = np.sqrt(mesh.cell_volumes[cells])
            h = np.maximum(h, np.sqrt(mesh.cell_volumes[cells]) * eps)
            xi = np.einsum('fi,fi->f', dx_face, nvec, optimize=True) / h
            center = np.arctanh(np.abs(2.0 * q - 1.0))
            qf = 0.5 * (1.0 + gamma * np.tanh(beta * xi + center))
            cand = lo + rng * qf
            cand = np.minimum(np.maximum(cand, lo), hi)
            return np.where(active, cand, out)

        o_idx = owner[interior]
        n_idx = nei[interior]
        dx_fo = eval_points[interior] - mesh.cell_centers[o_idx]
        dx_fn = eval_points[interior] - mesh.cell_centers[n_idx]
        for v in range(nvar):
            W_L[v, interior] = _side(v, o_idx, dx_fo)
            W_R[v, interior] = _side(v, n_idx, dx_fn)
        return W_L, W_R

    def reconstruct(self, mesh, W_cell, eq, eval_points=None):
        A_L, A_R = self._smooth_candidate().reconstruct(
            mesh, W_cell, eq, eval_points=eval_points)
        A0_L = A0_R = None
        if self.smooth_weak_face_bvd and self.smooth_weak_face_mlp:
            A0_L, A0_R = self._tmlpu_candidate(
                self.smooth_tvd,
                self.smooth_face_increment,
                self.smooth_idw_p,
                self.smooth_stencil,
                self.smooth_order,
                self.smooth_physical_vertex_bounds,
                False,
                self.smooth_extremum_relax,
                self.smooth_extremum_relax_curved_otsu,
                vertex_mlp_face_local=(
                    False if self.smooth_cell_vertex_mlp else None),
                vertex_mlp_face_local_otsu=None,
                vertex_mlp_face_local_otsu_mode=None,
            ).reconstruct(mesh, W_cell, eq, eval_points=eval_points)
        sharp_otsu = (self.vertex_mlp_face_local_otsu
                      if self.sharp_vertex_mlp_face_local_otsu is None
                      else bool(self.sharp_vertex_mlp_face_local_otsu))
        sharp_otsu_mode = (
            self.vertex_mlp_face_local_otsu_mode
            if self.sharp_vertex_mlp_face_local_otsu_mode is None
            else str(self.sharp_vertex_mlp_face_local_otsu_mode))
        B_L, B_R = self._tmlpu_candidate(
            self.sharp_tvd, self.sharp_face_increment,
            self.sharp_idw_p,
            self.sharp_stencil,
            self.sharp_order,
            vertex_mlp_face_local_otsu=sharp_otsu,
            vertex_mlp_face_local_otsu_mode=sharp_otsu_mode).reconstruct(
                mesh, W_cell, eq, eval_points=eval_points)

        owner = mesh.face_owner
        nei = mesh.face_neighbour
        interior = np.where(nei >= 0)[0]
        if interior.size == 0:
            return A_L, A_R

        nvar, n_cells = W_cell.shape
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

        def _bvd_scores(left, right):
            tbv = np.zeros((nvar, n_cells), dtype=float)
            jump = np.abs(left[:, interior] - right[:, interior])
            jump_w = jump * bvd_weight[None, :]
            if not self.moment_bvd:
                for vv in range(nvar):
                    np.add.at(tbv[vv], o_idx, jump_w[vv])
                    np.add.at(tbv[vv], n_idx, jump_w[vv])
                return tbv, None
            mx = np.zeros((nvar, n_cells), dtype=float)
            my = np.zeros((nvar, n_cells), dtype=float)
            face_c = score_points
            d_o = face_c - mesh.cell_centers[o_idx]
            d_n = face_c - mesh.cell_centers[n_idx]
            for vv in range(nvar):
                j = jump_w[vv]
                np.add.at(tbv[vv], o_idx, j)
                np.add.at(tbv[vv], n_idx, j)
                np.add.at(mx[vv], o_idx, j * d_o[:, 0])
                np.add.at(my[vv], o_idx, j * d_o[:, 1])
                np.add.at(mx[vv], n_idx, j * d_n[:, 0])
                np.add.at(my[vv], n_idx, j * d_n[:, 1])
            mbv = np.sqrt(mx * mx + my * my)
            if self.moment_bvd_normalize_length:
                h = np.sqrt(np.maximum(
                    np.asarray(mesh.cell_volumes, dtype=float),
                    np.finfo(float).tiny))
                mbv = mbv / h[None, :]
            return tbv, mbv

        if self.smooth_curved_value_alt_bvd and self.smooth_weak_face_mlp:
            C_L, C_R = self._tmlpu_candidate(
                self.smooth_tvd,
                self.smooth_face_increment,
                self.smooth_idw_p,
                self.smooth_stencil,
                self.smooth_order,
                self.smooth_physical_vertex_bounds,
                True,
                self.smooth_extremum_relax,
                self.smooth_extremum_relax_curved_otsu,
                weak_face_mlp_curved_value_otsu=True,
                weak_face_mlp_value_upper_otsu=False,
                physical_vertex_bounds_value_continuous_otsu=False,
                vertex_mlp_face_local=(
                    False if self.smooth_cell_vertex_mlp else None),
                vertex_mlp_face_local_otsu=None,
                vertex_mlp_face_local_otsu_mode=None,
            ).reconstruct(mesh, W_cell, eq, eval_points=eval_points)
            tbv_Ac, mbv_Ac = _bvd_scores(A_L, A_R)
            tbv_C, mbv_C = _bvd_scores(C_L, C_R)
            if self.moment_bvd:
                mode = str(self.moment_bvd_mode).lower()
                if mode == 'moment':
                    use_C = mbv_C < mbv_Ac
                elif mode == 'product':
                    use_C = (tbv_C * mbv_C) < (tbv_Ac * mbv_Ac)
                elif mode == 'combined':
                    use_C = (tbv_C * tbv_C + mbv_C * mbv_C
                             < tbv_Ac * tbv_Ac + mbv_Ac * mbv_Ac)
                else:
                    use_C = (tbv_C < tbv_Ac) & (mbv_C < mbv_Ac)
            else:
                use_C = tbv_C < tbv_Ac
            for v in range(nvar):
                A_L[v, interior] = np.where(
                    use_C[v, o_idx], C_L[v, interior], A_L[v, interior])
                A_R[v, interior] = np.where(
                    use_C[v, n_idx], C_R[v, interior], A_R[v, interior])

        if self.smooth_physical_value_alt_bvd and self.smooth_weak_face_mlp:
            C_L, C_R = self._tmlpu_candidate(
                self.smooth_tvd,
                self.smooth_face_increment,
                self.smooth_idw_p,
                self.smooth_stencil,
                self.smooth_order,
                self.smooth_physical_vertex_bounds,
                True,
                self.smooth_extremum_relax,
                self.smooth_extremum_relax_curved_otsu,
                weak_face_mlp_curved_value_otsu=False,
                weak_face_mlp_value_upper_otsu=False,
                physical_vertex_bounds_value_continuous_otsu=True,
                vertex_mlp_face_local=(
                    False if self.smooth_cell_vertex_mlp else None),
                vertex_mlp_face_local_otsu=None,
                vertex_mlp_face_local_otsu_mode=None,
            ).reconstruct(mesh, W_cell, eq, eval_points=eval_points)
            tbv_Ap, mbv_Ap = _bvd_scores(A_L, A_R)
            tbv_C, mbv_C = _bvd_scores(C_L, C_R)
            if self.moment_bvd:
                mode = str(self.moment_bvd_mode).lower()
                if mode == 'moment':
                    use_C = mbv_C < mbv_Ap
                elif mode == 'product':
                    use_C = (tbv_C * mbv_C) < (tbv_Ap * mbv_Ap)
                elif mode == 'combined':
                    use_C = (tbv_C * tbv_C + mbv_C * mbv_C
                             < tbv_Ap * tbv_Ap + mbv_Ap * mbv_Ap)
                else:
                    use_C = (tbv_C < tbv_Ap) & (mbv_C < mbv_Ap)
            else:
                use_C = tbv_C < tbv_Ap
            for v in range(nvar):
                A_L[v, interior] = np.where(
                    use_C[v, o_idx], C_L[v, interior], A_L[v, interior])
                A_R[v, interior] = np.where(
                    use_C[v, n_idx], C_R[v, interior], A_R[v, interior])

        if A0_L is not None and A0_R is not None:
            tbv_Aw, mbv_Aw = _bvd_scores(A_L, A_R)
            tbv_A0, mbv_A0 = _bvd_scores(A0_L, A0_R)
            if self.moment_bvd:
                mode = str(self.moment_bvd_mode).lower()
                if mode == 'moment':
                    use_A0 = mbv_A0 < mbv_Aw
                elif mode == 'product':
                    use_A0 = (tbv_A0 * mbv_A0) < (tbv_Aw * mbv_Aw)
                elif mode == 'combined':
                    use_A0 = (tbv_A0 * tbv_A0 + mbv_A0 * mbv_A0
                              < tbv_Aw * tbv_Aw + mbv_Aw * mbv_Aw)
                else:
                    use_A0 = (tbv_A0 < tbv_Aw) & (mbv_A0 < mbv_Aw)
            else:
                use_A0 = tbv_A0 < tbv_Aw
            if self.smooth_weak_face_bvd_linear_otsu:
                eta = self._linear_residual_indicator(mesh, W_cell)
                if eta is not None:
                    use_A0 &= eta <= self._otsu_range_threshold(eta)
            if (self.smooth_weak_face_bvd_high_value_otsu
                    or self.smooth_weak_face_bvd_high_value_or_range_otsu):
                high_value = np.zeros((nvar, n_cells), dtype=bool)
                for v in range(nvar):
                    values = np.sort(W_cell[v][np.isfinite(W_cell[v])])
                    if values.size < 2 or values[0] == values[-1]:
                        continue
                    prefix = np.cumsum(values)
                    total = prefix[-1]
                    counts = np.arange(1, values.size)
                    left_mean = prefix[:-1] / counts
                    right_count = values.size - counts
                    right_mean = (total - prefix[:-1]) / right_count
                    between = counts * right_count * (left_mean - right_mean) ** 2
                    between[values[:-1] == values[1:]] = -1.0
                    idx = int(np.argmax(between))
                    cutoff = 0.5 * (values[idx] + values[idx + 1])
                    high_value[v] = W_cell[v] >= cutoff
                if self.smooth_weak_face_bvd_high_value_or_range_otsu:
                    ctx = self._tmlpu_candidate(
                        self.smooth_tvd, self.smooth_face_increment,
                        self.smooth_idw_p, self.smooth_stencil,
                        self.smooth_order)._unstructured_cache(mesh)
                    nb_safe = ctx['nb_safe']
                    valid_nb = ctx['valid_nb']
                    W_self = W_cell[:, :, None]
                    W_nb = np.where(
                        valid_nb[None, :, :], W_cell[:, nb_safe], W_self)
                    local_range = np.max(
                        np.concatenate([W_self, W_nb], axis=2), axis=2
                    ) - np.min(
                        np.concatenate([W_self, W_nb], axis=2), axis=2)
                    high_range = (
                        local_range >= self._otsu_range_threshold(local_range))
                    use_A0 &= high_value | high_range
                else:
                    use_A0 &= high_value
            for v in range(nvar):
                A_L[v, interior] = np.where(
                    use_A0[v, o_idx], A0_L[v, interior], A_L[v, interior])
                A_R[v, interior] = np.where(
                    use_A0[v, n_idx], A0_R[v, interior], A_R[v, interior])

        if self.smooth_linear_curved_split_otsu:
            Alt_L, Alt_R = self._tmlpu_candidate(
                self.smooth_tvd,
                self.smooth_face_increment,
                self.smooth_idw_p,
                self.smooth_stencil,
                self.smooth_order,
                False,
                False,
                self.smooth_extremum_relax,
                self.smooth_extremum_relax_curved_otsu,
                vertex_mlp_face_local=(
                    False if self.smooth_cell_vertex_mlp else None),
                vertex_mlp_face_local_otsu=None,
                vertex_mlp_face_local_otsu_mode=None,
            ).reconstruct(mesh, W_cell, eq, eval_points=eval_points)
            eta = self._linear_residual_indicator(mesh, W_cell)
            if eta is not None:
                locally_linear = eta <= self._otsu_range_threshold(eta)
                for v in range(nvar):
                    A_L[v, interior] = np.where(
                        locally_linear[v, o_idx],
                        Alt_L[v, interior], A_L[v, interior])
                    A_R[v, interior] = np.where(
                        locally_linear[v, n_idx],
                        Alt_R[v, interior], A_R[v, interior])

        tbv_A, mbv_A = _bvd_scores(A_L, A_R)
        tbv_B, mbv_B = _bvd_scores(B_L, B_R)
        tbv_cut = np.median(tbv_A, axis=1)[:, None]
        sharp_factor = np.where(tbv_A > tbv_cut, self.sharp_bvd_factor, 1.0)
        if self.moment_bvd:
            if str(self.moment_bvd_mode).lower() == 'moment':
                use_B = mbv_B < mbv_A
            elif str(self.moment_bvd_mode).lower() == 'product':
                score_A = tbv_A * mbv_A
                score_B = tbv_B * mbv_B
                if str(self.sharp_bvd_factor_mode).lower() == 'otsu_ratio':
                    ratio = np.divide(
                        score_B, score_A,
                        out=np.full_like(score_B, np.inf),
                        where=score_A > 64.0 * np.finfo(float).eps)
                    sharp_factor = np.ones_like(score_A)
                    high_tbv = tbv_A > tbv_cut
                    for vv in range(nvar):
                        vals = ratio[vv, high_tbv[vv] & np.isfinite(ratio[vv])]
                        threshold = self._otsu_values_threshold(vals, default=1.0)
                        sharp_factor[vv, high_tbv[vv]] = min(1.0, max(0.0, threshold))
                elif str(self.sharp_bvd_factor_mode).lower() == 'linear_residual':
                    eta = self._linear_residual_indicator(mesh, W_cell)
                    if eta is not None:
                        cutoff = np.median(eta, axis=1)[:, None]
                        locally_linear = eta <= cutoff
                        sharp_factor = np.where(locally_linear,
                                                self.sharp_bvd_factor, 1.0)
                use_B = score_B < (sharp_factor * score_A)
            elif str(self.moment_bvd_mode).lower() == 'combined':
                use_B = (tbv_B * tbv_B + mbv_B * mbv_B
                         < (sharp_factor * tbv_A) ** 2 + mbv_A * mbv_A)
            else:
                use_B = (tbv_B < sharp_factor * tbv_A) & (mbv_B < mbv_A)
        else:
            use_B = tbv_B < sharp_factor * tbv_A
        if self.bvd_smoothness_gate or self.bvd_linear_gate:
            eta = self._linear_residual_indicator(mesh, W_cell)
            if eta is not None:
                cutoff = np.median(eta, axis=1)[:, None]
                if self.bvd_smoothness_gate:
                    use_B &= eta > cutoff
                if self.bvd_linear_gate:
                    use_B &= eta <= cutoff
        if self.bvd_range_gate:
            ctx = self._tmlpu_candidate(
                self.smooth_tvd, self.smooth_face_increment,
                self.smooth_idw_p, self.smooth_stencil,
                self.smooth_order)._unstructured_cache(mesh)
            nb_safe = ctx['nb_safe']
            valid_nb = ctx['valid_nb']
            W_self = W_cell[:, :, None]
            W_nb = np.where(valid_nb[None, :, :], W_cell[:, nb_safe], W_self)
            local_min = np.minimum(W_self.min(axis=2), W_nb.min(axis=2))
            local_max = np.maximum(W_self.max(axis=2), W_nb.max(axis=2))
            local_range = local_max - local_min
            global_range = (
                np.max(W_cell, axis=1) - np.min(W_cell, axis=1))[:, None]
            use_B &= local_range >= 0.5 * np.maximum(global_range, 1.0e-14)
        if self.bvd_range_median_gate:
            ctx = self._tmlpu_candidate(
                self.smooth_tvd, self.smooth_face_increment,
                self.smooth_idw_p, self.smooth_stencil,
                self.smooth_order)._unstructured_cache(mesh)
            nb_safe = ctx['nb_safe']
            valid_nb = ctx['valid_nb']
            W_self = W_cell[:, :, None]
            W_nb = np.where(valid_nb[None, :, :], W_cell[:, nb_safe], W_self)
            W_stencil = np.concatenate([W_self, W_nb], axis=2)
            local_min = np.minimum(W_self.min(axis=2), W_nb.min(axis=2))
            local_max = np.maximum(W_self.max(axis=2), W_nb.max(axis=2))
            local_range = local_max - local_min
            use_B &= local_range >= np.median(local_range, axis=1)[:, None]

        W_L = A_L.copy()
        W_R = A_R.copy()
        for v in range(nvar):
            if self.face_consistent_bvd:
                use_face = use_B[v, o_idx] & use_B[v, n_idx]
                W_L[v, interior] = np.where(
                    use_face, B_L[v, interior], A_L[v, interior])
                W_R[v, interior] = np.where(
                    use_face, B_R[v, interior], A_R[v, interior])
            else:
                W_L[v, interior] = np.where(
                    use_B[v, o_idx], B_L[v, interior], A_L[v, interior])
                W_R[v, interior] = np.where(
                    use_B[v, n_idx], B_R[v, interior], A_R[v, interior])
        if self.interface_tvd is not None:
            if self.interface_thinc:
                C_L, C_R = self._thinc_candidate(
                    mesh, W_cell, eval_points=eval_points)
            else:
                C_L, C_R = self._tmlpu_candidate(
                    self.interface_tvd, self.sharp_face_increment,
                    self.sharp_idw_p, self.sharp_stencil,
                    self.sharp_order).reconstruct(
                        mesh, W_cell, eq, eval_points=eval_points)
            ctx = self._tmlpu_candidate(
                self.smooth_tvd, self.smooth_face_increment,
                self.smooth_idw_p, self.smooth_stencil,
                self.smooth_order)._unstructured_cache(mesh)
            nb_safe = ctx['nb_safe']
            valid_nb = ctx['valid_nb']
            W_self = W_cell[:, :, None]
            W_nb = np.where(valid_nb[None, :, :], W_cell[:, nb_safe], W_self)
            W_stencil = np.concatenate([W_self, W_nb], axis=2)
            local_min = np.minimum(W_self.min(axis=2), W_nb.min(axis=2))
            local_max = np.maximum(W_self.max(axis=2), W_nb.max(axis=2))
            local_range = local_max - local_min
            global_range = (
                np.max(W_cell, axis=1) - np.min(W_cell, axis=1))[:, None]
            if self.interface_range_otsu_gate:
                use_C = local_range >= self._otsu_range_threshold(local_range)
            else:
                use_C = local_range >= self.interface_range_fraction * np.maximum(
                    global_range, 1.0e-14)
            value_gap_cell = None
            if (self.interface_value_gap_gate
                    or self.interface_value_gap_relax_consistency):
                value_gap_cell = self._dominant_gap_mask(W_stencil, local_range)
            dominant_gap_cell = value_gap_cell
            if ((self.interface_update_bvd_dominant_exempt
                    or self.interface_face_consistent_except_dominant_gap)
                    and dominant_gap_cell is None):
                dominant_gap_cell = self._dominant_gap_mask(
                    W_stencil, local_range)
            if self.interface_value_separability_gate:
                sep = self._otsu_separability_scores(W_stencil)
                sep_gate = np.zeros_like(use_C, dtype=bool)
                active = local_range > (
                    64.0 * np.finfo(float).eps
                    * (1.0 + np.abs(local_min) + np.abs(local_max)))
                for vv in range(nvar):
                    vals = sep[vv, active[vv]]
                    cutoff = self._otsu_values_threshold(vals, default=np.inf)
                    sep_gate[vv] = sep[vv] >= cutoff
                use_C &= sep_gate
            separable_gap_cell = None
            if self.interface_face_consistent_except_separable_gap:
                sep = self._otsu_separability_scores(W_stencil)
                separable_gap_cell = np.zeros_like(use_C, dtype=bool)
                active = local_range > (
                    64.0 * np.finfo(float).eps
                    * (1.0 + np.abs(local_min) + np.abs(local_max)))
                for vv in range(nvar):
                    vals = sep[vv, active[vv]]
                    cutoff = self._otsu_values_threshold(vals, default=np.inf)
                    separable_gap_cell[vv] = sep[vv] >= cutoff
            if self.interface_range_median_gate:
                use_C &= local_range >= np.median(local_range, axis=1)[:, None]
            jump_gate = np.ones((nvar, interior.size), dtype=bool)
            if self.interface_jump_otsu_gate:
                face_jump = np.abs(W_cell[:, o_idx] - W_cell[:, n_idx])
                for vv in range(nvar):
                    threshold = self._otsu_values_threshold(
                        face_jump[vv], default=np.inf)
                    jump_gate[vv] = face_jump[vv] >= threshold
            if self.interface_face_jump_dominant_gate:
                cell_jump = np.abs(W_cell[:, nb_safe] - W_self) * valid_nb
                max_jump = np.max(cell_jump, axis=2)
                second_jump = np.zeros_like(max_jump)
                if cell_jump.shape[2] > 1:
                    part = np.partition(cell_jump, -2, axis=2)
                    second_jump = part[:, :, -2]
                eps = (64.0 * np.finfo(float).eps
                       * (1.0 + np.abs(local_min) + np.abs(local_max)))
                owner_dom = (
                    (np.abs(W_cell[:, o_idx] - W_cell[:, n_idx])
                     >= max_jump[:, o_idx] - eps[:, o_idx])
                    & (max_jump[:, o_idx] > second_jump[:, o_idx] + eps[:, o_idx])
                )
                neigh_dom = (
                    (np.abs(W_cell[:, o_idx] - W_cell[:, n_idx])
                     >= max_jump[:, n_idx] - eps[:, n_idx])
                    & (max_jump[:, n_idx] > second_jump[:, n_idx] + eps[:, n_idx])
                )
                jump_gate &= owner_dom & neigh_dom
            thin_gap_face = np.zeros((nvar, interior.size), dtype=bool)
            if self.interface_thin_gap_boost:
                gap_cell = (
                    self._dominant_gap_mask(W_stencil, local_range)
                    if self.interface_thin_gap_dominant_gate else None)
                eta_gap = self._linear_residual_indicator(mesh, W_cell)
                eta_cut = (
                    np.median(eta_gap, axis=1)[:, None]
                    if eta_gap is not None else None)
                d_nb = mesh.cell_centers[nb_safe] - mesh.cell_centers[:, None, :]
                d_norm = np.linalg.norm(d_nb, axis=2)
                d_unit = np.divide(
                    d_nb, np.maximum(d_norm, 1.0e-300)[:, :, None],
                    out=np.zeros_like(d_nb),
                    where=d_norm[:, :, None] > 0.0)
                opposite = (
                    np.einsum('cki,cli->ckl', d_unit, d_unit, optimize=True)
                    < 0.0)
                for vv in range(nvar):
                    mid = 0.5 * (local_min[vv] + local_max[vv])
                    high_nb = valid_nb & (W_nb[vv] > mid[:, None])
                    low_cell = W_cell[vv] <= mid
                    has_opposite_high = np.any(
                        high_nb[:, :, None] & high_nb[:, None, :] & opposite,
                        axis=(1, 2))
                    thin_cell = low_cell & has_opposite_high
                    if gap_cell is not None:
                        thin_cell &= gap_cell[vv]
                    if eta_gap is not None:
                        thin_cell &= eta_gap[vv] > eta_cut[vv, 0]
                    if self.interface_thin_gap_face_only:
                        high_from_o = W_cell[vv, n_idx] > mid[o_idx]
                        high_from_n = W_cell[vv, o_idx] > mid[n_idx]
                        thin_gap_face[vv] = (
                            (thin_cell[o_idx] & high_from_o)
                            | (thin_cell[n_idx] & high_from_n)
                        )
                    else:
                        thin_gap_face[vv] = thin_cell[o_idx] | thin_cell[n_idx]
            if self.interface_value_gap_gate and value_gap_cell is not None:
                if self.interface_thin_gap_boost:
                    reopen = np.zeros_like(value_gap_cell, dtype=bool)
                    for vv in range(nvar):
                        np.logical_or.at(reopen[vv], o_idx, thin_gap_face[vv])
                        np.logical_or.at(reopen[vv], n_idx, thin_gap_face[vv])
                    use_C &= value_gap_cell | reopen
                else:
                    use_C &= value_gap_cell
            if self.interface_smoothness_gate:
                eta = self._linear_residual_indicator(mesh, W_cell)
                if eta is not None:
                    cutoff = np.median(eta, axis=1)[:, None]
                    use_C &= eta > cutoff
            if self.interface_nonlinear_gate:
                eta = self._linear_residual_indicator(mesh, W_cell)
                if eta is not None:
                    use_C &= eta > 64.0 * np.finfo(float).eps
            if self.interface_smooth_extrema_guard:
                eta = self._linear_residual_indicator(mesh, W_cell)
                if eta is not None:
                    cutoff = self._otsu_range_threshold(eta)
                    smooth_class = eta <= cutoff
                    eps = (64.0 * np.finfo(float).eps
                           * (1.0 + np.abs(local_min) + np.abs(local_max)))
                    at_extremum = (
                        (W_cell >= local_max - eps)
                        | (W_cell <= local_min + eps)
                    )
                    use_C &= ~(smooth_class & at_extremum)
            if self.interface_bvd_gate:
                tbv_W, mbv_W = _bvd_scores(W_L, W_R)
                tbv_C, mbv_C = _bvd_scores(C_L, C_R)
                if self.moment_bvd:
                    mode = str(self.moment_bvd_mode).lower()
                    if mode == 'moment':
                        gate_ok = mbv_C < mbv_W
                    elif mode == 'product':
                        gate_ok = (tbv_C * mbv_C) < (tbv_W * mbv_W)
                    elif mode == 'combined':
                        gate_ok = (tbv_C * tbv_C + mbv_C * mbv_C
                                   < tbv_W * tbv_W + mbv_W * mbv_W)
                    else:
                        gate_ok = (tbv_C < tbv_W) & (mbv_C < mbv_W)
                else:
                    gate_ok = tbv_C < tbv_W
                if self.interface_bvd_gate_otsu_exempt:
                    high_range = local_range >= self._otsu_range_threshold(local_range)
                    use_C &= high_range | gate_ok
                else:
                    use_C &= gate_ok
                if (self.interface_thin_gap_boost
                        and self.interface_thin_gap_bvd_exempt):
                    reopen = np.zeros_like(use_C, dtype=bool)
                    for vv in range(nvar):
                        np.logical_or.at(reopen[vv], o_idx, thin_gap_face[vv])
                        np.logical_or.at(reopen[vv], n_idx, thin_gap_face[vv])
                    use_C |= reopen
            if self.interface_update_bvd_gate:
                utbv_W = self._scalar_update_bvd_scores(
                    mesh, W_cell, eq, W_L, W_R, eval_points=eval_points)
                utbv_C = self._scalar_update_bvd_scores(
                    mesh, W_cell, eq, C_L, C_R, eval_points=eval_points)
                if utbv_W is not None and utbv_C is not None:
                    update_ok = utbv_C < utbv_W
                    if dominant_gap_cell is not None:
                        update_ok |= dominant_gap_cell
                    use_C &= update_ok
                    if (self.interface_thin_gap_boost
                            and self.interface_thin_gap_update_exempt):
                        reopen = np.zeros_like(use_C, dtype=bool)
                        for vv in range(nvar):
                            np.logical_or.at(
                                reopen[vv], o_idx, thin_gap_face[vv])
                            np.logical_or.at(
                                reopen[vv], n_idx, thin_gap_face[vv])
                        use_C |= reopen
            split_smooth = None
            if self.interface_residual_split_bound:
                eta = self._linear_residual_indicator(mesh, W_cell)
                if eta is not None:
                    split_mode = str(self.interface_residual_split_mode).lower()
                    if split_mode == 'otsu':
                        split_cutoff = self._otsu_range_threshold(eta)
                    elif split_mode == 'median':
                        split_cutoff = np.median(eta, axis=1)[:, None]
                    else:
                        raise ValueError(
                            "interface_residual_split_mode must be 'otsu' "
                            f"or 'median', got {self.interface_residual_split_mode!r}")
                    split_smooth = eta <= split_cutoff
            C_L_raw_int = C_L[:, interior]
            C_R_raw_int = C_R[:, interior]
            if self.interface_pair_bound:
                pair_min = np.minimum(W_cell[:, o_idx], W_cell[:, n_idx])
                pair_max = np.maximum(W_cell[:, o_idx], W_cell[:, n_idx])
                C_L_int = np.minimum(np.maximum(C_L_raw_int, pair_min), pair_max)
                C_R_int = np.minimum(np.maximum(C_R_raw_int, pair_min), pair_max)
            else:
                C_L_int = C_L_raw_int
                C_R_int = C_R_raw_int
            for v in range(nvar):
                if (self.interface_residual_split_bound
                        and split_smooth is not None
                        and self.interface_pair_bound):
                    smooth_o = split_smooth[v, o_idx]
                    smooth_n = split_smooth[v, n_idx]
                    use_o = use_C[v, o_idx] & jump_gate[v]
                    use_n = use_C[v, n_idx] & jump_gate[v]
                    raw_o = use_o & (~smooth_o | thin_gap_face[v])
                    raw_n = use_n & (~smooth_n | thin_gap_face[v])
                    if self.interface_face_consistent:
                        dominant_o = (
                            dominant_gap_cell[v, o_idx]
                            if (self.interface_face_consistent_except_dominant_gap
                                and dominant_gap_cell is not None)
                            else np.zeros_like(use_o, dtype=bool))
                        dominant_n = (
                            dominant_gap_cell[v, n_idx]
                            if (self.interface_face_consistent_except_dominant_gap
                                and dominant_gap_cell is not None)
                            else np.zeros_like(use_n, dtype=bool))
                        if separable_gap_cell is not None:
                            dominant_o = dominant_o | separable_gap_cell[v, o_idx]
                            dominant_n = dominant_n | separable_gap_cell[v, n_idx]
                        pair_face = (use_o & use_n
                                     & smooth_o & smooth_n
                                     & ~thin_gap_face[v]
                                     & ~(dominant_o | dominant_n))
                        W_L[v, interior] = np.where(
                            pair_face, C_L_int[v], W_L[v, interior])
                        W_R[v, interior] = np.where(
                            pair_face, C_R_int[v], W_R[v, interior])
                        split_o = use_o & smooth_o & dominant_o
                        split_n = use_n & smooth_n & dominant_n
                        W_L[v, interior] = np.where(
                            split_o, C_L_int[v], W_L[v, interior])
                        W_R[v, interior] = np.where(
                            split_n, C_R_int[v], W_R[v, interior])
                        if (self.interface_value_gap_relax_consistency
                                and value_gap_cell is not None):
                            relax_o = value_gap_cell[v, o_idx]
                            relax_n = value_gap_cell[v, n_idx]
                            pair_o = use_o & smooth_o & relax_o
                            pair_n = use_n & smooth_n & relax_n
                            W_L[v, interior] = np.where(
                                pair_o, C_L_int[v], W_L[v, interior])
                            W_R[v, interior] = np.where(
                                pair_n, C_R_int[v], W_R[v, interior])
                    else:
                        pair_o = use_C[v, o_idx] & smooth_o
                        pair_n = use_C[v, n_idx] & smooth_n
                        W_L[v, interior] = np.where(
                            pair_o, C_L_int[v], W_L[v, interior])
                        W_R[v, interior] = np.where(
                            pair_n, C_R_int[v], W_R[v, interior])
                    W_L[v, interior] = np.where(
                        raw_o, C_L_raw_int[v], W_L[v, interior])
                    W_R[v, interior] = np.where(
                        raw_n, C_R_raw_int[v], W_R[v, interior])
                    continue
                if self.interface_face_consistent:
                    use_face = (use_C[v, o_idx] & use_C[v, n_idx]
                                & jump_gate[v])
                    W_L[v, interior] = np.where(
                        use_face, C_L_int[v], W_L[v, interior])
                    W_R[v, interior] = np.where(
                        use_face, C_R_int[v], W_R[v, interior])
                else:
                    W_L[v, interior] = np.where(
                        use_C[v, o_idx] & jump_gate[v],
                        C_L_int[v], W_L[v, interior])
                    W_R[v, interior] = np.where(
                        use_C[v, n_idx] & jump_gate[v],
                        C_R_int[v], W_R[v, interior])
        if self.local_face_bound:
            ctx = self._tmlpu_candidate(
                self.smooth_tvd, self.smooth_face_increment,
                self.smooth_idw_p, self.smooth_stencil,
                self.smooth_order)._unstructured_cache(mesh)
            nb_safe = ctx['nb_safe']
            valid_nb = ctx['valid_nb']
            W_self = W_cell[:, :, None]
            W_nb = np.where(valid_nb[None, :, :], W_cell[:, nb_safe], W_self)
            W_stencil = np.concatenate([W_self, W_nb], axis=2)
            cell_min = W_stencil.min(axis=2)
            cell_max = W_stencil.max(axis=2)
            for v in range(nvar):
                W_L[v, interior] = np.minimum(
                    np.maximum(W_L[v, interior], cell_min[v, o_idx]),
                    cell_max[v, o_idx])
                W_R[v, interior] = np.minimum(
                    np.maximum(W_R[v, interior], cell_min[v, n_idx]),
                    cell_max[v, n_idx])
        if self.global_face_bound:
            wmin = np.min(W_cell, axis=1)[:, None]
            wmax = np.max(W_cell, axis=1)[:, None]
            W_L = np.minimum(np.maximum(W_L, wmin), wmax)
            W_R = np.minimum(np.maximum(W_R, wmin), wmax)
        if self.unit_interval_face_bound and getattr(eq, 'nvar', 0) == 1:
            # [0,1] face clamp is a scalar-advection (phi in [0,1]) assumption.
            # It must NOT touch Euler primitives (rho/p/velocity span far beyond
            # [0,1]); clamping them corrupts post-shock states and diverges.
            W_L = np.minimum(np.maximum(W_L, 0.0), 1.0)
            W_R = np.minimum(np.maximum(W_R, 0.0), 1.0)
        W_L, W_R = self._apply_scalar_update_bound(
            mesh, W_cell, eq, W_L, W_R, eval_points=eval_points)
        return W_L, W_R


def _build_vertex_neighbours(mesh, n_rings: int = 1):
    """Cells sharing any vertex with each cell.

    n_rings = 1 : direct vertex-neighbours (Park-Yoon-Kim 2010 default).
    n_rings = 2 : 2-ring  — vertex-neighbours of the 1-ring (much wider
                  stencil; ~25–30 cells / triangle on criss-cross).
    """
    if not getattr(mesh, 'cell_nodes', None):
        return None
    vertex_cells = {}
    for c, nodes in enumerate(mesh.cell_nodes):
        for v in nodes:
            vertex_cells.setdefault(int(v), []).append(c)
    ring1 = []
    for c, nodes in enumerate(mesh.cell_nodes):
        s = set()
        for v in nodes:
            for c2 in vertex_cells[int(v)]:
                if c2 != c:
                    s.add(c2)
        ring1.append(s)
    if n_rings == 1:
        return [sorted(s) for s in ring1]
    out = []
    for c in range(mesh.n_cells):
        s = set(ring1[c])
        for c2 in ring1[c]:
            s.update(ring1[c2])
        s.discard(c)
        out.append(sorted(s))
    return out


def _limited_linear_2d(mesh, W_cell, eq, eval_points=None, *,
                       limiter='bj', stencil='face', vertex_bounds=False,
                       n_rings=1, venkat_K=1.0,
                       tmlpu_face_bound=False,
                       contact_compress=0.0,
                       contact_compress_rho_lo=0.06,
                       contact_compress_rho_hi=0.30,
                       contact_compress_p_tol=0.18):
    """Shared 2D limited-linear reconstruction for BJ/Venkat/MLP-u baselines.

    contact_compress > 0 enables a single-pass artificial-compression
    (anti-diffusion) of the limited-linear face values at *contact*
    interfaces only.  A face is flagged as a contact when the normalised
    density jump is large AND the normalised pressure jump is small
    (Euler), or when the scalar value jump is large (passive scalar).  At
    flagged faces each cell's downwind face value is pushed toward the
    neighbour value by `contact_compress` (Harten-style artificial
    compression); the subsequent `tmlpu_face_bound` clamp keeps every face
    value inside the local MLP min/max so no new extremum is created.
    Shocks (large pressure jump) are excluded, so shock fronts stay clean.
    """
    if mesh.dim != 2:
        return FirstOrder().reconstruct(mesh, W_cell, eq, eval_points=eval_points)
    if mesh.kind not in ('structured_2d', 'unstructured_2d'):
        return FirstOrder().reconstruct(mesh, W_cell, eq, eval_points=eval_points)
    if eval_points is None:
        eval_points = mesh.face_centers

    nvar, N = W_cell.shape
    n_faces = mesh.n_faces
    owner = mesh.face_owner
    nei = mesh.face_neighbour
    cc = mesh.cell_centers

    cache_key = f'_limited_linear_cache_{stencil}_{n_rings}_{int(vertex_bounds)}'
    ctx = getattr(mesh, cache_key, None)
    if ctx is None:
        if stencil in ('vertex', 'vertex2') and getattr(mesh, 'cell_nodes', None):
            nb_lists = _build_vertex_neighbours(
                mesh, n_rings=1 if stencil == 'vertex' else 2)
        else:
            nb_lists = mesh.cell_neighbours
        max_nb = max((len(nbs) for nbs in nb_lists), default=1)
        max_nb = max(max_nb, 1)
        nb = np.full((N, max_nb), -1, dtype=int)
        for c, nbs in enumerate(nb_lists):
            valid = [int(k) for k in nbs if int(k) >= 0]
            nb[c, :len(valid)] = valid
        valid = nb >= 0
        nb_safe = np.where(valid, nb, 0)
        d = (cc[nb_safe] - cc[:, None, :]) * valid[:, :, None]
        ATA = np.einsum('cki,ckj->cij', d, d)
        ATA_inv = np.zeros_like(ATA)
        det = ATA[:, 0, 0] * ATA[:, 1, 1] - ATA[:, 0, 1] * ATA[:, 1, 0]
        ok = np.abs(det) > 1e-30
        det_safe = np.where(ok, det, 1.0)
        ATA_inv[:, 0, 0] = ATA[:, 1, 1] / det_safe
        ATA_inv[:, 1, 1] = ATA[:, 0, 0] / det_safe
        ATA_inv[:, 0, 1] = -ATA[:, 0, 1] / det_safe
        ATA_inv[:, 1, 0] = -ATA[:, 1, 0] / det_safe
        ATA_inv = np.where(ok[:, None, None], ATA_inv, 0.0)

        sample_offsets = []
        sample_vertex_ids = []
        if getattr(mesh, 'cell_nodes', None):
            max_v = max(len(vs) for vs in mesh.cell_nodes)
            vertex_ids = np.full((N, max_v), -1, dtype=int)
            offsets = np.zeros((N, max_v, 2), dtype=float)
            for c, vs in enumerate(mesh.cell_nodes):
                vertex_ids[c, :len(vs)] = vs
                offsets[c, :len(vs)] = mesh.nodes[vs] - cc[c]
            sample_offsets = offsets
            sample_vertex_ids = vertex_ids
        else:
            max_f = max((len(fs) for fs in mesh.cell_faces), default=1)
            offsets = np.zeros((N, max_f, 2), dtype=float)
            vertex_ids = np.full((N, max_f), -1, dtype=int)
            for c, fs in enumerate(mesh.cell_faces):
                pts = mesh.face_centers[fs] - cc[c]
                offsets[c, :len(fs)] = pts
                vertex_ids[c, :len(fs)] = np.arange(len(fs))
            sample_offsets = offsets
            sample_vertex_ids = vertex_ids

        vertex_cell_safe = None
        vertex_cell_valid = None
        if vertex_bounds and getattr(mesh, 'cell_nodes', None):
            n_nodes = mesh.nodes.shape[0]
            v2c = [[] for _ in range(n_nodes)]
            for c, vs in enumerate(mesh.cell_nodes):
                for v in vs:
                    v2c[int(v)].append(c)
            max_v2c = max((len(xs) for xs in v2c), default=1)
            v2c_arr = np.full((n_nodes, max_v2c), -1, dtype=int)
            for v, cs in enumerate(v2c):
                v2c_arr[v, :len(cs)] = cs
            vertex_cell_valid = v2c_arr >= 0
            vertex_cell_safe = np.where(vertex_cell_valid, v2c_arr, 0)

        ctx = dict(nb_safe=nb_safe, valid=valid, d=d, ATA_inv=ATA_inv,
                   sample_offsets=sample_offsets,
                   sample_vertex_ids=sample_vertex_ids,
                   vertex_cell_safe=vertex_cell_safe,
                   vertex_cell_valid=vertex_cell_valid)
        setattr(mesh, cache_key, ctx)

    nb_safe = ctx['nb_safe']
    valid = ctx['valid']
    d = ctx['d']
    ATA_inv = ctx['ATA_inv']
    sample_offsets = ctx['sample_offsets']
    sample_vertex_ids = ctx['sample_vertex_ids']

    W_L = np.empty((nvar, n_faces), dtype=float)
    W_R = np.empty((nvar, n_faces), dtype=float)
    n_idx_def = np.maximum(nei, 0)
    for v in range(nvar):
        W_L[v] = W_cell[v, owner]
        W_R[v] = np.where(nei >= 0, W_cell[v, n_idx_def], W_cell[v, owner])

    delta_nb = (W_cell[:, nb_safe] - W_cell[:, :, None]) * valid[None, :, :]
    rhs = np.einsum('ckj,vck->vcj', d, delta_nb, optimize=True)
    grad = np.einsum('cij,vcj->vci', ATA_inv, rhs, optimize=True)

    W_self = W_cell[:, :, None]
    W_nb_filled = np.where(valid[None, :, :], W_cell[:, nb_safe], W_self)
    W_stencil = np.concatenate([W_self, W_nb_filled], axis=2)
    phi_min_cell = W_stencil.min(axis=2)
    phi_max_cell = W_stencil.max(axis=2)

    sample_delta = np.einsum('vci,csi->vcs', grad, sample_offsets,
                             optimize=True)
    phi_cell = np.ones((nvar, N), dtype=float)
    if vertex_bounds and ctx['vertex_cell_safe'] is not None:
        v2c_safe = ctx['vertex_cell_safe']
        v2c_valid = ctx['vertex_cell_valid']
        safe_vertex_ids = np.where(sample_vertex_ids >= 0, sample_vertex_ids, 0)
        for var in range(nvar):
            vals = W_cell[var, v2c_safe]
            vals = np.where(v2c_valid, vals, W_cell[var, v2c_safe[:, :1]])
            vmin = vals.min(axis=1)
            vmax = vals.max(axis=1)
            lo = vmin[safe_vertex_ids]
            hi = vmax[safe_vertex_ids]
            phis = _limiter_phi(sample_delta[var], W_cell[var, :, None],
                                lo, hi, limiter, mesh.cell_volumes,
                                venkat_K)
            phis = np.where(sample_vertex_ids >= 0, phis, 1.0)
            phi_cell[var] = phis.min(axis=1)
    else:
        for var in range(nvar):
            lo = phi_min_cell[var, :, None]
            hi = phi_max_cell[var, :, None]
            phis = _limiter_phi(sample_delta[var], W_cell[var, :, None],
                                lo, hi, limiter, mesh.cell_volumes,
                                venkat_K)
            phi_cell[var] = phis.min(axis=1)

    interior = np.where(nei >= 0)[0]
    if interior.size == 0:
        return W_L, W_R
    o = owner[interior]
    n = nei[interior]
    dx_o = eval_points[interior] - cc[o]
    dx_n = eval_points[interior] - cc[n]
    for var in range(nvar):
        W_L[var, interior] = (
            W_cell[var, o]
            + phi_cell[var, o] * np.einsum('fi,fi->f', grad[var, o], dx_o)
        )
        W_R[var, interior] = (
            W_cell[var, n]
            + phi_cell[var, n] * np.einsum('fi,fi->f', grad[var, n], dx_n)
        )
    if contact_compress > 0.0:
        # Single-pass artificial compression at contact interfaces only.
        eps = 1e-30
        if nvar >= 4:
            # Euler primitives W = (rho, u, v, p): contact = rho jump w/o p jump.
            rho_o = W_cell[0, o]
            rho_n = W_cell[0, n]
            p_o = W_cell[nvar - 1, o]
            p_n = W_cell[nvar - 1, n]
            rj = np.abs(rho_n - rho_o) / np.maximum(
                np.maximum(np.abs(rho_o), np.abs(rho_n)), eps)
            pj = np.abs(p_n - p_o) / np.maximum(
                np.maximum(np.abs(p_o), np.abs(p_n)), eps)
            rho_gate = np.clip(
                (rj - contact_compress_rho_lo)
                / max(contact_compress_rho_hi - contact_compress_rho_lo, eps),
                0.0, 1.0)
            shock_gate = np.clip(
                1.0 - pj / max(contact_compress_p_tol, eps), 0.0, 1.0)
            cw = rho_gate * shock_gate
        else:
            phi_o = W_cell[0, o]
            phi_n = W_cell[0, n]
            scale = max(float(np.max(np.abs(W_cell[0]))), eps)
            vj = np.abs(phi_n - phi_o) / scale
            cw = np.clip(
                (vj - contact_compress_rho_lo)
                / max(contact_compress_rho_hi - contact_compress_rho_lo, eps),
                0.0, 1.0)
        s = float(contact_compress) * cw
        for var in range(nvar):
            # downwind-biased steepening: each side toward the other cell value
            W_L[var, interior] = (
                W_L[var, interior]
                + s * (W_cell[var, n] - W_L[var, interior]))
            W_R[var, interior] = (
                W_R[var, interior]
                + s * (W_cell[var, o] - W_R[var, interior]))
    if tmlpu_face_bound:
        eps = 1e-30
        for var in range(nvar):
            center_o = W_cell[var, o]
            delta_o = W_L[var, interior] - center_o
            allowed_o = np.where(delta_o >= 0.0,
                                 phi_max_cell[var, o] - center_o,
                                 center_o - phi_min_cell[var, o])
            theta_o = np.where(np.abs(delta_o) > eps,
                               np.maximum(allowed_o, 0.0)
                               / np.maximum(np.abs(delta_o), eps),
                               1.0)
            theta_o = np.clip(theta_o, 0.0, 1.0)
            W_L[var, interior] = center_o + theta_o * delta_o

            center_n = W_cell[var, n]
            delta_n = W_R[var, interior] - center_n
            allowed_n = np.where(delta_n >= 0.0,
                                 phi_max_cell[var, n] - center_n,
                                 center_n - phi_min_cell[var, n])
            theta_n = np.where(np.abs(delta_n) > eps,
                               np.maximum(allowed_n, 0.0)
                               / np.maximum(np.abs(delta_n), eps),
                               1.0)
            theta_n = np.clip(theta_n, 0.0, 1.0)
            W_R[var, interior] = center_n + theta_n * delta_n
    return W_L, W_R


def _limiter_phi(delta, center, lower, upper, limiter, volumes, venkat_K):
    eps = 1e-30
    if limiter == 'venkat':
        h = np.sqrt(np.maximum(volumes, eps))[:, None]
        eps2 = (venkat_K * h) ** 3
        y = np.abs(delta)
        allowed = np.where(delta >= 0.0, upper - center, center - lower)
        allowed = np.maximum(allowed, 0.0)
        num = (allowed * allowed + eps2) * y + 2.0 * y * y * allowed
        den = y * (allowed * allowed + 2.0 * y * y + allowed * y + eps2)
        phi = np.where(y > eps, num / np.maximum(den, eps), 1.0)
    else:
        allowed = np.where(delta >= 0.0, upper - center, center - lower)
        phi = np.where(np.abs(delta) > eps,
                       np.maximum(allowed, 0.0) / np.maximum(np.abs(delta), eps),
                       1.0)
    return np.clip(phi, 0.0, 1.0)


# ─── Registry helper ───────────────────────────────────────────────────────
def get_reconstruction(name: str, **kwargs) -> Reconstruction:
    """Construct a Reconstruction object by name."""
    table = {
        'first_order':  FirstOrder,
        'minmod_tvd_1d': MinmodTVD1D,
        'mlp_u':        MLPU,
        'barth_jespersen': BarthJespersen,
        'barth':        BarthJespersen,
        'bj':           BarthJespersen,
        'venkatakrishnan': Venkatakrishnan,
        'venkat':       Venkatakrishnan,
        'mlp_u1':       MLPU1,
        'mlp_u1_tmlpu': MLPU1TMLPU,
        'mlp_u2':       MLPU2,
        't_mlp_u':      TMLPU,
        't_mlp_u_bvd':  TMLPUBVD,
        'tmlpu_bvd':    TMLPUBVD,
        't_mlp_u_smooth_sharp_bvd': TMLPUSmoothSharpBVD,
        'tmlpu_smooth_sharp_bvd': TMLPUSmoothSharpBVD,
    }
    name = name.lower()
    if name not in table:
        raise ValueError(f"unknown reconstruction '{name}'; available: {list(table)}")
    return table[name](**kwargs)
