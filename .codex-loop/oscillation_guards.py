"""Shared high-frequency oscillation guards for 1D validation scripts."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def _contiguous_segments(mask: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    return [seg for seg in np.split(idx, cuts) if seg.size > 0]


def _field_scale(ref: np.ndarray, floor: float) -> float:
    ref = np.asarray(ref, dtype=float)
    finite = ref[np.isfinite(ref)]
    if finite.size == 0:
        return max(float(floor), 1.0)
    return max(
        float(np.max(finite) - np.min(finite)),
        float(floor),
        1.0e-300,
    )


def _auto_sharp_mask(x: np.ndarray, refs: Sequence[np.ndarray], dx: float,
                     centers: Sequence[float] = (),
                     *, cells: int = 8) -> np.ndarray:
    """Build a sharp-region mask from exact discontinuities and known centers."""
    x = np.asarray(x, dtype=float)
    n = x.size
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    dx = max(float(dx), 1.0e-300)
    grow = max(int(cells), 1)
    for ref in refs:
        r = np.asarray(ref, dtype=float)
        if r.size != n or n < 3:
            continue
        amp = float(np.max(r) - np.min(r))
        if not np.isfinite(amp) or amp <= 1.0e-300:
            continue
        edge = np.abs(np.diff(r))
        med = float(np.median(edge))
        threshold = max(0.15 * amp, 8.0 * med, 1.0e-14)
        edge_idx = np.flatnonzero(edge > threshold)
        for i in edge_idx:
            lo = max(0, int(i) - grow)
            hi = min(n, int(i) + grow + 2)
            mask[lo:hi] = True
    half_width = grow * dx
    for c in centers:
        if c is None:
            continue
        try:
            cf = float(c)
        except (TypeError, ValueError):
            continue
        if math.isfinite(cf):
            mask[np.abs(x - cf) <= half_width] = True
    return mask


def _smooth_hf_metric(num: np.ndarray, ref: np.ndarray, region: np.ndarray,
                      scale: float) -> tuple[float, float]:
    if num.size < 4:
        return 0.0, 0.0
    mid = region[1:-1] & region[:-2] & region[2:]
    if int(np.count_nonzero(mid)) < 3:
        return 0.0, 0.0
    residual = np.asarray(num, dtype=float) - np.asarray(ref, dtype=float)
    d2 = residual[1:-1] - 0.5 * (residual[:-2] + residual[2:])
    vals = np.abs(d2[mid])
    return (
        float(np.max(vals) / scale),
        float(np.sqrt(np.mean(vals * vals)) / scale),
    )


def _count_turns(y: np.ndarray, slope_tol: float) -> int:
    d = np.diff(np.asarray(y, dtype=float))
    active = d[np.abs(d) > max(float(slope_tol), 0.0)]
    if active.size < 2:
        return 0
    s = np.sign(active)
    return int(np.count_nonzero(s[1:] * s[:-1] < 0.0))


def _smooth_local_metrics(
    num: np.ndarray,
    ref: np.ndarray,
    region: np.ndarray,
    floor: float,
    *,
    window_cells: int = 21,
    relative_scale_floor: float = 0.0,
) -> tuple[float, float, int]:
    """Return local smooth-region oscillation metrics.

    The global second-difference metric can miss small-amplitude ringing inside
    a large-scale rarefaction.  This local check compares total variation over
    monotone exact-solution windows and counts numerical slope reversals only
    where the exact profile is monotone in the same window.
    """
    num = np.asarray(num, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if num.size < 5:
        return 0.0, 0.0, 0
    max_local_hf = 0.0
    max_tv_excess = 0.0
    max_turns = 0
    floor = max(float(floor), 1.0e-300)
    for seg in _contiguous_segments(region):
        if seg.size < 5:
            continue
        width = min(max(int(window_cells), 5), int(seg.size))
        for start in range(0, int(seg.size) - width + 1):
            ids = seg[start:start + width]
            nb = num[ids]
            rb = ref[ids]
            rb_diff = np.diff(rb)
            tv_ref = float(np.sum(np.abs(rb_diff))) if rb_diff.size else 0.0
            span_ref = float(abs(rb[-1] - rb[0])) if rb.size else 0.0
            local_var = float(np.max(rb) - np.min(rb)) if rb.size else 0.0
            local_mag = max(
                float(np.max(np.abs(rb))) if rb.size else 0.0,
                float(np.max(np.abs(nb))) if nb.size else 0.0,
                floor,
            )
            local_scale = max(
                tv_ref,
                local_var,
                floor,
                max(float(relative_scale_floor), 0.0) * local_mag,
                1.0e-300,
            )
            # Treat the exact profile as locally monotone if its TV is its
            # endpoint span up to roundoff/interpolation noise.
            ref_monotone = (tv_ref - span_ref) <= 1.0e-10 * local_scale
            if not ref_monotone:
                continue
            tv_num = float(np.sum(np.abs(np.diff(nb)))) if nb.size > 1 else 0.0
            physical_tv = max(tv_ref, span_ref)
            max_tv_excess = max(
                max_tv_excess,
                max(0.0, tv_num - physical_tv) / local_scale,
            )
            residual = nb - rb
            if residual.size >= 3:
                d2 = residual[1:-1] - 0.5 * (residual[:-2] + residual[2:])
                max_local_hf = max(max_local_hf, float(np.max(np.abs(d2))) / local_scale)
            # Ignore roundoff-level derivative sign changes; count visible
            # reversals in monotone rarefaction/plateau windows.
            mean_abs_slope = tv_num / max(float(nb.size - 1), 1.0)
            slope_tol = max(
                1.0e-10 * max(float(np.max(np.abs(nb))), 1.0),
                1.0e-8 * local_scale,
                5.0e-2 * mean_abs_slope,
            )
            turns = _count_turns(nb, slope_tol)
            max_turns = max(max_turns, turns)
    return float(max_local_hf), float(max_tv_excess), int(max_turns)


def _sharp_segment_metrics(num: np.ndarray, ref: np.ndarray, segment: np.ndarray,
                           scale: float) -> tuple[float, float, int]:
    if segment.size < 2:
        return 0.0, 0.0, 0
    nb = np.asarray(num, dtype=float)[segment]
    rb = np.asarray(ref, dtype=float)[segment]
    lo = float(np.min(rb))
    hi = float(np.max(rb))
    tv_ref = float(np.sum(np.abs(np.diff(rb)))) if rb.size > 1 else 0.0
    physical_tv = max(tv_ref, hi - lo, 1.0e-300)
    jump = max(hi - lo, scale)
    overshoot = max(0.0, float(np.max(nb)) - hi, lo - float(np.min(nb))) / jump
    tv = float(np.sum(np.abs(np.diff(nb)))) if nb.size > 1 else 0.0
    tv_excess = max(0.0, tv - physical_tv) / max(physical_tv, scale)
    d = np.diff(nb)
    slope_tol = 0.01 * jump
    active = d[np.abs(d) > slope_tol]
    if active.size < 2:
        turns = 0
    else:
        s = np.sign(active)
        turns = int(np.count_nonzero(s[1:] * s[:-1] < 0.0))
    return float(overshoot), float(tv_excess), turns


def high_frequency_oscillation_guard(
    x: np.ndarray,
    fields: Mapping[str, tuple[np.ndarray, np.ndarray, float]],
    *,
    sharp_centers: Sequence[float] = (),
    smooth_hf_limit: float = 0.08,
    smooth_local_tv_excess_limit: float = 0.50,
    smooth_local_turn_limit: int = 4,
    smooth_local_relative_scale_floor: float = 0.0,
    sharp_overshoot_limit: float = 0.12,
    sharp_tv_excess_limit: float = 0.75,
    sharp_turn_limit: int = 2,
) -> dict[str, float | bool | int]:
    """Check smooth and sharp regions for nonphysical high-frequency content.

    ``fields`` maps a variable name to ``(numerical, exact, scale_floor)``.
    Sharp regions are inferred from exact discontinuities using the same rule
    for all cases.  Smooth regions are tested with second differences of the
    exact residual, while sharp regions are tested for overshoot, excess total
    variation, and repeated slope reversals.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return {"hf_oscillation_ok": True, "hf_sharp_cells": int(n)}
    dx = float(np.median(np.diff(x))) if n > 1 else 1.0
    refs = [np.asarray(v[1], dtype=float) for v in fields.values()]
    sharp_grow_cells = 24
    sharp = _auto_sharp_mask(x, refs, dx, sharp_centers, cells=sharp_grow_cells)
    for num_raw, _, _ in fields.values():
        # A numerically captured shock/contact can be smeared or displaced by a
        # few cells relative to the exact discontinuity.  Treat steep numerical
        # gradients as part of the sharp region as well; otherwise a monotone
        # shock tail is misclassified as smooth-region high-frequency content.
        sharp |= _auto_sharp_mask(
            x, [np.asarray(num_raw, dtype=float)], dx, (),
            cells=sharp_grow_cells)
    smooth = ~sharp
    if n > 4:
        smooth[:2] = False
        smooth[-2:] = False

    metrics: dict[str, float | bool | int] = {
        "hf_sharp_cells": int(np.count_nonzero(sharp)),
        "hf_smooth_cells": int(np.count_nonzero(smooth)),
    }
    ok = True
    for name, (num_raw, ref_raw, floor) in fields.items():
        num = np.asarray(num_raw, dtype=float)
        ref = np.asarray(ref_raw, dtype=float)
        scale = _field_scale(ref, float(floor))
        smooth_max, smooth_rms = _smooth_hf_metric(num, ref, smooth, scale)
        local_hf, local_tv_excess, local_turns = _smooth_local_metrics(
            num, ref, smooth, float(floor),
            relative_scale_floor=smooth_local_relative_scale_floor)
        max_overshoot = 0.0
        max_tv_excess = 0.0
        max_turns = 0
        for seg in _contiguous_segments(sharp):
            overshoot, tv_excess, turns = _sharp_segment_metrics(num, ref, seg, scale)
            max_overshoot = max(max_overshoot, overshoot)
            max_tv_excess = max(max_tv_excess, tv_excess)
            max_turns = max(max_turns, turns)
        field_ok = (
            smooth_max <= smooth_hf_limit
            and (
                local_turns <= smooth_local_turn_limit
                or (
                    local_hf <= smooth_hf_limit
                    and local_tv_excess <= smooth_local_tv_excess_limit
                )
            )
            and max_overshoot <= sharp_overshoot_limit
            and max_tv_excess <= sharp_tv_excess_limit
            and max_turns <= sharp_turn_limit
        )
        ok = bool(ok and field_ok)
        metrics[f"{name}_smooth_hf_max"] = float(smooth_max)
        metrics[f"{name}_smooth_hf_rms"] = float(smooth_rms)
        metrics[f"{name}_smooth_local_hf_max"] = float(local_hf)
        metrics[f"{name}_smooth_local_tv_excess"] = float(local_tv_excess)
        metrics[f"{name}_smooth_local_turns"] = int(local_turns)
        metrics[f"{name}_sharp_overshoot"] = float(max_overshoot)
        metrics[f"{name}_sharp_tv_excess"] = float(max_tv_excess)
        metrics[f"{name}_sharp_turns"] = int(max_turns)
        metrics[f"{name}_hf_ok"] = bool(field_ok)
    metrics["hf_oscillation_ok"] = bool(ok)
    return metrics
