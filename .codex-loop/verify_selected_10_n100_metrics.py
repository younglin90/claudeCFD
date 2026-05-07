#!/usr/bin/env python3
"""Aggregate N=100 metrics for selected 1D validations.

This is the autoresearch target gate for cases:

    01, 02, 04, 05, 07, 13, 14, 15, 24, 25

It intentionally forces one consistent numerical method through environment
defaults: IMEX-SSP3, adaptive-BVD alpha, T-MLP-u primitive reconstruction, and
no characteristic reconstruction.  Each case still writes only
results/1D/{case}/diff_vs_exact.png via the underlying verifier.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEX_LOOP = ROOT / ".codex-loop"


def _load_module(name: str, filename: str):
    for path in (str(ROOT), str(CODEX_LOOP)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(name, CODEX_LOOP / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _set_default_env() -> None:
    defaults = {
        "MPLCONFIGDIR": "/tmp/mpl",
        "FIVE_EQ_IMEX_TIME_INTEGRATOR": "imex_ssp3",
        "FIVE_EQ_IMEX_ALPHA_SCHEME": "adaptive_bvd",
        "FIVE_EQ_IMEX_PRIMITIVE_SCHEME": "tmlpu",
        "FIVE_EQ_IMEX_TMLPU_TVD": "vanleer",
        "FIVE_EQ_IMEX_ACOUSTIC_TVD": "mc",
        "FIVE_EQ_IMEX_ACOUSTIC_WAF": "1",
        "FIVE_EQ_IMEX_CHARACTERISTIC_RECON": "0",
        "FIVE_EQ_IMEX_PRIMITIVE_LMP": "off",
        "FIVE_EQ_IMEX_PURE_HANCOCK": "1",
        "FIVE_EQ_CASE04_N": "100",
        "FIVE_EQ_CASE05_N": "100",
        "FIVE_EQ_CASE07_N": "100",
        "FIVE_EQ_CASE13_N": "100",
        "FIVE_EQ_CASE14_N": "100",
        "FIVE_EQ_CASE15_N": "100",
        "FIVE_EQ_CASE24_N": "100",
        "FIVE_EQ_CASE25_N": "100",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _as_float(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _walk_numbers(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key), value
            yield from _walk_numbers(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _walk_numbers(value)


def _case_quality(result: dict) -> dict[str, float]:
    """Map heterogeneous verifier outputs to the user's 40/30/20/10 buckets."""
    diffusion = 0.0
    exact = 0.0
    peak = 0.0
    checker = 0.0
    for key, value in _walk_numbers(result):
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        v = abs(float(value))
        k = key.lower()
        # Diffusion: amplitude/range loss and correlation loss only.  Do not
        # treat raw field values or cell counts as diffusion.
        if (
            k.endswith("range_ratio")
            or k.endswith("amp_ratio")
            or k in {"rho_corr", "p_corr", "u_corr", "corr_rho", "corr_alpha", "corr_p", "corr_u"}
        ):
            if "range_ratio" in k or k.startswith("corr"):
                diffusion = max(diffusion, abs(1.0 - v))
            elif "amp_ratio" in k:
                diffusion = max(diffusion, max(0.0, 1.0 - v))
            else:
                diffusion = max(diffusion, max(0.0, 1.0 - v))
        if k.endswith("l1_ratio") or k == "mass_rel":
            diffusion = max(diffusion, min(v, 2.0))

        # Exact error: already normalized L2/Linf/profile/scaled metrics.
        if (
            k in {"l2p", "l2u", "lip", "liu"}
            or k.endswith("_profile_l2")
            or k.endswith("_scaled_l2")
            or k.endswith("_smooth_l2_rel")
            or k.endswith("_smooth_linf_rel")
            or k.endswith("_rel_linf")
            or k.endswith("_l2s")
            or k.endswith("_linf")
        ):
            exact = max(exact, v)

        # Peak/shock-location: only normalized overshoot ratios and cell deltas.
        # Raw maxima such as alpha_peak/contact_rho_peak_value are physical
        # state magnitudes and must not dominate the quality score.
        if (
            k.endswith("overshoot")
            or k.endswith("overshoot_ratio")
            or k.endswith("hump_rel_jump")
            or k.endswith("dip_rel_jump")
        ):
            peak = max(peak, v)
        if "shock_cells" in k or k.endswith("_delta_cells"):
            peak = max(peak, v / 10.0)

        # Checkerboard/HF: use normalized oscillation residuals, local TV
        # excess, and turn counts normalized by a 10-cell feature scale.  Exclude
        # diagnostic counts like hf_sharp_cells/hf_smooth_cells.
        if (
            k.endswith("_osc")
            or k.endswith("_cb")
            or k.endswith("_hf_max")
            or k.endswith("_hf_rms")
            or k.endswith("_local_hf_max")
            or k.endswith("_tv_excess")
            or k.endswith("_tv_excess_ratio")
            or k.endswith("_alt_amp")
        ):
            checker = max(checker, v)
        if k.endswith("_turns") or k.endswith("_local_turns"):
            checker = max(checker, v / 10.0)

    return {
        "diffusion_score": float(diffusion),
        "exact_score": float(exact),
        "peak_score": float(peak),
        "checker_score": float(checker),
        "weighted_score": float(
            40.0 * diffusion + 30.0 * exact + 20.0 * peak + 10.0 * checker
        ),
    }


def main() -> int:
    _set_default_env()
    v010306 = _load_module("verify_01_03_06_acceptance", "verify_01_03_06_acceptance.py")
    v0207 = _load_module("verify_02_07_acceptance", "verify_02_07_acceptance.py")
    v0826 = _load_module("verify_08_26_acceptance", "verify_08_26_acceptance.py")

    cases = [
        ("01", v010306.case_01),
        ("02", v0207.verify_02_A),
        ("04", v010306.case_04),
        ("05", v010306.case_05),
        ("07", v0207.verify_07_B),
        ("13", v0826.case_13),
        ("14", v0826.case_14),
        ("15", v0826.case_15),
        ("24", v0826.case_24),
        ("25", v0826.case_25),
    ]
    results = []
    fail_count = 0
    crash_count = 0
    quality_total = 0.0
    for case_id, fn in cases:
        print(f"=== CASE {case_id} N=100 ===", flush=True)
        try:
            result = fn()
            result = dict(result)
            ok = bool(result.get("pass"))
        except Exception as exc:
            ok = False
            crash_count += 1
            traceback.print_exc()
            result = {"case": case_id, "pass": False, "crash": repr(exc)}
        if not ok:
            fail_count += 1
        q = _case_quality(result)
        quality_total += q["weighted_score"]
        result["_quality"] = q
        result["_case_id"] = case_id
        results.append(result)
        print("CASE_JSON " + json.dumps(result, sort_keys=True, default=str), flush=True)

    # Lower is better. Hard failures dominate quality deltas.
    score = 10000.0 * fail_count + 10000.0 * crash_count + quality_total
    summary = {
        "score": float(score),
        "accepted": bool(fail_count == 0 and crash_count == 0),
        "fail_count": int(fail_count),
        "crash_count": int(crash_count),
        "pass_count": int(len(cases) - fail_count),
        "quality_total": float(quality_total),
        "characteristic_used": int(
            os.environ.get("FIVE_EQ_IMEX_CHARACTERISTIC_RECON", "0").lower()
            not in {"0", "false", "off", "no"}
        ),
    }
    for result in results:
        case_id = result["_case_id"]
        summary[f"case{case_id}_pass"] = 1 if bool(result.get("pass")) else 0
        summary[f"case{case_id}_quality"] = float(result["_quality"]["weighted_score"])
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
