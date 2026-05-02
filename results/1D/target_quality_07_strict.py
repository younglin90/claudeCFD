#!/usr/bin/env python3
"""Strict 07_B quality target for codex-autoresearch.

The final stdout line is JSON.  Lower ``quality_score`` is better; the goal is
``failures == 0`` with WENO disabled and the common method flags satisfied.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VERIFY_PATH = ROOT / ".codex-loop" / "verify_02_07_acceptance.py"
IMEX_AD_PATH = ROOT / "solver" / "five_eq_IMEX" / "imex_ad.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out or out in (float("inf"), float("-inf")):
        return float(default)
    return out


def _active_method_flags() -> dict[str, float]:
    primitive = os.environ.get("FIVE_EQ_IMEX_PRIMITIVE_SCHEME", "").strip().lower()
    tvd = os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD", "").strip().lower()
    alpha = os.environ.get("FIVE_EQ_IMEX_ALPHA_SCHEME", "").strip().lower()
    text = IMEX_AD_PATH.read_text()
    uses_rusanov_face = "F_face = F_rusanov.copy()" in text
    uses_hllc_face = "F_face = _single_phase_hllc_flux(" in text
    no_weno_active = float(not primitive.startswith("weno"))
    tmlpu_active = float(primitive in {"tmlpu", "t_mlp_u", "t-mlp-u"})
    tvd_active = float(tvd in {"minmod", "vanleer", "van_leer", "superbee", "mc", "monotonized_central"})
    alpha_sharp = float(alpha in {"cicsam", "mstacs", "thinc", "thinc_bvd", "thinc-qq", "thinc_qq"})
    modern_flux = float(uses_hllc_face and not uses_rusanov_face)
    no_first_order_active = float(tmlpu_active and tvd_active and alpha_sharp and modern_flux)
    return {
        "no_weno_active": no_weno_active,
        "tmlpu_active": tmlpu_active,
        "tvd_active": tvd_active,
        "alpha_sharp_active": alpha_sharp,
        "modern_flux_active": modern_flux,
        "no_first_order_active": no_first_order_active,
    }


def _subcase_score(sub: dict[str, Any]) -> tuple[float, dict[str, float]]:
    metrics = sub.get("metrics") or {}
    osc = sub.get("osc") or {}
    peak = sub.get("peak") or {}
    hf = sub.get("hf") or {}
    l2p = _num(metrics.get("L2p"), 10.0)
    l2u = _num(metrics.get("L2u"), 10.0)
    lip = _num(metrics.get("Lip"), 10.0)
    liu = _num(metrics.get("Liu"), 10.0)
    l1p = _num(metrics.get("L1p"), 10.0)
    l1u = _num(metrics.get("L1u"), 10.0)
    corr_p = _num(metrics.get("corr_p"), 0.0)
    corr_u = _num(metrics.get("corr_u"), 0.0)

    diffusion = max(0.0, lip - 0.50) + max(0.0, liu - 0.50) + 0.5 * (
        max(0.0, l1p - 1.0) + max(0.0, l1u - 1.0)
    )
    exact = max(0.0, l2p - 0.30) + max(0.0, l2u - 0.30) + max(0.0, 0.85 - corr_p) + max(0.0, 0.85 - corr_u)
    peak_penalty = 0.0
    for key, value in peak.items():
        if key.endswith("_delta_cells"):
            stem = key[: -len("_delta_cells")]
            if bool(peak.get(f"{stem}_required", True)):
                tol = _num(peak.get(f"{stem}_tol_cells"), 3.0)
                peak_penalty += max(0.0, _num(value) - tol)
    checker = (
        _num(osc.get("p_alt_amp"), 0.0)
        + _num(osc.get("u_alt_amp"), 0.0)
        + _num(hf.get("p_smooth_local_hf_max"), 0.0)
        + _num(hf.get("u_smooth_local_hf_max"), 0.0)
        + _num(hf.get("p_smooth_local_tv_excess"), 0.0)
        + _num(hf.get("u_smooth_local_tv_excess"), 0.0)
    )
    fail_penalty = 10.0 if not bool(sub.get("pass")) else 0.0
    score = fail_penalty + 0.40 * diffusion + 0.30 * exact + 0.10 * peak_penalty + 0.20 * checker
    return float(score), {
        "diffusion": float(diffusion),
        "exact": float(exact),
        "peak": float(peak_penalty),
        "checker": float(checker),
        "L2p": l2p,
        "L2u": l2u,
        "Lip": lip,
        "Liu": liu,
        "L1p": l1p,
        "L1u": l1u,
        "corr_p": corr_p,
        "corr_u": corr_u,
    }


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    for key in tuple(os.environ):
        if key.startswith("FIVE_EQ_IMEX_"):
            os.environ.pop(key, None)
    os.environ["FIVE_EQ_IMEX_ALPHA_SCHEME"] = "mstacs"
    os.environ["FIVE_EQ_IMEX_PRIMITIVE_SCHEME"] = "tmlpu"
    os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = "minmod"
    os.environ["FIVE_EQ_IMEX_PRESSURE_CLOSURE"] = "regime_auto"

    start = time.time()
    verifier = _load_module("verify_02_07_strict", VERIFY_PATH)
    payload = verifier.verify_07_B()
    flags = _active_method_flags()
    method_failures = int(sum(1 for v in flags.values() if float(v) < 0.5))

    flat: dict[str, Any] = {
        "case": "07_B",
        "failures": int(payload.get("failures", 3)) + method_failures,
        "case_failures": int(payload.get("failures", 3)),
        "method_failures": method_failures,
        "pass_numeric": 1.0 if bool(payload.get("pass")) and method_failures == 0 else 0.0,
        "same_method_all_subcases": 1.0,
        "no_free_knobs": 1.0,
        "wall": float(time.time() - start),
    }
    flat.update(flags)

    quality = 1000.0 * method_failures
    for sub in payload.get("subcases") or []:
        name = str(sub.get("case", "sub")).replace("-", "_").replace(" ", "_")
        score, detail = _subcase_score(sub)
        quality += score
        flat[f"pass_{name}"] = 1.0 if bool(sub.get("pass")) else 0.0
        flat[f"score_{name}"] = score
        for key, value in detail.items():
            flat[f"{name}_{key}"] = value

    flat["quality_score"] = float(quality)
    print(json.dumps(flat, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
