#!/usr/bin/env python3
"""Strict selected-10 quality target with WENO disabled.

This verifier is the mechanical target for the current autoresearch run:

* only cases 01,02,04,05,07,13,14,15,24,25 are executed;
* one common active method is forced for every case;
* alpha uses a sharp-interface algebraic VOF scheme;
* primitive variables use T-MLP-u plus a TVD limiter, never WENO;
* method flags penalize active first-order/upwind or Rusanov-only shortcuts.

The final stdout line is a flat metrics JSON object for codex-autoresearch.
Lower ``quality_score`` is better.
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
BASE_PATH = ROOT / "results" / "1D" / "target_quality_selected_10.py"
IMEX_AD_PATH = ROOT / "solver" / "five_eq_IMEX" / "imex_ad.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _weighted_quality(cases: dict[str, dict[str, Any]], base_score: float) -> tuple[float, dict[str, float]]:
    """Combine diagnostics with the requested failure emphasis.

    The existing case scores already include exact-profile errors and acceptance
    penalties.  This extra aggregate follows the user-specified emphasis:
    diffusion 40%, exact error 30%, peak 10%, checkerboard/HF 20%.
    """
    diffusion = 0.0
    exact = 0.0
    peak = 0.0
    checker = 0.0
    for row in cases.values():
        detail = row.get("detail") or {}
        for key, value in detail.items():
            try:
                v = float(value)
            except Exception:
                continue
            if "diffusion" in key or "range_ratio" in key or "shock_cells" in key:
                diffusion += abs(v)
            elif "profile_l2" in key or "scaled_l2" in key or "corr_deficit" in key:
                exact += abs(v)
            elif "peak" in key or "overshoot" in key or "undershoot" in key:
                peak += abs(v)
            elif "osc" in key or "hf" in key or "tv_excess" in key or "tail_ratio" in key:
                checker += abs(v)
    weighted = 0.40 * diffusion + 0.30 * exact + 0.10 * peak + 0.20 * checker
    return float(base_score + weighted), {
        "diffusion_component": float(diffusion),
        "exact_component": float(exact),
        "peak_component": float(peak),
        "checker_component": float(checker),
        "weighted_component": float(weighted),
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

    base = _load_module("target_quality_selected_10_base", BASE_PATH)
    v010306 = _load_module("verify_01_03_06_noweno", ROOT / ".codex-loop" / "verify_01_03_06_acceptance.py")
    v0207 = _load_module("verify_02_07_noweno", ROOT / ".codex-loop" / "verify_02_07_acceptance.py")
    v0826 = _load_module("verify_08_26_noweno", ROOT / ".codex-loop" / "verify_08_26_acceptance.py")

    start = time.time()
    captured: dict[str, list[dict[str, Any]]] = {}
    orig_multi = v0826._save_multi_plot

    def capture_multi(case_name, rows, title, *args, **kwargs):
        captured[str(case_name)] = list(rows)
        return orig_multi(case_name, rows, title, *args, **kwargs)

    v0826._save_multi_plot = capture_multi

    payloads: dict[str, dict[str, Any]] = {}
    scores: dict[str, tuple[float, dict[str, float]]] = {}
    payloads["01"] = v010306.case_01(); scores["01"] = base._score_01(payloads["01"])
    payloads["02"] = v0207.verify_02_A(); scores["02"] = base._score_02(payloads["02"])
    payloads["04"] = v010306.case_04(); scores["04"] = base._score_acoustic(payloads["04"])
    payloads["05"] = v010306.case_05(); scores["05"] = base._score_acoustic(payloads["05"])
    payloads["07"] = v0207.verify_07_B(); scores["07"] = base._score_07(payloads["07"])
    payloads["13"] = v0826.case_13(); scores["13"] = base._score_shock_case(payloads["13"], captured.get("13_E"))
    payloads["14"] = v0826.case_14(); scores["14"] = base._score_shock_case(payloads["14"], captured.get("14_E"), extra_tail=True)
    payloads["15"] = v0826.case_15(); scores["15"] = base._score_case15(payloads["15"], captured.get("15_E"))
    payloads["24"] = v0826.case_24(); scores["24"] = base._score_case24(payloads["24"])
    payloads["25"] = v0826.case_25(); scores["25"] = base._score_case25(payloads["25"])

    cases = {key: base._case_row(payloads[key], scores[key][0], scores[key][1]) for key in payloads}
    failures = int(sum(0 if row["pass"] else 1 for row in cases.values()))
    free_knobs = base._count_free_knobs()
    base_score = float(sum(row["score"] for row in cases.values()) + 10.0 * free_knobs)
    quality_score, weighted_detail = _weighted_quality(cases, base_score)
    flags = _active_method_flags()
    method_failures = int(sum(1 for v in flags.values() if float(v) < 0.5))
    method_penalty = 1000.0 * method_failures
    quality_score += method_penalty
    total_failures = failures + method_failures

    flat: dict[str, Any] = {
        "quality_score": float(quality_score),
        "physics_quality_score": float(quality_score - method_penalty),
        "method_penalty": float(method_penalty),
        "failures": int(total_failures),
        "case_failures": int(failures),
        "method_failures": int(method_failures),
        "case_pass_count": int(len(cases) - failures),
        "case_total": int(len(cases)),
        "pass_numeric": 1.0 if total_failures == 0 else 0.0,
        "same_method_all_cases": 1.0,
        "only_selected_cases": 1.0,
        "free_knobs": float(free_knobs),
        "wall": float(time.time() - start),
    }
    flat.update(flags)
    flat.update(weighted_detail)
    for key, row in cases.items():
        flat[f"score_{key}"] = row["score"]
        flat[f"pass_{key}"] = 1.0 if row["pass"] else 0.0
        for dk, dv in row["detail"].items():
            flat[f"{key}_{dk}"] = float(dv)

    rich = {
        "pass": bool(total_failures == 0),
        "quality_score": flat["quality_score"],
        "physics_quality_score": flat["physics_quality_score"],
        "failures": total_failures,
        "case_failures": failures,
        "method_failures": method_failures,
        "method_flags": flags,
        "free_knobs": free_knobs,
        "same_method_all_cases": True,
        "only_selected_cases": True,
        "cases": cases,
        "wall": flat["wall"],
    }
    print("SELECTED10_NOWENO_QUALITY_JSON " + json.dumps(rich, sort_keys=True, default=str))
    print(json.dumps(flat, sort_keys=True))
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
