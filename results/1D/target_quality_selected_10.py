#!/usr/bin/env python3
"""Quality verifier for selected 1D cases only: 01,02,04,05,07,13,14,15,24,25.

Final stdout line is metrics JSON for codex-autoresearch.  The metric is lower-is-better
and combines existing acceptance with diffusion/profile errors.  No validation case
outside the selected set is executed.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


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
        v = float(value)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _scaled_l2(num: Any, ref: Any, floor: float = 1.0) -> float:
    a = np.asarray(num, dtype=float)
    b = np.asarray(ref, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(mask)) == 0:
        return 10.0
    amp = max(float(np.nanmax(b[mask]) - np.nanmin(b[mask])), float(floor))
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)) / amp)


def _peak_ratio_penalty(num: Any, ref: Any, floor: float = 1.0) -> float:
    a = np.asarray(num, dtype=float)
    b = np.asarray(ref, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(mask)) == 0:
        return 1.0
    na = float(np.max(np.abs(a[mask])))
    nb = max(float(np.max(np.abs(b[mask]))), float(floor))
    return abs(na / nb - 1.0)


def _row_profile_score(row: dict[str, Any], *, fields=("rho", "u", "p"), floor=1.0) -> tuple[float, dict[str, float]]:
    exact = row.get("exact") or {}
    detail: dict[str, float] = {}
    score = 0.0
    for f in fields:
        if f == "rho":
            num = row.get("rho")
        elif f == "u":
            num = (row.get("W") or [None, None, None, None, None])[3]
        elif f == "p":
            num = (row.get("W") or [None, None, None, None, None])[4]
        elif f == "alpha":
            num = row.get("alpha_num", (row.get("W") or [None])[0])
        else:
            continue
        if num is None or f not in exact:
            continue
        l2 = _scaled_l2(num, exact[f], floor=floor)
        pr = _peak_ratio_penalty(num, exact[f], floor=floor)
        detail[f"{f}_profile_l2"] = l2
        detail[f"{f}_peak_ratio_penalty"] = pr
        score += l2 + 0.10 * pr
    return float(score), detail


def _score_01(payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    detail = {"p_rel": _num(payload.get("p_rel"), 1.0), "u_abs": _num(payload.get("u_abs"), 1.0), "osc": _num(payload.get("osc"), 1.0)}
    return detail["p_rel"] + detail["u_abs"] + detail["osc"], detail


def _score_02(payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    rho_range = _num(payload.get("rho_range_ratio"), 0.0)
    alpha_range = _num(payload.get("alpha_range_ratio"), 0.0)
    detail = {
        "rho_diffusion": max(0.0, 1.0 - rho_range),
        "alpha_diffusion": max(0.0, 1.0 - alpha_range),
        "rho_range_ratio": rho_range,
        "alpha_range_ratio": alpha_range,
        "p_rel_linf": _num(payload.get("p_rel_linf"), 1.0),
        "u_abs_linf": _num(payload.get("u_abs_linf"), 1.0),
    }
    return 2.0 * detail["rho_diffusion"] + detail["alpha_diffusion"] + detail["p_rel_linf"] + detail["u_abs_linf"], detail


def _score_acoustic(payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    detail = {
        "p_scaled_l2": _num(payload.get("p_scaled_l2"), 10.0),
        "u_scaled_l2": _num(payload.get("u_scaled_l2"), 10.0),
        "p_amp_error": abs(_num(payload.get("p_amp_ratio"), 0.0) - 1.0),
        "u_amp_error": abs(_num(payload.get("u_amp_ratio"), 0.0) - 1.0),
        "p_corr_deficit": max(0.0, 1.0 - _num(payload.get("p_corr"), 0.0)),
        "u_corr_deficit": max(0.0, 1.0 - _num(payload.get("u_corr"), 0.0)),
        "osc": _num(payload.get("osc"), 0.0),
    }
    score = detail["p_scaled_l2"] + detail["u_scaled_l2"] + 0.5 * (detail["p_amp_error"] + detail["u_amp_error"]) + 0.25 * (detail["p_corr_deficit"] + detail["u_corr_deficit"]) + detail["osc"]
    return float(score), detail


def _score_07(payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    subcases = payload.get("subcases") or []
    detail: dict[str, float] = {}
    score = 0.0
    for sub in subcases:
        name = str(sub.get("case", "sub")).replace("-", "_").replace(" ", "_")
        metrics = sub.get("metrics") or {}
        osc = sub.get("osc") or {}
        l2p = _num(metrics.get("L2p"), 10.0)
        l2u = _num(metrics.get("L2u"), 10.0)
        lip = _num(metrics.get("Lip"), 10.0)
        liu = _num(metrics.get("Liu"), 10.0)
        cp = max(0.0, 1.0 - _num(metrics.get("corr_p"), 0.0))
        cu = max(0.0, 1.0 - _num(metrics.get("corr_u"), 0.0))
        altp = _num(osc.get("p_alt_amp"), 0.0)
        altu = _num(osc.get("u_alt_amp"), 0.0)
        passed = bool(sub.get("pass", False))
        s = l2p + l2u + 0.25 * (lip + liu) + 0.5 * (cp + cu) + 0.25 * (altp + altu) + (0.0 if passed else 100.0)
        score += s
        detail[f"{name}_score"] = float(s)
        detail[f"{name}_L2p"] = l2p
        detail[f"{name}_L2u"] = l2u
        detail[f"{name}_Lip"] = lip
        detail[f"{name}_Liu"] = liu
    if not subcases:
        return 100.0, {"missing_subcases": 1.0}
    return float(score), detail


def _score_shock_case(payload: dict[str, Any], rows: list[dict[str, Any]] | None, *, extra_tail=False) -> tuple[float, dict[str, float]]:
    detail = {
        "p_osc": _num(payload.get("p_osc"), 1.0),
        "rho_osc": _num(payload.get("rho_osc"), 1.0),
        "undershoot": max(0.0, _num(payload.get("undershoot"), 1.0)),
    }
    score = detail["p_osc"] + detail["rho_osc"] + detail["undershoot"]
    if "contact_rho_peak_overshoot_ratio" in payload:
        detail["contact_rho_peak_overshoot_ratio"] = _num(
            payload.get("contact_rho_peak_overshoot_ratio"), 1.0)
        detail["contact_rho_tv_excess_ratio"] = _num(
            payload.get("contact_rho_tv_excess_ratio"), 1.0)
        score += detail["contact_rho_peak_overshoot_ratio"]
    if rows:
        for idx, row in enumerate(rows):
            s, d = _row_profile_score(row, fields=("rho", "u", "p", "alpha"), floor=1.0)
            score += s
            for k, v in d.items():
                detail[f"row{idx}_{k}"] = v
    if extra_tail:
        tail = _num(payload.get("u_tail_ratio_ref_stagnant"), 1.0)
        detail["u_tail_ratio_ref_stagnant"] = tail
        score += 2.0 * tail
    return float(score), detail


def _score_case15(payload: dict[str, Any], rows: list[dict[str, Any]] | None) -> tuple[float, dict[str, float]]:
    detail = {
        "p_osc": _num(payload.get("p_osc"), 1.0),
        "rho_osc": _num(payload.get("rho_osc"), 1.0),
        "center_u_jump": _num(payload.get("center_u_jump"), 100.0),
        "center_ref_jump": _num(payload.get("center_ref_jump"), 1.0),
    }
    jump_penalty = detail["center_u_jump"] / max(detail["center_ref_jump"], 1.0)
    score = detail["p_osc"] + detail["rho_osc"] + 0.05 * jump_penalty
    if rows:
        for idx, row in enumerate(rows):
            s, d = _row_profile_score(row, fields=("rho", "u", "p", "alpha"), floor=1.0)
            score += s
            for k, v in d.items():
                detail[f"row{idx}_{k}"] = v
    return float(score), detail


def _score_case24(payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    detail: dict[str, float] = {}
    score = 0.0
    for sub in payload.get("subcases") or []:
        name = str(sub.get("name", "sub")).split("=")[-1].replace(".", "p")
        vals = {
            "p_profile_l2": _num(sub.get("p_profile_l2"), 10.0),
            "u_profile_l2": _num(sub.get("u_profile_l2"), 10.0),
            "rho_profile_l2": _num(sub.get("rho_profile_l2"), 10.0),
            "p_hf_osc": _num(sub.get("p_hf_osc"), 1.0),
            "rho_hf_osc": _num(sub.get("rho_hf_osc"), 1.0),
            "shock_cells": _num(sub.get("shock_cells"), 100.0),
        }
        s = vals["p_profile_l2"] + vals["u_profile_l2"] + vals["rho_profile_l2"] + vals["p_hf_osc"] + vals["rho_hf_osc"] + 0.01 * vals["shock_cells"] + (0.0 if sub.get("pass") else 100.0)
        score += s
        for k, v in vals.items():
            detail[f"psi_{name}_{k}"] = v
    if not payload.get("subcases"):
        return 100.0, {"missing_subcases": 1.0}
    return float(score), detail


def _score_case25(payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    detail = {
        "p_scaled_l2": _num(payload.get("p_scaled_l2"), 10.0),
        "u_scaled_l2": _num(payload.get("u_scaled_l2"), 10.0),
        "rho_scaled_l2": _num(payload.get("rho_scaled_l2"), 10.0),
        "shock_delta_cells": _num(payload.get("shock_delta_cells"), 100.0),
        "reflected_shock_delta_cells": _num(payload.get("reflected_shock_delta_cells"), 100.0),
        "iface_delta_cells": _num(payload.get("iface_delta_cells"), 100.0),
        "interface_instability": _num(payload.get("interface_instability"), 1.0),
        "p_osc": _num(payload.get("p_osc"), 1.0),
        "rho_osc": _num(payload.get("rho_osc"), 1.0),
    }
    score = detail["p_scaled_l2"] + detail["u_scaled_l2"] + detail["rho_scaled_l2"] + 0.01 * (detail["shock_delta_cells"] + detail["reflected_shock_delta_cells"] + detail["iface_delta_cells"]) + detail["interface_instability"] + detail["p_osc"] + detail["rho_osc"]
    return float(score), detail


def _case_row(payload: dict[str, Any], score: float, detail: dict[str, float]) -> dict[str, Any]:
    passed = bool(payload.get("pass", False))
    return {
        "case": str(payload.get("case", "unknown")),
        "pass": passed,
        "raw_score": float(score),
        "score": float(score + (0.0 if passed else 100.0)),
        "detail": detail,
    }


def _count_free_knobs() -> int:
    text = (ROOT / "solver" / "five_eq_IMEX" / "imex_ad.py").read_text()
    bad = ["PURE_PRIMITIVE_BLEND"]
    return sum(1 for token in bad if token in text)


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    # Do not let user/env tuning override the common method during verification.
    for key in tuple(os.environ):
        if key.startswith("FIVE_EQ_IMEX_"):
            os.environ.pop(key, None)
    os.environ["FIVE_EQ_IMEX_ALPHA_SCHEME"] = "mstacs"
    os.environ["FIVE_EQ_IMEX_PRIMITIVE_SCHEME"] = "weno3"
    os.environ["FIVE_EQ_IMEX_PRESSURE_CLOSURE"] = "regime_auto"
    start = time.time()

    v010306 = _load_module("verify_01_03_06_selected", ROOT / ".codex-loop" / "verify_01_03_06_acceptance.py")
    v0207 = _load_module("verify_02_07_selected", ROOT / ".codex-loop" / "verify_02_07_acceptance.py")
    v0826 = _load_module("verify_08_26_selected", ROOT / ".codex-loop" / "verify_08_26_acceptance.py")

    captured: dict[str, list[dict[str, Any]]] = {}
    orig_multi = v0826._save_multi_plot

    def capture_multi(case_name, rows, title, *args, **kwargs):
        captured[str(case_name)] = list(rows)
        return orig_multi(case_name, rows, title, *args, **kwargs)

    v0826._save_multi_plot = capture_multi

    payloads: dict[str, dict[str, Any]] = {}
    scores: dict[str, tuple[float, dict[str, float]]] = {}

    payloads["01"] = v010306.case_01(); scores["01"] = _score_01(payloads["01"])
    payloads["02"] = v0207.verify_02_A(); scores["02"] = _score_02(payloads["02"])
    payloads["04"] = v010306.case_04(); scores["04"] = _score_acoustic(payloads["04"])
    payloads["05"] = v010306.case_05(); scores["05"] = _score_acoustic(payloads["05"])
    payloads["07"] = v0207.verify_07_B(); scores["07"] = _score_07(payloads["07"])
    payloads["13"] = v0826.case_13(); scores["13"] = _score_shock_case(payloads["13"], captured.get("13_E"))
    payloads["14"] = v0826.case_14(); scores["14"] = _score_shock_case(payloads["14"], captured.get("14_E"), extra_tail=True)
    payloads["15"] = v0826.case_15(); scores["15"] = _score_case15(payloads["15"], captured.get("15_E"))
    payloads["24"] = v0826.case_24(); scores["24"] = _score_case24(payloads["24"])
    payloads["25"] = v0826.case_25(); scores["25"] = _score_case25(payloads["25"])

    cases = {key: _case_row(payloads[key], scores[key][0], scores[key][1]) for key in payloads}
    failures = int(sum(0 if row["pass"] else 1 for row in cases.values()))
    free_knobs = _count_free_knobs()
    quality_score = float(sum(row["score"] for row in cases.values()) + 10.0 * free_knobs)
    flat: dict[str, Any] = {
        "quality_score": quality_score,
        "failures": failures,
        "case_pass_count": int(len(cases) - failures),
        "case_total": int(len(cases)),
        "pass_numeric": 1.0 if failures == 0 else 0.0,
        "same_method_all_cases": 1.0,
        "only_selected_cases": 1.0,
        "free_knobs": float(free_knobs),
        "wall": float(time.time() - start),
    }
    for key, row in cases.items():
        flat[f"score_{key}"] = row["score"]
        flat[f"pass_{key}"] = 1.0 if row["pass"] else 0.0
        for dk, dv in row["detail"].items():
            flat[f"{key}_{dk}"] = float(dv)

    rich = {
        "pass": bool(failures == 0),
        "quality_score": quality_score,
        "failures": failures,
        "free_knobs": free_knobs,
        "same_method_all_cases": True,
        "only_selected_cases": True,
        "cases": cases,
        "wall": flat["wall"],
    }
    print("SELECTED10_QUALITY_JSON " + json.dumps(rich, sort_keys=True, default=str))
    print(json.dumps(flat, sort_keys=True))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
