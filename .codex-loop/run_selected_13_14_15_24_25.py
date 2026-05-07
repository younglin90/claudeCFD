#!/usr/bin/env python3
"""Run the selected 1D validation gates and print aggregate JSON on the last line."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
VERIFY = ROOT / "verify_08_26_acceptance.py"
CASES = ["13", "14", "15", "24", "25"]


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("verify_08_26_acceptance", VERIFY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {VERIFY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_verify_module()
    results = []
    failures = 0
    crashes = 0
    for case in CASES:
        print(f"=== CASE {case} ===", flush=True)
        try:
            result = module.CASES[case]()
            ok = bool(result.get("pass"))
            print("CASE_JSON " + json.dumps(result, sort_keys=True), flush=True)
        except Exception as exc:  # keep the full gate running so the aggregate is meaningful
            ok = False
            crashes += 1
            result = {"case": case, "pass": False, "crash": repr(exc)}
            traceback.print_exc()
            print("CASE_JSON " + json.dumps(result, sort_keys=True), flush=True)
        if not ok:
            failures += 1
        results.append(result)
    summary = {
        "fail_count": failures,
        "pass_count": len(CASES) - failures,
        "crash_count": crashes,
    }
    for case, result in zip(CASES, results):
        summary[f"case{case}_pass"] = 1 if bool(result.get("pass")) else 0
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
