#!/usr/bin/env python3
"""Restricted quality gate for dense two-phase cases 14 and 15 only.

This verifier intentionally does not run case 24 or any broad shock sweep.
The pass gates are diffusion-aware: they allow smeared discontinuities and
attenuated extrema, but still reject the user-visible artifacts under review:

- 14_E: spurious velocity rise/fall in the stagnant post-interface tail.
- 15_E: high-frequency pressure oscillation around the cavitation region.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY_08_26 = os.path.join(ROOT, ".codex-loop", "verify_08_26_acceptance.py")
VERIFY_14_15_24 = os.path.join(ROOT, ".codex-loop", "verify_quality_14_15_24.py")


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    v = _load_module("verify_08_26_acceptance", VERIFY_08_26)
    q = _load_module("verify_quality_14_15_24", VERIFY_14_15_24)

    rows = [v.case_14(), q._case15_quality(v)]
    failures = int(sum(0 if bool(row.get("pass")) else 1 for row in rows))
    result = {
        "metric": failures,
        "quality_failures": failures,
        "pass": failures == 0,
        "cases": rows,
        "scope": "restricted: cases 14_E and 15_E only",
        "diffusion_policy": (
            "Allow numerical diffusion/smeared discontinuities; fail only on "
            "large 14_E tail artifact or 15_E high-frequency pressure ringing."
        ),
    }
    print("QUALITY_JSON " + json.dumps(result, sort_keys=True))
    print(failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
