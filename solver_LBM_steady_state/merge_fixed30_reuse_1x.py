"""Merge fixed30 scaling results while reusing unchanged 1x caches.

The 2x/3x physics-scaling correction changes forcing, wall speed, and
tolerance, so those levels must be recomputed.  The 1x problem definition is
unchanged; this script intentionally allows legacy 1x cache files to be used
when the current hash key is absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper_60case_benchmark import METHODS
from verify_fixed30_scaling_strict import (
    BASE_CASE_IDS,
    CACHE,
    OUT,
    _load_cached,
    case_factory_scaled,
    row_for,
    score,
    write_outputs,
)


def _load_latest_cache(case_id: str, method: str):
    matches = sorted(
        CACHE.glob(f"{case_id}__{method}__*.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return None
    data = np.load(matches[0], allow_pickle=False)
    f = data["f"]
    hist = [tuple(row) for row in data["hist"].tolist()]
    wall = float(data["wall"])
    return f, hist, wall, matches[0]


def load_cache(case_id: str, method: str, allow_legacy: bool):
    current = _load_cached(case_id, method)
    if current is not None:
        f, hist, wall = current
        return f, hist, wall, "current"
    if allow_legacy:
        legacy = _load_latest_cache(case_id, method)
        if legacy is not None:
            f, hist, wall, path = legacy
            return f, hist, wall, path.name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-legacy-1x", action="store_true", default=True)
    parser.add_argument("--strict-current-1x", action="store_true")
    args = parser.parse_args()

    allow_legacy_1x = args.allow_legacy_1x and not args.strict_current_1x
    rows = []
    missing = []
    cache_sources = {}
    selected_case_ids = []

    for base_id in BASE_CASE_IDS:
        for level in (1, 2, 3):
            case_id, label, tol, factory = case_factory_scaled(base_id, level)
            selected_case_ids.append(case_id)
            allow_legacy = allow_legacy_1x and level == 1

            ref_loaded = load_cache(case_id, "picard_lbm", allow_legacy)
            if ref_loaded is None:
                missing.append({"case_id": case_id, "method": "picard_lbm"})
                continue
            ref_f, ref_hist, ref_wall, ref_source = ref_loaded
            ref_case = factory()
            cache_sources[f"{case_id}__picard_lbm"] = ref_source

            for method in METHODS:
                loaded = load_cache(case_id, method, allow_legacy)
                if loaded is None:
                    missing.append({"case_id": case_id, "method": method})
                    continue
                f, hist, wall, source = loaded
                case = ref_case if method == "picard_lbm" else factory()
                cache_sources[f"{case_id}__{method}"] = source
                rows.append(
                    row_for(
                        base_id,
                        level,
                        case_id,
                        label,
                        tol,
                        ref_case,
                        ref_f,
                        method,
                        case,
                        f,
                        hist,
                        wall,
                    )
                )

    metrics = score(rows, selected_case_ids)
    metrics["missing"] = missing
    metrics["missing_count"] = len(missing)
    metrics["allow_legacy_1x"] = bool(allow_legacy_1x)
    metrics["cache_sources"] = cache_sources
    metrics["merge_note"] = (
        "1x cache reuse is allowed because level=1 keeps original forcing, wall velocity, "
        "tolerance, cavity Re/nu, and native initial fields unchanged."
    )
    write_outputs(rows, metrics)
    print(f"[merged] rows={len(rows)} missing={len(missing)} -> {OUT / 'summary.csv'}", flush=True)
    print(json.dumps({k: metrics[k] for k in ["all_pass", "pass_count", "case_count", "missing_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
