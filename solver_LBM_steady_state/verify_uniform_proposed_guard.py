"""Guard that the active proposed path is a single uniform methodology."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FORBIDDEN_TOKENS = (
    "_cavity_",
    "_channel_",
    "_couette_",
    "_tjunction_",
    "_masked_",
    "_is_wall_driven",
    "_is_force_free_moving_wall_shear",
    "case_id",
    "base_case",
)


def _uniform_source() -> str:
    text = (ROOT / "solver_proposed_single.py").read_text(encoding="utf-8")
    start = text.index("def _solve_uniform_ap_schur")
    end = text.index("\ndef solve_proposed_single", start)
    return text[start:end]


def check_source() -> list[str]:
    src = _uniform_source()
    violations = [token for token in FORBIDDEN_TOKENS if token in src]
    runner = (ROOT / "run_ap_schur_proposed_only.py").read_text(encoding="utf-8")
    for call in ("f, hist = _unified_macro_l2_convergence_audit", "f, hist = _post_tail_block_rre"):
        idx = runner.find(call)
        if idx >= 0:
            prefix = runner[max(0, idx - 120):idx]
            if "if not uniform_variant" not in prefix:
                violations.append(f"uniform_runner_unconditional:{call}")
    return violations


def check_summary(summary: Path | None) -> list[str]:
    if summary is None:
        return []
    if not summary.exists():
        return [f"missing_summary:{summary}"]
    bad = []
    for row in csv.DictReader(summary.open(newline="", encoding="utf-8")):
        if row.get("method") == "proposed" and row.get("method_variant") != "uniform_ap_schur_rre":
            bad.append(f"{row.get('case_id', 'unknown')}:{row.get('method_variant', '')}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="")
    args = parser.parse_args()
    summary = Path(args.summary) if args.summary else None
    source_violations = check_source()
    summary_violations = check_summary(summary)
    out = {
        "uniform_source_violations": source_violations,
        "summary_violations": summary_violations,
        "violation_count": len(source_violations) + len(summary_violations),
    }
    print(json.dumps(out, sort_keys=True))
    return 1 if out["violation_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
