#!/usr/bin/env python3
"""Aggregate acceptance verifier for the selected 1D validation set.

The final stdout line is the named validation failure count for:
01, 02, 04, 05, 07, 13, 14, 15, 24, and 25.

Case 07 contains three documented subcases, and case 24 contains five mixture
subcases.  Each named validation contributes one failure if any of its internal
subcases fail, matching the user's selected case list.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SELECTED = ("01", "02", "04", "05", "07", "13", "14", "15", "24", "25")


def _run(cmd: list[str]) -> tuple[int, str, float]:
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout, time.time() - start


def _parse_json(output: str) -> dict:
    parsed = {}
    for line in output.splitlines():
        if line.startswith("CASE_JSON "):
            parsed = json.loads(line[len("CASE_JSON "):])
        elif line.startswith("ACCEPTANCE_JSON "):
            parsed = json.loads(line[len("ACCEPTANCE_JSON "):])
    return parsed


def _case_command(case: str) -> list[str]:
    py = sys.executable
    if case in {"01", "04", "05"}:
        return [py, ".codex-loop/verify_01_03_06_acceptance.py", "--case", case]
    if case == "02_07":
        return [py, ".codex-loop/verify_02_07_acceptance.py"]
    return [py, ".codex-loop/verify_08_26_acceptance.py", "--case", case]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=",".join(SELECTED),
        help="Comma-separated named cases. Use 02 and/or 07 to run the shared 02_07 verifier.",
    )
    args = parser.parse_args()
    requested = tuple(c.strip() for c in args.cases.split(",") if c.strip())
    run_02_07 = "02" in requested or "07" in requested
    commands: list[tuple[str, list[str]]] = []
    for case in requested:
        if case in {"02", "07"}:
            continue
        commands.append((case, _case_command(case)))
    if run_02_07:
        insert_at = 0
        for i, (name, _) in enumerate(commands):
            if name in {"13", "14", "15", "24", "25"}:
                insert_at = i
                break
            insert_at = i + 1
        commands.insert(insert_at, ("02_07", _case_command("02_07")))

    summary: dict[str, dict] = {}
    failures = 0
    for name, cmd in commands:
        rc, out, wall = _run(cmd)
        print(out, end="")
        parsed = _parse_json(out)
        if name == "02_07":
            pass02 = bool(parsed.get("results", {}).get("02", {}).get("pass"))
            pass07 = bool(parsed.get("results", {}).get("07", {}).get("pass"))
            if "02" in requested:
                summary["02"] = {
                    "pass": pass02,
                    "wall": parsed.get("results", {}).get("02", {}).get("wall"),
                }
                failures += 0 if pass02 else 1
            if "07" in requested:
                summary["07"] = {
                    "pass": pass07,
                    "failures": parsed.get("results", {}).get("07", {}).get("failures"),
                }
                failures += 0 if pass07 else 1
        else:
            ok = bool(parsed.get("pass")) and rc == 0
            summary[name] = {"pass": ok, "wall": wall}
            for key in (
                "failures",
                "case13_goal_failure_score",
                "p_osc",
                "rho_osc",
                "steps",
                "terminated_reason",
            ):
                if key in parsed:
                    summary[name][key] = parsed[key]
            failures += 0 if ok else 1
        print(f"SELECTED_CASE {name} rc={rc} wall={wall:.2f}s", flush=True)

    compact = {
        "cases": requested,
        "failures": failures,
        "goal_reached": failures == 0,
        "summary": summary,
    }
    print("SELECTED_ACCEPTANCE_JSON " + json.dumps(compact, sort_keys=True, default=str))
    print(failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
