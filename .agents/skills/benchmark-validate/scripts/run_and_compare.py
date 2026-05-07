#!/usr/bin/env python3
"""Run claudeCFD regression gates and write a validation report."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BASE_COMMANDS = [
    {
        "name": "uniform_flow",
        "cmd": [sys.executable, "tests/test_uniform_flow.py"],
        "timeout": 180,
    },
    {
        "name": "amplification_matrix",
        "cmd": [sys.executable, "tests/test_amplification_matrix.py"],
        "timeout": 240,
    },
    {
        "name": "transport_eigenmode",
        "cmd": [sys.executable, "tests/test_transport_eigenmode.py"],
        "timeout": 240,
    },
    {
        "name": "02A_nasg",
        "cmd": [
            sys.executable,
            "results/run_02_07_five_eq_imex.py",
            "--case", "02",
            "--variant02", "nasg",
            "--tend02", "1.0",
            "--dt-fixed02", "0.01",
        ],
        "timeout": 240,
    },
]

EXTRA_07 = {
    "name": "07B_smoke_n50",
    "cmd": [
        sys.executable,
        "results/run_02_07_five_eq_imex.py",
        "--case", "07",
        "--n07", "50",
        "--cfl07", "0.1",
        "--imp-dissipation", "0.02",
        "--pe-projection-mode", "contact",
        "--max-steps07", "1000",
    ],
    "timeout": 360,
}


def _pipeline_dir() -> Path:
    return Path(os.environ.get("PIPELINE_DIR", ".agents/pipeline"))


def _run(entry: dict) -> dict:
    start = time.time()
    try:
        proc = subprocess.run(
            entry["cmd"],
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=entry["timeout"],
        )
        return {
            "name": entry["name"],
            "cmd": entry["cmd"],
            "returncode": proc.returncode,
            "wall": time.time() - start,
            "output": proc.stdout,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": entry["name"],
            "cmd": entry["cmd"],
            "returncode": 124,
            "wall": time.time() - start,
            "output": (exc.stdout or "") + "\nTIMEOUT",
        }


def _markdown(results: list[dict]) -> str:
    lines = [
        "# Validation Report",
        "",
        f"Generated: {_dt.datetime.now(_dt.UTC).isoformat()}",
        "",
        "| gate | status | returncode | wall_s |",
        "|---|---:|---:|---:|",
    ]
    for item in results:
        status = "PASS" if item["returncode"] == 0 else "FAIL"
        lines.append(f"| {item['name']} | {status} | {item['returncode']} | {item['wall']:.2f} |")
    lines.append("")
    for item in results:
        lines.extend([
            f"## {item['name']}",
            "",
            "```text",
            item["output"].rstrip(),
            "```",
            "",
        ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-07", action="store_true")
    args = parser.parse_args()

    commands = list(BASE_COMMANDS)
    if args.include_07:
        commands.append(EXTRA_07)

    out_dir = _pipeline_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [_run(entry) for entry in commands]
    payload = {
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "include_07": args.include_07,
        "results": results,
    }
    (out_dir / "validation_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "validation_report.md").write_text(
        _markdown(results),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'validation_report.md'}")
    return 0 if all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
