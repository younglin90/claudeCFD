#!/usr/bin/env python3
"""Run or audit an OpenFOAM snappyHexMesh meshing case."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


SNAPPY_COMMANDS = [
    ("blockMesh", []),
    ("surfaceFeatureExtract", []),
    ("snappyHexMesh", ["-overwrite"]),
    ("checkMesh", []),
]

CFMESH_COMMANDS = [
    ("cartesianMesh", []),
    ("checkMesh", []),
]


def case_ready(case_dir: Path, workflow: str) -> list[str]:
    if workflow == "cfmesh":
        required = [
            case_dir / "system" / "meshDict",
            case_dir / "constant" / "triSurface",
        ]
    else:
        required = [
            case_dir / "system" / "blockMeshDict",
            case_dir / "system" / "surfaceFeatureExtractDict",
            case_dir / "system" / "snappyHexMeshDict",
            case_dir / "constant" / "triSurface",
        ]
    missing = [str(path) for path in required if not path.exists()]
    return missing


def write_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--workflow", choices=["snappy", "cfmesh"], default="snappy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    commands = CFMESH_COMMANDS if args.workflow == "cfmesh" else SNAPPY_COMMANDS
    missing_case_files = case_ready(case_dir, args.workflow)
    if missing_case_files:
        write_json({
            "status": "error",
            "error": "case directory is not snappyHexMesh-ready",
            "case_dir": str(case_dir),
            "missing": missing_case_files,
        })
        return 2

    missing_commands = [cmd for cmd, _ in commands if shutil.which(cmd) is None]
    planned = [{"command": " ".join([cmd, *extra]), "available": shutil.which(cmd) is not None}
               for cmd, extra in commands]
    if args.dry_run:
        write_json({
            "status": "pass",
            "dry_run": True,
            "case_dir": str(case_dir),
            "workflow": args.workflow,
            "planned_commands": planned,
            "missing_commands": missing_commands,
            "polyMesh_exists": (case_dir / "constant" / "polyMesh" / "points").exists(),
        })
        return 0

    if missing_commands:
        write_json({
            "status": "blocked",
            "blocked": True,
            "case_dir": str(case_dir),
            "workflow": args.workflow,
            "missing_commands": missing_commands,
            "planned_commands": planned,
            "polyMesh_exists": (case_dir / "constant" / "polyMesh" / "points").exists(),
            "unblock": (
                "Install/source OpenFOAM/cfMesh so the required meshing utilities "
                "are on PATH, then rerun this command."
            ),
        })
        return 3

    log_dir = Path(args.log_dir).resolve() if args.log_dir else case_dir / "meshing_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.time()
    for cmd, extra in commands:
        full_cmd = [cmd, *extra]
        step_started = time.time()
        proc = subprocess.run(
            full_cmd,
            cwd=case_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
        stem = cmd + ("_" + "_".join(extra).replace("-", "") if extra else "")
        stdout_path = log_dir / f"{stem}.stdout.log"
        stderr_path = log_dir / f"{stem}.stderr.log"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        result = {
            "command": " ".join(full_cmd),
            "returncode": proc.returncode,
            "seconds": time.time() - step_started,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
        results.append(result)
        if proc.returncode != 0:
            write_json({
                "status": "fail",
                "case_dir": str(case_dir),
                "workflow": args.workflow,
                "failed_command": result,
                "steps": results,
                "log_dir": str(log_dir),
                "polyMesh_exists": (case_dir / "constant" / "polyMesh" / "points").exists(),
            })
            return proc.returncode or 1

    poly_mesh = case_dir / "constant" / "polyMesh"
    ok = (poly_mesh / "points").exists() and (poly_mesh / "faces").exists()
    write_json({
        "status": "pass" if ok else "fail",
        "case_dir": str(case_dir),
        "workflow": args.workflow,
        "polyMesh": str(poly_mesh) if ok else "",
        "polyMesh_exists": ok,
        "seconds": time.time() - started,
        "steps": results,
        "log_dir": str(log_dir),
    })
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
