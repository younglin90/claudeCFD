#!/usr/bin/env python3
"""Regression guard for CAD-resolved nozzle surface/snappy case generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: test_nozzle_cad_generator.py <repo-root> <output-dir>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(root / "apps" / "generate_nozzle_cad.py"),
        "--output-dir",
        str(out_dir),
        "--segments",
        "40",
    ]
    proc = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    payload = json.loads(proc.stdout)
    assert payload["status"] == "pass"
    expected = {
        "liquid_inlet",
        "inner_nozzle_wall",
        "nozzle_electrode",
        "collector_ground",
        "open_atmosphere",
    }
    assert expected == set(payload["patches"].keys())
    stl = Path(payload["stl"])
    stl_text = stl.read_text(encoding="utf-8")
    for patch in expected:
        assert f"solid {patch}" in stl_text
        assert f"endsolid {patch}" in stl_text
    surface_check = subprocess.run(
        [
            sys.executable,
            str(root / "apps" / "validate_nozzle_cad_surface.py"),
            str(stl),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if surface_check.returncode != 0:
        print(surface_check.stdout)
        print(surface_check.stderr, file=sys.stderr)
        return surface_check.returncode
    surface_payload = json.loads(surface_check.stdout)
    assert surface_payload["status"] == "pass"
    assert surface_payload["boundary_edge_count"] == 0
    assert surface_payload["nonmanifold_edge_count"] == 0
    assert surface_payload["degenerate_triangles"] == 0

    case_dir = Path(payload["snappy_case_dir"])
    cfmesh_case_dir = Path(payload["cfmesh_case_dir"])
    required = [
        case_dir / "constant" / "triSurface" / stl.name,
        case_dir / "system" / "blockMeshDict",
        case_dir / "system" / "snappyHexMeshDict",
        case_dir / "system" / "surfaceFeatureExtractDict",
        case_dir / "system" / "meshQualityDict",
        case_dir / "0" / "U",
        case_dir / "0" / "p",
        case_dir / "0" / "alpha",
        case_dir / "0" / "phi",
        case_dir / "0" / "rhoE",
    ]
    for path in required:
        assert path.exists(), path
    cfmesh_required = [
        cfmesh_case_dir / "constant" / "triSurface" / stl.name,
        cfmesh_case_dir / "system" / "meshDict",
        cfmesh_case_dir / "0" / "U",
        cfmesh_case_dir / "0" / "phi",
    ]
    for path in cfmesh_required:
        assert path.exists(), path
    mesh_dict = (cfmesh_case_dir / "system" / "meshDict").read_text(encoding="utf-8")
    assert "surfaceFile" in mesh_dict
    assert "maxCellSize" in mesh_dict
    assert "localRefinement" in mesh_dict
    assert "boundaryLayers" in mesh_dict

    snappy = (case_dir / "system" / "snappyHexMeshDict").read_text(encoding="utf-8")
    for patch in expected:
        assert patch in snappy
    assert "addLayers true" in snappy
    assert "inner_nozzle_wall { nSurfaceLayers 3; }" in snappy
    assert "nozzle_electrode { nSurfaceLayers 3; }" in snappy

    phi = (case_dir / "0" / "phi").read_text(encoding="utf-8")
    assert "liquid_inlet" in phi
    assert "nozzle_electrode" in phi
    assert "collector_ground" in phi
    assert "fixedValue" in phi

    runner = subprocess.run(
        [
            sys.executable,
            str(root / "apps" / "run_openfoam_meshing.py"),
            "--case-dir",
            str(case_dir),
            "--dry-run",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if runner.returncode != 0:
        print(runner.stdout)
        print(runner.stderr, file=sys.stderr)
        return runner.returncode
    runner_payload = json.loads(runner.stdout)
    assert runner_payload["status"] == "pass"
    assert runner_payload["dry_run"] is True
    assert [step["command"] for step in runner_payload["planned_commands"]] == [
        "blockMesh",
        "surfaceFeatureExtract",
        "snappyHexMesh -overwrite",
        "checkMesh",
    ]
    cfmesh_runner = subprocess.run(
        [
            sys.executable,
            str(root / "apps" / "run_openfoam_meshing.py"),
            "--case-dir",
            str(cfmesh_case_dir),
            "--workflow",
            "cfmesh",
            "--dry-run",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if cfmesh_runner.returncode != 0:
        print(cfmesh_runner.stdout)
        print(cfmesh_runner.stderr, file=sys.stderr)
        return cfmesh_runner.returncode
    cfmesh_payload = json.loads(cfmesh_runner.stdout)
    assert cfmesh_payload["status"] == "pass"
    assert [step["command"] for step in cfmesh_payload["planned_commands"]] == [
        "cartesianMesh",
        "checkMesh",
    ]
    print(json.dumps({
        "status": "pass",
        "stl": str(stl),
        "snappy_case_dir": str(case_dir),
        "cfmesh_case_dir": str(cfmesh_case_dir),
        "patches": payload["patches"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
