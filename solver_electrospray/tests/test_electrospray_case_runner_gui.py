#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_electrospray_case_runner_gui.py CASE_RUNNER MESH_VALIDATE OUTDIR")
    case_runner = Path(sys.argv[1])
    mesh_validate = Path(sys.argv[2])
    outdir = Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)

    validate_case = {
        "case_name": "gui_validate_mesh_regression",
        "run_mode": "validate_mesh",
        "mesh_mode": "builtin_hex",
        "nx": 3,
        "ny": 4,
        "nz": 2,
        "lx": 1.0,
        "ly": 2.0,
        "lz": 0.5,
        "skew": 0.02,
        "patch_roles": {"ymin": "inlet", "ymax": "outlet"},
    }
    validate_path = outdir / "validate_case.json"
    validate_path.write_text(json.dumps(validate_case, indent=2) + "\n", encoding="utf-8")
    validate_out = outdir / "validate_run"
    run([case_runner, "--case", validate_path, "--output-dir", validate_out])
    validate_summary = json.loads((validate_out / "summary.json").read_text(encoding="utf-8"))
    assert validate_summary["status"] == "pass"
    assert validate_summary["run_mode"] == "validate_mesh"
    assert validate_summary["cells"] == 24
    assert validate_summary["patch_count"] == 6

    mesh_proc = run([
        mesh_validate,
        "--builtin-hex",
        "--nx",
        "3",
        "--ny",
        "4",
        "--nz",
        "2",
        "--skew",
        "0.02",
    ])
    mesh_summary = json.loads(mesh_proc.stdout)
    assert mesh_summary["status"] == "pass"
    assert mesh_summary["cells"] == 24

    smoke_case = {
        "case_name": "gui_candido_smoke_regression",
        "run_mode": "candido_smoke",
        "mesh_mode": "builtin_hex",
        "nx": 4,
        "ny": 6,
        "nz": 4,
        "skew": 0.0,
        "target_ca_e": 0.25,
        "steps": 1,
        "cfl": 0.15,
        "radial_window_outer_diameters": 1.25,
        "vof_compression": 1.0,
        "vof_post_sharpening_sweeps": 0,
        "use_dimensional_electrical_scaling": False,
        "use_electric_relaxation_timestep_limit": True,
        "use_poisson_face_maxwell_force": True,
        "use_poisson_hybrid_maxwell_force": False,
        "use_tomar_conducting_surface_force": False,
    }
    smoke_path = outdir / "smoke_case.json"
    smoke_path.write_text(json.dumps(smoke_case, indent=2) + "\n", encoding="utf-8")
    smoke_out = outdir / "smoke_run"
    run([case_runner, "--case", smoke_path, "--output-dir", smoke_out])
    smoke_summary = json.loads((smoke_out / "summary.json").read_text(encoding="utf-8"))
    assert smoke_summary["status"] == "pass"
    assert smoke_summary["run_mode"] == "candido_smoke"
    assert smoke_summary["cells"] == 96
    assert smoke_summary["steps"] == 1
    assert smoke_summary["min_alpha"] >= -1e-12
    assert smoke_summary["max_alpha"] <= 1.0 + 1e-12
    assert (smoke_out / "history.csv").exists()

    print(
        "gui_case_runner_validate_cells="
        f"{validate_summary['cells']} smoke_cells={smoke_summary['cells']} "
        f"smoke_mass_drift={smoke_summary['alpha_mass_drift']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
