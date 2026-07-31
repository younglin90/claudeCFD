#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def scalar_field(name: str, internal: str, xmin: str, xmax: str) -> str:
    return f"""FoamFile {{ version 2.0; format ascii; class volScalarField; object {name}; }}
dimensions [0 0 0 0 0 0 0];
internalField {internal};
boundaryField
{{
  xmin {{ type fixedValue; value uniform {xmin}; }}
  xmax {{ type fixedValue; value uniform {xmax}; }}
  ymin {{ type zeroGradient; }}
  ymax {{ type zeroGradient; }}
  zmin {{ type symmetryPlane; }}
  zmax {{ type symmetryPlane; }}
}}
"""


def build_case(case_dir: Path) -> None:
    poly = case_dir / "constant" / "polyMesh"
    write(
        poly / "points",
        """FoamFile { version 2.0; format ascii; class vectorField; object points; }
12
(
(0 0 0)
(1 0 0)
(1 1 0)
(0 1 0)
(0 0 1)
(1 0 1)
(1 1 1)
(0 1 1)
(2 0 0)
(2 1 0)
(2 0 1)
(2 1 1)
)
""",
    )
    write(
        poly / "faces",
        """FoamFile { version 2.0; format ascii; class faceList; object faces; }
11
(
4(1 5 6 2)
4(0 3 7 4)
4(8 10 11 9)
4(0 4 5 1)
4(1 5 10 8)
4(3 2 6 7)
4(2 9 11 6)
4(0 1 2 3)
4(1 8 9 2)
4(4 7 6 5)
4(5 6 11 10)
)
""",
    )
    write(
        poly / "owner",
        """FoamFile { version 2.0; format ascii; class labelList; object owner; }
11
(
0
0
1
0
1
0
1
0
1
0
1
)
""",
    )
    write(
        poly / "neighbour",
        """FoamFile { version 2.0; format ascii; class labelList; object neighbour; }
1
(
1
)
""",
    )
    write(
        poly / "boundary",
        """FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }
6
(
xmin { type patch; nFaces 1; startFace 1; }
xmax { type patch; nFaces 1; startFace 2; }
ymin { type wall; nFaces 2; startFace 3; }
ymax { type wall; nFaces 2; startFace 5; }
zmin { type symmetryPlane; nFaces 2; startFace 7; }
zmax { type symmetryPlane; nFaces 2; startFace 9; }
)
""",
    )
    write(
        case_dir / "0" / "U",
        """FoamFile { version 2.0; format ascii; class volVectorField; object U; }
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
  xmin { type fixedValue; value uniform (0 1 0); }
  xmax { type zeroGradient; }
  ymin { type noSlip; }
  ymax { type noSlip; }
  zmin { type symmetryPlane; }
  zmax { type symmetryPlane; }
}
""",
    )
    write(case_dir / "0" / "p", scalar_field("p", "uniform 0", "0", "0"))
    write(case_dir / "0" / "alpha", scalar_field("alpha", "uniform 0", "1", "0"))
    write(case_dir / "0" / "phi", scalar_field("phi", "uniform 0", "2180", "0"))
    write(case_dir / "0" / "rhoE", scalar_field("rhoE", "nonuniform List<scalar> 2 ( 0.1 -0.2 )", "0", "0"))


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_openfoam_case_runner.py CASE_RUNNER OUTDIR")
    runner = Path(sys.argv[1])
    out_root = Path(sys.argv[2])
    case_dir = out_root / "openfoam_case"
    run_dir = out_root / "openfoam_case_run"
    build_case(case_dir)
    proc = subprocess.run(
        [runner, "--case-dir", case_dir, "--output-dir", run_dir],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"case runner failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["run_mode"] == "openfoam_case_validate"
    assert summary["cells"] == 2
    assert summary["fields_read"] == 5
    assert summary["missing_patch_boundary_entries"] == 0
    assert summary["unknown_patch_boundary_entries"] == 0
    u_patches = {row["patch"] for row in summary["fields"]["U"]["boundary"]}
    assert {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"} <= u_patches
    assert (run_dir / "openfoam_boundary_fields.csv").exists()
    print(
        "openfoam_case_runner_status=pass "
        f"cells={summary['cells']} fields_read={summary['fields_read']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
