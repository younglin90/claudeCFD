#!/usr/bin/env python3
"""Local web GUI server for electrospray case setup and execution."""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import math
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
GUI_DIR = ROOT / "apps" / "gui"
RUNS = ROOT / "runs"
UPLOADS = RUNS / "uploads"
GENERATED = RUNS / "generated_meshes"
GENERATED_CAD = RUNS / "generated_cad"


# ---------------------------------------------------------------------------
# Single source of truth for the case parameter form.
# Each block maps to a container <div> in apps/gui/index.html; the client
# fetches this schema via /api/ui-config and builds the form inputs from it,
# so every parameter's name/type/default lives in exactly one place (here).
# ---------------------------------------------------------------------------
FIELD_SCHEMA = [
    {
        "container": "fields-case",
        "fields": [
            {"id": "case_name", "label": "case name", "kind": "text", "default": "candido_gui_smoke"},
            {"id": "run_mode", "label": "run mode", "kind": "select", "default": "candido_smoke",
             "options": ["candido_smoke", "validate_mesh", "validate_openfoam_case"]},
        ],
    },
    {
        "container": "fields-mesh-modes",
        "fields": [
            {"id": "mesh_mode", "label": "mesh mode", "kind": "select", "default": "builtin_hex",
             "options": ["builtin_hex", "candido_nozzle_template", "openfoam_polyMesh"]},
            {"id": "openfoam_polyMesh", "label": "OpenFOAM polyMesh path", "kind": "text", "default": ""},
            {"id": "openfoam_case_dir", "label": "OpenFOAM case path", "kind": "text", "default": ""},
            {"id": "cad_meshing_workflow", "label": "CAD meshing workflow", "kind": "select", "default": "snappy",
             "options": ["snappy", "cfmesh"]},
        ],
    },
    {
        "container": "fields-mesh-dims",
        "fields": [
            {"id": "nx", "label": "nx", "kind": "number", "default": 8, "int": True, "min": 1},
            {"id": "ny", "label": "ny", "kind": "number", "default": 16, "int": True, "min": 1},
            {"id": "nz", "label": "nz", "kind": "number", "default": 8, "int": True, "min": 1},
            {"id": "lx", "label": "lx", "kind": "number", "default": 1.0},
            {"id": "ly", "label": "ly", "kind": "number", "default": 4.0},
            {"id": "lz", "label": "lz", "kind": "number", "default": 1.0},
            {"id": "skew", "label": "skew", "kind": "number", "default": 0.02},
        ],
    },
    {
        "container": "fields-electrospray",
        "fields": [
            {"id": "target_ca_e", "label": "target CaE", "kind": "number", "default": 0.35},
            {"id": "steps", "label": "steps", "kind": "number", "default": 4, "int": True, "min": 0},
            {"id": "cfl", "label": "CFL", "kind": "number", "default": 0.2},
            {"id": "radial_window_outer_diameters", "label": "radial window OD", "kind": "number", "default": 1.5},
            {"id": "pseudo_viscosity", "label": "pseudo viscosity", "kind": "number", "default": 0.03},
            {"id": "vof_compression", "label": "VoF compression", "kind": "number", "default": 1.0},
            {"id": "vof_post_sharpening", "label": "VoF post sharpening", "kind": "number", "default": 0.0},
            {"id": "vof_post_sharpening_sweeps", "label": "sharpen sweeps", "kind": "number", "default": 1, "int": True, "min": 0},
            {"id": "alpha_interface_width_outer_diameters", "label": "interface width OD", "kind": "number", "default": 0.22},
        ],
    },
    {
        "container": "fields-electrospray-flags",
        "fields": [
            {"id": "use_vof_inlet_boundary_alpha", "label": "VoF inlet alpha", "kind": "check", "default": False},
            {"id": "use_open_atmospheric_boundary_flux", "label": "open boundary flux", "kind": "check", "default": True},
        ],
    },
    {
        "container": "fields-geometry",
        "fields": [
            {"id": "inner_diameter", "label": "inner diameter", "kind": "number", "default": 160e-6},
            {"id": "outer_diameter", "label": "outer diameter", "kind": "number", "default": 260e-6},
            {"id": "nozzle_length", "label": "nozzle length", "kind": "number", "default": 300e-6},
            {"id": "collector_distance", "label": "collector distance", "kind": "number", "default": 1.5e-3},
            {"id": "collector_diameter", "label": "collector diameter", "kind": "number", "default": 5.0e-3},
            {"id": "collector_speed", "label": "collector speed", "kind": "number", "default": 20.0e-3},
            {"id": "contact_angle_deg", "label": "contact angle deg", "kind": "number", "default": 51.0},
            {"id": "validation_voltage", "label": "voltage", "kind": "number", "default": 2180.0},
            {"id": "validation_flow_rate", "label": "flow rate", "kind": "number", "default": 16.1e-12},
        ],
    },
    {
        "container": "fields-material",
        "fields": [
            {"id": "surface_tension", "label": "surface tension", "kind": "number", "default": 64.5e-3},
            {"id": "liquid_density", "label": "liquid density", "kind": "number", "default": 1208.4},
            {"id": "gas_density", "label": "gas density", "kind": "number", "default": 1.225},
            {"id": "liquid_viscosity", "label": "liquid viscosity", "kind": "number", "default": 60.0e-3},
            {"id": "gas_viscosity", "label": "gas viscosity", "kind": "number", "default": 0.012e-3},
            {"id": "liquid_relative_permittivity", "label": "liquid epsilon r", "kind": "number", "default": 55.6},
            {"id": "gas_relative_permittivity", "label": "gas epsilon r", "kind": "number", "default": 1.0},
            {"id": "liquid_conductivity", "label": "liquid conductivity", "kind": "number", "default": 60.0e-6},
            {"id": "gas_conductivity", "label": "gas conductivity", "kind": "number", "default": 1.0e-15},
        ],
    },
    {
        "container": "fields-charge",
        "fields": [
            {"id": "normalized_liquid_conductivity", "label": "liquid sigma*", "kind": "number", "default": 1.0},
            {"id": "normalized_gas_conductivity", "label": "gas sigma*", "kind": "number", "default": 1.0e-6},
            {"id": "charge_limit_base", "label": "charge limit base", "kind": "number", "default": 50.0},
            {"id": "charge_subcycles", "label": "charge subcycles", "kind": "number", "default": 1, "int": True, "min": 1},
            {"id": "interface_charge_redistribution_liquid_floor", "label": "charge redist floor", "kind": "number", "default": 0.02},
            {"id": "interfacial_ohmic_charge_source_scale", "label": "ohmic source scale", "kind": "number", "default": 1.0},
            {"id": "electric_relaxation_timestep_safety", "label": "electric dt safety", "kind": "number", "default": 1.0},
            {"id": "electric_drive_reference_scale", "label": "electric drive ref", "kind": "number", "default": 15.2 * 0.25},
            {"id": "electric_drive_ca_exponent", "label": "electric drive exponent", "kind": "number", "default": 1.25},
            {"id": "poisson_tangential_limit_factor", "label": "tangential limit", "kind": "number", "default": 1.0},
            {"id": "poisson_tangential_limit_floor_fraction", "label": "tangential floor", "kind": "number", "default": 0.05},
            {"id": "surface_tension_drive_scale", "label": "surface drive scale", "kind": "number", "default": 0.20},
        ],
    },
    {
        "container": "fields-charge-flags",
        "fields": [
            {"id": "use_dimensional_electrical_scaling", "label": "dimensional scaling", "kind": "check", "default": False},
            {"id": "conservative_charge_bounding", "label": "charge bounding", "kind": "check", "default": False},
            {"id": "quasi_implicit_charge_relaxation", "label": "charge relaxation", "kind": "check", "default": False},
            {"id": "quasi_implicit_bulk_conduction", "label": "bulk conduction", "kind": "check", "default": False},
            {"id": "use_rayleigh_charge_limit", "label": "Rayleigh limit", "kind": "check", "default": False},
            {"id": "use_interface_localized_charge_redistribution", "label": "interface charge redist", "kind": "check", "default": False},
            {"id": "use_interfacial_ohmic_charge_source", "label": "interfacial ohmic source", "kind": "check", "default": False},
            {"id": "use_conductivity_potential_charge_closure", "label": "potential charge closure", "kind": "check", "default": False},
            {"id": "suppress_nozzle_conductive_charge_flux", "label": "suppress nozzle current", "kind": "check", "default": False},
            {"id": "collector_only_conductive_charge_flux", "label": "collector-only current", "kind": "check", "default": False},
            {"id": "apply_conductive_boundary_filters_in_implicit_ohmic", "label": "implicit filter", "kind": "check", "default": False},
            {"id": "use_poisson_face_conductive_current", "label": "Poisson face current", "kind": "check", "default": False},
            {"id": "implicit_ohmic_charge_projection", "label": "implicit ohmic projection", "kind": "check", "default": False},
            {"id": "refresh_potential_after_charge_advance", "label": "refresh phi after charge", "kind": "check", "default": False},
            {"id": "use_electric_relaxation_timestep_limit", "label": "electric dt limit", "kind": "check", "default": True},
            {"id": "use_poisson_face_maxwell_force", "label": "Poisson face force", "kind": "check", "default": True},
            {"id": "use_poisson_hybrid_maxwell_force", "label": "hybrid force", "kind": "check", "default": False},
            {"id": "use_poisson_bounded_vector_maxwell_force", "label": "bounded vector force", "kind": "check", "default": False},
            {"id": "use_tomar_conducting_surface_force", "label": "Tomar force", "kind": "check", "default": False},
        ],
    },
    {
        "container": "fields-boundary",
        "fields": [
            {"id": "preconditioned_jet_tip_y_over_inner_diameter", "label": "jet tip y/Di", "kind": "number", "default": -1.0},
            {"id": "preconditioned_jet_radius_inner_diameters", "label": "jet radius Di", "kind": "number", "default": 0.65},
            {"id": "preconditioned_jet_interface_width_inner_diameters", "label": "jet interface width Di", "kind": "number", "default": 0.20},
            {"id": "preconditioned_jet_velocity_scale", "label": "jet velocity scale", "kind": "number", "default": 1.0},
            {"id": "contact_angle_curvature_wall_band_cells", "label": "contact wall band cells", "kind": "number", "default": 1.5},
        ],
    },
    {
        "container": "fields-boundary-flags",
        "fields": [
            {"id": "use_boundary_charge_advection", "label": "boundary charge advection", "kind": "check", "default": False},
            {"id": "use_fully_developed_inlet_velocity_boundary", "label": "fully developed inlet", "kind": "check", "default": False},
            {"id": "use_moving_collector_wall", "label": "moving collector wall", "kind": "check", "default": False},
            {"id": "use_preconditioned_paper_current_jet", "label": "preconditioned paper jet", "kind": "check", "default": False},
            {"id": "use_contact_angle_curvature", "label": "contact angle curvature", "kind": "check", "default": False},
        ],
    },
]


VELOCITY_BC_TYPES = ["fixedValue", "zeroGradient", "noSlip", "slip", "movingWall", "inletOutlet"]
SCALAR_BC_TYPES = ["fixedValue", "zeroGradient", "inletOutlet", "boundedOutlet", "symmetry"]
POTENTIAL_BC_TYPES = ["fixedValue", "zeroGradient", "electrode", "grounded", "insulating", "symmetry"]
BC_ROLES = ["unassigned", "inlet", "outlet", "nozzle", "collector", "wall", "symmetry", "electrode"]

# Voltage-dependent preset values use this sentinel; the live validation_voltage
# is substituted by both the server and the client when a preset is applied.
VOLTAGE_SENTINEL = "$voltage"


def bc_role_presets() -> dict:
    return {
        "inlet": {
            "velocity": {"type": "fixedValue", "value": [0, 1, 0]},
            "pressure": {"type": "zeroGradient", "value": 0},
            "alpha": {"type": "fixedValue", "value": 1},
            "potential": {"type": "fixedValue", "value": VOLTAGE_SENTINEL},
            "charge": {"type": "zeroGradient", "value": 0},
        },
        "outlet": {
            "velocity": {"type": "zeroGradient", "value": [0, 0, 0]},
            "pressure": {"type": "fixedValue", "value": 0},
            "alpha": {"type": "zeroGradient", "value": 0},
            "potential": {"type": "zeroGradient", "value": 0},
            "charge": {"type": "zeroGradient", "value": 0},
        },
        "nozzle": {
            "velocity": {"type": "fixedValue", "value": [0, 1, 0]},
            "pressure": {"type": "zeroGradient", "value": 0},
            "alpha": {"type": "fixedValue", "value": 1},
            "potential": {"type": "fixedValue", "value": VOLTAGE_SENTINEL},
            "charge": {"type": "zeroGradient", "value": 0},
        },
        "collector": {
            "velocity": {"type": "movingWall", "value": [0, 0, 0]},
            "pressure": {"type": "zeroGradient", "value": 0},
            "alpha": {"type": "zeroGradient", "value": 0},
            "potential": {"type": "fixedValue", "value": 0},
            "charge": {"type": "zeroGradient", "value": 0},
        },
        "electrode": {
            "velocity": {"type": "noSlip", "value": [0, 0, 0]},
            "pressure": {"type": "zeroGradient", "value": 0},
            "alpha": {"type": "zeroGradient", "value": 0},
            "potential": {"type": "fixedValue", "value": VOLTAGE_SENTINEL},
            "charge": {"type": "zeroGradient", "value": 0},
        },
        "symmetry": {
            "velocity": {"type": "slip", "value": [0, 0, 0]},
            "pressure": {"type": "zeroGradient", "value": 0},
            "alpha": {"type": "zeroGradient", "value": 0},
            "potential": {"type": "zeroGradient", "value": 0},
            "charge": {"type": "zeroGradient", "value": 0},
        },
        "wall": {
            "velocity": {"type": "noSlip", "value": [0, 0, 0]},
            "pressure": {"type": "zeroGradient", "value": 0},
            "alpha": {"type": "zeroGradient", "value": 0},
            "potential": {"type": "zeroGradient", "value": 0},
            "charge": {"type": "zeroGradient", "value": 0},
        },
    }


def resolve_bc_preset(role: str, voltage: float) -> dict:
    """Deep-copy a role preset and substitute the voltage sentinel."""
    presets = bc_role_presets()
    preset = json.loads(json.dumps(presets.get(role, presets["wall"])))
    for field in preset.values():
        if field.get("value") == VOLTAGE_SENTINEL:
            field["value"] = voltage
    return preset


@functools.lru_cache(maxsize=1)
def _solver_defaults() -> dict:
    """Mirror the C++ solver struct defaults (single source of truth) so the GUI never
    drifts from the solver. Falls back to the schema defaults if the binary is missing."""
    try:
        exe = executable("electrospray_case_runner")
    except FileNotFoundError:
        return {}
    try:
        proc = subprocess.run(
            [str(exe), "--print-defaults"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout)
    except Exception:
        pass
    return {}


def default_case() -> dict:
    case = {field["id"]: field["default"] for block in FIELD_SCHEMA for field in block["fields"]}
    # Overlay the C++ solver defaults (SSOT) for every key the solver owns.
    for key, value in _solver_defaults().items():
        if key in case:
            case[key] = value
    case["patch_roles"] = {}
    case["boundary_conditions"] = {}
    return case


def ui_config() -> dict:
    return {
        "schema": FIELD_SCHEMA,
        "bc_presets": bc_role_presets(),
        "bc_roles": BC_ROLES,
        "voltage_sentinel": VOLTAGE_SENTINEL,
        "velocity_types": VELOCITY_BC_TYPES,
        "scalar_types": SCALAR_BC_TYPES,
        "potential_types": POTENTIAL_BC_TYPES,
    }


def sanitize_case_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return clean[:80] or "gui_case"


def executable(name: str) -> Path:
    exe = BUILD / name
    if exe.exists():
        return exe
    exe_win = BUILD / f"{name}.exe"
    if exe_win.exists():
        return exe_win
    raise FileNotFoundError(f"{name} not found under {BUILD}; run cmake --build build first")


def run_json_command(args: list[str], timeout: int = 120) -> dict:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "error",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    if proc.returncode != 0 and payload.get("status") not in {"error", "fail", "blocked"}:
        payload["status"] = "error"
    payload.setdefault("returncode", proc.returncode)
    if proc.stderr:
        payload["stderr"] = proc.stderr
    return payload


def validate_mesh(case: dict) -> dict:
    mesh_mode = case.get("mesh_mode", "builtin_hex")
    exe = executable("electrospray_mesh_validate")
    out_dir = RUNS / sanitize_case_name(case.get("case_name", "gui_case"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "mesh_validation.csv"
    if mesh_mode == "openfoam_polyMesh":
        poly = str(case.get("openfoam_polyMesh", "")).strip()
        poly = _ascii_polymesh_dir(poly, out_dir)
        args = [str(exe), "--openfoam-polyMesh", poly, "--output", str(out_csv)]
    else:
        args = [
            str(exe),
            "--builtin-hex",
            "--nx",
            str(int(case.get("nx", 8))),
            "--ny",
            str(int(case.get("ny", 16))),
            "--nz",
            str(int(case.get("nz", 8))),
            "--lx",
            str(float(case.get("lx", 1.0))),
            "--ly",
            str(float(case.get("ly", 4.0))),
            "--lz",
            str(float(case.get("lz", 1.0))),
            "--skew",
            str(float(case.get("skew", 0.0))),
            "--output",
            str(out_csv),
        ]
    payload = run_json_command(args)
    payload["mesh_validation_csv"] = str(out_csv)
    payload["boundary_conditions"] = case.get("boundary_conditions", {})
    payload["boundary_condition_application"] = (
        "recorded_for_case_json; validate_mesh does not solve equations"
    )
    return payload


def _face_center(points: list[list[float]], face: list[int]) -> list[float]:
    if not face:
        return [0.0, 0.0, 0.0]
    c = [0.0, 0.0, 0.0]
    for idx in face:
        p = points[idx]
        c[0] += p[0]
        c[1] += p[1]
        c[2] += p[2]
    inv = 1.0 / len(face)
    return [c[0] * inv, c[1] * inv, c[2] * inv]


def _bounds(points: list[list[float]]) -> dict:
    if not points:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    lo = list(points[0])
    hi = list(points[0])
    for p in points[1:]:
        for i in range(3):
            lo[i] = min(lo[i], p[i])
            hi[i] = max(hi[i], p[i])
    return {"min": lo, "max": hi}


def _compact_preview(points: list[list[float]], faces: list[dict], patches: list[dict],
                     max_faces: int = 6000) -> dict:
    total_faces = len(faces)
    if total_faces > max_faces:
        stride = max(1, math.ceil(total_faces / max_faces))
        faces = faces[::stride]
    else:
        stride = 1

    used = sorted({idx for face in faces for idx in face["vertices"]})
    remap = {old: new for new, old in enumerate(used)}
    compact_points = [points[i] for i in used]
    compact_faces = []
    for face in faces:
        verts = [remap[i] for i in face["vertices"] if i in remap]
        compact_faces.append({
            "id": face["id"],
            "patch": face["patch"],
            "vertices": verts,
        })
    return {
        "points": compact_points,
        "faces": compact_faces,
        "patches": patches,
        "bounds": _bounds(compact_points),
        "preview_face_count": len(compact_faces),
        "total_boundary_faces": total_faces,
        "decimation_stride": stride,
    }


def _builtin_hex_preview(case: dict) -> dict:
    nx = max(1, int(case.get("nx", 8)))
    ny = max(1, int(case.get("ny", 16)))
    nz = max(1, int(case.get("nz", 8)))
    lx = float(case.get("lx", 1.0))
    ly = float(case.get("ly", 4.0))
    lz = float(case.get("lz", 1.0))
    skew = float(case.get("skew", 0.0))
    npx, npy, npz = nx + 1, ny + 1, nz + 1

    def pid(i: int, j: int, k: int) -> int:
        return k * npx * npy + j * npx + i

    points: list[list[float]] = []
    for k in range(npz):
        z = lz * k / nz
        for j in range(npy):
            y = ly * j / ny
            for i in range(npx):
                x = lx * i / nx
                bump = (
                    skew
                    * math.sin(math.pi * x / lx)
                    * math.sin(math.pi * y / ly)
                    * math.sin(math.pi * z / lz)
                    if lx != 0.0 and ly != 0.0 and lz != 0.0
                    else 0.0
                )
                points.append([
                    x + 0.18 * bump / nx,
                    y + 0.14 * bump / ny,
                    z + 0.11 * bump / nz,
                ])

    faces: list[dict] = []
    patch_counts = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0}

    def add_face(patch: str, verts: list[int]) -> None:
        patch_counts[patch] += 1
        faces.append({"id": len(faces), "patch": patch, "vertices": verts})

    for j in range(ny):
        for k in range(nz):
            add_face("xmin", [pid(0, j, k), pid(0, j, k + 1), pid(0, j + 1, k + 1), pid(0, j + 1, k)])
            add_face("xmax", [pid(nx, j, k), pid(nx, j + 1, k), pid(nx, j + 1, k + 1), pid(nx, j, k + 1)])
    for i in range(nx):
        for k in range(nz):
            add_face("ymin", [pid(i, 0, k), pid(i + 1, 0, k), pid(i + 1, 0, k + 1), pid(i, 0, k + 1)])
            add_face("ymax", [pid(i, ny, k), pid(i, ny, k + 1), pid(i + 1, ny, k + 1), pid(i + 1, ny, k)])
    for i in range(nx):
        for j in range(ny):
            add_face("zmin", [pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0)])
            add_face("zmax", [pid(i, j, nz), pid(i + 1, j, nz), pid(i + 1, j + 1, nz), pid(i, j + 1, nz)])

    patches = [{"name": name, "faces": patch_counts[name]} for name in patch_counts]
    return _compact_preview(points, faces, patches)


def _strip_foam_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def _matching_paren(text: str, open_index: int) -> int:
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced OpenFOAM list parentheses")


def _read_foam_text(path: Path) -> str:
    """Read an OpenFOAM file as text, transparently handling .gz compression and
    rejecting binary-format files with an actionable message."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    fmt = re.search(r"\bformat\s+(\w+)\s*;", text)
    if fmt and fmt.group(1).lower() == "binary":
        raise ValueError(
            f"{path.name} is in OpenFOAM binary format, which the GUI loader cannot "
            "parse. Re-export the mesh as ASCII (set 'writeFormat ascii;' in "
            "system/controlDict before meshing, or run 'foamFormatConvert -ascii')."
        )
    return text


def _resolve_foam_file(poly_dir: Path, name: str) -> Path:
    """Locate an OpenFOAM polyMesh file, accepting plain or .gz, and descending
    into constant/polyMesh when given a case directory instead of the mesh dir."""
    for base in (poly_dir, poly_dir / "constant" / "polyMesh"):
        plain = base / name
        if plain.exists():
            return plain
        compressed = base / f"{name}.gz"
        if compressed.exists():
            return compressed
    raise FileNotFoundError(
        f"OpenFOAM polyMesh file '{name}' not found under {poly_dir} "
        f"(looked for '{name}' and '{name}.gz', and inside constant/polyMesh)"
    )


def _is_binary_foam(path: Path) -> bool:
    with open(path, "rb") as handle:
        head = handle.read(4096).decode("latin-1", "ignore")
    match = re.search(r"\bformat\s+(\w+)\s*;", head)
    return bool(match and match.group(1).lower() == "binary")


def _ascii_polymesh_dir(poly: str, out_dir: Path) -> str:
    """Return a polyMesh directory the C++ validator can read: resolves a case dir
    vs the mesh dir, decompresses any *.gz into an ASCII copy, and rejects binary."""
    src = _resolve_foam_file(Path(poly), "points").parent
    names = ["points", "faces", "owner", "neighbour", "boundary"]
    resolved = {name: _resolve_foam_file(src, name) for name in names}
    if all(path.suffix != ".gz" for path in resolved.values()):
        for path in resolved.values():
            if _is_binary_foam(path):
                _read_foam_text(path)  # raises the actionable binary-format error
        return str(src)
    target = out_dir / "_ascii_polyMesh"
    target.mkdir(parents=True, exist_ok=True)
    for name, path in resolved.items():
        (target / name).write_text(_read_foam_text(path), encoding="utf-8")
    return str(target)


def _openfoam_list_body(path: Path) -> str:
    text = _strip_foam_comments(_read_foam_text(path))
    match = re.search(r"(?:^|\n)\s*\d+\s*\n?\s*\(", text)
    if not match:
        raise ValueError(f"cannot find OpenFOAM list in {path}")
    open_index = text.find("(", match.start())
    close_index = _matching_paren(text, open_index)
    return text[open_index + 1:close_index]


def _read_openfoam_points(path: Path) -> list[list[float]]:
    body = _openfoam_list_body(path)
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    points = []
    for match in re.finditer(rf"\(\s*({number})\s+({number})\s+({number})\s*\)", body):
        points.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    if not points:
        raise ValueError(f"no points parsed from {path}")
    return points


def _read_openfoam_faces(path: Path) -> list[list[int]]:
    body = _openfoam_list_body(path)
    faces = []
    for match in re.finditer(r"(\d+)\s*\(([^()]*)\)", body):
        n = int(match.group(1))
        verts = [int(v) for v in match.group(2).split()]
        if len(verts) == n and n >= 3:
            faces.append(verts)
    if not faces:
        raise ValueError(f"no faces parsed from {path}")
    return faces


def _read_openfoam_boundary(path: Path) -> list[dict]:
    text = _strip_foam_comments(_read_foam_text(path))
    patches = []
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\{([^{}]*)\}", text, flags=re.S):
        name = match.group(1)
        body = match.group(2)
        n_faces = re.search(r"\bnFaces\s+(\d+)\s*;", body)
        start_face = re.search(r"\bstartFace\s+(\d+)\s*;", body)
        if n_faces and start_face:
            patches.append({
                "name": name,
                "faces": int(n_faces.group(1)),
                "startFace": int(start_face.group(1)),
            })
    if not patches:
        raise ValueError(f"no boundary patches parsed from {path}")
    return patches


def _openfoam_preview(poly_dir: Path) -> dict:
    if not poly_dir.exists():
        raise ValueError(f"OpenFOAM polyMesh path does not exist: {poly_dir}")
    points = _read_openfoam_points(_resolve_foam_file(poly_dir, "points"))
    all_faces = _read_openfoam_faces(_resolve_foam_file(poly_dir, "faces"))
    ranges = _read_openfoam_boundary(_resolve_foam_file(poly_dir, "boundary"))
    faces: list[dict] = []
    patches = []
    for patch in ranges:
        name = patch["name"]
        count = patch["faces"]
        start = patch["startFace"]
        patches.append({"name": name, "faces": count})
        for fi in range(start, min(start + count, len(all_faces))):
            faces.append({"id": fi, "patch": name, "vertices": all_faces[fi]})
    return _compact_preview(points, faces, patches)


def _foam_header(class_name: str, object_name: str) -> str:
    return (
        f"FoamFile {{ version 2.0; format ascii; class {class_name}; "
        f"object {object_name}; }}\n"
    )


def _write_foam_points(path: Path, points: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write(_foam_header("vectorField", "points"))
        out.write(f"{len(points)}\n(\n")
        for p in points:
            out.write(f"({p[0]:.16g} {p[1]:.16g} {p[2]:.16g})\n")
        out.write(")\n")


def _write_foam_faces(path: Path, faces: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write(_foam_header("faceList", "faces"))
        out.write(f"{len(faces)}\n(\n")
        for face in faces:
            out.write(f"{len(face)}(" + " ".join(str(i) for i in face) + ")\n")
        out.write(")\n")


def _write_foam_labels(path: Path, object_name: str, values: list[int]) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write(_foam_header("labelList", object_name))
        out.write(f"{len(values)}\n(\n")
        for value in values:
            out.write(f"{value}\n")
        out.write(")\n")


def _write_foam_boundary(path: Path, patch_ranges: list[dict]) -> None:
    patch_types = {
        "liquid_inlet": "patch",
        "nozzle_electrode": "wall",
        "collector_ground": "wall",
        "open_atmosphere": "patch",
    }
    with path.open("w", encoding="utf-8") as out:
        out.write(_foam_header("polyBoundaryMesh", "boundary"))
        out.write(f"{len(patch_ranges)}\n(\n")
        for patch in patch_ranges:
            patch_type = patch_types.get(patch["name"], "patch")
            out.write(
                f"{patch['name']} {{ type {patch_type}; "
                f"nFaces {patch['nFaces']}; startFace {patch['startFace']}; }}\n"
            )
        out.write(")\n")


def _generated_patch_roles() -> dict:
    return {
        "liquid_inlet": "inlet",
        "nozzle_electrode": "electrode",
        "collector_ground": "collector",
        "open_atmosphere": "outlet",
    }


def _generated_boundary_conditions(case: dict) -> dict:
    voltage = float(case.get("validation_voltage", 0.0))
    return {
        patch: resolve_bc_preset(role, voltage)
        for patch, role in _generated_patch_roles().items()
    }


def generate_nozzle_polymesh(case: dict) -> dict:
    case_name = sanitize_case_name(case.get("case_name", "gui_case"))
    nx = max(8, int(case.get("nx", 16)))
    ny = max(4, int(case.get("ny", 32)))
    nz = max(8, int(case.get("nz", nx)))
    inner_d = max(float(case.get("inner_diameter", 160e-6)), 1e-12)
    outer_d = max(float(case.get("outer_diameter", 260e-6)), inner_d * 1.05)
    collector_distance = max(float(case.get("collector_distance", 1.5e-3)), outer_d)
    collector_radius = max(float(case.get("collector_diameter", 5e-3)) * 0.5, outer_d * 0.5)
    radial_window = max(float(case.get("radial_window_outer_diameters", 1.5)), 1.15)
    radius = max(0.5 * outer_d * radial_window, 0.6 * outer_d)
    inner_radius = 0.5 * inner_d
    outer_radius = 0.5 * outer_d

    def pid(i: int, j: int, k: int) -> int:
      return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i

    raw_points: list[list[float]] = []
    for k in range(nz + 1):
        z = -radius + 2.0 * radius * k / nz
        for j in range(ny + 1):
            y = collector_distance * j / ny
            for i in range(nx + 1):
                x = -radius + 2.0 * radius * i / nx
                raw_points.append([x, y, z])

    selected: dict[tuple[int, int, int], int] = {}
    for k in range(nz):
        zc = -radius + 2.0 * radius * (k + 0.5) / nz
        for j in range(ny):
            for i in range(nx):
                xc = -radius + 2.0 * radius * (i + 0.5) / nx
                if math.hypot(xc, zc) <= radius * (1.0 + 1e-12):
                    selected[(i, j, k)] = len(selected)
    if not selected:
        raise ValueError("generated nozzle mesh contains no cells; increase nx/nz or radial window")

    local_faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 7, 6, 2],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ]
    face_map: dict[tuple[int, ...], dict] = {}
    face_order: list[tuple[int, ...]] = []
    for (i, j, k), cell_id in selected.items():
        cell_points = [
            pid(i, j, k),
            pid(i + 1, j, k),
            pid(i + 1, j + 1, k),
            pid(i, j + 1, k),
            pid(i, j, k + 1),
            pid(i + 1, j, k + 1),
            pid(i + 1, j + 1, k + 1),
            pid(i, j + 1, k + 1),
        ]
        for local in local_faces:
            verts = [cell_points[q] for q in local]
            key = tuple(sorted(verts))
            if key in face_map:
                face_map[key]["neighbour"] = cell_id
            else:
                face_map[key] = {"vertices": verts, "owner": cell_id, "neighbour": -1}
                face_order.append(key)

    dx = 2.0 * radius / nx
    dz = 2.0 * radius / nz
    inlet_threshold = max(inner_radius, 0.76 * min(dx, dz))

    def classify_boundary(vertices: list[int]) -> str:
        c = _face_center(raw_points, vertices)
        r = math.hypot(c[0], c[2])
        tol = 1e-9 * max(collector_distance, radius, 1.0)
        if abs(c[1]) <= tol:
            if r <= inlet_threshold:
                return "liquid_inlet"
            if r <= outer_radius:
                return "nozzle_electrode"
            return "open_atmosphere"
        if abs(c[1] - collector_distance) <= tol:
            if r <= collector_radius:
                return "collector_ground"
            return "open_atmosphere"
        return "open_atmosphere"

    internal = []
    boundary_by_patch = {
        "liquid_inlet": [],
        "nozzle_electrode": [],
        "collector_ground": [],
        "open_atmosphere": [],
    }
    for key in face_order:
        face = face_map[key]
        if face["neighbour"] >= 0:
            internal.append(face)
        else:
            boundary_by_patch[classify_boundary(face["vertices"])].append(face)

    ordered_faces = internal[:]
    patch_ranges = []
    for name, faces in boundary_by_patch.items():
        patch_ranges.append({"name": name, "startFace": len(ordered_faces), "nFaces": len(faces)})
        ordered_faces.extend(faces)

    used_points = sorted({idx for face in ordered_faces for idx in face["vertices"]})
    remap = {old: new for new, old in enumerate(used_points)}
    points = [raw_points[idx] for idx in used_points]
    faces = [[remap[idx] for idx in face["vertices"]] for face in ordered_faces]
    owners = [face["owner"] for face in ordered_faces]
    neighbours = [face["neighbour"] for face in internal]

    target = GENERATED / case_name / "constant" / "polyMesh"
    target.mkdir(parents=True, exist_ok=True)
    _write_foam_points(target / "points", points)
    _write_foam_faces(target / "faces", faces)
    _write_foam_labels(target / "owner", "owner", owners)
    _write_foam_labels(target / "neighbour", "neighbour", neighbours)
    _write_foam_boundary(target / "boundary", patch_ranges)

    metadata = {
        "status": "pass",
        "generator": "internal_candido_nozzle_template",
        "generated_polyMesh": str(target),
        "cells": len(selected),
        "points": len(points),
        "faces": len(faces),
        "internal_faces": len(internal),
        "boundary_faces": len(faces) - len(internal),
        "patches": [{"name": p["name"], "faces": p["nFaces"]} for p in patch_ranges],
        "dimensions": {
            "inner_diameter": inner_d,
            "outer_diameter": outer_d,
            "collector_distance": collector_distance,
            "collector_diameter": collector_radius * 2.0,
            "air_domain_radius": radius,
        },
        "notes": [
            "Cartesian cut-cylinder full-3D template with stair-stepped cylindrical farfield.",
            "Nozzle is represented by split top patches, not a resolved internal capillary wall.",
        ],
        "patch_roles": _generated_patch_roles(),
        "boundary_conditions": _generated_boundary_conditions(case),
    }
    (target.parent.parent / "mesh_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def generate_cad_nozzle_case(case: dict) -> dict:
    case_name = sanitize_case_name(case.get("case_name", "gui_case"))
    out_dir = GENERATED_CAD / case_name
    outer_d = max(float(case.get("outer_diameter", 260e-6)), 1e-12)
    collector_d = max(float(case.get("collector_diameter", 5.0e-3)), outer_d)
    farfield_radius = max(
        0.5 * collector_d,
        outer_d * max(float(case.get("radial_window_outer_diameters", 1.5)), 1.0),
    )
    segments = max(32, min(256, int(max(float(case.get("nx", 8)), float(case.get("nz", 8))) * 8)))
    args = [
        sys.executable,
        str(ROOT / "apps" / "generate_nozzle_cad.py"),
        "--output-dir", str(out_dir),
        "--inner-diameter", str(float(case.get("inner_diameter", 160e-6))),
        "--outer-diameter", str(outer_d),
        "--nozzle-length", str(float(case.get("nozzle_length", 300e-6))),
        "--collector-distance", str(float(case.get("collector_distance", 1.5e-3))),
        "--collector-diameter", str(collector_d),
        "--farfield-radius", str(farfield_radius),
        "--segments", str(segments),
        "--voltage", str(float(case.get("validation_voltage", 2180.0))),
    ]
    payload = run_json_command(args, timeout=120)
    stl_path = str(payload.get("stl", "")).strip()
    if stl_path:
        surface_validation = run_json_command(
            [
                sys.executable,
                str(ROOT / "apps" / "validate_nozzle_cad_surface.py"),
                stl_path,
            ],
            timeout=120,
        )
        payload["surface_validation"] = surface_validation
        if surface_validation.get("status") != "pass":
            payload["status"] = "fail"
    payload["cad_workflow"] = "snappyHexMesh_ready_surface_case"
    payload["snappy_commands"] = ["blockMesh", "surfaceFeatureExtract", "snappyHexMesh -overwrite", "checkMesh"]
    payload["cfmesh_commands"] = ["cartesianMesh", "checkMesh"]
    payload["patch_roles"] = {
        "liquid_inlet": "inlet",
        "inner_nozzle_wall": "wall",
        "nozzle_electrode": "electrode",
        "collector_ground": "collector",
        "open_atmosphere": "outlet",
    }
    payload["boundary_conditions"] = _generated_boundary_conditions(case)
    payload["boundary_conditions"]["inner_nozzle_wall"] = resolve_bc_preset(
        "electrode", float(case.get("validation_voltage", 0.0))
    )
    return payload


def run_cad_meshing(case: dict) -> dict:
    case_dir = str(case.get("openfoam_case_dir", "")).strip()
    workflow = str(case.get("cad_meshing_workflow", "snappy")).strip() or "snappy"
    if workflow not in {"snappy", "cfmesh"}:
        raise ValueError("cad_meshing_workflow must be 'snappy' or 'cfmesh'")
    if not case_dir:
        raise ValueError("openfoam_case_dir is required; generate a CAD/snappy case first")
    args = [
        sys.executable,
        str(ROOT / "apps" / "run_openfoam_meshing.py"),
        "--case-dir", case_dir,
        "--workflow", workflow,
    ]
    payload = run_json_command(args, timeout=1200)
    if payload.get("status") == "pass" and payload.get("polyMesh"):
        payload["openfoam_polyMesh"] = payload["polyMesh"]
    return payload


def mesh_preview(case: dict) -> dict:
    mesh_mode = case.get("mesh_mode", "builtin_hex")
    if mesh_mode == "openfoam_polyMesh":
        poly = str(case.get("openfoam_polyMesh", "")).strip()
        if not poly and str(case.get("openfoam_case_dir", "")).strip():
            poly = str(Path(str(case["openfoam_case_dir"])) / "constant" / "polyMesh")
        if not poly:
            raise ValueError("openfoam_polyMesh is required for OpenFOAM mesh preview")
        preview = _openfoam_preview(Path(poly))
    else:
        preview = _builtin_hex_preview(case)
    preview["status"] = "pass"
    preview["mesh_mode"] = mesh_mode
    preview["boundary_conditions"] = case.get("boundary_conditions", {})
    preview["patch_roles"] = case.get("patch_roles", {})
    return preview


def write_case(case: dict) -> Path:
    case_name = sanitize_case_name(case.get("case_name", "gui_case"))
    case["case_name"] = case_name
    out_dir = RUNS / case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    case_path = out_dir / "case.json"
    case_path.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if case.get("boundary_conditions"):
        (out_dir / "boundary_conditions.json").write_text(
            json.dumps(case["boundary_conditions"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return case_path


def upload_polymesh(payload: dict) -> dict:
    case_name = sanitize_case_name(payload.get("case_name", "gui_case"))
    files = payload.get("files", [])
    if not isinstance(files, list) or not files:
        raise ValueError("no files uploaded")
    target = UPLOADS / case_name / "constant" / "polyMesh"
    target.mkdir(parents=True, exist_ok=True)
    required = {"points", "faces", "owner", "neighbour", "boundary"}
    written: set[str] = set()
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            continue
        name = Path(str(item.get("name", ""))).name
        if name not in required:
            continue
        text = str(item.get("text", ""))
        total_bytes += len(text.encode("utf-8"))
        if total_bytes > 200 * 1024 * 1024:
            raise ValueError("uploaded polyMesh is larger than 200 MB")
        (target / name).write_text(text, encoding="utf-8")
        written.add(name)
    missing = sorted(required - written)
    if missing:
        raise ValueError("missing OpenFOAM polyMesh files: " + ", ".join(missing))
    case = dict(payload.get("case", {}))
    case["case_name"] = case_name
    case["mesh_mode"] = "openfoam_polyMesh"
    case["openfoam_polyMesh"] = str(target)
    validation = validate_mesh(case)
    validation["uploaded_polyMesh"] = str(target)
    validation["uploaded_files"] = sorted(written)
    return validation


def _copy_selected_openfoam_files(files: list, target: Path, required_suffixes: set[str]) -> set[str]:
    written: set[str] = set()
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("relative_path") or item.get("name", "")).replace("\\", "/")
        text = str(item.get("text", ""))
        matched = None
        for suffix in required_suffixes:
            if rel.endswith(suffix):
                matched = suffix
                break
        if matched is None:
            continue
        total_bytes += len(text.encode("utf-8"))
        if total_bytes > 250 * 1024 * 1024:
            raise ValueError("uploaded OpenFOAM case is larger than 250 MB")
        out = target / matched
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        written.add(matched)
    return written


def validate_openfoam_case(case_dir: Path, case_name: str) -> dict:
    exe = executable("electrospray_case_runner")
    out_dir = RUNS / sanitize_case_name(case_name) / "openfoam_case_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(exe), "--case-dir", str(case_dir), "--output-dir", str(out_dir)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        payload = {"status": "error"}
    payload["returncode"] = proc.returncode
    payload["openfoam_case_dir"] = str(case_dir)
    payload["output_dir"] = str(out_dir)
    payload["stdout"] = proc.stdout
    if proc.stderr:
        payload["stderr"] = proc.stderr
    if proc.returncode != 0:
        payload["status"] = "error"
    return payload


def upload_openfoam_case(payload: dict) -> dict:
    case_name = sanitize_case_name(payload.get("case_name", "gui_case"))
    files = payload.get("files", [])
    if not isinstance(files, list) or not files:
        raise ValueError("no files uploaded")
    target = UPLOADS / case_name / "openfoam_case"
    required = {
        "constant/polyMesh/points",
        "constant/polyMesh/faces",
        "constant/polyMesh/owner",
        "constant/polyMesh/neighbour",
        "constant/polyMesh/boundary",
        "0/U",
        "0/p",
        "0/alpha",
        "0/phi",
        "0/rhoE",
    }
    written = _copy_selected_openfoam_files(files, target, required)
    missing = sorted(required - written)
    if missing:
        raise ValueError("missing OpenFOAM case files: " + ", ".join(missing))
    result = validate_openfoam_case(target, case_name)
    result["uploaded_openfoam_case"] = str(target)
    result["uploaded_files"] = sorted(written)
    return result


def run_case(case: dict) -> dict:
    case_path = write_case(case)
    out_dir = case_path.parent
    exe = executable("electrospray_case_runner")
    if case.get("run_mode") == "validate_openfoam_case":
        case_dir_text = str(case.get("openfoam_case_dir", "")).strip()
        if not case_dir_text:
            raise ValueError("openfoam_case_dir is required for validate_openfoam_case")
        case_dir = Path(case_dir_text)
        case_out_dir = out_dir / "openfoam_case_validation"
        proc = subprocess.run(
            [str(exe), "--case-dir", str(case_dir), "--output-dir", str(case_out_dir)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        summary_path = case_out_dir / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            payload = {"status": "error"}
        payload["returncode"] = proc.returncode
        payload["case_json"] = str(case_path)
        payload["openfoam_case_dir"] = str(case_dir)
        payload["output_dir"] = str(case_out_dir)
        payload["stdout"] = proc.stdout
        if proc.stderr:
            payload["stderr"] = proc.stderr
        if proc.returncode != 0:
            payload["status"] = "error"
        return payload

    proc = subprocess.run(
        [str(exe), "--case", str(case_path), "--output-dir", str(out_dir)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
        check=False,
    )
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        payload = {"status": "error"}
    payload["returncode"] = proc.returncode
    payload["case_json"] = str(case_path)
    payload["output_dir"] = str(out_dir)
    payload["stdout"] = proc.stdout
    if proc.stderr:
        payload["stderr"] = proc.stderr
    if proc.returncode != 0:
        payload["status"] = "error"
    payload["boundary_conditions"] = case.get("boundary_conditions", {})
    payload["boundary_condition_application"] = (
        "recorded_in_case_json; candido_smoke currently uses its built-in boundary model"
    )
    history_path = out_dir / "history.csv"
    if history_path.exists():
        payload["history"] = _parse_history_csv(history_path)
    return payload


def list_saved_cases() -> dict:
    cases = []
    if RUNS.exists():
        for case_json in sorted(RUNS.glob("*/case.json")):
            cases.append(case_json.parent.name)
    return {"status": "pass", "cases": cases}


def load_saved_case(name: str) -> dict:
    safe = sanitize_case_name(name)
    case_path = RUNS / safe / "case.json"
    if not case_path.exists():
        raise ValueError(f"no saved case named '{safe}'")
    return json.loads(case_path.read_text(encoding="utf-8"))


def _parse_history_csv(path: Path, max_rows: int = 4000) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    columns = lines[0].split(",")
    data_lines = [ln for ln in lines[1:] if ln.strip()]
    stride = max(1, math.ceil(len(data_lines) / max_rows))
    rows = []
    for ln in data_lines[::stride]:
        row = []
        for p in ln.split(","):
            try:
                row.append(float(p))
            except ValueError:
                row.append(p)
        rows.append(row)
    return {"columns": columns, "rows": rows, "row_count": len(data_lines), "stride": stride}


def run_history(name: str) -> dict:
    safe = sanitize_case_name(name)
    hist = RUNS / safe / "history.csv"
    if not hist.exists():
        raise ValueError(f"no run history for case '{safe}'")
    return {"status": "pass", "history": _parse_history_csv(hist)}


class Handler(BaseHTTPRequestHandler):
    server_version = "ElectrosprayGUI/1.0"

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/default-case":
            self.send_json(default_case())
            return
        if path == "/api/ui-config":
            self.send_json(ui_config())
            return
        if path == "/api/list-cases":
            self.send_json(list_saved_cases())
            return
        if path == "/api/load-case":
            try:
                name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
                self.send_json(load_saved_case(name))
            except Exception as exc:
                self.send_json({"status": "error", "error": str(exc)}, status=404)
            return
        if path == "/api/run-history":
            try:
                name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
                self.send_json(run_history(name))
            except Exception as exc:
                self.send_json({"status": "error", "error": str(exc)}, status=404)
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path in {"/", "/index.html"}:
            file_path = GUI_DIR / "index.html"
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json({"status": "error", "error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            payload = self.read_json()
            if path == "/api/validate-mesh":
                self.send_json(validate_mesh(payload))
            elif path == "/api/mesh-preview":
                self.send_json(mesh_preview(payload))
            elif path == "/api/generate-nozzle-mesh":
                metadata = generate_nozzle_polymesh(payload)
                case = dict(payload)
                case["mesh_mode"] = "openfoam_polyMesh"
                case["openfoam_polyMesh"] = metadata["generated_polyMesh"]
                case["patch_roles"] = metadata["patch_roles"]
                case["boundary_conditions"] = metadata["boundary_conditions"]
                validation = validate_mesh(case)
                metadata.update(validation)
                metadata["status"] = validation.get("status", metadata["status"])
                metadata["generated_polyMesh"] = case["openfoam_polyMesh"]
                metadata["patch_roles"] = case["patch_roles"]
                metadata["boundary_conditions"] = case["boundary_conditions"]
                self.send_json(metadata)
            elif path == "/api/generate-cad-nozzle-case":
                self.send_json(generate_cad_nozzle_case(payload))
            elif path == "/api/run-cad-meshing":
                self.send_json(run_cad_meshing(payload))
            elif path == "/api/upload-polymesh":
                self.send_json(upload_polymesh(payload))
            elif path == "/api/upload-openfoam-case":
                self.send_json(upload_openfoam_case(payload))
            elif path == "/api/save-case":
                case_path = write_case(payload)
                self.send_json({"status": "pass", "case_json": str(case_path)})
            elif path == "/api/run-case":
                self.send_json(run_case(payload))
            else:
                self.send_json({"status": "error", "error": "not found"}, status=404)
        except Exception as exc:  # pragma: no cover - surfaced to GUI
            self.send_json({"status": "error", "error": str(exc)}, status=500)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    GENERATED_CAD.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Electrospray GUI: http://{args.host}:{args.port}")
    print("Build first if needed: cmake --build build")
    server.serve_forever()


if __name__ == "__main__":
    main()
