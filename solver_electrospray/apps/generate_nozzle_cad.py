#!/usr/bin/env python3
"""Generate a CAD-like electrospray nozzle surface package.

The output is intentionally dependency-free: a parametric OpenSCAD source for
editing, plus multi-solid STL/OBJ surfaces that can be consumed by OpenFOAM
surface workflows such as snappyHexMesh.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normal(a: tuple[float, float, float],
           b: tuple[float, float, float],
           c: tuple[float, float, float]) -> tuple[float, float, float]:
    n = cross(vec_sub(b, a), vec_sub(c, a))
    mag = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    if mag <= 0.0:
        return (0.0, 0.0, 0.0)
    return (n[0] / mag, n[1] / mag, n[2] / mag)


def p(radius: float, theta: float, y: float) -> tuple[float, float, float]:
    return (radius * math.cos(theta), y, radius * math.sin(theta))


def add_cylinder_side(solids: dict[str, list[tuple]], name: str, radius: float,
                      y0: float, y1: float, segments: int, outward: bool = True) -> None:
    tris = solids.setdefault(name, [])
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        p00 = p(radius, a0, y0)
        p01 = p(radius, a1, y0)
        p10 = p(radius, a0, y1)
        p11 = p(radius, a1, y1)
        if outward:
            tris.append((p00, p01, p11))
            tris.append((p00, p11, p10))
        else:
            tris.append((p00, p11, p01))
            tris.append((p00, p10, p11))


def add_disk(solids: dict[str, list[tuple]], name: str, radius: float, y: float,
             segments: int, normal_sign_y: float) -> None:
    tris = solids.setdefault(name, [])
    center = (0.0, y, 0.0)
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        q0 = p(radius, a0, y)
        q1 = p(radius, a1, y)
        if normal_sign_y >= 0.0:
            tris.append((center, q1, q0))
        else:
            tris.append((center, q0, q1))


def add_annulus(solids: dict[str, list[tuple]], name: str, r_inner: float, r_outer: float,
                y: float, segments: int, normal_sign_y: float) -> None:
    tris = solids.setdefault(name, [])
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        i0 = p(r_inner, a0, y)
        i1 = p(r_inner, a1, y)
        o0 = p(r_outer, a0, y)
        o1 = p(r_outer, a1, y)
        if normal_sign_y >= 0.0:
            tris.append((i0, o1, o0))
            tris.append((i0, i1, o1))
        else:
            tris.append((i0, o0, o1))
            tris.append((i0, o1, i1))


def write_stl(path: Path, solids: dict[str, list[tuple]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for name, tris in solids.items():
            out.write(f"solid {name}\n")
            for tri in tris:
                n = normal(*tri)
                out.write(f"  facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}\n")
                out.write("    outer loop\n")
                for v in tri:
                    out.write(f"      vertex {v[0]:.12e} {v[1]:.12e} {v[2]:.12e}\n")
                out.write("    endloop\n")
                out.write("  endfacet\n")
            out.write(f"endsolid {name}\n")


def write_obj(path: Path, solids: dict[str, list[tuple]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write("# Electrospray nozzle CAD-like surface package\n")
        vertex_index = 1
        for name, tris in solids.items():
            out.write(f"g {name}\n")
            for tri in tris:
                for v in tri:
                    out.write(f"v {v[0]:.12e} {v[1]:.12e} {v[2]:.12e}\n")
                out.write(f"f {vertex_index} {vertex_index + 1} {vertex_index + 2}\n")
                vertex_index += 3


def write_scad(path: Path, meta: dict) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write("// Parametric electrospray nozzle/collector geometry.\n")
        out.write("// Units are meters; OpenSCAD will display them as model units.\n")
        out.write(f"inner_diameter = {meta['inner_diameter']:.16g};\n")
        out.write(f"outer_diameter = {meta['outer_diameter']:.16g};\n")
        out.write(f"nozzle_length = {meta['nozzle_length']:.16g};\n")
        out.write(f"collector_distance = {meta['collector_distance']:.16g};\n")
        out.write(f"collector_diameter = {meta['collector_diameter']:.16g};\n")
        out.write(f"farfield_radius = {meta['farfield_radius']:.16g};\n")
        out.write("$fn = 128;\n\n")
        out.write("module nozzle_electrode() {\n")
        out.write("  translate([0, -nozzle_length, 0]) rotate([-90,0,0])\n")
        out.write("  difference() {\n")
        out.write("    cylinder(h=nozzle_length, d=outer_diameter);\n")
        out.write("    translate([0,0,-1e-9]) cylinder(h=nozzle_length + 2e-9, d=inner_diameter);\n")
        out.write("  }\n")
        out.write("}\n\n")
        out.write("module collector_ground() {\n")
        out.write("  translate([0, collector_distance, 0]) rotate([-90,0,0])\n")
        out.write("    cylinder(h=outer_diameter*0.08, d=collector_diameter, center=true);\n")
        out.write("}\n\n")
        out.write("module open_atmosphere_hint() {\n")
        out.write("  %translate([0, (-nozzle_length + collector_distance)/2, 0]) rotate([-90,0,0])\n")
        out.write("    cylinder(h=nozzle_length + collector_distance, r=farfield_radius);\n")
        out.write("}\n\n")
        out.write("nozzle_electrode();\n")
        out.write("collector_ground();\n")
        out.write("open_atmosphere_hint();\n")


def foam_header(class_name: str, object_name: str, location: str = "") -> str:
    loc = f"    location    \"{location}\";\n" if location else ""
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"{loc}"
        f"    object      {object_name};\n"
        "}\n\n"
    )


def write_block_mesh_dict(path: Path, meta: dict) -> None:
    r = meta["farfield_radius"] * 1.08
    y_min = -meta["nozzle_length"]
    y_max = meta["collector_distance"]
    nx = 28
    ny = max(24, int(round((y_max - y_min) / max(2.0 * r / nx, 1e-12))))
    nz = 28
    vertices = [
        (-r, y_min, -r),
        (r, y_min, -r),
        (r, y_max, -r),
        (-r, y_max, -r),
        (-r, y_min, r),
        (r, y_min, r),
        (r, y_max, r),
        (-r, y_max, r),
    ]
    with path.open("w", encoding="utf-8") as out:
        out.write(foam_header("dictionary", "blockMeshDict", "system"))
        out.write("convertToMeters 1;\n\n")
        out.write("vertices\n(\n")
        for v in vertices:
            out.write(f"    ({v[0]:.12e} {v[1]:.12e} {v[2]:.12e})\n")
        out.write(");\n\n")
        out.write("blocks\n(\n")
        out.write(f"    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)\n")
        out.write(");\n\n")
        out.write("edges ();\n\n")
        out.write("boundary\n(\n")
        out.write("    background\n")
        out.write("    {\n")
        out.write("        type patch;\n")
        out.write("        faces\n")
        out.write("        (\n")
        out.write("            (0 4 7 3)\n")
        out.write("            (1 2 6 5)\n")
        out.write("            (0 1 5 4)\n")
        out.write("            (3 7 6 2)\n")
        out.write("            (0 3 2 1)\n")
        out.write("            (4 5 6 7)\n")
        out.write("        );\n")
        out.write("    }\n")
        out.write(");\n\n")
        out.write("mergePatchPairs ();\n")


def write_surface_feature_extract_dict(path: Path, stl_name: str) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write(foam_header("dictionary", "surfaceFeatureExtractDict", "system"))
        out.write(f"{stl_name}\n")
        out.write("{\n")
        out.write("    extractionMethod extractFromSurface;\n")
        out.write("    extractFromSurfaceCoeffs\n")
        out.write("    {\n")
        out.write("        includedAngle 150;\n")
        out.write("    }\n")
        out.write("    writeObj yes;\n")
        out.write("}\n")


def write_mesh_quality_dict(path: Path) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write(foam_header("dictionary", "meshQualityDict", "system"))
        out.write("maxNonOrtho 70;\n")
        out.write("maxBoundarySkewness 20;\n")
        out.write("maxInternalSkewness 4;\n")
        out.write("maxConcave 80;\n")
        out.write("minVol 1e-30;\n")
        out.write("minTetQuality -1e30;\n")
        out.write("minArea -1;\n")
        out.write("minTwist 0.02;\n")
        out.write("minDeterminant 0.001;\n")
        out.write("minFaceWeight 0.02;\n")
        out.write("minVolRatio 0.01;\n")
        out.write("minTriangleTwist -1;\n")
        out.write("nSmoothScale 4;\n")
        out.write("errorReduction 0.75;\n")


def write_snappy_hex_mesh_dict(path: Path, meta: dict, stl_name: str) -> None:
    y_mid = 0.50 * (-meta["nozzle_length"] + meta["collector_distance"])
    r = min(meta["farfield_radius"] * 0.20, meta["collector_diameter"] * 0.20)
    with path.open("w", encoding="utf-8") as out:
        out.write(foam_header("dictionary", "snappyHexMeshDict", "system"))
        out.write("castellatedMesh true;\n")
        out.write("snap true;\n")
        out.write("addLayers true;\n\n")
        out.write("geometry\n")
        out.write("{\n")
        out.write(f"    {stl_name}\n")
        out.write("    {\n")
        out.write("        type triSurfaceMesh;\n")
        out.write(f"        file \"{stl_name}\";\n")
        out.write("        regions\n")
        out.write("        {\n")
        for region in ["liquid_inlet", "inner_nozzle_wall", "nozzle_electrode",
                       "collector_ground", "open_atmosphere"]:
            out.write(f"            {region} {{ name {region}; }}\n")
        out.write("        }\n")
        out.write("    }\n")
        out.write("    nozzleTipRefinement\n")
        out.write("    {\n")
        out.write("        type searchableSphere;\n")
        out.write("        centre (0 0 0);\n")
        out.write(f"        radius {max(meta['outer_diameter'] * 1.5, 1e-12):.12e};\n")
        out.write("    }\n")
        out.write("    jetCoreRefinement\n")
        out.write("    {\n")
        out.write("        type searchableCylinder;\n")
        out.write(f"        point1 (0 {-meta['nozzle_length']:.12e} 0);\n")
        out.write(f"        point2 (0 {meta['collector_distance']:.12e} 0);\n")
        out.write(f"        radius {max(r, meta['outer_diameter']):.12e};\n")
        out.write("    }\n")
        out.write("}\n\n")
        out.write("castellatedMeshControls\n")
        out.write("{\n")
        out.write("    maxLocalCells 1000000;\n")
        out.write("    maxGlobalCells 3000000;\n")
        out.write("    minRefinementCells 0;\n")
        out.write("    nCellsBetweenLevels 3;\n")
        out.write("    features\n")
        out.write("    (\n")
        out.write(f"        {{ file \"{Path(stl_name).with_suffix('.eMesh').name}\"; level 2; }}\n")
        out.write("    );\n")
        out.write("    refinementSurfaces\n")
        out.write("    {\n")
        out.write(f"        {stl_name}\n")
        out.write("        {\n")
        out.write("            level (2 3);\n")
        out.write("            regions\n")
        out.write("            {\n")
        out.write("                liquid_inlet { level (3 4); patchInfo { type patch; } }\n")
        out.write("                inner_nozzle_wall { level (4 5); patchInfo { type wall; } }\n")
        out.write("                nozzle_electrode { level (4 5); patchInfo { type wall; } }\n")
        out.write("                collector_ground { level (2 3); patchInfo { type wall; } }\n")
        out.write("                open_atmosphere { level (1 2); patchInfo { type patch; } }\n")
        out.write("            }\n")
        out.write("        }\n")
        out.write("    }\n")
        out.write("    refinementRegions\n")
        out.write("    {\n")
        out.write("        nozzleTipRefinement { mode inside; levels ((1e15 5)); }\n")
        out.write("        jetCoreRefinement { mode inside; levels ((1e15 3)); }\n")
        out.write("    }\n")
        out.write(f"    locationInMesh (0 {y_mid:.12e} 0);\n")
        out.write("    allowFreeStandingZoneFaces true;\n")
        out.write("    resolveFeatureAngle 30;\n")
        out.write("}\n\n")
        out.write("snapControls\n")
        out.write("{\n")
        out.write("    nSmoothPatch 3;\n")
        out.write("    tolerance 2.0;\n")
        out.write("    nSolveIter 30;\n")
        out.write("    nRelaxIter 5;\n")
        out.write("    nFeatureSnapIter 10;\n")
        out.write("    implicitFeatureSnap false;\n")
        out.write("    explicitFeatureSnap true;\n")
        out.write("    multiRegionFeatureSnap true;\n")
        out.write("}\n\n")
        out.write("addLayersControls\n")
        out.write("{\n")
        out.write("    relativeSizes true;\n")
        out.write("    layers\n")
        out.write("    {\n")
        out.write("        inner_nozzle_wall { nSurfaceLayers 3; }\n")
        out.write("        nozzle_electrode { nSurfaceLayers 3; }\n")
        out.write("        collector_ground { nSurfaceLayers 2; }\n")
        out.write("    }\n")
        out.write("    expansionRatio 1.2;\n")
        out.write("    finalLayerThickness 0.3;\n")
        out.write("    minThickness 0.08;\n")
        out.write("    nGrow 0;\n")
        out.write("    featureAngle 60;\n")
        out.write("    nRelaxIter 5;\n")
        out.write("    nSmoothSurfaceNormals 1;\n")
        out.write("    nSmoothNormals 3;\n")
        out.write("    nSmoothThickness 10;\n")
        out.write("    maxFaceThicknessRatio 0.5;\n")
        out.write("    maxThicknessToMedialRatio 0.3;\n")
        out.write("    minMedianAxisAngle 90;\n")
        out.write("    nBufferCellsNoExtrude 0;\n")
        out.write("    nLayerIter 50;\n")
        out.write("}\n\n")
        out.write("meshQualityControls\n")
        out.write("{\n")
        out.write("    #include \"meshQualityDict\"\n")
        out.write("}\n\n")
        out.write("writeFlags (scalarLevels layerSets layerFields);\n")
        out.write("mergeTolerance 1e-6;\n")


def openfoam_vector_boundary(name: str, values: dict[str, tuple[str, str]]) -> str:
    body = ""
    for patch, (bc_type, value) in values.items():
        body += f"    {patch}\n    {{\n        type {bc_type};\n"
        if value:
            body += f"        value {value};\n"
        body += "    }\n"
    return body


def write_initial_fields(case_dir: Path, meta: dict, voltage: float = 2180.0) -> None:
    zero = case_dir / "0"
    zero.mkdir(parents=True, exist_ok=True)
    vector_zero = "uniform (0 0 0)"
    fields = {
        "U": (
            "volVectorField",
            "uniform (0 0 0)",
            {
                "liquid_inlet": ("fixedValue", "uniform (0 1 0)"),
                "inner_nozzle_wall": ("noSlip", ""),
                "nozzle_electrode": ("noSlip", ""),
                "collector_ground": ("movingWallVelocity", "uniform (0 0 0)"),
                "open_atmosphere": ("zeroGradient", ""),
                "background": ("zeroGradient", ""),
            },
        ),
        "p": (
            "volScalarField",
            "uniform 0",
            {
                "liquid_inlet": ("zeroGradient", ""),
                "inner_nozzle_wall": ("zeroGradient", ""),
                "nozzle_electrode": ("zeroGradient", ""),
                "collector_ground": ("zeroGradient", ""),
                "open_atmosphere": ("fixedValue", "uniform 0"),
                "background": ("fixedValue", "uniform 0"),
            },
        ),
        "alpha": (
            "volScalarField",
            "uniform 0",
            {
                "liquid_inlet": ("fixedValue", "uniform 1"),
                "inner_nozzle_wall": ("zeroGradient", ""),
                "nozzle_electrode": ("zeroGradient", ""),
                "collector_ground": ("zeroGradient", ""),
                "open_atmosphere": ("zeroGradient", ""),
                "background": ("zeroGradient", ""),
            },
        ),
        "phi": (
            "volScalarField",
            "uniform 0",
            {
                "liquid_inlet": ("fixedValue", f"uniform {voltage:.12g}"),
                "inner_nozzle_wall": ("fixedValue", f"uniform {voltage:.12g}"),
                "nozzle_electrode": ("fixedValue", f"uniform {voltage:.12g}"),
                "collector_ground": ("fixedValue", "uniform 0"),
                "open_atmosphere": ("zeroGradient", ""),
                "background": ("zeroGradient", ""),
            },
        ),
        "rhoE": (
            "volScalarField",
            "uniform 0",
            {
                "liquid_inlet": ("zeroGradient", ""),
                "inner_nozzle_wall": ("zeroGradient", ""),
                "nozzle_electrode": ("zeroGradient", ""),
                "collector_ground": ("zeroGradient", ""),
                "open_atmosphere": ("zeroGradient", ""),
                "background": ("zeroGradient", ""),
            },
        ),
    }
    for name, (field_class, internal, boundary) in fields.items():
        with (zero / name).open("w", encoding="utf-8") as out:
            out.write(foam_header(field_class, name, "0"))
            dims = "[0 1 -1 0 0 0 0]" if name == "U" else "[0 0 0 0 0 0 0]"
            out.write(f"dimensions {dims};\n")
            out.write(f"internalField {internal};\n")
            out.write("boundaryField\n{\n")
            out.write(openfoam_vector_boundary(name, boundary))
            out.write("}\n")


def write_control_dict(path: Path) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write(foam_header("dictionary", "controlDict", "system"))
        out.write("application     snappyHexMesh;\n")
        out.write("startFrom       startTime;\n")
        out.write("startTime       0;\n")
        out.write("stopAt          endTime;\n")
        out.write("endTime         1;\n")
        out.write("deltaT          1;\n")
        out.write("writeControl    timeStep;\n")
        out.write("writeInterval   1;\n")
        out.write("purgeWrite      0;\n")
        out.write("writeFormat     ascii;\n")
        out.write("writePrecision  8;\n")
        out.write("writeCompression off;\n")
        out.write("timeFormat      general;\n")
        out.write("timePrecision   6;\n")
        out.write("runTimeModifiable true;\n")


def write_cfmesh_dict(path: Path, meta: dict, stl_name: str) -> None:
    max_cell = max(meta["outer_diameter"] * 0.25, meta["farfield_radius"] / 48.0)
    boundary_cell = max(meta["outer_diameter"] * 0.08, max_cell / 6.0)
    min_cell = max(meta["inner_diameter"] * 0.04, boundary_cell / 4.0)
    first_layer = max(meta["inner_diameter"] * 0.015, min_cell)
    with path.open("w", encoding="utf-8") as out:
        out.write(foam_header("dictionary", "meshDict", "system"))
        out.write(f"surfaceFile \"constant/triSurface/{stl_name}\";\n\n")
        out.write(f"maxCellSize {max_cell:.12e};\n")
        out.write(f"boundaryCellSize {boundary_cell:.12e};\n")
        out.write(f"minCellSize {min_cell:.12e};\n")
        out.write(f"boundaryCellSizeRefinementThickness {max(meta['outer_diameter'], max_cell):.12e};\n\n")
        out.write("keepCellsIntersectingBoundary 1;\n\n")
        out.write("localRefinement\n")
        out.write("{\n")
        out.write("    liquid_inlet { additionalRefinementLevels 3; }\n")
        out.write("    inner_nozzle_wall { additionalRefinementLevels 4; }\n")
        out.write("    nozzle_electrode { additionalRefinementLevels 4; }\n")
        out.write("    collector_ground { additionalRefinementLevels 2; }\n")
        out.write("    open_atmosphere { additionalRefinementLevels 1; }\n")
        out.write("}\n\n")
        out.write("objectRefinements\n")
        out.write("{\n")
        out.write("    jet_core\n")
        out.write("    {\n")
        out.write("        type cone;\n")
        out.write(f"        p0 (0 {-meta['nozzle_length']:.12e} 0);\n")
        out.write(f"        p1 (0 {meta['collector_distance']:.12e} 0);\n")
        out.write(f"        radius0 {max(meta['outer_diameter'], 1e-12):.12e};\n")
        out.write(f"        radius1 {max(meta['outer_diameter'] * 2.0, 1e-12):.12e};\n")
        out.write(f"        cellSize {boundary_cell:.12e};\n")
        out.write(f"        refinementThickness {max(meta['outer_diameter'], boundary_cell):.12e};\n")
        out.write("    }\n")
        out.write("}\n\n")
        out.write("boundaryLayers\n")
        out.write("{\n")
        out.write("    nLayers 1;\n")
        out.write("    thicknessRatio 1.2;\n")
        out.write(f"    maxFirstLayerThickness {first_layer:.12e};\n")
        out.write("    allowDiscontinuity 1;\n")
        out.write("    patchBoundaryLayers\n")
        out.write("    {\n")
        out.write("        inner_nozzle_wall { nLayers 4; thicknessRatio 1.2; allowDiscontinuity 1; }\n")
        out.write("        nozzle_electrode { nLayers 4; thicknessRatio 1.2; allowDiscontinuity 1; }\n")
        out.write("        collector_ground { nLayers 2; thicknessRatio 1.2; allowDiscontinuity 1; }\n")
        out.write("        liquid_inlet { nLayers 0; }\n")
        out.write("        open_atmosphere { nLayers 0; }\n")
        out.write("    }\n")
        out.write("}\n\n")
        out.write("renameBoundary\n")
        out.write("{\n")
        out.write("    defaultName open_atmosphere;\n")
        out.write("    defaultType patch;\n")
        out.write("    newPatchNames\n")
        out.write("    (\n")
        out.write("        liquid_inlet { type patch; }\n")
        out.write("        inner_nozzle_wall { type wall; }\n")
        out.write("        nozzle_electrode { type wall; }\n")
        out.write("        collector_ground { type wall; }\n")
        out.write("        open_atmosphere { type patch; }\n")
        out.write("    );\n")
        out.write("}\n")


def write_snappy_case(case_dir: Path, source_stl: Path, meta: dict, voltage: float = 2180.0) -> None:
    system = case_dir / "system"
    tri = case_dir / "constant" / "triSurface"
    system.mkdir(parents=True, exist_ok=True)
    tri.mkdir(parents=True, exist_ok=True)
    stl_name = source_stl.name
    (tri / stl_name).write_text(source_stl.read_text(encoding="utf-8"), encoding="utf-8")
    write_block_mesh_dict(system / "blockMeshDict", meta)
    write_surface_feature_extract_dict(system / "surfaceFeatureExtractDict", stl_name)
    write_snappy_hex_mesh_dict(system / "snappyHexMeshDict", meta, stl_name)
    write_mesh_quality_dict(system / "meshQualityDict")
    write_control_dict(system / "controlDict")
    write_initial_fields(case_dir, meta, voltage)
    (case_dir / "README.md").write_text(
        "# Electrospray CAD-resolved nozzle mesh case\n\n"
        "This directory is an OpenFOAM snappyHexMesh-ready case generated from the "
        "named electrospray nozzle surfaces.\n\n"
        "Suggested OpenFOAM sequence:\n\n"
        "```bash\n"
        "blockMesh\n"
        "surfaceFeatureExtract\n"
        "snappyHexMesh -overwrite\n"
        "checkMesh\n"
        "```\n\n"
        "Expected named patches after snapping: `liquid_inlet`, `inner_nozzle_wall`, "
        "`nozzle_electrode`, `collector_ground`, and `open_atmosphere`.\n\n"
        "The Taylor cone/meniscus is intentionally not a CAD surface; it should be "
        "initialized as a VoF field.\n",
        encoding="utf-8",
    )


def write_cfmesh_case(case_dir: Path, source_stl: Path, meta: dict, voltage: float = 2180.0) -> None:
    system = case_dir / "system"
    tri = case_dir / "constant" / "triSurface"
    system.mkdir(parents=True, exist_ok=True)
    tri.mkdir(parents=True, exist_ok=True)
    stl_name = source_stl.name
    (tri / stl_name).write_text(source_stl.read_text(encoding="utf-8"), encoding="utf-8")
    write_cfmesh_dict(system / "meshDict", meta, stl_name)
    write_control_dict(system / "controlDict")
    write_initial_fields(case_dir, meta, voltage)
    (case_dir / "README.md").write_text(
        "# Electrospray CAD-resolved nozzle cfMesh case\n\n"
        "This directory is a cfMesh `cartesianMesh`-ready case generated from the "
        "named electrospray nozzle surfaces.\n\n"
        "Suggested cfMesh sequence:\n\n"
        "```bash\n"
        "cartesianMesh\n"
        "checkMesh\n"
        "```\n\n"
        "Expected named patches: `liquid_inlet`, `inner_nozzle_wall`, "
        "`nozzle_electrode`, `collector_ground`, and `open_atmosphere`.\n",
        encoding="utf-8",
    )


def build_geometry(args: argparse.Namespace) -> tuple[dict[str, list[tuple]], dict]:
    inner_r = args.inner_diameter * 0.5
    outer_r = args.outer_diameter * 0.5
    collector_r = args.collector_diameter * 0.5
    farfield_r = max(args.farfield_radius, collector_r, outer_r * 4.0)
    y_back = -args.nozzle_length
    y_tip = 0.0
    y_collector = args.collector_distance

    solids: dict[str, list[tuple]] = {}
    add_disk(solids, "liquid_inlet", inner_r, y_back, args.segments, normal_sign_y=-1.0)
    add_cylinder_side(solids, "inner_nozzle_wall", inner_r, y_back, y_tip, args.segments, outward=False)
    add_cylinder_side(solids, "nozzle_electrode", outer_r, y_back, y_tip, args.segments, outward=True)
    add_annulus(solids, "nozzle_electrode", inner_r, outer_r, y_tip, args.segments, normal_sign_y=-1.0)
    add_disk(solids, "collector_ground", collector_r, y_collector, args.segments, normal_sign_y=1.0)
    add_cylinder_side(solids, "open_atmosphere", farfield_r, y_back, y_collector, args.segments, outward=False)
    add_annulus(solids, "open_atmosphere", outer_r, farfield_r, y_back, args.segments, normal_sign_y=-1.0)

    meta = {
        "inner_diameter": args.inner_diameter,
        "outer_diameter": args.outer_diameter,
        "nozzle_length": args.nozzle_length,
        "collector_distance": args.collector_distance,
        "collector_diameter": args.collector_diameter,
        "farfield_radius": farfield_r,
        "segments": args.segments,
        "units": "m",
        "axis": "y",
        "patches": {name: len(tris) for name, tris in solids.items()},
        "intended_meshing": "OpenFOAM snappyHexMesh/cfMesh triSurface input",
        "limitations": [
            "Triangulated analytic surface package, not a STEP/BREP solid.",
            "Nozzle bore, inner wall, electrode wall, collector, and farfield are resolved as named surfaces.",
            "Meniscus/cone shape is not included; that remains an initial VoF field.",
        ],
    }
    return solids, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/generated_cad/candido_nozzle")
    parser.add_argument("--inner-diameter", type=float, default=160e-6)
    parser.add_argument("--outer-diameter", type=float, default=260e-6)
    parser.add_argument("--nozzle-length", type=float, default=300e-6)
    parser.add_argument("--collector-distance", type=float, default=1.5e-3)
    parser.add_argument("--collector-diameter", type=float, default=5.0e-3)
    parser.add_argument("--farfield-radius", type=float, default=2.5e-3)
    parser.add_argument("--segments", type=int, default=96)
    parser.add_argument("--voltage", type=float, default=2180.0)
    parser.add_argument("--no-snappy-case", action="store_true")
    parser.add_argument("--no-cfmesh-case", action="store_true")
    parser.add_argument("--snappy-case-dir", default="")
    parser.add_argument("--cfmesh-case-dir", default="")
    args = parser.parse_args()

    if args.inner_diameter <= 0.0:
        raise SystemExit("inner diameter must be positive")
    if args.outer_diameter <= args.inner_diameter:
        raise SystemExit("outer diameter must be larger than inner diameter")
    if args.nozzle_length <= 0.0 or args.collector_distance <= 0.0:
        raise SystemExit("lengths must be positive")
    if args.segments < 16:
        raise SystemExit("segments must be >= 16")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    solids, meta = build_geometry(args)

    stem = "electrospray_nozzle_full_cad"
    write_stl(out_dir / f"{stem}.stl", solids)
    write_obj(out_dir / f"{stem}.obj", solids)
    write_scad(out_dir / f"{stem}.scad", meta)
    (out_dir / f"{stem}_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    case_dir = None
    if not args.no_snappy_case:
        case_dir = Path(args.snappy_case_dir) if args.snappy_case_dir else out_dir / "openfoam_snappy_case"
        write_snappy_case(case_dir, out_dir / f"{stem}.stl", meta, args.voltage)
    cfmesh_case_dir = None
    if not args.no_cfmesh_case:
        cfmesh_case_dir = Path(args.cfmesh_case_dir) if args.cfmesh_case_dir else out_dir / "openfoam_cfmesh_case"
        write_cfmesh_case(cfmesh_case_dir, out_dir / f"{stem}.stl", meta, args.voltage)
    print(json.dumps({
        "status": "pass",
        "output_dir": str(out_dir.resolve()),
        "stl": str((out_dir / f"{stem}.stl").resolve()),
        "obj": str((out_dir / f"{stem}.obj").resolve()),
        "scad": str((out_dir / f"{stem}.scad").resolve()),
        "metadata": str((out_dir / f"{stem}_metadata.json").resolve()),
        "snappy_case_dir": str(case_dir.resolve()) if case_dir else "",
        "cfmesh_case_dir": str(cfmesh_case_dir.resolve()) if cfmesh_case_dir else "",
        "patches": meta["patches"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
