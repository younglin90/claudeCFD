#!/usr/bin/env python3
"""Validate the generated electrospray nozzle STL before meshing."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_SOLIDS = {
    "liquid_inlet",
    "inner_nozzle_wall",
    "nozzle_electrode",
    "collector_ground",
    "open_atmosphere",
}


def parse_ascii_stl(path: Path) -> dict[str, list[tuple[tuple[float, float, float], ...]]]:
    solids: dict[str, list[tuple[tuple[float, float, float], ...]]] = defaultdict(list)
    current = ""
    vertices: list[tuple[float, float, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("solid "):
            current = line.split(maxsplit=1)[1].strip()
            solids.setdefault(current, [])
        elif line.startswith("vertex "):
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"invalid STL vertex line: {raw}")
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line == "endfacet":
            if len(vertices) < 3:
                raise ValueError("STL facet ended before three vertices")
            if not current:
                raise ValueError("STL facet encountered before solid name")
            solids[current].append((vertices[-3], vertices[-2], vertices[-1]))
    return dict(solids)


def sub(a: tuple[float, float, float],
        b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a: tuple[float, float, float],
          b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def quantize(p: tuple[float, float, float], scale: float) -> tuple[int, int, int]:
    return (round(p[0] / scale), round(p[1] / scale), round(p[2] / scale))


def validate_surface(path: Path, tolerance: float | None = None) -> dict:
    solids = parse_ascii_stl(path)
    missing = sorted(REQUIRED_SOLIDS - set(solids))
    all_vertices = [v for triangles in solids.values() for tri in triangles for v in tri]
    if not all_vertices:
        return {"status": "fail", "error": "STL contains no vertices"}

    mins = [min(v[i] for v in all_vertices) for i in range(3)]
    maxs = [max(v[i] for v in all_vertices) for i in range(3)]
    diag = math.sqrt(sum((maxs[i] - mins[i]) ** 2 for i in range(3)))
    q = tolerance if tolerance is not None else max(diag * 1e-9, 1e-15)

    edge_counter: Counter[tuple[tuple[int, int, int], tuple[int, int, int]]] = Counter()
    oriented_counter: Counter[tuple[tuple[int, int, int], tuple[int, int, int]]] = Counter()
    degenerate = 0
    nonfinite = 0
    min_area = math.inf
    max_area = 0.0

    for triangles in solids.values():
        for tri in triangles:
            if any(not math.isfinite(x) for v in tri for x in v):
                nonfinite += 1
                continue
            area = 0.5 * norm(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0])))
            min_area = min(min_area, area)
            max_area = max(max_area, area)
            if area <= max(diag * diag * 1e-24, 1e-30):
                degenerate += 1
            qv = [quantize(v, q) for v in tri]
            for a, b in [(qv[0], qv[1]), (qv[1], qv[2]), (qv[2], qv[0])]:
                edge_counter[tuple(sorted((a, b)))] += 1
                oriented_counter[(a, b)] += 1

    boundary_edges = [edge for edge, count in edge_counter.items() if count == 1]
    nonmanifold_edges = [edge for edge, count in edge_counter.items() if count > 2]
    duplicate_oriented_edges = []
    for a, b in edge_counter:
        same = oriented_counter[(a, b)]
        reverse = oriented_counter[(b, a)]
        if same != reverse:
            duplicate_oriented_edges.append((a, b))

    triangle_counts = {name: len(tris) for name, tris in solids.items()}
    ok = (
        not missing
        and nonfinite == 0
        and degenerate == 0
        and len(boundary_edges) == 0
        and len(nonmanifold_edges) == 0
        and len(duplicate_oriented_edges) == 0
    )
    return {
        "status": "pass" if ok else "fail",
        "stl": str(path),
        "required_solids_present": not missing,
        "missing_solids": missing,
        "solid_count": len(solids),
        "triangle_counts": triangle_counts,
        "triangle_count_total": sum(triangle_counts.values()),
        "nonfinite_triangles": nonfinite,
        "degenerate_triangles": degenerate,
        "boundary_edge_count": len(boundary_edges),
        "nonmanifold_edge_count": len(nonmanifold_edges),
        "oriented_edge_mismatch_count": len(duplicate_oriented_edges),
        "min_triangle_area": 0.0 if min_area == math.inf else min_area,
        "max_triangle_area": max_area,
        "quantization_tolerance": q,
        "bounds": {"min": mins, "max": maxs},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl")
    parser.add_argument("--tolerance", type=float, default=None)
    args = parser.parse_args()
    payload = validate_surface(Path(args.stl), args.tolerance)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
