"""Compare every Re=1000 cavity method against Ghia centerline data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ghia_validation import get_ghia_data


BENCH_VTK = Path("paper_revision_data/bench60/vtk")
PLBE_VTK = Path("paper_revision_data/plbe_re1000_paper_mach/final_gamma00625_field.vtk")
OUT = Path("paper_revision_data/plbe_re1000_paper_mach/method_ghia_comparison.json")
OUT_MD = Path("paper_revision_data/plbe_re1000_paper_mach/method_ghia_comparison.md")


def parse_vtk(path: Path):
    text = path.read_text().splitlines()
    dims_line = next(l for l in text if l.startswith("DIMENSIONS"))
    nx, ny, _ = (int(s) for s in dims_line.split()[1:4])
    start = text.index("VECTORS velocity float") + 1
    vals = []
    for line in text[start : start + nx * ny]:
        parts = line.split()
        vals.append((float(parts[0]), float(parts[1])))
    arr = np.array(vals).reshape(ny, nx, 2)
    ux = arr[..., 0]
    uy = arr[..., 1]
    return ux, uy


def ghia_err(ux, uy, U_wall):
    n = ux.shape[0]
    y_g, u_g, x_g, v_g = get_ghia_data(1000)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    u_center = ux[:, n // 2] / U_wall
    v_center = uy[n // 2, :] / U_wall
    u_interp = np.interp(y_g, y, u_center)
    v_interp = np.interp(x_g, x, v_center)
    return {
        "u_rms": float(np.sqrt(np.mean((u_interp - u_g) ** 2))),
        "v_rms": float(np.sqrt(np.mean((v_interp - v_g) ** 2))),
        "u_linf": float(np.max(np.abs(u_interp - u_g))),
        "v_linf": float(np.max(np.abs(v_interp - v_g))),
        "centerline_max": float(max(np.max(np.abs(u_interp - u_g)), np.max(np.abs(v_interp - v_g)))),
        "u_max_abs": float(np.max(np.abs(u_center))),
    }


def main():
    methods = [
        ("picard_lbm", BENCH_VTK / "cavity_re1000_n129__picard_lbm.vtk", 0.1, "U=0.1 bounce-back, baseline LBE Picard"),
        ("anderson_lbm", BENCH_VTK / "cavity_re1000_n129__anderson_lbm.vtk", 0.1, "U=0.1 bounce-back, Anderson acceleration"),
        ("preconditioned_lbm", BENCH_VTK / "cavity_re1000_n129__preconditioned_lbm.vtk", 0.1, "U=0.1 bounce-back, proxy PLBE (NOT paper-faithful)"),
        ("inexact_newton_lbe", BENCH_VTK / "cavity_re1000_n129__inexact_newton_lbe.vtk", 0.1, "U=0.1 bounce-back, inexact Newton"),
        ("dual_time_mg_lbm", BENCH_VTK / "cavity_re1000_n129__dual_time_mg_lbm.vtk", 0.1, "U=0.1 bounce-back, dual-time MG"),
        ("proposed_safenn_scmk", BENCH_VTK / "cavity_re1000_n129__proposed.vtk", 0.1, "U=0.1 bounce-back, proposed SafeNN/SCMK"),
        ("plbe_paper_faithful", PLBE_VTK, 0.008371578903249575, "Guo-Zhao-Shi PLBE, M*=0.058, gamma=0.0625, NEQ-extrap walls"),
    ]
    rows = []
    for name, path, U_wall, note in methods:
        if not path.exists():
            rows.append({"method": name, "path": str(path), "U_wall": U_wall, "note": note, "available": False})
            continue
        ux, uy = parse_vtk(path)
        rows.append({
            "method": name,
            "path": str(path),
            "U_wall": U_wall,
            "note": note,
            "available": True,
            **ghia_err(ux, uy, U_wall),
        })
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = ["# Re=1000 cavity: method-vs-Ghia comparison",
             "",
             "All errors normalized by lid speed U_wall. Ghia 1982 table values are reference.",
             "",
             "| Method | Setup | u_rms | v_rms | u_Linf | v_Linf | Paper-like? |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if not r["available"]:
            lines.append(f"| {r['method']} | {r['note']} | n/a | n/a | n/a | n/a | missing |")
            continue
        ok = (r["centerline_max"] < 0.05)
        verdict = "YES" if ok else ("MARGINAL" if r["centerline_max"] < 0.1 else "NO")
        lines.append(
            f"| {r['method']} | {r['note']} | {r['u_rms']:.3e} | {r['v_rms']:.3e} | "
            f"{r['u_linf']:.3e} | {r['v_linf']:.3e} | {verdict} |"
        )
    lines += ["",
              "Verdict thresholds: YES if max centerline deviation < 0.05, MARGINAL < 0.10, else NO.",
              "",
              "Note: methods marked 'U=0.1 bounce-back' solve the legacy benchmark, not the",
              "paper's low-Mach PLBE cavity. Only `plbe_paper_faithful` reproduces the",
              "Guo-Zhao-Shi setup directly. The legacy methods are still informative because",
              "Ghia's reference is grid- and Mach-converged enough that all stable steady",
              "solutions should approach the same normalized centerline profile."]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
