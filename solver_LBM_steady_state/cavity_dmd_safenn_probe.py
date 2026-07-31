"""Probe guarded Koopman/DMD proposals on lid-driven cavity Re=1000 N=129."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

import paper_60case_benchmark as bench
from solver_unified_safe_nn import _residual_norm


OUT = Path("paper_revision_data") / "bench2_focus" / "cavity_dmd_safenn_probe"


def project_finite(f):
    return np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0)


def picard(case, f, steps):
    f = project_finite(f)
    for _ in range(steps):
        f = case.lbe_step(f)
        if not np.all(np.isfinite(f)):
            return project_finite(f)
    return project_finite(f)


def velocity_rel_l2(case_ref, f_ref, case, f):
    fluid = np.ones((case_ref.N, case_ref.N), dtype=bool)
    return bench.velocity_error(case_ref, f_ref, case, f, fluid_mask=fluid)["rel_l2"]


def collect_snaps(case, sample_steps, stride):
    f = project_finite(case.initial_field())
    snaps = []
    for step in range(1, sample_steps + 1):
        f = case.lbe_step(f)
        if step % stride == 0:
            snaps.append(project_finite(f.copy()))
    return snaps


def dmd_predict(snaps, future_steps, stride, rank, eig_clip=None):
    x = np.stack([s.ravel() for s in snaps[:-1]], axis=1)
    y = np.stack([s.ravel() for s in snaps[1:]], axis=1)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    r = min(rank, len(s))
    u = u[:, :r]
    s = s[:r]
    v = vt[:r, :].T
    a = u.T @ y @ v @ np.diag(1.0 / s)
    if eig_clip is not None:
        vals, vecs = np.linalg.eig(a)
        mag = np.abs(vals)
        vals = np.where(mag > eig_clip, vals * (eig_clip / mag), vals)
        a = np.real_if_close(vecs @ np.diag(vals) @ np.linalg.inv(vecs)).real
    z = u.T @ snaps[-1].ravel()
    for _ in range(max(0, int(round(future_steps / stride)))):
        z = a @ z
        z = np.nan_to_num(z, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    return project_finite((u @ z).reshape(snaps[-1].shape))


def macro_aitken(snaps, damping=1.0):
    f0, f1, f2 = snaps[-3], snaps[-2], snaps[-1]
    d1 = f1 - f0
    d2 = f2 - 2.0 * f1 + f0
    eps = 1.0e-14 * np.maximum(1.0, np.abs(f2))
    denom = np.where(np.abs(d2) > eps, d2, np.sign(d2 + eps) * eps)
    pred = f0 - d1 * d1 / denom
    return project_finite((1.0 - damping) * f2 + damping * pred)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    case_id = "cavity_re1000_n129"
    label, tol, factory = bench.case_factory(case_id)
    case_ref = factory()
    f_ref, hist_ref, _ = bench.run_method("picard_lbm", case_ref, tol, bench.max_steps_for(case_id), verbose=False)
    baseline_lbe = int(hist_ref[-1][2])
    rows = []
    t0 = time.perf_counter()

    configs = []
    for sample in [6000, 7000, 8000, 8500]:
        for stride in [125, 250, 500]:
            for rank in [4, 8, 12, 16, 24, 32]:
                for future in [2000, 4000, 6000, 8000, 10000]:
                    for polish in [0, 250, 500, 750]:
                        configs.append(("dmd", sample, stride, rank, future, polish, None))
                        configs.append(("dmd_clip", sample, stride, rank, future, polish, 1.0))
    for sample in [7000, 8000, 8500]:
        for stride in [250, 500]:
            for damping in [0.05, 0.10, 0.20, 0.50]:
                for polish in [0, 250, 500, 750]:
                    configs.append(("aitken", sample, stride, damping, 0, polish, None))

    cache = {}
    for cfg in configs:
        kind = cfg[0]
        sample = cfg[1]
        stride = cfg[2]
        key = (sample, stride)
        if key not in cache:
            case = factory()
            cache[key] = collect_snaps(case, sample, stride)
        snaps = cache[key]
        case = factory()
        failed = False
        if kind in {"dmd", "dmd_clip"}:
            _, _, _, rank, future, polish_steps, clip = cfg
            try:
                f = dmd_predict(snaps, future, stride, int(rank), eig_clip=clip)
                name = f"{kind}_sample{sample}_stride{stride}_r{rank}_future{future}_polish{polish_steps}"
            except Exception:
                failed = True
                f = case.initial_field()
                polish_steps = 0
                name = f"{kind}_sample{sample}_stride{stride}_failed"
        else:
            _, _, _, damping, _, polish_steps, _ = cfg
            f = macro_aitken(snaps, damping=float(damping))
            name = f"aitken_sample{sample}_stride{stride}_damp{damping}_polish{polish_steps}"
        if polish_steps:
            f = picard(case, f, int(polish_steps))
        _, res = _residual_norm(case, f)
        rel = velocity_rel_l2(case_ref, f_ref, case, f)
        lbe = int(sample + len(snaps) + polish_steps + 1)
        speed = baseline_lbe / max(lbe, 1)
        rows.append(
            {
                "name": name,
                "family": kind,
                "sample_steps": sample,
                "stride": stride,
                "lbe_calls": lbe,
                "speedup_vs_picard": speed,
                "final_residual": float(res),
                "rel_l2_vs_picard": float(rel),
                "converged": bool(np.isfinite(res) and res < 5.0 * tol),
                "accurate": bool(np.isfinite(rel) and rel <= 0.05),
                "strict_2x": bool(np.isfinite(res) and res < 5.0 * tol and np.isfinite(rel) and rel <= 0.05 and speed >= 2.0),
                "failed": failed,
            }
        )

    rows.sort(key=lambda r: (not r["strict_2x"], r["rel_l2_vs_picard"], r["lbe_calls"]))
    fields = list(rows[0].keys())
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    metrics = {
        "case": case_id,
        "baseline_lbe": baseline_lbe,
        "two_x_budget_lbe": baseline_lbe / 2.0,
        "variant_count": len(rows),
        "strict_2x_count": sum(int(r["strict_2x"]) for r in rows),
        "accurate_count": sum(int(r["accurate"]) for r in rows),
        "best_strict_2x": next((r for r in rows if r["strict_2x"]), None),
        "best_by_accuracy": rows[0],
        "elapsed_wall_seconds": time.perf_counter() - t0,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (OUT / "report.md").open("w", encoding="utf-8") as fh:
        fh.write("# Cavity Re=1000 Guarded DMD/SafeNN Probe\n\n")
        fh.write(f"Picard reference LBE: {baseline_lbe}\n\n")
        fh.write(f"2x budget LBE: {baseline_lbe / 2.0:.1f}\n\n")
        fh.write(f"Variants tested: {len(rows)}\n\n")
        fh.write(f"Strict 2x + relL2<=0.05 variants: {metrics['strict_2x_count']}\n\n")
        fh.write("## Top 20 By Accuracy\n\n")
        fh.write("| name | family | LBE | speedup | residual | relL2 | strict |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | ---: | --- |\n")
        for r in rows[:20]:
            fh.write(
                f"| {r['name']} | {r['family']} | {r['lbe_calls']} | "
                f"{r['speedup_vs_picard']:.3f} | {r['final_residual']:.3e} | "
                f"{r['rel_l2_vs_picard']:.3e} | {r['strict_2x']} |\n"
            )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
