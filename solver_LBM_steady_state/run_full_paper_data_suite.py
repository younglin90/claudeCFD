#!/usr/bin/env python3
"""Run and export the paper-grade LBM benchmark suite.

The suite writes calculation outputs to paper_revision_data first, then exports
per-case paper artifacts to papers_data/<case_id>.  It uses the current solver
hash in the benchmark cache keys and fails export if a hash-matched cache is
missing or if residual histories are not strict-monotone in raw wall time.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PAPER_BASE_CASES = [
    "channel_poiseuille_rect",
    "couette_n32",
    "cavity_re100_n33",
    "cavity_re400_n49",
    "cavity_re1000_n129",
    "multi_cylinder_n32",
    "backward_step_n64",
    "cylinder_wake_n64",
    "t_junction_rect",
]


def _env(threads: int, cpu_list: str | None = None):
    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = str(threads)
    env["MPLBACKEND"] = "Agg"
    if cpu_list:
        env["GOMP_CPU_AFFINITY"] = cpu_list.replace(",", " ")
        env["OMP_PLACES"] = "cores"
        env["OMP_PROC_BIND"] = "close"
    return env


def _run(cmd: list[str], env: dict):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _run_capture(cmd: list[str], env: dict, cpu_list: str | None = None):
    full_cmd = cmd
    if cpu_list:
        full_cmd = ["taskset", "-c", cpu_list] + cmd
    print("[run]", " ".join(full_cmd), flush=True)
    proc = subprocess.run(full_cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    print(proc.stdout[-3000:], flush=True)


def _case_ids_from_summary(base_cases: set[str], levels: set[int]):
    path = Path("paper_revision_data") / "no_force_scaling_benchmark" / "summary.csv"
    if not path.exists():
        raise RuntimeError(f"missing {path}")
    case_ids = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("base_case_id") in base_cases and int(float(row.get("scaling_level", 0))) in levels:
                cid = row["case_id"]
                if cid not in case_ids:
                    case_ids.append(cid)
    return case_ids


def _job_out_dir(base_case: str, level: int, method: str | None = None) -> Path:
    stem = f"{base_case}__{level}x"
    if method:
        stem = f"{stem}__{method}"
    return Path("paper_revision_data") / "no_force_scaling_benchmark_jobs" / stem


def _main_out_dir() -> Path:
    return Path("paper_revision_data") / "no_force_scaling_benchmark"


def _parse_cpu_list(cpu_list: str) -> list[int]:
    cpus: list[int] = []
    for part in cpu_list.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            cpus.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            cpus.append(int(part))
    return sorted(dict.fromkeys(cpus))


def _format_cpu_list(cpus: list[int]) -> str:
    if not cpus:
        return ""
    chunks = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        chunks.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    chunks.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(chunks)


def _split_cpu_sets(cpu_list: str, workers: int) -> list[str]:
    cpus = _parse_cpu_list(cpu_list)
    if not cpus:
        return [None] * workers
    workers = max(1, min(workers, len(cpus)))
    base = len(cpus) // workers
    extra = len(cpus) % workers
    sets = []
    pos = 0
    for idx in range(workers):
        size = base + (1 if idx < extra else 0)
        sets.append(cpus[pos : pos + size])
        pos += size
    return [_format_cpu_list(s) for s in sets]


def _run_one_job(args_tuple):
    base_case, level, methods, no_cache, no_resume, threads, cpu_list, method_label, suppress_reference_row = args_tuple
    out_dir = _job_out_dir(base_case, level, method_label)
    cache_dir = _main_out_dir() / "npz_cache"
    cmd = [
        sys.executable,
        "paper_60case_benchmark_no_force_scaling.py",
        "--levels",
        str(level),
        "--base-cases",
        base_case,
        "--methods",
        methods,
        "--run-mode",
        "strict",
        "--out-dir",
        str(out_dir),
        "--cache-dir",
        str(cache_dir),
    ]
    if method_label is not None:
        cmd.append("--exact-methods")
    if suppress_reference_row:
        cmd.append("--suppress-reference-row")
    if no_cache:
        cmd.append("--no-cache")
    if no_resume:
        cmd.append("--no-resume")
    _run_capture(cmd, _env(threads, cpu_list), cpu_list=cpu_list)
    return str(out_dir)


def _copy_tree_contents(src: Path, dst: Path):
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _merge_job_outputs(job_dirs: list[str]):
    out = _main_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "histories").mkdir(parents=True, exist_ok=True)
    (out / "vtk").mkdir(parents=True, exist_ok=True)
    (out / "npz_cache").mkdir(parents=True, exist_ok=True)

    rows = []
    metrics = {"job_dirs": job_dirs}
    for jd_s in job_dirs:
        jd = Path(jd_s)
        summary = jd / "summary.csv"
        if summary.exists():
            with summary.open("r", encoding="utf-8", newline="") as fh:
                rows.extend(list(csv.DictReader(fh)))
        _copy_tree_contents(jd / "histories", out / "histories")
        _copy_tree_contents(jd / "vtk", out / "vtk")
        _copy_tree_contents(jd / "npz_cache", out / "npz_cache")

    if not rows:
        raise RuntimeError("no rows produced by worker jobs")
    fields = list(rows[0].keys())
    rows = sorted(rows, key=lambda r: (r["base_case_id"], int(float(r["scaling_level"])), r["case_id"], r["method"]))
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    (out / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    metrics.update(
        {
            "case_count": len(set(r["case_id"] for r in rows)),
            "row_count": len(rows),
            "methods": sorted(set(r["method"] for r in rows)),
            "levels": sorted(set(int(float(r["scaling_level"])) for r in rows)),
            "base_cases": sorted(set(r["base_case_id"] for r in rows)),
        }
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[merged] {len(rows)} rows into {out}", flush=True)
    return rows


def _write_manifest(case_ids: list[str], args):
    out = Path("papers_data") / "full_paper_suite_manifest.json"
    manifest = {
        "base_cases": args.base_cases,
        "levels": args.levels,
        "case_ids": case_ids,
        "threads": args.threads,
        "source": "paper_revision_data/no_force_scaling_benchmark",
        "export_policy": {
            "cache": "current solver hash + recorded cache label required",
            "wall_seconds": "raw solver wall time only, strict monotone",
            "figures": "png only",
            "fields": "csv and vtk",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[saved] {out}", flush=True)


def _run_d12_cylinder(levels: list[int], threads: int):
    env = _env(threads)
    for level in levels:
        nx = 312 * level
        ny = 168 * level
        d = 12 * level
        tol = 1.0e-7 / float(level)
        max_steps = 160000 * level * level
        cid = f"cylinder_wake_Re40_D{d}_Nx{nx}_Ny{ny}__{level}x"
        _run(
            [
                sys.executable,
                "run_cylinder_wake_re40_d12_rect_publish.py",
                "--Nx",
                str(nx),
                "--Ny",
                str(ny),
                "--D",
                str(float(d)),
                "--tol",
                f"{tol:.12g}",
                "--max-steps",
                str(max_steps),
                "--case-id",
                cid,
                "--no-clean",
            ],
            env,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="1,2,3")
    parser.add_argument("--base-cases", default=",".join(PAPER_BASE_CASES))
    parser.add_argument("--methods", default="picard_lbm,anderson_lbm,preconditioned_lbm,inexact_newton_lbe,dual_time_mg_lbm,proposed")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument("--cpu-list", default=None, help="Logical CPU list for this suite, e.g. 0-31 or 16-47.")
    parser.add_argument("--split-methods", action="store_true", help="Run each method as an independent worker job.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--include-d12-cylinder", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    base_cases = [x.strip() for x in args.base_cases.split(",") if x.strip()]
    workers = max(1, int(args.workers))
    threads_per_worker = int(args.threads_per_worker or max(1, args.threads // workers))
    cpu_sets = _split_cpu_sets(args.cpu_list, workers) if args.cpu_list else [None] * workers
    if args.cpu_list:
        print(f"[affinity] suite cpus={args.cpu_list} worker_cpu_sets={cpu_sets}", flush=True)
    env = _env(args.threads, args.cpu_list)

    if not args.export_only:
        jobs = []
        picard_jobs = []
        other_jobs = []
        idx = 0
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        for base_case in base_cases:
            for level in levels:
                if args.split_methods:
                    for method in methods:
                        job = (
                            base_case,
                            level,
                            method,
                            args.no_cache,
                            args.no_resume,
                            threads_per_worker,
                            cpu_sets[idx % len(cpu_sets)],
                            method,
                            method != "picard_lbm",
                        )
                        if method == "picard_lbm":
                            picard_jobs.append(job)
                        else:
                            other_jobs.append(job)
                        idx += 1
                else:
                    jobs.append((base_case, level, args.methods, args.no_cache, args.no_resume, threads_per_worker, cpu_sets[idx % len(cpu_sets)], None, False))
                    idx += 1
        if args.split_methods:
            jobs = picard_jobs + other_jobs
        if workers == 1:
            job_dirs = [_run_one_job(job) for job in jobs]
        else:
            if args.split_methods:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                    picard_dirs = list(pool.map(_run_one_job, picard_jobs))
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                    other_dirs = list(pool.map(_run_one_job, other_jobs))
                job_dirs = picard_dirs + other_dirs
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                    job_dirs = list(pool.map(_run_one_job, jobs))
        _merge_job_outputs(job_dirs)

    case_ids = _case_ids_from_summary(set(base_cases), set(levels))
    for case_id in case_ids:
        _run([sys.executable, "export_paper_scaling_case_to_papers_data.py", "--case-id", case_id, "--force"], env)

    if args.include_d12_cylinder and not args.export_only:
        _run_d12_cylinder(levels, args.threads)

    _write_manifest(case_ids, args)


if __name__ == "__main__":
    main()
