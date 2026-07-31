#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE_CASES = [
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


def _read_summary(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _stream_process(cmd: list[str], log_path: Path, *, env: dict, label: str, completed: int, total: int, case_start: float, global_start: float):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        selector = selectors.DefaultSelector()
        if proc.stdout is not None:
            selector.register(proc.stdout, selectors.EVENT_READ)
        last_status = 0.0
        while True:
            now = time.time()
            for key, _ in selector.select(timeout=1.0):
                line = key.fileobj.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
            if now - last_status >= 60.0:
                elapsed = now - global_start
                case_elapsed = now - case_start
                if completed > 0:
                    avg = elapsed / completed
                    eta = max(total - completed, 0) * avg
                    eta_s = f"{eta/60.0:.1f} min"
                else:
                    eta_s = "estimating"
                msg = (
                    f"[progress] case {completed + 1}/{total} {label} "
                    f"case_elapsed={case_elapsed/60.0:.1f} min "
                    f"elapsed={elapsed/60.0:.1f} min ETA={eta_s}\n"
                )
                sys.stdout.write(msg)
                sys.stdout.flush()
                log.write(msg)
                log.flush()
                last_status = now
            if proc.poll() is not None:
                rest = proc.stdout.read() if proc.stdout is not None else ""
                if rest:
                    sys.stdout.write(rest)
                    sys.stdout.flush()
                    log.write(rest)
                    log.flush()
                break
        selector.close()
        return int(proc.returncode or 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="paper_revision_data/_coord_round99_rel_macro_all_1x")
    parser.add_argument("--base-cases", default=",".join(BASE_CASES))
    parser.add_argument("--levels", default="1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_root = ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    base_cases = [x.strip() for x in args.base_cases.split(",") if x.strip()]
    env = os.environ.copy()
    env.update(
        {
            "SAFE_NN_UNIFIED_MACRO_L2_CONVERGENCE": "0",
            "SAFE_NN_CAVITY_PLATEAU_TAIL": "0",
            "SAFE_NN_RELATIVE_MACRO_L2_CONVERGENCE": "1",
            "SAFE_NN_RELATIVE_MACRO_PROGRESS": "1",
            "MPLBACKEND": "Agg",
        }
    )
    global_start = time.time()
    completed_rows = []
    for idx, base in enumerate(base_cases):
        case_out = out_root / base
        if case_out.exists() and args.force:
            import shutil

            shutil.rmtree(case_out)
        case_start = time.time()
        print(f"[case-start] {idx + 1}/{len(base_cases)} {base}", flush=True)
        cmd = [
            sys.executable,
            "paper_60case_benchmark_no_force_scaling.py",
            "--base-cases",
            base,
            "--levels",
            args.levels,
            "--methods",
            "all",
            "--no-cache",
            "--no-resume",
            "--no-vtk",
            "--out-dir",
            str(case_out),
        ]
        rc = _stream_process(
            cmd,
            out_root / f"{base}.log",
            env=env,
            label=base,
            completed=idx,
            total=len(base_cases),
            case_start=case_start,
            global_start=global_start,
        )
        if rc != 0:
            raise SystemExit(f"benchmark failed for {base} rc={rc}")
        rows = _read_summary(case_out / "summary.csv")
        case_ids = sorted({row["case_id"] for row in rows})
        for case_id in case_ids:
            export_cmd = [
                sys.executable,
                "export_paper_scaling_case_to_papers_data.py",
                "--case-id",
                case_id,
                "--source-dir",
                str(case_out),
                "--force",
            ]
            rc = _stream_process(
                export_cmd,
                out_root / f"{base}__export.log",
                env=env,
                label=f"export {case_id}",
                completed=idx,
                total=len(base_cases),
                case_start=case_start,
                global_start=global_start,
            )
            if rc != 0:
                raise SystemExit(f"export failed for {case_id} rc={rc}")
        completed_rows.extend(rows)
        with (out_root / "summary_completed_so_far.csv").open("w", encoding="utf-8", newline="") as fh:
            if completed_rows:
                wr = csv.DictWriter(fh, fieldnames=list(completed_rows[0].keys()))
                wr.writeheader()
                wr.writerows(completed_rows)
        elapsed = time.time() - global_start
        avg = elapsed / float(idx + 1)
        eta = (len(base_cases) - idx - 1) * avg
        print(
            f"[case-done] {idx + 1}/{len(base_cases)} {base} "
            f"case_elapsed={(time.time() - case_start)/60.0:.1f} min "
            f"elapsed={elapsed/60.0:.1f} min ETA={eta/60.0:.1f} min",
            flush=True,
        )
    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        if completed_rows:
            wr = csv.DictWriter(fh, fieldnames=list(completed_rows[0].keys()))
            wr.writeheader()
            wr.writerows(completed_rows)


if __name__ == "__main__":
    main()
