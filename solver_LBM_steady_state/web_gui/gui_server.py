#!/usr/bin/env python3
"""Local web GUI for the steady-state LBM benchmark drivers.

The server is intentionally dependency-free.  It serves the static UI and
starts existing project scripts in background subprocesses, keeping the core
solver files untouched.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import signal
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
RUN_ROOT = ROOT / "web_gui_runs"

CASES: dict[str, dict[str, Any]] = {
    "channel_poiseuille_rect": {
        "label": "Plane Poiseuille inlet/outlet rectangular channel",
        "family": "channel",
        "baseN": 32,
        "mesh": {"Ny": "32*level", "Nx": "192*level"},
        "defaultTol": 1.0e-7,
    },
    "couette_n32": {
        "label": "Couette flow",
        "family": "couette",
        "baseN": 32,
        "mesh": {"N": "32*level"},
        "defaultTol": 1.0e-7,
    },
    "cavity_re100_n33": {
        "label": "Lid-driven cavity Re=100",
        "family": "cavity",
        "baseN": 33,
        "mesh": {"N": "1 + level*(33-1)"},
        "defaultTol": {1: 1.0e-8, 2: 5.0e-9, 3: 3.333e-9},
    },
    "cavity_re400_n49": {
        "label": "Lid-driven cavity Re=400",
        "family": "cavity",
        "baseN": 49,
        "mesh": {"N": "1 + level*(49-1)"},
        "defaultTol": {1: 1.0e-8, 2: 5.0e-9, 3: 3.333e-9},
    },
    "cavity_re1000_n129": {
        "label": "Lid-driven cavity Re=1000",
        "family": "cavity",
        "baseN": 129,
        "mesh": {"N": "1 + level*(129-1)"},
        "defaultTol": {1: 1.0e-8, 2: 5.0e-9, 3: 3.333e-9},
    },
    "multi_cylinder_n32": {
        "label": "Multi-cylinder masked flow",
        "family": "masked",
        "baseN": 32,
        "mesh": {"N": "32*level"},
        "defaultTol": 1.0e-7,
    },
    "backward_step_n64": {
        "label": "Backward-facing step",
        "family": "masked",
        "baseN": 64,
        "mesh": {"N": "64*level"},
        "defaultTol": 1.0e-7,
    },
    "cylinder_wake_n64": {
        "label": "Cylinder wake Re=40 analogue",
        "family": "masked",
        "baseN": 64,
        "mesh": {"N": "64*level"},
        "defaultTol": 1.0e-7,
    },
    "t_junction_rect": {
        "label": "Strict inlet/outlet T-junction",
        "family": "t-junction",
        "baseN": 128,
        "mesh": {"Ny": "64*level", "Nx": "96*level", "W": "16*level"},
        "defaultTol": 1.0e-7,
    },
}

METHODS = [
    {"id": "proposed", "label": "Proposed AP-Schur-only"},
    {"id": "picard_lbm", "label": "Picard LBM"},
    {"id": "anderson_lbm", "label": "Anderson LBM"},
    {"id": "preconditioned_lbm", "label": "Preconditioned LBM"},
    {"id": "inexact_newton_lbe", "label": "Inexact Newton LBE"},
    {"id": "dual_time_mg_lbm", "label": "Dual-time MG LBM"},
]

ENV_PARAMS: list[dict[str, Any]] = [
    {"name": "SAFE_NN_UNIFORM_PROPOSED", "type": "bool", "default": True, "group": "method"},
    {"name": "SAFE_NN_DISABLE_RRE", "type": "bool", "default": True, "group": "method"},
    {"name": "SAFE_NN_ENABLE_AP_SCHUR", "type": "bool", "default": True, "group": "ap-schur"},
    {"name": "SAFE_NN_AP_SCHUR_RECTANGULAR", "type": "bool", "default": True, "group": "ap-schur"},
    {"name": "SAFE_NN_AP_SCHUR_MASK_AWARE", "type": "bool", "default": True, "group": "ap-schur"},
    {"name": "SAFE_NN_AP_SCHUR_LOCAL_DEFLATION", "type": "bool", "default": False, "group": "ap-schur"},
    {"name": "SAFE_NN_AP_SCHUR_MAX_ATTEMPTS", "type": "int", "default": 8, "min": 0, "max": 128, "group": "ap-schur"},
    {"name": "SAFE_NN_AP_SCHUR_KRYLOV_MAX", "type": "int", "default": 8, "min": 2, "max": 64, "group": "ap-schur"},
    {"name": "SAFE_NN_AP_SCHUR_KINETIC_SUBSTEPS", "type": "int", "default": 4, "min": 0, "max": 64, "group": "ap-schur"},
    {"name": "SAFE_NN_AP_SCHUR_RTOL", "type": "float", "default": 2.0e-3, "min": 1.0e-8, "max": 1.0, "group": "ap-schur"},
    {"name": "SAFE_NN_UNIFORM_ROUNDS", "type": "int", "default": 160, "min": 1, "max": 10000, "group": "outer-loop"},
    {"name": "SAFE_NN_UNIFORM_STALE_LIMIT", "type": "int", "default": 40, "min": 1, "max": 1000, "group": "outer-loop"},
    {"name": "SAFE_NN_UNIFORM_RRE_DEPTH", "type": "int", "default": 5, "min": 2, "max": 16, "group": "outer-loop"},
    {"name": "SAFE_NN_BURN_SCALE", "type": "float", "default": 1.0, "min": 0.05, "max": 16.0, "group": "outer-loop"},
    {"name": "SAFE_NN_PICARD_SCALE", "type": "float", "default": 1.0, "min": 0.05, "max": 16.0, "group": "outer-loop"},
    {"name": "SAFE_NN_MAX_OUTER_SCALE", "type": "float", "default": 1.0, "min": 0.05, "max": 16.0, "group": "outer-loop"},
    {"name": "SAFE_NN_TAIL_STEPS", "type": "int", "default": -1, "min": -1, "max": 10_000_000, "group": "tail"},
    {"name": "SAFE_NN_FINAL_TAIL_STEPS", "type": "int", "default": -1, "min": -1, "max": 10_000_000, "group": "tail"},
    {"name": "SAFE_NN_ENABLE_TAIL", "type": "bool", "default": True, "group": "tail"},
    {"name": "SAFE_NN_BENCHMARK_MAX_STEPS_OVERRIDE", "type": "int", "default": "", "min": 1, "max": 50_000_000, "group": "benchmark"},
    {"name": "NUMBA_NUM_THREADS", "type": "int", "default": 24, "min": 1, "max": 256, "group": "runtime"},
    {"name": "OMP_NUM_THREADS", "type": "int", "default": 24, "min": 1, "max": 256, "group": "runtime"},
    {"name": "OPENBLAS_NUM_THREADS", "type": "int", "default": 1, "min": 1, "max": 256, "group": "runtime"},
    {"name": "MKL_NUM_THREADS", "type": "int", "default": 1, "min": 1, "max": 256, "group": "runtime"},
]


def _discover_env_params() -> list[dict[str, Any]]:
    """Find SAFE_NN_* controls that are present in project Python files."""
    seen = {p["name"] for p in ENV_PARAMS}
    found: list[dict[str, Any]] = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        parts = set(rel.parts)
        if {"__pycache__", ".git", "web_gui"} & parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in sorted(set(re.findall(r"\bSAFE_NN_[A-Z0-9_]+\b", text))):
            if name in seen:
                continue
            seen.add(name)
            found.append({"name": name, "type": "str", "default": "", "group": "discovered"})
    return found


def _env_params() -> list[dict[str, Any]]:
    return [*ENV_PARAMS, *_discover_env_params()]


def _discover_scripts() -> list[dict[str, Any]]:
    """Expose runnable project scripts and their visible argparse options."""
    scripts: list[dict[str, Any]] = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        parts = set(rel.parts)
        if {"__pycache__", ".git", "web_gui"} & parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "add_argument" not in text and "if __name__" not in text:
            continue
        options = sorted(set(re.findall(r"add_argument\(\s*[\"'](--[A-Za-z0-9][A-Za-z0-9_-]*)[\"']", text)))
        scripts.append({"path": str(rel), "options": options})
    return scripts


@dataclass
class Job:
    id: str
    command: list[str]
    env_overrides: dict[str, str]
    output_dir: Path
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    returncode: int | None = None
    log: list[str] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    finished_at: float | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "returncode": self.returncode,
            "createdAt": self.created_at,
            "finishedAt": self.finished_at,
            "command": self.command,
            "env": self.env_overrides,
            "outputDir": str(self.output_dir.relative_to(ROOT)) if self.output_dir.is_relative_to(ROOT) else str(self.output_dir),
            "log": self.log[-1200:],
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _get_job(job_id: str) -> Job | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def _job_files(job: Job) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not job.output_dir.exists():
        return files
    root = job.output_dir.resolve()
    for path in sorted(job.output_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve()
            if root not in resolved.parents:
                continue
            stat = path.stat()
            rel = path.relative_to(job.output_dir).as_posix()
        except OSError:
            continue
        files.append(
            {
                "path": rel,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "url": f"/api/jobs/{job.id}/files/{quote(rel)}",
                "kind": path.suffix.lower().lstrip(".") or "file",
            }
        )
        if len(files) >= 500:
            break
    return files


def _job_file_path(job: Job, rel: str) -> Path:
    root = job.output_dir.resolve()
    target = (job.output_dir / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError("file path escapes job output directory")
    if not target.is_file():
        raise FileNotFoundError(rel)
    return target


def _json_response(handler: SimpleHTTPRequestHandler, data: Any, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    n = int(handler.headers.get("Content-Length") or "0")
    if n <= 0:
        return {}
    return json.loads(handler.rfile.read(n).decode("utf-8"))


def _clean_list(values: Any, allowed: set[str], default: list[str]) -> list[str]:
    if values in (None, "", "all"):
        return list(default)
    if isinstance(values, str):
        raw = [x.strip() for x in values.split(",") if x.strip()]
    else:
        raw = [str(x).strip() for x in values if str(x).strip()]
    bad = [x for x in raw if x not in allowed]
    if bad:
        raise ValueError(f"unknown values: {bad}")
    return raw


def _clean_levels(values: Any) -> list[int]:
    if values in (None, "", "all"):
        return [1, 2, 3]
    raw = [values] if isinstance(values, int) else values
    if isinstance(values, str):
        raw = [x.strip() for x in values.split(",") if x.strip()]
    levels = [int(x) for x in raw]
    bad = [x for x in levels if x not in {1, 2, 3}]
    if bad:
        raise ValueError(f"levels must be 1, 2, or 3; got {bad}")
    return levels


def _env_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
    }
    allowed = {p["name"]: p for p in _env_params()}
    for name, value in (payload.get("env") or {}).items():
        if name not in allowed:
            continue
        if value is None or value == "":
            continue
        if allowed[name]["type"] == "bool":
            env[name] = "1" if bool(value) else "0"
        else:
            env[name] = str(value)
    for name, value in (payload.get("extraEnv") or {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
            raise ValueError(f"invalid environment variable name: {name}")
        if value is None or value == "":
            continue
        env[str(name)] = str(value)
    return env


def _extra_args(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("extraArgs") or []
    if isinstance(raw, str):
        return shlex.split(raw)
    return [str(x) for x in raw]


def _script_path(script: str) -> Path:
    rel = Path(script)
    if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".py":
        raise ValueError(f"invalid script path: {script}")
    path = (ROOT / rel).resolve()
    if not path.is_file() or ROOT not in path.parents:
        raise ValueError(f"script not found inside project: {script}")
    if "web_gui" in path.relative_to(ROOT).parts:
        raise ValueError("web_gui internals are not runnable custom scripts")
    return path


def _build_command(payload: dict[str, Any], job_id: str) -> tuple[list[str], dict[str, str], Path]:
    mode = str(payload.get("mode") or "benchmark")
    cases = _clean_list(payload.get("cases"), set(CASES), list(CASES))
    levels = _clean_levels(payload.get("levels"))
    env = _env_from_payload(payload)
    out_dir = RUN_ROOT / job_id
    extra_args = _extra_args(payload)

    if mode == "proposed_only":
        cmd = [
            sys.executable,
            "-u",
            "run_ap_schur_proposed_only.py",
            "--levels",
            ",".join(str(x) for x in levels),
            "--base-cases",
            ",".join(cases),
            "--out-dir",
            str(out_dir),
        ]
        if payload.get("overwrite", True):
            cmd.append("--overwrite")
        cmd.extend(extra_args)
        return cmd, env, out_dir

    if mode == "benchmark":
        methods = _clean_list(payload.get("methods"), {m["id"] for m in METHODS}, ["proposed"])
        cmd = [
            sys.executable,
            "-u",
            "paper_60case_benchmark_no_force_scaling.py",
            "--levels",
            ",".join(str(x) for x in levels),
            "--base-cases",
            ",".join(cases),
            "--methods",
            ",".join(methods),
            "--out-dir",
            str(out_dir),
            "--cache-dir",
            str(out_dir / "npz_cache"),
        ]
        if payload.get("exactMethods", True):
            cmd.append("--exact-methods")
        if payload.get("noVtk", False):
            cmd.append("--no-vtk")
        if payload.get("noCache", False):
            cmd.append("--no-cache")
        if payload.get("noResume", False):
            cmd.append("--no-resume")
        if payload.get("overwrite", False):
            cmd.extend(["--recompute-methods", ",".join(methods)])
        tol_scale = payload.get("methodTolScale")
        if tol_scale not in (None, ""):
            cmd.extend(["--method-tol-scale", str(tol_scale)])
        run_mode = payload.get("runMode")
        if run_mode:
            cmd.extend(["--run-mode", str(run_mode)])
        picard_ref_tol_scale = payload.get("picardRefTolScale")
        if picard_ref_tol_scale not in (None, ""):
            cmd.extend(["--picard-ref-tol-scale", str(picard_ref_tol_scale)])
        if payload.get("suppressReferenceRow", False):
            cmd.append("--suppress-reference-row")
        cmd.extend(extra_args)
        return cmd, env, out_dir

    if mode == "custom_script":
        script = _script_path(str(payload.get("script") or ""))
        cmd = [sys.executable, "-u", str(script.relative_to(ROOT)), *extra_args]
        return cmd, env, out_dir

    raise ValueError(f"unknown run mode: {mode}")


def _run_job(job: Job) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(job.env_overrides)
    with JOBS_LOCK:
        job.status = "running"
        job.log.append("$ " + " ".join(job.command))
        if job.env_overrides:
            job.log.append("# env " + json.dumps(job.env_overrides, ensure_ascii=False, sort_keys=True))
    try:
        proc = subprocess.Popen(
            job.command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        job.process = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            with JOBS_LOCK:
                job.log.append(line.rstrip("\n"))
                if len(job.log) > 5000:
                    job.log = job.log[-5000:]
        rc = proc.wait()
        with JOBS_LOCK:
            job.returncode = rc
            job.status = "completed" if rc == 0 else "failed"
            job.finished_at = time.time()
            job.log.append(f"# process exited with code {rc}")
    except Exception as exc:
        with JOBS_LOCK:
            job.status = "failed"
            job.returncode = -1
            job.finished_at = time.time()
            job.log.append(f"# server exception: {type(exc).__name__}: {exc}")


def _start_job(payload: dict[str, Any]) -> Job:
    job_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    command, env, out_dir = _build_command(payload, job_id)
    job = Job(id=job_id, command=command, env_overrides=env, output_dir=out_dir)
    with JOBS_LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    return job


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[gui] {self.address_string()} - {fmt % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/schema":
            _json_response(
                self,
                {
                    "cases": CASES,
                    "methods": METHODS,
                    "envParams": _env_params(),
                    "scripts": _discover_scripts(),
                    "runModes": [
                        {"id": "proposed_only", "label": "Proposed-only runner"},
                        {"id": "benchmark", "label": "All-method benchmark runner"},
                        {"id": "custom_script", "label": "Custom project script"},
                    ],
                },
            )
            return
        if path == "/api/jobs":
            with JOBS_LOCK:
                jobs = [job.public() for job in sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)]
            _json_response(self, jobs)
            return
        if path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            if len(parts) >= 4 and parts[0] == "api" and parts[1] == "jobs" and parts[3] == "files":
                job = _get_job(parts[2])
                if job is None:
                    _json_response(self, {"error": "job not found"}, HTTPStatus.NOT_FOUND)
                    return
                if len(parts) == 4:
                    _json_response(self, _job_files(job))
                    return
                rel = "/".join(parts[4:])
                try:
                    target = _job_file_path(job, rel)
                except FileNotFoundError:
                    _json_response(self, {"error": "file not found"}, HTTPStatus.NOT_FOUND)
                    return
                except ValueError as exc:
                    _json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(target.stat().st_size))
                self.send_header("Content-Disposition", f'inline; filename="{target.name}"')
                self.end_headers()
                with target.open("rb") as fh:
                    while True:
                        chunk = fh.read(1024 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                return
        if path.startswith("/api/jobs/"):
            job_id = path.split("/")[-1]
            job = _get_job(job_id)
            if job is None:
                _json_response(self, {"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            _json_response(self, job.public())
            return
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/start":
                payload = _read_json(self)
                job = _start_job(payload)
                _json_response(self, job.public(), HTTPStatus.CREATED)
                return
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    proc = job.process if job else None
                if job is None:
                    _json_response(self, {"error": "job not found"}, HTTPStatus.NOT_FOUND)
                    return
                if proc and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    with JOBS_LOCK:
                        job.status = "cancelled"
                        job.finished_at = time.time()
                        job.log.append("# cancel requested")
                _json_response(self, job.public())
                return
            _json_response(self, {"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            _json_response(self, {"error": str(exc), "type": type(exc).__name__}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LBM GUI running at http://{args.host}:{args.port}", flush=True)
    print(f"Project root: {ROOT}", flush=True)
    print(f"Run output: {RUN_ROOT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
