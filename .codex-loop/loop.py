#!/usr/bin/env python3
"""Bounded Codex repair loop for claudeCFD.

This supervisor repeatedly runs a validation command, feeds failing logs to
`codex exec`, records state, and stops on explicit safety conditions.
It is intentionally not a true infinite loop.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOOP_DIR = ROOT / ".codex-loop"
LOG_DIR = LOOP_DIR / "logs"
RULES_FILE = LOOP_DIR / "LOOP_RULES.md"
STATE_MD = LOOP_DIR / "LOOP_STATE.md"
STATE_JSON = LOOP_DIR / "loop_state.json"

DEFAULT_TEST_CMD = "python3 .agents/skills/benchmark-validate/scripts/run_and_compare.py"
DEFAULT_MAX_ITERS = 10
DEFAULT_REPEAT_LIMIT = 3
DEFAULT_CHANGED_LINE_LIMIT = 800
DEFAULT_LOG_TAIL = 12000


def git_cmd(args: list[str]) -> list[str]:
    return ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def run_cmd(cmd: str | list[str], *, timeout: int | None = None,
            log_path: Path | None = None, shell: bool = False) -> tuple[int, str]:
    if shell:
        if isinstance(cmd, list):
            cmd = " ".join(shlex.quote(str(part)) for part in cmd)
        if os.name == "nt":
            cmd = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                   "-Command", cmd]
        else:
            cmd = ["bash", "-lc", cmd]
        shell = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            shell=shell,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        output = proc.stdout or ""
        status = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\nTIMEOUT\n"
        status = 124
    except FileNotFoundError as exc:
        output = f"{type(exc).__name__}: {exc}\n"
        status = 127
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8", errors="replace")
    return status, output


def git_out(args: list[str], *, allow_fail: bool = False) -> str:
    status, output = run_cmd(git_cmd(args))
    if status != 0 and not allow_fail:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{output}")
    return output.strip()


def ensure_git_repo() -> None:
    status, output = run_cmd(git_cmd(["rev-parse", "--is-inside-work-tree"]))
    if status != 0 or "true" not in output:
        raise SystemExit("Not inside a git repository.")


def load_state() -> dict[str, Any]:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    return {"created_at": utc_now(), "iterations": [], "failure_hashes": []}


def new_state() -> dict[str, Any]:
    return {"created_at": utc_now(), "iterations": [], "failure_hashes": []}


def save_state(state: dict[str, Any]) -> None:
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def append_state_md(entry: dict[str, Any]) -> None:
    if not STATE_MD.exists():
        STATE_MD.write_text("# Codex Loop State\n\n## Iteration History\n",
                            encoding="utf-8")
    changed_files = entry.get("changed_files") or []
    lines = [
        "",
        f"### Iteration {entry['iteration']} - {entry['timestamp']}",
        "",
        f"- test_status: {entry['test_status']}",
        f"- codex_status: {entry.get('codex_status')}",
        f"- failure_hash: `{entry['failure_hash']}`",
        f"- changed_lines: {entry.get('changed_lines', 0)}",
        f"- stop_reason: {entry.get('stop_reason', '')}",
        f"- hypothesis: {entry.get('hypothesis', 'see codex log')}",
        f"- remaining_suspected_cause: {entry.get('remaining_suspected_cause', 'unknown')}",
        "- files_changed:",
    ]
    if changed_files:
        lines.extend([f"  - `{name}`" for name in changed_files])
    else:
        lines.append("  - none")
    STATE_MD.write_text(STATE_MD.read_text(encoding="utf-8") + "\n".join(lines) + "\n",
                        encoding="utf-8")


def normalized_failure_hash(log: str) -> str:
    keys = []
    markers = (
        "FAILED", "ERROR", "AssertionError", "Traceback", "error:",
        "undefined reference", "ModuleNotFoundError", "ImportError",
        "FAIL", "NaN", "overflow",
    )
    for raw in log.splitlines():
        line = raw.strip()
        if any(marker in line for marker in markers):
            keys.append(line)
    text = "\n".join(keys[-120:]) if keys else log[-4000:]
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def changed_files() -> list[str]:
    status, output = run_cmd(git_cmd(["diff", "--name-only"]))
    if status != 0:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def changed_line_count() -> int:
    status, output = run_cmd(git_cmd(["diff", "--numstat"]))
    if status != 0:
        return 0
    total = 0
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        for part in parts[:2]:
            if part.isdigit():
                total += int(part)
    return total


def diff_stat() -> str:
    status, output = run_cmd(git_cmd(["diff", "--stat"]))
    return output if status == 0 else ""


def dirty_status() -> str:
    status, output = run_cmd(git_cmd(["status", "--short"]))
    return output if status == 0 else ""


def make_prompt(test_cmd: str, test_log: str, state: dict[str, Any],
                log_tail: int) -> str:
    rules = RULES_FILE.read_text(encoding="utf-8") if RULES_FILE.exists() else ""
    state_md = STATE_MD.read_text(encoding="utf-8") if STATE_MD.exists() else ""
    recent = json.dumps(state.get("iterations", [])[-3:], indent=2,
                        ensure_ascii=False)
    return f"""You are running inside the claudeCFD repository via a bounded repair loop.

Read and follow `.codex-loop/LOOP_RULES.md` before acting.

Loop rules:
```md
{rules}
```

Current loop state:
```md
{state_md[-6000:]}
```

Recent machine-readable history:
```json
{recent}
```

Current git status:
```text
{dirty_status()}
```

Goal:
- Make the test command pass.
- Use the smallest correct code change.
- Do not weaken tests.
- Do not modify frozen directories.
- Do not run destructive git commands.
- Avoid broad refactoring.
- If changing numerical code, explain the residual/operator impact in comments or the final response.

Test command:
```bash
{test_cmd}
```

Latest failing log:
```text
{test_log[-log_tail:]}
```

Required actions:
1. Identify the most likely root cause from the log.
2. Modify only the minimal relevant files.
3. Run targeted verification if feasible under the sandbox.
4. Update `.codex-loop/LOOP_STATE.md` with hypothesis, files changed, test result, and remaining suspected cause.
"""


def codex_exec(args: argparse.Namespace, prompt: str, log_path: Path) -> tuple[int, str]:
    cmd = [
        args.codex_bin,
        "exec",
        "--sandbox", args.sandbox,
        "--ask-for-approval", args.approval,
    ]
    for item in args.codex_arg:
        cmd.extend(shlex.split(item))
    cmd.append(prompt)
    return run_cmd(cmd, timeout=args.codex_timeout, log_path=log_path)


def commit_iteration(iteration: int) -> tuple[int, str]:
    status, output = run_cmd(git_cmd(["add", "-A"]))
    if status != 0:
        return status, output
    return run_cmd(git_cmd(["commit", "-m", f"codex-loop: iteration {iteration} fix attempt"]))


def write_patch_snapshot(iteration: int) -> None:
    status, output = run_cmd(git_cmd(["diff"]))
    if status == 0 and output:
        path = LOG_DIR / f"diff_iter_{iteration}.patch"
        path.write_text(output, encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iters", type=int, default=int(os.environ.get("MAX_ITERS", DEFAULT_MAX_ITERS)))
    parser.add_argument("--test-cmd", default=os.environ.get("TEST_CMD", DEFAULT_TEST_CMD))
    parser.add_argument("--test-timeout", type=int, default=int(os.environ.get("TEST_TIMEOUT", "600")))
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", "codex"))
    parser.add_argument("--codex-timeout", type=int, default=int(os.environ.get("CODEX_TIMEOUT", "900")))
    parser.add_argument("--sandbox", default=os.environ.get("CODEX_SANDBOX", "workspace-write"))
    parser.add_argument("--approval", default=os.environ.get("CODEX_APPROVAL", "never"))
    parser.add_argument("--codex-arg", action="append", default=[],
                        help="Extra argument string passed to codex exec. Repeatable.")
    parser.add_argument("--repeat-limit", type=int, default=DEFAULT_REPEAT_LIMIT)
    parser.add_argument("--changed-line-limit", type=int, default=DEFAULT_CHANGED_LINE_LIMIT)
    parser.add_argument("--log-tail", type=int, default=DEFAULT_LOG_TAIL)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Allow starting while git status is already dirty.")
    parser.add_argument("--commit-each-iteration", action="store_true",
                        help="Commit each successful Codex edit attempt. Default is off.")
    parser.add_argument("--check-only", action="store_true",
                        help="Run the test command once and exit without calling Codex.")
    parser.add_argument("--reset-state", action="store_true",
                        help="Start with a fresh in-memory loop state and overwrite loop_state.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ensure_git_repo()

    if not args.allow_dirty and dirty_status().strip():
        print("Working tree is dirty. Commit/stash changes or pass --allow-dirty.", file=sys.stderr)
        print(dirty_status(), file=sys.stderr)
        return 6

    state = new_state() if args.reset_state else load_state()
    for iteration in range(1, args.max_iters + 1):
        print(f"========== Iteration {iteration}/{args.max_iters} ==========")

        test_log_path = LOG_DIR / f"test_iter_{iteration}.log"
        test_status, test_log = run_cmd(args.test_cmd, timeout=args.test_timeout,
                                        log_path=test_log_path, shell=True)

        if test_status == 0:
            print("Tests passed.")
            state["completed_at"] = utc_now()
            state["result"] = "pass"
            save_state(state)
            return 0

        fail_hash = normalized_failure_hash(test_log)
        state.setdefault("failure_hashes", []).append(fail_hash)
        save_state(state)

        if args.check_only:
            print("Tests failed; --check-only prevents Codex repair.")
            return test_status

        if len(state["failure_hashes"]) >= args.repeat_limit:
            recent_hashes = state["failure_hashes"][-args.repeat_limit:]
            if len(set(recent_hashes)) == 1:
                print(f"Same normalized failure repeated {args.repeat_limit} times. Stopping.")
                save_state(state)
                return 5

        before = run_cmd(git_cmd(["rev-parse", "HEAD"]))[1].strip()
        prompt = make_prompt(args.test_cmd, test_log, state, args.log_tail)
        prompt_path = LOG_DIR / f"prompt_iter_{iteration}.md"
        prompt_path.write_text(prompt, encoding="utf-8")

        codex_log_path = LOG_DIR / f"codex_iter_{iteration}.md"
        codex_status, _codex_out = codex_exec(args, prompt, codex_log_path)

        files = changed_files()
        lines = changed_line_count()
        write_patch_snapshot(iteration)
        entry = {
            "iteration": iteration,
            "timestamp": utc_now(),
            "test_status": test_status,
            "codex_status": codex_status,
            "failure_hash": fail_hash,
            "changed_lines": lines,
            "changed_files": files,
            "before": before,
            "diff_stat": diff_stat(),
        }

        if lines == 0:
            entry["stop_reason"] = "codex_made_no_changes"
            state.setdefault("iterations", []).append(entry)
            save_state(state)
            append_state_md(entry)
            print("Codex produced no changes. Stopping.")
            return 3

        if lines > args.changed_line_limit:
            entry["stop_reason"] = "changed_line_limit_exceeded"
            state.setdefault("iterations", []).append(entry)
            save_state(state)
            append_state_md(entry)
            print(f"Changed lines {lines} exceeded limit {args.changed_line_limit}. Stopping without rollback.")
            return 4

        if args.commit_each_iteration:
            commit_status, commit_log = commit_iteration(iteration)
            (LOG_DIR / f"commit_iter_{iteration}.log").write_text(commit_log, encoding="utf-8")
            entry["commit_status"] = commit_status
            entry["after"] = run_cmd(git_cmd(["rev-parse", "HEAD"]))[1].strip()

        entry["stop_reason"] = ""
        state.setdefault("iterations", []).append(entry)
        save_state(state)
        append_state_md(entry)

    print("Reached max iterations without success.")
    state["completed_at"] = utc_now()
    state["result"] = "max_iterations"
    save_state(state)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
