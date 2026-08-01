#!/usr/bin/env python3
"""Round 9 (Phase 2 Stage 4, consolidation) measurement/reporting tool.

Unlike yadv_r5..r8_verify.py (each a copy-paste with a hardcoded worktree path that goes
stale the moment the worktree is deleted), this script derives the repo root from __file__,
so it works unmodified in any worktree or the main tree. No solver code is touched by this
script; it only runs the existing denner1d_validate/_dump binaries with existing env vars.

Modes: --verify --sweep --table --timing --iters (combine freely, e.g. --sweep --table).
"""
import json
import os
import re
import subprocess
import sys
import time

_NAN_RE = re.compile(r"-?nan")  # Python's json module accepts "NaN"/"-Infinity" but not the
                                 # lowercase "nan"/"-nan" this C++ build prints (printf %g style)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build-cpp", "cpp", "denner_1d")
VALIDATE = os.path.join(BUILD, "denner1d_validate")
DUMP = os.path.join(BUILD, "denner1d_dump")
RUN = os.path.join(BUILD, "denner1d_run")
REF_ROOT = "/home/younglin90/work/claude_code/claudeCFD/solver_denner"
REF_DUMP = os.path.join(REF_ROOT, "build-cpp", "cpp", "denner_1d", "denner1d_dump")
SCRATCH = "/tmp/yadv_r9"

ACID_ENV_VARS = ("ACID_YADV", "ACID_YADV_ALPHA_IMPLICIT", "ACID_YADV_ALPHA_IMPLICIT_T",
                  "ACID_NO_AJAC", "ACID_RHIST", "ACID_BLK_STEP")

CONFIGS = [
    ("A", "OFF", {}),
    ("B", "ON", {"ACID_YADV": "1"}),
    ("C", "ON+IMPLICIT", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1"}),
    ("D", "ON+IMPLICIT+FD", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1", "ACID_NO_AJAC": "1"}),
    ("E", "OFF+FD", {"ACID_NO_AJAC": "1"}),
    ("F", "ON+IMPLICIT+T", {"ACID_YADV": "1", "ACID_YADV_ALPHA_IMPLICIT": "1",
                             "ACID_YADV_ALPHA_IMPLICIT_T": "1"}),
    ("G", "ON+FD", {"ACID_YADV": "1", "ACID_NO_AJAC": "1"}),
]
# EXPECTED updated round 20 (docs/YADV_RESEARCH.md sect.30): promoting the T-ceiling-saturation
# stall check (F2'', formerly ACID_TSAT_STALL) to unconditional changed D and E -- both were
# silently accepting a saturated iterate that later NaN-diverged; catching it as a stall now lets
# the existing dt-halving retry find an admissible step instead, so case28 (D) and case27 (E) flip
# from FAIL to PASS. A/B/C/F are unchanged (measured byte-identical to their round 19 values).
# Pre-round-20 values (last valid through commit ea38c04, i.e. through round 19):
#   D: (12, {"14", "15", "24", "27", "28", "33", "34"})
#   E: (13, {"15", "24", "27", "28", "33", "34"})
#   (config G did not exist before round 20)
EXPECTED = {
    "A": (19, set()),
    "B": (15, {"15", "24", "33", "34"}),
    "C": (14, {"14", "15", "24", "33", "34"}),
    "D": (13, {"14", "15", "24", "27", "33", "34"}),
    "E": (14, {"15", "24", "28", "33", "34"}),
    "F": (14, {"14", "15", "24", "33", "34"}),
    "G": (15, {"15", "24", "33", "34"}),
}
ALL_CASES = ["01", "02", "04", "05", "07", "13", "14", "15", "24", "25", "26", "27", "28",
             "30", "31", "33", "34", "35", "36"]
VERIFY_CASES = ["01", "02", "13", "14", "15", "24", "25", "33", "34"]


def base_env(overlay=None):
    env = dict(os.environ, DENNER_ACID="1")
    for k in ACID_ENV_VARS:
        env.pop(k, None)  # never inherit a stale flag from the caller's shell
    if overlay:
        env.update(overlay)
    return env


def run_validate(overlay, only=None, out=None):
    env = base_env(overlay)
    cmd = [VALIDATE]
    if only:
        cmd += ["--only", only]
    if out:
        cmd += ["--out", out]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    dt = time.perf_counter() - t0
    return dt, r.stdout


def parse_metrics(stdout):
    cases = {}
    pass_count = total = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(_NAN_RE.sub("NaN", line))
            except json.JSONDecodeError:
                continue
            if "case" in d:
                cases[d["case"]] = d
        elif line.startswith("DENNER1D_CPP_METRIC"):
            for tok in line.split():
                if tok.startswith("pass_count="):
                    pass_count = int(tok.split("=")[1])
                elif tok.startswith("total="):
                    total = int(tok.split("=")[1])
    return cases, pass_count, total


def do_sweep():
    os.makedirs(SCRATCH, exist_ok=True)
    results = {}
    for tag, label, overlay in CONFIGS:
        outdir = os.path.join(SCRATCH, f"out_{tag}")
        os.makedirs(outdir, exist_ok=True)
        dt, stdout = run_validate(overlay, out=outdir)
        cases, pass_count, total = parse_metrics(stdout)
        fail_set = {cid for cid, d in cases.items() if d.get("pass") is False}
        results[tag] = {"label": label, "cases": cases, "pass_count": pass_count,
                         "total": total, "fail_set": sorted(fail_set), "wall_s": dt}
        exp_pc, exp_fail = EXPECTED[tag]
        ok = (pass_count == exp_pc) and (fail_set == exp_fail)
        print(f"[{tag}] {label}: pass_count={pass_count}/{total} fail={sorted(fail_set)} "
              f"wall={dt:.2f}s  {'GATE OK' if ok else 'GATE MISMATCH'}")
        if not ok:
            print(f"    expected pass_count={exp_pc} fail={sorted(exp_fail)}")
    with open(os.path.join(SCRATCH, "r9_raw.json"), "w") as f:
        json.dump(results, f, indent=2)
    all_ok = all(
        results[tag]["pass_count"] == EXPECTED[tag][0]
        and set(results[tag]["fail_set"]) == EXPECTED[tag][1]
        for tag in results
    )
    print("ALL GATES OK" if all_ok else "SOME GATES MISMATCHED -- investigate before publishing")
    return results, all_ok


def load_raw():
    with open(os.path.join(SCRATCH, "r9_raw.json")) as f:
        return json.load(f)


def do_table():
    results = load_raw()
    tags = [t for t, _, _ in CONFIGS if t in results]
    print("\n### Table 1 -- consolidated pass/fail\n")
    header = "| case | " + " | ".join(f"{t} pass" for t in tags) + " | B l2_p | C l2_p | B corr_p | C corr_p |"
    print(header)
    print("|---" * (len(tags) + 5) + "|")
    for cid in ALL_CASES:
        row = [cid]
        for t in tags:
            d = results[t]["cases"].get(cid, {})
            row.append("PASS" if d.get("pass") else "FAIL")
        b = results.get("B", {}).get("cases", {}).get(cid, {})
        c = results.get("C", {}).get("cases", {}).get(cid, {})
        row.append(f'{b.get("l2_p", float("nan")):.4g}')
        row.append(f'{c.get("l2_p", float("nan")):.4g}')
        row.append(f'{b.get("corr_p", float("nan")):.4g}')
        row.append(f'{c.get("corr_p", float("nan")):.4g}')
        print("| " + " | ".join(row) + " |")
    for t in tags:
        r = results[t]
        print(f"pass_count[{t}]={r['pass_count']}/{r['total']}  fail={r['fail_set']}")

    print("\n### Table 3 -- B->C delta (cases where B and C disagree on pass, or l2_p/corr_p differ)\n")
    print("| case | B pass | C pass | B l2_p | C l2_p | B corr_p | C corr_p |")
    print("|---|---|---|---|---|---|---|")
    for cid in ALL_CASES:
        b = results.get("B", {}).get("cases", {}).get(cid, {})
        c = results.get("C", {}).get("cases", {}).get(cid, {})
        if not b or not c:
            continue
        diff_pass = b.get("pass") != c.get("pass")
        diff_l2p = abs(b.get("l2_p", 0) - c.get("l2_p", 0)) > 1e-9
        diff_corrp = abs(b.get("corr_p", 0) - c.get("corr_p", 0)) > 1e-9
        if diff_pass or diff_l2p or diff_corrp:
            print(f'| {cid} | {b.get("pass")} | {c.get("pass")} | {b.get("l2_p"):.6g} | '
                  f'{c.get("l2_p"):.6g} | {b.get("corr_p"):.6g} | {c.get("corr_p"):.6g} |')


def do_verify():
    ok_all = True
    print("(1) ACID_YADV unset vs solver_denner published binary")
    for cid in VERIFY_CASES:
        env = base_env()
        mine = subprocess.run([DUMP, cid], capture_output=True, text=True, env=env, cwd=ROOT).stdout
        env_ref = dict(os.environ, DENNER_ACID="1")
        for k in ACID_ENV_VARS:
            env_ref.pop(k, None)
        ref = subprocess.run([REF_DUMP, cid], capture_output=True, text=True, env=env_ref,
                              cwd=REF_ROOT).stdout
        identical = mine == ref
        ok_all &= identical
        print(f"    case{cid}: {'BYTE-IDENTICAL' if identical else 'DIFFERS'}")
    print("(2) ACID_YADV=1 vs ACID_YADV unset (same binary), case01")
    env_on = base_env({"ACID_YADV": "1"})
    on01 = subprocess.run([DUMP, "01"], capture_output=True, text=True, env=env_on, cwd=ROOT).stdout
    env_off = base_env()
    off01 = subprocess.run([DUMP, "01"], capture_output=True, text=True, env=env_off, cwd=ROOT).stdout
    identical01 = on01 == off01
    ok_all &= identical01
    print(f"    case01: {'BYTE-IDENTICAL' if identical01 else 'DIFFERS'}")
    print("VERIFY OK" if ok_all else "VERIFY MISMATCH")
    return ok_all


def do_timing():
    os.makedirs(SCRATCH, exist_ok=True)
    print("\n### Table -- per-case wall clock, min of 3 repeats (s)\n")
    print("| case | B | C | C/B | D | D/C | comparable |")
    print("|---|---|---|---|---|---|---|")
    both_pass_b = both_pass_c = both_pass_d = 0.0
    n_comparable = 0
    for cid in ALL_CASES:
        if cid in ("04", "05", "07", "35", "36"):
            continue  # TR-BDF2 -> FD Jacobian regardless of config; not part of this comparison
        times = {}
        for tag, _, overlay in CONFIGS:
            if tag not in ("B", "C", "D"):
                continue
            reps = []
            for _ in range(3):
                dt, _ = run_validate(overlay, only=cid, out=os.path.join(SCRATCH, f"t_{tag}"))
                reps.append(dt)
            times[tag] = min(reps)
        b, c, d = times["B"], times["C"], times["D"]
        raw = load_raw() if os.path.exists(os.path.join(SCRATCH, "r9_raw.json")) else None
        comparable = False
        if raw:
            bp = raw["B"]["cases"].get(cid, {}).get("pass")
            cp = raw["C"]["cases"].get(cid, {}).get("pass")
            comparable = bool(bp) and bool(cp)
        if comparable:
            both_pass_b += b
            both_pass_c += c
            both_pass_d += d
            n_comparable += 1
        print(f"| {cid} | {b:.3f} | {c:.3f} | {c/b:.3f} | {d:.3f} | {d/c:.3f} | "
              f"{'yes' if comparable else 'no'} |")
    if n_comparable:
        print(f"\nboth-pass subset ({n_comparable} cases): B total={both_pass_b:.2f}s "
              f"C total={both_pass_c:.2f}s ratio={both_pass_c/both_pass_b:.3f} "
              f"D total={both_pass_d:.2f}s D/C={both_pass_d/both_pass_c:.3f}")


def do_iters():
    sample_steps = [0, 1, 2, 5, 10, 25, 50, 100]
    sample_cases = ["13", "25", "14", "15", "02", "24"]
    print("\n### Sampled inner-Newton iteration counts (ACID_RHIST, steps "
          f"{sample_steps}, NOT a suite mean)\n")
    print("| case | config | mean iters (sampled) | notes |")
    print("|---|---|---|---|")
    for cid in sample_cases:
        for tag, _, overlay in CONFIGS:
            if tag not in ("B", "C", "D"):
                continue
            counts = []
            capped = False
            cap = 40 if tag == "D" else 150
            for step in sample_steps:
                env = base_env(overlay)
                env["ACID_RHIST"] = "1"
                env["ACID_BLK_STEP"] = str(step)
                r = subprocess.run([RUN, cid], capture_output=True, text=True, env=env, cwd=ROOT)
                n = sum(1 for line in r.stderr.splitlines() if line.startswith("RHIST"))
                if n > 0:
                    counts.append(n)
                    if n >= cap:
                        capped = True
            mean = sum(counts) / len(counts) if counts else float("nan")
            note = "hit cap" if capped else ""
            print(f"| {cid} | {tag} | {mean:.1f} | {note} |")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--sweep" in args:
        do_sweep()
    if "--table" in args:
        do_table()
    if "--verify" in args:
        ok = do_verify()
        if not ok:
            sys.exit(1)
    if "--timing" in args:
        do_timing()
    if "--iters" in args:
        do_iters()
    if not args:
        print(__doc__)
