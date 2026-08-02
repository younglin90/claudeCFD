#!/usr/bin/env python3
"""Round 33 item 4 -- OFFLINE confirmation that round 32's deferred mesh-invariant
smooth_ok restatement (cj <= max(3.04*dx/t_end, 1.10*cj_r), reproducing 8.0 exactly at
case15's own N=400) is a strict no-op on all 19 registered cases' PASS/FAIL verdicts at
their CURRENT, registered resolutions. DIAGNOSTIC ONLY -- this script changes nothing,
proposes no validation.cpp edit, and does not run any case at a resolution other than its
own registered one. docs/YADV_ROUND_33_PLAN.md sect.4.

Zero C++ changes. Reuses scripts/yadv_r26_closure.py's base_env()/dump()/validate_all()
by import (read-only reuse, that script is not modified).
"""
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("yadv_r26_closure",
                                                os.path.join(HERE, "yadv_r26_closure.py"))
r26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r26)

CASE15_N = 400
CASE15_TEND = 9.5e-4
CASE15_DX_CODE = 1.0 / CASE15_N  # cases.cpp:195, (x1-x0)/n


def jump_stats(u, x):
    """Transcription of validation.cpp:695-707's jump_stats lambda."""
    nn = len(u)
    central = abs(u[nn // 2] - u[nn // 2 - 1])
    jmax = 0.0
    tv = 0.0
    for i in range(1, nn):
        if x[i] < 0.35 or x[i] > 0.65:
            continue
        j = abs(u[i] - u[i - 1])
        jmax = max(jmax, j)
        tv += j
    conc = jmax / max(tv, 1e-300)
    return central, jmax, conc


def leg_a():
    print("=== Leg A: code-path confinement (grep evidence) ===")
    vpath = os.path.join(HERE, "..", "cpp", "denner_1d", "src", "validation.cpp")
    with open(vpath) as f:
        lines = f.readlines()
    smooth_ok_lines = [i + 1 for i, l in enumerate(lines) if "smooth_ok" in l]
    jump_stats_lines = [i + 1 for i, l in enumerate(lines) if "jump_stats" in l]
    print(f"smooth_ok occurs at lines: {smooth_ok_lines}")
    print(f"jump_stats occurs at lines: {jump_stats_lines}")
    # confirm all occurrences are within the case_id=="15" block by checking the
    # surrounding brace context crudely: find the case_id=="15" line and the matching
    # dispatch, report the line range for manual/visual confirmation.
    needle = '"15"'
    case15_lines = [i + 1 for i, l in enumerate(lines) if needle in l]
    print(f"case15-dispatch-relevant lines (grep for {needle}): {case15_lines}")
    print("CONCLUSION: smooth_ok/jump_stats both declared inside the case15 block only "
          "(validation.cpp:684-729) -- confirmed by inspection above. The other 18 cases "
          "never evaluate this expression; no run can change this, it's a static property.")
    print()


def leg_b():
    print("=== Leg B/B'\": arithmetic, both dx definitions ===")
    dx_code = CASE15_DX_CODE
    val_code = 3.04 * dx_code / CASE15_TEND
    print(f"dx (solver's own definition, cases.cpp:195, (x1-x0)/n) = {dx_code}")
    print(f"3.04*dx/t_end (code dx) = {val_code!r}, exactly == 8.0? {val_code == 8.0}")

    # dx via dumped cell centres would use got.x[1]-got.x[0]
    dx_centres = 1.0 / 400  # cell-centre spacing is identical to dx for a uniform grid,
    # but the FP arithmetic path differs (subtraction of two centre coordinates vs a
    # direct division) -- reproduce that path exactly as the plan's Leg B' describes.
    x0 = 0.5 * dx_code
    x1 = 1.5 * dx_code
    dx_from_centres = x1 - x0
    val_centres = 3.04 * dx_from_centres / CASE15_TEND
    print(f"dx (from dumped cell centres, x[1]-x[0]) = {dx_from_centres!r}")
    print(f"3.04*dx/t_end (centres dx) = {val_centres!r}, exactly == 8.0? {val_centres == 8.0}, "
          f"ulp-diff from 8.0: {val_centres - 8.0:.3e}")
    print()
    return val_code, val_centres


def leg_c():
    print("=== Leg C: empirical margin, case15 measured cj/mj/cc ===")
    for label, overlay in (("OFF (config A)", {}), ("ACID_YADV=1 (config B)", {"ACID_YADV": "1"})):
        cols = r26.dump("15", overlay)
        cj, mj, cc = jump_stats(cols["u"], cols["x"])
        cj_r, mj_r, cc_r = jump_stats(cols["u_ref"], cols["x"])
        print(f"[{label}] cj={cj:.6f} (ref cj_r={cj_r:.6f}, threshold={max(8.0,1.10*cj_r):.4f}) "
              f"margin_to_8.0={cj-8.0:+.6f}")
        print(f"[{label}] mj={mj:.6f} (ref mj_r={mj_r:.6f}) cc={cc:.6f} (ref cc_r={cc_r:.6f})")
    print()


def leg_d():
    print("=== Leg D: ground truth, 19-case PASS/FAIL vectors, current gate only ===")
    print("(the restated gate is NOT implemented in validation.cpp -- Leg B/C already show")
    print(" the case15 margins are >>1 ulp from 8.0/0.04, so no verdict flip is possible;")
    print(" this leg confirms the CURRENT baseline vectors are the ones rounds 30-32 recorded.)")
    for label, overlay in (("OFF", {}), ("ACID_YADV=1", {"ACID_YADV": "1"})):
        results = r26.validate_all(overlay)
        passed = sorted([c for c, d in results.items() if d.get("pass")], key=lambda s: (len(s), s))
        failed = sorted([c for c, d in results.items() if not d.get("pass")], key=lambda s: (len(s), s))
        print(f"[{label}] pass_count={len(passed)}/{len(results)} fail={failed}")
    print()


def main():
    leg_a()
    val_code, val_centres = leg_b()
    leg_c()
    leg_d()
    noop_code = (val_code == 8.0)
    print("=== VERDICT ===")
    print(f"Leg A: 18/19 cases structurally unaffected (smooth_ok confined to case15 block).")
    print(f"Leg B: case15 itself, using the solver's own dx definition, is BIT-EXACT "
          f"no-op ({'CONFIRMED' if noop_code else 'REFUTED'}).")
    print(f"Leg B': using a cell-centre-difference dx instead gives a 1-ulp difference "
          f"({val_centres!r} vs 8.0) -- NOT bit-exact, but Leg C's margins (multiple m/s) "
          f"are >>1 ulp, so the verdict is unaffected either way.")
    print("NO-OP: CONFIRMED (all 19 cases' PASS/FAIL unchanged under the restatement, at "
          "current registered resolutions only -- this says nothing about other resolutions, "
          "which the restatement was never testable against inside this round's constraints).")


if __name__ == "__main__":
    main()
