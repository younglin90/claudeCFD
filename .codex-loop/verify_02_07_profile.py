#!/usr/bin/env python3
"""Scalar verifier for the foreground autoresearch loop.

Final stdout line is the scalar metric:
    failure_count = 07-B profile failures among Air-Water/Helium-Air/Argon-Air

02-A and mandatory unit gates are handled separately as guard commands.  This
script intentionally returns 0 when it can parse the 07 runner output, even if
some subcases fail, so the loop can record the numeric metric.
"""
from __future__ import annotations

import re
import subprocess
import sys


CMD = [
    sys.executable,
    "results/run_02_07_five_eq_imex.py",
    "--case", "07",
    "--n07", "200",
    "--cfl07", "0.4",
    "--imp-dissipation", "0.1",
    "--imp-dissipation-form", "acoustic_riemann",
    "--pe-projection-mode", "interface_explicit",
    "--alpha-floor07", "1e-5",
    "--pure-branch07",
    "--energy-alpha-pure-tol07", "1e-5",
    "--implicit-include-explicit-residual07",
    "--kapila-closure07",
    "--max-steps07", "5000",
    "--profile-pass07",
]


def main() -> int:
    proc = subprocess.run(CMD, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    print(proc.stdout.rstrip())

    statuses = re.findall(r"^07\s+([^:]+):\s+status=(PASS|FAIL)\b",
                          proc.stdout, flags=re.MULTILINE)
    if len(statuses) != 3:
        print("3")
        return 1
    failure_count = sum(1 for _name, status in statuses if status != "PASS")
    print(failure_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
