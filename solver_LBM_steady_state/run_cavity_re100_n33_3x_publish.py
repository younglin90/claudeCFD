#!/usr/bin/env python3
"""Run lid-driven cavity Re=100, N=129 (3x) and export paper-ready artifacts.

This thin wrapper reuses the 2x publication runner while overriding the case
constants to keep the 1x/2x/3x pipeline identical.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_impl():
    path = Path(__file__).with_name("run_cavity_re100_n33_2x_publish.py")
    spec = importlib.util.spec_from_file_location("cavity_re100_n33_2x_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mod = _load_impl()
    mod.CASE_ID = "cavity_re100_n33__3x"
    mod.CASE_LABEL = "Lid-driven cavity Re=100 N=129__3x"
    mod.N = 129
    mod.OUT_ROOT = Path("papers_data") / "lid_driven_Re100_N33__3x"
    mod.FIELD_DIR = mod.OUT_ROOT / "fields"
    mod.FIG_DIR = mod.OUT_ROOT / "figure"
    mod.HIST_DIR = mod.OUT_ROOT / "histories"
    mod.VTK_DIR = mod.OUT_ROOT / "vtk"
    mod.main()


if __name__ == "__main__":
    main()
