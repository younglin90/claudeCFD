# -*- coding: utf-8 -*-
"""Path resolution for the archived paper2 figure scripts.

RESULTS   - the cpp/results tree, resolved for whichever OS is running the script
CACHE_DIR - a `_cache/` directory beside this file, replacing the per-session scratchpad
            the scripts originally used
"""
import os

_WIN = r"\\wsl.localhost\ubuntu\home\younglin90\work\claude_code\claudeCFD\cpp\results"
_NIX = "/home/younglin90/work/claude_code/claudeCFD/cpp/results"

RESULTS = _NIX if os.path.isdir(_NIX) else _WIN
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
# scripts write their regenerated figures here
FIG_DIR = os.path.join(CACHE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
