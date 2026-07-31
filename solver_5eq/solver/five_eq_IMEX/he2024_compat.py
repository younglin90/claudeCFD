"""Compatibility loader for frozen He2024 Phase 1/2 modules.

This avoids executing `solver.He2024.__init__` (which may import legacy
runtime-only symbols) while still reusing:
  - solver/He2024/eos_general.py
  - solver/He2024/primitive_W.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types


_ROOT = pathlib.Path(__file__).resolve().parents[1] / "He2024"
_PKG_NAME = "solver.He2024"


def _ensure_he2024_pkg() -> None:
    if _PKG_NAME in sys.modules:
        return
    pkg = types.ModuleType(_PKG_NAME)
    pkg.__path__ = [str(_ROOT)]
    pkg.__package__ = _PKG_NAME
    pkg.__file__ = str(_ROOT / "__init__.py")
    sys.modules[_PKG_NAME] = pkg


def _load_submodule(name: str, file_name: str):
    full_name = f"{_PKG_NAME}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    _ensure_he2024_pkg()
    spec = importlib.util.spec_from_file_location(full_name, _ROOT / file_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {full_name} from {_ROOT / file_name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_eos_general():
    return _load_submodule("eos_general", "eos_general.py")


def load_primitive_W():
    # Ensure eos_general exists first for primitive_W relative import.
    _load_submodule("eos_general", "eos_general.py")
    return _load_submodule("primitive_W", "primitive_W.py")

