---
description: Which trees are editable vs frozen in solver_5eq, and why
paths:
  - "solver/**"
  - "validation/**"
  - "tests/**"
---

# Edit boundaries — solver_5eq

## Editable
`solver/five_eq_IMEX/`, `solver/five_eq_IMEX_v2/`, `tests/`, `docs/`, `results/`, `.codex-loop/`.

## Frozen — do NOT modify
- `solver/He2024/` — frozen EOS Phase-1 output. Consume it read-only through
  `solver/five_eq_IMEX/he2024_compat.py`, which loads `eos_general.py` / `primitive_W.py`
  directly and **bypasses `__init__`**. Never `import solver.He2024`.
  Keep `solver/__init__.py` empty (no legacy imports) so the bypass keeps working.
- `validation/` — validation specs are read-only (1D 26-case + 2D/3D + `INDEX.md`).

Editing either frozen tree is a mistake even when it looks like the quick fix —
route the change through the editable trees or ask first.
