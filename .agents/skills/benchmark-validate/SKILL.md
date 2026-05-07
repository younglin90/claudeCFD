---
name: benchmark-validate
description: Run claudeCFD regression gates and summarize pass/fail deltas. Use when validating solver changes, checking 02-A/07-B behavior, or producing benchmark evidence for a research-improvement cycle.
---

# Benchmark Validate

## Workflow

1. Run `scripts/run_and_compare.py` from the repository root.
2. Review `.agents/pipeline/validation_report.md`.
3. Treat 02-A, uniform flow, BE1 amplification, and transport eigenmode as mandatory gates.
4. Treat 07-B as diagnostic until strict PASS criteria are met.

## Command

```bash
python3 .agents/skills/benchmark-validate/scripts/run_and_compare.py
```

Include 07 smoke only when runtime is acceptable:

```bash
python3 .agents/skills/benchmark-validate/scripts/run_and_compare.py --include-07
```

## Required Interpretation

- If 02-A fails, stop and report regression.
- If BE1 `rho(A)` exceeds the configured gate, stop and report PE instability.
- If 07 changes without 02-A regression, report subcase metrics separately. Do not claim a global improvement from one subcase.
