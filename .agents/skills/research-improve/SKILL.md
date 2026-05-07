---
name: research-improve
description: Convert a vetted CFD research idea into a bounded claudeCFD implementation sprint. Use when asked to implement a paper idea, run a research-improvement cycle, create a sprint contract, or improve five_eq_IMEX based on scout_report.json.
---

# Research Improve

## Workflow

1. Read `.agents/pipeline/scout_report.json`.
2. Select exactly one implementable idea.
3. Write `.agents/pipeline/sprint_contract.md` before editing code.
4. Limit code changes to allowed paths: `solver/five_eq_IMEX/`, `tests/`, `docs/`, and active `results/` drivers.
5. Change one numerical mechanism per cycle.
6. Run `$benchmark-validate`.
7. Append outcome to `.agents/pipeline/cycle_log.md`.

## Sprint Contract

Include:

- target issue and hypothesis
- paper/source basis
- files to modify
- exact regression gates
- expected metric movement
- rollback/stop condition

## Stop Conditions

- 02-A regression breaks.
- BE1 amplification gate exceeds the accepted threshold.
- A test failure cannot be attributed to the intended mechanism.
- The change requires frozen directories.

Do not perform destructive rollback automatically. Leave changes inspectable, write the failure reason, and add the failed idea to `.agents/pipeline/blocklist.json`.

## References

- Read `references/pe_operator_priorities.md` before selecting an implementation target.
