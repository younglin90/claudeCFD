---
name: research-scout
description: Search and rank recent compressible multiphase CFD papers for claudeCFD research-improvement cycles. Use when asked to find papers, scout literature, identify unimplemented numerical methods, or prepare a research candidate list for the five_eq_IMEX solver.
---

# Research Scout

## Workflow

1. Run `scripts/search_papers.py` from the repository root.
2. Use `.agents/pipeline/implemented_methods.json` and `.agents/pipeline/blocklist.json` to filter repeated or rejected ideas.
3. Write candidates to `.agents/pipeline/scout_report.json`.
4. If network is unavailable, write a failure report and stop. Do not invent paper metadata.

## Command

```bash
python3 .agents/skills/research-scout/scripts/search_papers.py
```

Override state/output location only when explicitly needed:

```bash
PIPELINE_DIR=pipeline python3 .agents/skills/research-scout/scripts/search_papers.py
```

## Ranking Policy

- Prefer papers that address pressure-equilibrium preservation, all-Mach fluxes, ACID/APEC energy consistency, path-conservative alpha source terms, pressure Helmholtz blocks, or positivity-preserving multiphase fluxes.
- Reject ideas that require modifying frozen solver directories unless the user explicitly changes repository rules.
- Rank implementation candidates by relevance to `solver/five_eq_IMEX/`, expected validation impact on 02-A/07-B, and implementation scope.
