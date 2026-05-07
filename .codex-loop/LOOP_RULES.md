# Codex Loop Rules

## Goal

Run a bounded repair loop that fixes failing validation gates with minimal,
numerically consistent changes.

## Numerical Invariants

- Preserve conservative flux form for mass, momentum, and total energy.
- Do not replace conservative divergence with split form unless explicitly justified.
- Do not loosen test tolerances to hide instability.
- Do not remove physical source terms without a discrete residual argument.
- For IMEX changes, explain explicit/implicit operator consistency.
- For PE-related changes, report the impact on `p_U dot L` or the amplification/eigenmode gates.
- Keep `solver/He2024/`, legacy solver folders, `validation/`, and other frozen paths unchanged unless the user explicitly approves otherwise.

## Allowed Default Scope

- `solver/five_eq_IMEX/`
- `tests/`
- `results/` active drivers and diagnostics
- `docs/`
- `.codex-loop/`
- `.agents/`

## Forbidden Behavior

- Do not run `git reset --hard`, `git clean -fdx`, force push, or destructive recursive deletes.
- Do not modify tests only to make failures disappear.
- Do not broaden a fix into a large refactor unless the loop stops and asks for human review.
- Do not use external network access unless the configured sandbox and user policy explicitly allow it.

## Mandatory Gates

Default test command:

```bash
python3 .agents/skills/benchmark-validate/scripts/run_and_compare.py
```

Core gates inside that command:

- `tests/test_uniform_flow.py`
- `tests/test_amplification_matrix.py`
- `tests/test_transport_eigenmode.py`
- `results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01`

07-B can be included as a diagnostic gate, but failure there is not yet a global stop condition unless the loop objective says so.

## Stop Conditions

- Tests pass.
- Same normalized failure repeats 3 times.
- One iteration changes more than the configured line limit.
- Codex produces no changes.
- Codex exits nonzero and no actionable diff is produced.
- Maximum iteration count is reached.

## State Updates

After each attempt, update `.codex-loop/LOOP_STATE.md` with:

- hypothesis
- files changed
- test result
- remaining suspected cause
- next recommended action
