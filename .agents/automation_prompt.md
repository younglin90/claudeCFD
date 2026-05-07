# Codex Automation Prompt

Use this prompt for a scheduled or manually triggered research loop.

```text
Run one bounded claudeCFD research-improvement cycle.

1. Use $research-scout to update .agents/pipeline/scout_report.json.
2. Use $research-improve to select exactly one implementable idea.
3. Write .agents/pipeline/sprint_contract.md before editing code.
4. Implement only within allowed paths from AGENTS.md.
5. Use $benchmark-validate to run mandatory gates.
6. Append the result to .agents/pipeline/cycle_log.md.

Stop immediately if 02-A or BE1 amplification regresses. Do not use destructive rollback commands.
```

For bounded self-repair of failing gates, use:

```text
python3 .codex-loop/loop.py --allow-dirty --reset-state --max-iters 5
```

Use `--check-only` to validate the configured test command without invoking Codex.
