#!/usr/bin/env python3
"""PreToolUse gate for the Agent tool during the YADV round loop.

Gated by a STATE FILE (../round-loop-active), not an env var: hooks are spawned
as fresh subprocesses by the Claude Code harness and do not inherit ad-hoc shell
exports set via the Bash tool (Bash tool state does not persist between calls
either). A file on disk is the only thing both the round-loop skill and every
hook invocation can reliably see and toggle.

When the state file is absent: allow everything (inert, no-op outside the loop).
When present: only Agent calls with subagent_type == "Plan" are allowed; every
other subagent_type is denied. This mechanically enforces "the round loop
implements directly in-session; only the Planner may run as a subagent."
"""
import json
import os
import sys

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "round-loop-active")


def main() -> None:
    if not os.path.exists(STATE_FILE):
        sys.exit(0)  # loop not active -> allow everything

    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail open: never block on a hook-side parse bug

    subagent_type = (payload.get("tool_input") or {}).get("subagent_type")
    if subagent_type == "Plan":
        sys.exit(0)  # allow

    result = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "YADV round loop is active (.claude/round-loop-active present): "
                "only Agent(subagent_type='Plan') is allowed. Got "
                f"subagent_type={subagent_type!r}. Implementation must happen "
                "directly in this session -- delete .claude/round-loop-active "
                "to lift this gate."
            ),
        }
    }
    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
