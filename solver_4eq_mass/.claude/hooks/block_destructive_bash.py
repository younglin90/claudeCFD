#!/usr/bin/env python3
"""PreToolUse gate for the Bash tool during the YADV round loop.

Same state-file gating as agent_plan_only.py (see that file's docstring for why
an env var doesn't work across hook subprocess invocations). Blocks the
destructive / externally-visible command classes the round loop must never run
unattended: pushing to a remote, force-resetting history, and rm -rf. This is a
backstop, not the primary safety mechanism -- the round-loop skill itself is
expected to never issue these commands; this hook exists in case a round drifts
after many iterations.

Matching is done on the command with heredoc BODIES stripped first (see
strip_heredocs): a `git commit -m "$(cat <<'EOF' ... EOF)"` whose message text
merely *mentions* "git push" (e.g. describing this very hook) must not trip the
gate -- only the heredoc content is stripped, so an actual `git push` typed
outside a heredoc still matches. Found live: the first real commit this hook
guarded blocked itself on its own commit message describing what it blocks.
"""
import json
import os
import re
import sys

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "round-loop-active")

HEREDOC_START_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")


def strip_heredocs(command: str) -> str:
    """Remove heredoc BODY text (between the `<<[-]MARKER` line and the lone
    `MARKER` closing line) so quoted prose inside commit messages etc. can't
    trigger the deny patterns. The `<<MARKER` token itself is kept so a
    genuine `git push <<EOF` (nonsensical, but hypothetically dangerous)
    would still be visible to the matcher."""
    lines = command.split("\n")
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        m = HEREDOC_START_RE.search(line)
        if m:
            out.append(line)
            marker = m.group(2)
            i += 1
            while i < n and lines[i].strip() != marker:
                i += 1
            i += 1  # skip the closing marker line itself
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


DENY_PATTERNS = [
    (re.compile(r"\bgit\s+push\b"),
     "git push (remote reflect) is blocked during the round loop -- local commits only; "
     "the user pushes manually when ready."),
    (re.compile(r"\bgit\s+reset\s+--hard\b"),
     "git reset --hard is blocked during the round loop."),
    (re.compile(r"\brm\s+-rf\b"),
     "rm -rf is blocked during the round loop."),
]


def main() -> None:
    if not os.path.exists(STATE_FILE):
        sys.exit(0)  # loop not active -> allow everything

    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail open: never block on a hook-side parse bug

    raw_command = (payload.get("tool_input") or {}).get("command") or ""
    command = strip_heredocs(raw_command)

    for pattern, reason in DENY_PATTERNS:
        if pattern.search(command):
            result = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"YADV round loop safety net: {reason} "
                        f"Command was: {command!r}. Delete .claude/round-loop-active "
                        "to lift this gate."
                    ),
                }
            }
            print(json.dumps(result))
            sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
