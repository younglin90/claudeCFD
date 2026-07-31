---
description: WSL execution guardrails
---

## WSL Pitfalls

### (a) Inline arg mangling in `wsl.exe -d ubuntu bash -lc "..."`

NEVER put `$vars`, /tmp paths with vars, or curl `@`-file args inline in
`wsl.exe -d ubuntu bash -lc "..."` — MSYS path-mangling strips them silently.

Symptoms: empty output, `cannot access ''`, command sees wrong args.

ALWAYS Write a script file (e.g. `/tmp/mbq/x.sh`) then run:

```
wsl.exe -d ubuntu bash -lc 'bash /tmp/mbq/x.sh'
```

### (b) 9p stale-build trap

Headers edited over `\\wsl.localhost` UNC may be READ STALE by cmake/make
inside WSL → old binary runs → phantom results.

Guard before trusting any number:
1. `touch` the header inside WSL
2. rebuild
3. `md5sum` the binary and confirm it CHANGED

### (c) "Looks dead" background process

A background process that "looks dead" via an inline-grep check may be an
artifact of (a) — re-check with a script file before declaring it crashed.
