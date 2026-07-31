---
description: Known pitfalls and guardrails for the Denner C++ solver and WSL toolchain
paths:
  - "cpp/denner_1d/**"
  - "*.sh"
  - "*.py"
---

# Denner Solver — Lessons Learned

Guardrails from the development history of this C++ Denner solver. These record
failed approaches and toolchain pitfalls so they are not repeated. All items are
verified facts — do not delete and do not retry the documented dead-ends blindly.

## Toolchain (WSL from Windows)

- The WSL login shell prints noise `your NNNN x1 screen size is bogus. expect trouble`
  on init; it pollutes `grep`/`cat`/`tr` output. Do NOT parse WSL command output with
  inline shell pipes for anything fragile — write a `.sh` or `.py` script file and run
  it, or use `2>/dev/null` and match only the exact expected line.
- Inline `for` loops inside `wsl.exe -d ubuntu bash -lc '...'` frequently expand loop
  variables to EMPTY (e.g. `for v in 1 2 3; do echo $v` prints blanks). Put the loop in
  a `.sh` script file instead.
- Nested single/double quotes in `bash -lc '...'` break (especially Python `-c` with
  JSON). Write a script file.
- Shell output redirection `> file` from `wsl.exe` is INTERMITTENTLY unreliable
  (produces 0-byte files). For capturing solver/dump output into Python, use
  `subprocess.run([...], capture_output=True, text=True)` inside a Python script,
  NOT shell redirection.
- Windows-side paths passed to `wsl.exe bash /home/...` get mangled by Git-Bash path
  translation. Wrap the WSL path in single quotes:
  `wsl.exe -d ubuntu bash -lc 'bash /home/.../script.sh'`.

## Build / numerics

- Do NOT add `-march=native` (or FMA-enabling flags): fused multiply-add breaks case01
  machine-exactness (`linf_p` must stay 0). The only measured gain was FMA, which is not
  worth losing bit-exact reproducibility.
- The residual `compute_R` is the single source of truth (defect-correction Newton): an
  approximate/finite-difference Jacobian changes only iteration COUNT, never the
  converged solution. So Jacobian changes are safe to A/B by convergence speed while
  keeping 10/10 byte-identical.
- Per-case knobs live in `SolverConfig` (`types.hpp`): `cfl`, `coupled`, `bdf2`,
  `minmod`, `lowdiss`, `ap_advection`, `dhat_scale`. Prefer per-case flags over global
  changes so other cases stay byte-unchanged.

## Physics / scheme (documented dead-ends — do not retry blindly)

- MWI pressure dissipation scales with dt (`dhat ~ dt`, transient-dominated aP), so SMALL
  time steps UNDER-damp the pressure-velocity coupling at strong shocks (case25
  reflected-shock overshoot). Raising Courant toward Denner's own value, or the
  `ap_advection` lever (Denner Eq.21's own e_P definition — physical), restores damping.
  This is the collocated small-dt checkerboard, consistent with Bartholomew et al.
  (JCP 375, 2018). `dhat_scale` (a bare tuned multiplier) also damps but is a NON-PHYSICAL
  fudge factor — REMOVED from all case defaults by user rule (physical coefficients only);
  it remains only as a research env knob (`ACID_DHK`), never for validation runs.
- Upwinding the face PRESSURE (pface) or the advecting velocity (ubar) at a shock is NOT
  valid — pressure is not advected; it breaks shock speed/position or diverges. Keep the
  conservative central pface.
- case07 residual wake wiggle (~1 Pa) is intrinsic BDF2 time-dispersion, shared by
  Denner's own scheme (he uses BE/BDF2 only, judged by amplitude-only gate). Removing it
  needs an L-stable integrator (TR-BDF2), which our `compute_R` "trans + full-flux"
  structure does not cleanly accept — do not attempt without restructuring the energy
  pressure-work source.
- case15 (double rarefaction) reference is a grid self-consistency test, not exact
  validation; the 4-eq model has no phase change, so the expansion-core pressure hits the
  EOS floor, not a physical vapour pressure. Do not present it as cavitation validation.
