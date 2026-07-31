---
description: CFD run policy
---

## CFD Run Rules

- **Ask before long validation runs**: propose bench/resolution/variant/expected
  wall and wait for user "go" (short smokes with `CFD_MAXSTEP` are fine without
  asking).

- **NEVER recompute an already-computed scheme/case** — reuse the existing dump
  (user is angry about wasted electricity). Honor explicit scope.

- **EVERY run persists full-state VTK + metrics** (wall, steps, p_min, ens, L1,
  env recipe, cmd) per case into its results dir + a `meta.txt` with the recipe
  and binary md5.

- **Monitor dt every run**: dt collapse / elapsed >> normal / p_min<0 =
  divergence → kill NOW, report.

- **Judge KH rolls VISUALLY** (Read the PNG) not by rollup count; enstrophy is
  the honest scalar.

- **Comparison figures**: only compare same mesh/flux/frame; proposed scheme
  goes rightmost (BASE→VDB→proposed order).

- **Figure upload**: tmpfiles.org ONLY (catbox banned), `curl -4` (IPv6 hangs),
  grep id `[A-Za-z0-9]+`, 3 retries; render + link after EVERY calc.

- **Threads**: paper-res 2D = 24 physical cores (0,2,...,46); measure scaling
  at the REAL problem size before changing; never spread-bind.

- **3D runs ≤15 min smoke by default**; production 3D only when user
  explicitly asks.
