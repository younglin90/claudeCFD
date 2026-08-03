#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
for f in 13_E_shocktube_hp_air_lp_water 14_E_shocktube_hp_water_lp_air 24_H_hypersonic_mixture_ms10; do
  echo "=== $f ==="
  grep -iE "출처|denner|courant|co ?=|cfl|time.?step|7\.[0-9]" "validation/1D/$f.md" | head -5
done
