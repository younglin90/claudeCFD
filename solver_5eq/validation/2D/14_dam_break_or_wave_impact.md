# 2D-14: Dam-Break / Wave-Impact Low-Mach Free-Surface Limit

## Purpose

Verify nearly incompressible low-Mach free-surface behavior with gravity, large density ratio, and wall boundaries.

## Domain and Grid

```text
x in [0, 3.22]
y in [0, 1.0]
Development grid: 160 x 50
Target grid: 320 x 100
Boundary: reflective walls or open top according to reference
```

## Initial Condition

```text
water column: x <= 1.228, y <= 0.55
alpha_water = 1 - eps inside water
alpha_water = eps outside
u = 0
p = hydrostatic
T_water = T_air = 300 K
gravity downward
```

## Reference Solution

Compare water-front position and pressure/water-height probes against dam-break reference data when available.

## Output Plot

```text
results/2D/14_dam_break_or_wave_impact/diff_vs_exact.png
```

## PASS Criteria

- No negative pressure/density/temperature.
- Alpha remains bounded.
- Water front advances in the physically correct direction.
- Phase mass changes only through open boundaries.
