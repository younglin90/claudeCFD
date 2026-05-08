# 2D-07: Low-Mach Liquid-Gas Alternating Bands

## Purpose

Verify low-Mach liquid-gas interface transport in a periodic channel with alternating bands. This case targets numerical diffusion, pressure uniformity, and low-Mach robustness for repeated interfaces.

## Domain and Grid

```text
x in [0, 1]
y in [0, 1]
Development grid: 96 x 32
Target grid: 192 x 64
Boundary: periodic in x and y
```

## Initial Condition

Alternating vertical material bands:

```text
alpha1 = 1 - eps for floor(4*x) even
alpha1 = eps     for floor(4*x) odd
eps = 1e-6
p = p0
ux = Ma*c_ref
uy = 0
Ma = 1e-3
```

Thermal/density contrast may be imposed through EOS temperatures.

## Exact Solution

Bands translate periodically in the x direction. Pressure remains uniform.

## Output Plot

```text
results/2D/07_lm_liquid_gas_bands/diff_vs_exact.png
```

## PASS Criteria

- Band location error remains within `1 dx`.
- `max(abs(p-p0))/p0 <= 1e-8` in pressure-equilibrium smoke mode.
- No checkerboard pressure/velocity.
- Interface diffusion is confined to a few grid cells.
