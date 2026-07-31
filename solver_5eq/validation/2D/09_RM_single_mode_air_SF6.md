# 2D-09: Single-Mode Richtmyer-Meshkov Instability, Air/SF6

## Purpose

Verify shock interaction with a sinusoidally perturbed material interface and suppress spurious pressure oscillations for variable density/EOS interfaces.

## Domain and Grid

```text
x in [0, 4]
y in [-0.5, 0.5]
Development grid: 160 x 40
Target grid: 640 x 160
Boundary: transmissive in x, periodic in y
```

## Initial Condition

```text
interface: x_I(y) = 2.0 + a0*cos(2*pi*y/lambda)
a0 = 0.05
lambda = 1.0
left/right materials: air and SF6-like gas
incident shock Mach: Ms ~= 1.24
```

## Reference Solution

Use published Richtmyer-Meshkov growth data or a high-resolution same-solver reference.

## Output Plot

```text
results/2D/09_rm_single_mode/diff_vs_exact.png
```

## PASS Criteria

- Shock position error `<= 2 dx` against reference.
- Interface amplitude growth trend matches reference.
- No material-interface pressure spike before shock impact.
- No negative thermodynamic state.
