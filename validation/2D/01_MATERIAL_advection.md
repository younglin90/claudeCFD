# 2D-01: Pressure-Equilibrium Material Advection

## Purpose

Verify 2D sharp-interface transport under pressure equilibrium. This is the first 2D alpha-transport, periodic-boundary, pressure-preservation, and plotting check.

## Domain and Grid

```text
x in [0, 1]
y in [0, 1]
Development grid: 32 x 32 or 64 x 64
Default grid: 128 x 128
Boundary: periodic in x and y
```

## Initial Condition

Circular material patch:

```text
r = sqrt(dper(x, xc)^2 + dper(y, yc)^2)
alpha1 = 1 - eps  if r <= R
alpha1 = eps      otherwise
eps = 1.0e-6
xc = 0.30
yc = 0.45
R  = 0.16
```

Uniform primitive variables:

```text
ux = 0.35
uy = 0.20
p  = 1.0e5 Pa
T1 = 300 K
T2 = 300 K
```

EOS options:

```text
Smoke: ideal / ideal with different gamma
Production: NASG water / ideal air after pressure-equilibrium recovery is stable
```

## Final Time

```text
Smoke: t_end = 0.08
Periodic-return option: choose t_end such that ux*t_end and uy*t_end are integer-period displacements
```

## Exact Solution

The exact alpha field is the initial disk translated by:

```text
Delta x = ux * t_end
Delta y = uy * t_end
```

with periodic wrapping. Pressure, temperatures, and velocities remain constant.

## Output Plot

Save as:

```text
results/2D/01_material_advection/diff_vs_exact.png
```

Required panels:

- alpha numerical vs exact
- alpha absolute error
- pressure perturbation `p - p0`
- velocity perturbation magnitude

## PASS Criteria

- `L1(alpha - alpha_exact) <= 2.0e-2` on `64 x 64`.
- `L1(alpha - alpha_exact) <= 1.0e-2` on `128 x 128`.
- `max(abs(p-p0))/p0 <= 1.0e-8` for ideal/ideal smoke.
- No pressure spike or dip at the material interface.
- No checkerboard velocity pattern.
- Phase mass relative error `<= 1.0e-10` for periodic boundaries.
