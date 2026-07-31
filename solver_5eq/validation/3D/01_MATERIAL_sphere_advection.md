# 3D-01: Pressure-Equilibrium Sphere Advection

## Purpose

Verify 3D sharp-interface transport and pressure-equilibrium preservation.

## Domain and Grid

```text
x in [0, 1]
y in [0, 1]
z in [0, 1]
Development grid: 24 x 24 x 24
Default grid: 64 x 64 x 64
Boundary: periodic in all directions
```

## Initial Condition

Spherical material patch:

```text
center = (0.35, 0.45, 0.55)
R = 0.18
alpha1 = 1 - eps inside sphere
alpha1 = eps outside
eps = 1.0e-6
ux = 0.25
uy = 0.15
uz = 0.10
p  = 1.0e5 Pa
T1 = 300 K
T2 = 300 K
```

## Final Time

```text
Smoke: t_end = 0.05
Longer periodic advection: t_end = 1.0 if desired
```

## Exact Solution

The sphere is translated by `(ux*t, uy*t, uz*t)` with periodic wrapping. Pressure, temperatures, and velocities remain constant.

## Output Plot

Save as:

```text
results/3D/01_sphere_advection/diff_vs_exact.png
```

Required panels:

- mid-plane alpha numerical vs exact
- mid-plane alpha absolute error
- pressure perturbation slice
- integrated phase mass history

## PASS Criteria

- `L1(alpha-alpha_exact) <= 2.5e-2` on `64^3`.
- Pressure perturbation `max(abs(p-p0))/p0 <= 1.0e-8` for ideal/ideal smoke.
- Phase mass relative error `<= 1.0e-10`.
- No checkerboard velocity pattern.
