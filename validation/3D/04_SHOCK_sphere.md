# 3D-04: Shock-Sphere Interaction

## Purpose

Verify fully 3D compressible shock-interface interaction, shock deformation, baroclinic vorticity generation, and robust large-density-ratio behavior.

## Domain and Grid

3D extension of the shock-bubble setup:

```text
x in [0, 0.325]
y in [0, 0.089]
z in [0, 0.089]
Development grid: 325 x 89 x 89
Target grid: 650 x 178 x 178 or finer if feasible
```

Boundary conditions:

```text
left: post-shock inflow or transmissive after shock insertion
right: transmissive
y/z: slip wall or transmissive according to reference
```

## Initial Condition

Spherical bubble:

```text
center = (0.225, 0.0445, 0.0445)
R = 0.025
alpha_bubble = 1 - eps inside sphere
alpha_bubble = eps outside
eps = 1.0e-6
```

Air shock:

```text
M_s = 1.22
x_s = 0.05
```

Use the same pre-shock/post-shock formulas as `validation/2D/04_SHOCK_bubble.md`.

## Final Time

```text
t_end = 0.20 to 0.30 nondimensional
```

## Reference Solution

No exact solution. Compare against:

- high-resolution same-solver reference
- literature shock-sphere morphology
- shock position and bubble centroid

## Output Plot

Save as:

```text
results/3D/04_shock_sphere/diff_vs_exact.png
```

Required panels:

- density/pressure slices through the sphere center
- alpha isosurface
- vorticity magnitude slice
- shock position history

## PASS Criteria

- Shock position error `<= 2 dx` against high-resolution reference.
- Bubble centroid error `<= 3 dx`.
- No nonphysical pressure spike at the material interface before shock impact.
- No negative pressure, density, or temperature.
- Alpha remains bounded.
