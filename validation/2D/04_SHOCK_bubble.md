# 2D-04: Shock-Bubble Interaction

## Purpose

Verify compressible shock-interface interaction, shock speed, baroclinic vorticity generation, and suppression of nonphysical pressure oscillations.

## Domain and Grid

Dimensionless shock-bubble setup:

```text
x in [0, 0.325]
y in [0, 0.089]
Development grid: 325 x 89
Production grid: 650 x 178
```

Boundary conditions:

```text
left: post-shock inflow or transmissive after shock insertion
right: transmissive
top/bottom: slip wall or transmissive, matching the chosen reference setup
```

## Materials

Default nondimensional ideal-gas data:

```text
air:        gamma = 1.4,   rho = 1.0,    p = 1.0
bubble gas: gamma = 1.249, rho = 3.1538, p = 1.0
```

## Initial Bubble

```text
center = (0.225, 0.0445)
R      = 0.025
alpha_bubble = 1 - eps inside bubble
alpha_bubble = eps outside
eps = 1.0e-6
```

## Initial Shock

Planar Mach `M_s = 1.22` shock in air, initially at:

```text
x_s = 0.05
```

For `gamma=1.4`, nondimensional pre-shock air:

```text
rho_R = 1.0
p_R   = 1.0
u_R   = 0.0
```

Post-shock left state:

```text
p_L/p_R       = 1 + 2*gamma/(gamma+1)*(M_s^2 - 1)
rho_L/rho_R   = ((gamma+1)*M_s^2)/((gamma-1)*M_s^2 + 2)
u_L           = 2*c_R/(gamma+1)*(M_s^2 - 1)/M_s
c_R           = sqrt(gamma*p_R/rho_R)
```

## Final Time

```text
Development: t_end = 0.14 nondimensional
Target/reference: t_end = 0.20 to 0.30 nondimensional
```

Run until transmitted/reflected waves are clearly separated from the bubble.

## Reference Solution

No exact closed-form solution. Compare against:

- high-resolution same-solver run
- published shock-bubble morphology
- shock position and bubble centroid histories

## Output Plot

Save as:

```text
results/2D/04_shock_bubble/diff_vs_exact.png
```

Required panels:

- density or density-gradient Schlieren-like field
- alpha contour
- pressure field
- vorticity field
- centerline pressure/density cuts

## PASS Criteria

- No nonphysical pressure oscillation at the material interface before shock arrival.
- Shock position error `<= 1.5 dx` against high-resolution reference.
- Bubble centroid error `<= 2.0 dx` against high-resolution reference.
- No negative pressure, density, or temperature.
- Interface remains bounded and no isolated checkerboard alpha cells appear.
