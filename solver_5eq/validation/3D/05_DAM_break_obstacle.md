# 3D-05: Dam Break With Obstacle

## Purpose

Verify air-water large-density-ratio free-surface flow with gravity, wall boundary conditions, obstacle impact, pressure trend, and long-time robustness.

## Domain and Grid

Standard dam-break-with-obstacle geometry:

```text
domain: 3.22 m x 1.00 m x 1.00 m
x: streamwise
y: spanwise
z: vertical
```

Development grid:

```text
Nx x Ny x Nz = 161 x 50 x 50
```

Target/reference grid:

```text
Nx x Ny x Nz ~= 282 x 100 x 83 or finer
```

## Initial Condition

Water column:

```text
0 <= x <= 1.228 m
0 <= z <= 0.55 m
alpha_water = 1 - eps inside water column
alpha_water = eps outside
eps = 1.0e-6
ux = uy = uz = 0
p = hydrostatic water/air pressure
T_water = 300 K
T_air = 300 K
```

Obstacle:

```text
size = 0.16 m x 0.40 m x 0.16 m
attached to floor
spanwise centered
streamwise location chosen to match reference probe layout
```

Gravity:

```text
g_z = -9.81 m/s^2
```

## Boundary Conditions

```text
bottom and obstacle: no-penetration wall
side walls: slip wall
top: atmospheric pressure outlet or open boundary
downstream wall: wall or outlet according to reference setup
```

## Final Time

```text
t_end = 6.0 s
```

## Reference Solution

Compare to experimental or high-resolution reference using:

- water-height probes
- pressure probes on/near the obstacle
- free-surface snapshots

Recommended probes:

```text
H2 near x = 2.22 m, y = 0.5 m
H4 near x = 0.56 m, y = 0.5 m
P3 before obstacle
P7 above/on obstacle
```

Exact analytical solution is not expected.

## Output Plot

Save as:

```text
results/3D/05_dam_break_obstacle/diff_vs_exact.png
```

Required panels:

- free-surface snapshots
- water height at H2/H4 vs reference
- pressure at P3/P7 vs reference
- phase mass history

## PASS Criteria

- No negative pressure, density, or temperature.
- Alpha remains bounded.
- Water mass loss only through open-boundary flux if open boundaries are used.
- Arrival time at obstacle within `5%` of reference.
- First pressure peak time within `5%` of reference.
- Pressure peak magnitude trend qualitatively consistent; quantitative target should be set against a chosen high-resolution reference.
