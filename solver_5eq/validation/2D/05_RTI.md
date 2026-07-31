# 2D-05: Rayleigh-Taylor Instability

## Purpose

Verify gravity source coupling, hydrostatic pressure balance, sharp-interface large-density-ratio evolution, and long-time nonlinear stability.

## Domain and Grid

```text
x in [0, 1]
y in [0, 2]
Development grid: 64 x 128
Default grid: 128 x 256
Target grid: 256 x 512
```

Boundary conditions:

```text
x: periodic
y: slip wall or reflective wall
```

## Initial Condition

Heavy fluid over light fluid:

```text
rho_heavy / rho_light = 3
Atwood = 0.5
g = -1 in y direction
```

Perturbed interface:

```text
y_I(x) = 1.0 + 0.05*cos(2*pi*x)
alpha_heavy = 1 - eps if y > y_I
alpha_heavy = eps     if y < y_I
eps = 1.0e-6
ux = uy = 0
```

Pressure must be initialized hydrostatically:

```text
dp/dy = -rho*g_y
```

## Final Time

```text
Development: t_end = 0.5 nondimensional
Target/reference: t_end = 3.0 to 5.0 nondimensional
```

## Reference Solution

Compare bubble and spike locations against high-resolution reference or literature Atwood-0.5 RTI data.

## Output Plot

Save as:

```text
results/2D/05_rti/diff_vs_exact.png
```

Required panels:

- alpha contours over time
- density field
- pressure perturbation from hydrostatic state
- bubble/spike position history

## PASS Criteria

- Initial hydrostatic state must not generate spurious velocity above `1.0e-6` before perturbation growth.
- Bubble and spike positions within `2 dx` of high-resolution reference.
- No pressure checkerboard near the interface.
- No alpha overshoot/undershoot beyond `1.0e-10`.
- Heavy and light phase masses conserved to `1.0e-8` in the closed domain.
