# 3D-02: 3D Deformation-Reversal Interface Transport

## Purpose

Verify 3D interface deformation, mass conservation, and recovery under a strong time-reversed velocity field.

## Domain and Grid

```text
x in [0, 1]
y in [0, 1]
z in [0, 1]
Development grid: 48^3 or 64^3
Default grid: 128^3
Boundary: periodic or slip according to the velocity-field definition
```

## Initial Condition

```text
sphere center = (0.5, 0.75, 0.25)
R = 0.15
alpha1 = 1 - eps inside sphere
alpha1 = eps outside
eps = 1.0e-6
p = constant
T1 = constant
T2 = constant
```

## Prescribed Velocity Field

Use the Liovic/Rider-Kothe 3D deformation field:

```text
ux = -sin(2*pi*y)*sin(pi*x)^2*cos(pi*t/T)
uy =  sin(2*pi*x)*sin(pi*y)^2*cos(pi*t/T)
r  = sqrt((x-0.5)^2 + (y-0.5)^2)
uz = Umax*(1 - r/0.5)^2*cos(pi*t/T)
```

Use `uz=0` outside `r>0.5` if needed.

Recommended:

```text
T = 3.0
Umax = 1.0
t_end = T
```

## Exact Solution

At final time, exact alpha returns to the initial alpha field.

## Output Plot

Save as:

```text
results/3D/02_deformation_reversal/diff_vs_exact.png
```

Required panels:

- alpha isosurface at `t=0`
- alpha isosurface at `t=T/2`
- alpha mid-plane numerical vs exact at `t=T`
- final alpha absolute error slice

## PASS Criteria

- Final `L1(alpha-alpha_initial) <= 4.0e-2` on `64^3`.
- Final `L1(alpha-alpha_initial) <= 2.5e-2` on `128^3`.
- Phase mass relative error `<= 1.0e-8`.
- No severe filament breakup into disconnected numerical droplets.
