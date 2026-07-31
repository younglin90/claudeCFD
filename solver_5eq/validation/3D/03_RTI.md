# 3D-03: 3D Rayleigh-Taylor Instability

## Purpose

Verify 3D gravity-driven instability growth, hydrostatic balance, and large-density-ratio interface dynamics.

## Domain and Grid

```text
x in [0, L]
y in [0, L]
z in [0, 4L]
L = 1
Development grid: 64 x 64 x 256
Target grid: 128 x 128 x 512
```

Boundary conditions:

```text
x/y: periodic
z: slip/no-penetration or wall, matching reference
```

## Initial Condition

Heavy fluid above light fluid:

```text
rho_heavy / rho_light = 3
Atwood = 0.5
g = -1 in z direction
```

Perturbed interface around `z=2L`:

```text
eta(x,y) = 0.05*L*(cos(2*pi*x/L) + cos(2*pi*y/L))
z_I(x,y) = 2L + eta(x,y)
alpha_heavy = 1 - eps if z > z_I
alpha_heavy = eps     if z < z_I
eps = 1.0e-6
ux = uy = uz = 0
```

Initialize pressure hydrostatically.

## Final Time

```text
t_end = 4.0 nondimensional
```

## Reference Solution

Compare bubble, spike, and saddle positions against high-resolution reference or literature values.

## Output Plot

Save as:

```text
results/3D/03_rti/diff_vs_exact.png
```

Required panels:

- alpha isosurfaces at multiple times
- bubble/spike/saddle position history
- pressure perturbation slices
- phase mass history

## PASS Criteria

- Initial hydrostatic equilibrium should not produce spurious velocity above `1.0e-6` before instability growth.
- Bubble, spike, and saddle positions within `2 dx` to `3 dx` of reference.
- No pressure checkerboard at the interface.
- No negative thermodynamic state.
- Phase mass relative error `<= 1.0e-8`.
