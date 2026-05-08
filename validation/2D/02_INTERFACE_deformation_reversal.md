# 2D-02: Deformation-Reversal Interface Transport

## Purpose

Verify interface sharpness under strong stretching and final-time recovery. This is the primary 2D VOF/sharp-interface test before shock interaction.

## Domain and Grid

```text
x in [0, 1]
y in [0, 1]
Development grid: 128 x 128
Default grid: 128 x 128
Target grid: 256 x 256
Boundary: periodic
```

## Initial Condition

```text
alpha1 = 1 - eps inside radius R=0.15 centered at (0.5, 0.75)
alpha1 = eps outside
eps = 1.0e-6
p = 1.0e5 Pa
T1 = 300 K
T2 = 300 K
```

## Prescribed Velocity Field

Use the divergence-free single-vortex deformation field:

```text
ux(x,y,t) =  sin(pi*x)^2 * sin(2*pi*y) * cos(pi*t/T)
uy(x,y,t) = -sin(pi*y)^2 * sin(2*pi*x) * cos(pi*t/T)
```

Recommended:

```text
Development: T = 4.0
Target/literature-strength: T = 8.0
t_end = T
```

At `t=T/2`, the interface is maximally stretched. At `t=T`, the exact alpha field returns to the initial field.

## Exact Solution

No closed-form pointwise field is required during deformation. At final time, the exact solution is the initial alpha field.

## Output Plot

Save as:

```text
results/2D/02_deformation_reversal/diff_vs_exact.png
```

Required panels:

- alpha at `t=0`
- alpha at `t=T/2`
- alpha numerical vs exact at `t=T`
- final alpha absolute error

## PASS Criteria

- Final `L1(alpha-alpha_initial) <= 4.0e-2` on `128 x 128`.
- Final `L1(alpha-alpha_initial) <= 2.5e-2` on `256 x 256`.
- Interface must not fragment into detached nonphysical islands.
- `min(alpha1) >= -1.0e-12`, `max(alpha1) <= 1 + 1.0e-12`.
- Phase mass relative error `<= 1.0e-8`.
