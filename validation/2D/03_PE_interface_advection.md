# 2D-03: Pressure-Equilibrium Interface Advection

## Purpose

Verify pressure-equilibrium preservation across a moving material interface with large density and EOS contrast. This is the 2D analogue of the 1D pressure-equilibrium advection tests.

## Domain and Grid

```text
x in [0, 1]
y in [0, 1]
Development grid: 80 x 80
Default grid: 128 x 128
Boundary: periodic
```

## Initial Condition

Oblique material interface:

```text
phi(x,y) = x + y - 0.9
alpha1 = 1 - eps  if phi < 0
alpha1 = eps      otherwise
eps = 1.0e-6
```

Uniform primitive variables:

```text
p  = 1.0e5 Pa
ux = 0.20
uy = 0.10
T1 = 300 K
T2 = 300 K
```

EOS:

```text
phase 1: NASG water or stiffened-gas water
phase 2: ideal air
```

## Final Time

```text
t_end = 1.0
```

The interface translates periodically by `(0.2, 0.1)`.

## Exact Solution

The material interface is exactly translated with periodic wrapping. Pressure and velocity remain uniform.

## Output Plot

Save as:

```text
results/2D/03_pe_interface_advection/diff_vs_exact.png
```

Required panels:

- alpha numerical vs exact
- pressure perturbation
- velocity perturbation magnitude
- mixture density numerical vs exact

## PASS Criteria

- `max(abs(p-p0))/p0 <= 1.0e-6` for NASG/ideal.
- `max(sqrt((ux-ux0)^2 + (uy-uy0)^2)) <= 1.0e-7`.
- No pressure spike or dip at the material interface.
- `L1(alpha-alpha_exact) <= 2.0e-2` on `128 x 128`.
- Mixture density error is localized only to the smeared interface.
