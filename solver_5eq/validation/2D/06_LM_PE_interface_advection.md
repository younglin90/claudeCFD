# 2D-06: Ultra-Low-Mach Pressure-Equilibrium Interface Advection

## Purpose

Verify all-Mach behavior at ultra-low Mach number for a moving material interface. The primary failure modes are pressure checkerboard, spurious velocity, and pressure spikes at density/EOS jumps.

## Domain and Grid

```text
x in [0, 1]
y in [0, 1]
Development grid: 64 x 64
Target grid: 128 x 128
Boundary: periodic in x and y
```

## Initial Condition

Oblique or circular material interface:

```text
alpha1 = 1 - eps in material 1
alpha1 = eps     in material 2
eps = 1e-6
p = p0
T1 = T2 = 300 K
```

Velocity is chosen from the target Mach number:

```text
Ma = 1e-4 to 1e-3
ux = Ma * c_ref
uy = 0.5 * Ma * c_ref
```

## Exact Solution

The interface is translated periodically by `(ux*t, uy*t)`. Pressure and velocity remain spatially uniform.

## Output Plot

```text
results/2D/06_lm_pe_interface_advection/diff_vs_exact.png
```

Required panels:

- alpha numerical vs exact
- pressure perturbation `p-p0`
- velocity perturbation magnitude

## PASS Criteria

- `max(abs(p-p0))/p0 <= 1e-8`.
- `max(|u-u0|)/max(|u0|,1) <= 1e-8`.
- No pressure checkerboard near the interface.
- Phase mass relative error `<= 1e-10`.
