# 2D-11: Linear Rayleigh-Taylor Instability

## Purpose

Verify near-zero-Mach gravity-driven instability growth and hydrostatic balance in the linear regime.

## Domain and Grid

```text
x in [0, 1]
y in [0, 2]
Development grid: 64 x 128
Target grid: 256 x 512
Boundary: periodic in x, reflective/slip in y
```

## Initial Condition

```text
rho_heavy/rho_light = 3
interface y_I = 1 + a0*cos(2*pi*x)
a0 = 0.005 to 0.01
u = 0
hydrostatic pressure initialization
```

## Reference Solution

Compare early-time amplitude growth against linear RTI theory or high-resolution reference.

## Output Plot

```text
results/2D/11_linear_rti/diff_vs_exact.png
```

## PASS Criteria

- Hydrostatic pressure does not generate spurious velocity before growth.
- Growth rate matches reference trend.
- No pressure checkerboard at the interface.
