# 2D-12: Linear Kelvin-Helmholtz Instability

## Purpose

Verify subsonic shear-layer interface dynamics and low-Mach pressure stability.

## Domain and Grid

```text
x in [0, 1]
y in [0, 1]
Development grid: 128 x 128
Target grid: 256 x 256
Boundary: periodic in x, slip or periodic in y
```

## Initial Condition

Two layers separated by a perturbed interface:

```text
y_I = 0.5 + a0*sin(2*pi*x)
a0 = 0.01
ux = +U0 above interface
ux = -U0 below interface
Ma = U0 / c_ref < 0.1
p = p0
```

## Reference Solution

Compare early-time interface amplitude growth with linear KH theory or high-resolution reference.

## Output Plot

```text
results/2D/12_linear_kh/diff_vs_exact.png
```

## PASS Criteria

- KH roll-up/growth trend appears without pressure checkerboard.
- No unphysical pressure spike at the material interface.
- Interface thickness remains controlled.
