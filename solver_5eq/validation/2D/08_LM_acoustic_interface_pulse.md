# 2D-08: Low-Mach Acoustic Pulse Through Material Interface

## Purpose

Verify low-amplitude acoustic propagation through a material interface. This is the 2D extension of the 1D acoustic reflection/transmission tests.

## Domain and Grid

```text
x in [0, 1.5]
y in [0, 0.5]
Development grid: 160 x 48
Target grid: 320 x 96
Boundary: transmissive in x, periodic or slip in y
```

## Initial Condition

```text
material interface at x = 0.75
p = p0 + dp * exp(-((x-x0)^2 + (y-y0)^2)/sigma^2)
dp/p0 = 1e-5 to 1e-4
u = acoustic-consistent small perturbation or zero for smoke
```

## Reference Solution

Compare against a high-resolution 2D solver result or a 1D linear-acoustic projection along the centerline.

## Output Plot

```text
results/2D/08_lm_acoustic_interface_pulse/diff_vs_exact.png
```

## PASS Criteria

- Pulse position and amplitude agree with reference within prescribed tolerance.
- No high-frequency pressure oscillation at the interface.
- No spurious transverse velocity checkerboard.
