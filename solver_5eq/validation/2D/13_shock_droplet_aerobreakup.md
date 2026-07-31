# 2D-13: Shock-Droplet Aerobreakup Smoke Validation

## Purpose

Verify strong gas-liquid density-ratio interface deformation under shock/supersonic gas loading. This is a severe all-speed sharp-interface test.

## Domain and Grid

```text
x in [0, 1]
y in [0, 0.4]
Development grid: 200 x 80
Target grid: 800 x 320
Boundary: transmissive in x, slip/reflective in y
```

## Initial Condition

```text
droplet center = (0.35, 0.2)
R = 0.06
left post-shock/high-speed gas state
right pre-shock gas state
liquid-like droplet density ratio O(100-1000)
```

## Reference Solution

Compare against high-resolution reference or literature aerobreakup morphology.

## Output Plot

```text
results/2D/13_shock_droplet_aerobreakup/diff_vs_exact.png
```

## PASS Criteria

- No negative pressure/density/temperature.
- Shock and droplet deformation remain physically plausible.
- No severe interface pressure spike or checkerboard.
