# 2D-10: Haas-Sturtevant / Quirk-Karni Shock-Bubble Interaction

## Purpose

Reference shock-bubble validation for all-Mach compressible multiphase flow. This strengthens `2D-04` with a named literature target.

## Domain and Grid

```text
x in [0, 0.325]
y in [0, 0.089]
Development grid: 325 x 89 or reduced smoke grid
Target grid: 650 x 178
```

## Initial Condition

Gas cylinder or bubble in air:

```text
bubble center = (0.225, 0.0445)
R = 0.025
shock Mach Ms = 1.22
```

Use helium, R22, or SF6 bubble according to the selected reference.

## Output Plot

```text
results/2D/10_shock_bubble_haas_sturtevant/diff_vs_exact.png
```

## PASS Criteria

- Shock location error `<= 2 dx` against high-resolution/reference.
- Bubble centroid and deformation trend match reference.
- No nonphysical pressure spike at the interface.
