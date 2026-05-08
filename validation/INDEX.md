# Validation Index

Validation specifications for the five-equation multiphase solver.

| Dimension | Index | Notes |
|---|---|---|
| 1D | `1D/INDEX.md` | Existing 1D validation suite |
| 2D | `2D/INDEX.md` | New multidimensional interface, shock, and RTI validation specs |
| 3D | `3D/INDEX.md` | New 3D interface, shock, RTI, and dam-break validation specs |

Plot rule:

- 1D: `results/1D/{case_name}/diff_vs_exact.png`
- 2D: `results/2D/{case_name}/diff_vs_exact.png`
- 3D: `results/3D/{case_name}/diff_vs_exact.png`

All validation drivers should overwrite `diff_vs_exact.png`; no round-specific plot names.
