# 3D Lid-Driven Cavity Reference

Source: TNO report ECN-E--11-042, Figure 20, using Albensoeder and Kuhlmann
Chebyshev-collocation profile points for the cubical 3D lid-driven cavity.

Current digitized file:

- `albensoeder_kuhlmann_fig20_digitized.csv`

Reference convention used by the diagnostic tests:

- Reynolds number: `Re=1000`
- Physical domain in the report: `[-0.5, 0.5]^3`
- Code sampling domain: shifted to `[0, 1]^3`
- Lid plane: report `x=-0.5`, represented as code `x=0`
- Lid velocity: report `v=1`, represented as code `uy=1`
- Profile (a): `v` along the x-axis, stored as `x_center, uy`
- Profile (b): `u` along the y-axis, stored as `y_center, ux`

Status:

- The CSV is a digitized diagnostic reference, not a final Leg 1 pass gate.
- A Leg 1 cavity pass still requires the live `ctest` output and CSV logs to show
  a relative L2 error <= 0.02 against this trusted reference while all prior 2D and
  3D non-regression tests remain green.
