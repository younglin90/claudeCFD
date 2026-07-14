# Multiphase 2-D/3-D Validation Specifications

This document defines the recommended 2-D and 3-D validation cases for the
`solver/five_eq_IMEX` multidimensional extension.  The goal is to progress from
clean interface transport to compressible shock-interface interaction and then
gravity-driven/free-surface tests.

The frozen `validation/` tree is not modified.  When these cases mature, they
can be copied into `validation/2D/` and `validation/3D/` after explicit approval.

## Common Numerical Requirements

- Use the same solver configuration for all cases unless the case explicitly
  isolates a source term.
- Use second-order or higher spatial reconstruction for primitive variables.
- Use `T-MLP-u + TVD limiter` or an equivalent bounded high-order reconstruction.
- Use a sharp-interface alpha scheme path for material interfaces.
- Avoid Rusanov as the primary material/advection flux.
- Preserve `0 <= alpha_k <= 1` and `sum(alpha_k)=1`.
- Preserve positive `p`, `rho_k`, `T_k`.
- Save plots only as:
  - `results/2D/{case_name}/diff_vs_exact.png`
  - `results/3D/{case_name}/diff_vs_exact.png`

Unless otherwise stated, use:

- phase 1: heavy/liquid-like material
- phase 2: light/gas-like material
- alpha variable: `alpha1`
- primitive state:
  - 2-D: `W=(alpha1,T1,T2,ux,uy,p)`
  - 3-D: `W=(alpha1,T1,T2,ux,uy,uz,p)`

## Recommended Execution Order

2-D:

1. `2D_01_material_advection`
2. `2D_02_deformation_reversal`
3. `2D_03_pressure_equilibrium_interface_advection`
4. `2D_04_shock_bubble`
5. `2D_05_rayleigh_taylor`

3-D:

1. `3D_01_sphere_advection`
2. `3D_02_deformation_reversal`
3. `3D_03_rayleigh_taylor`
4. `3D_04_shock_sphere`
5. `3D_05_dam_break_obstacle`

# 2-D Cases

## `2D_01_material_advection`

### Purpose

Verify 2-D sharp-interface transport under pressure equilibrium.  This is the
first multidimensional alpha, primitive reconstruction, periodic-boundary, and
plot-contract check.

### Domain

- `x in [0,1]`
- `y in [0,1]`
- default grid: `N=128 x 128`
- development grid: `N=32 x 32` or `N=64 x 64`
- boundary: periodic in `x` and `y`

### Initial Condition

Circular material patch:

```text
r = sqrt(dper(x, xc)^2 + dper(y, yc)^2)
alpha1 = 1 - eps    if r <= R
alpha1 = eps        otherwise
eps = 1e-6
xc = 0.30
yc = 0.45
R  = 0.16
```

Uniform primitive variables:

```text
ux = 0.35
uy = 0.20
p  = 1.0e5 Pa
T1 = 300 K
T2 = 300 K
```

EOS options:

- smoke: ideal/ideal with different `gamma`
- production: NASG water / ideal air if pressure-equilibrium recovery is stable

### Final Time

Use one of:

- short smoke: `t_end = 0.08`
- full periodic return: choose `t_end` such that `(ux*t_end, uy*t_end)` is an
  integer-period displacement if possible.

### Exact Solution

The exact alpha field is the initial disk translated by:

```text
Delta x = ux * t_end
Delta y = uy * t_end
```

with periodic wrapping.  `p`, `T1`, `T2`, `ux`, `uy` should remain constant.

### Required Plots

- alpha numerical vs exact
- alpha absolute error
- pressure perturbation `p - p0`
- velocity perturbation magnitude

### PASS Criteria

- `L1(alpha - alpha_exact) <= 2.0e-2` on `N=64`.
- `L1(alpha - alpha_exact) <= 1.0e-2` on `N=128`.
- `max(abs(p-p0))/p0 <= 1.0e-8` for ideal/ideal pressure-equilibrium smoke.
- No pressure spike at the interface.
- No checkerboard velocity pattern.
- Phase mass relative error `<= 1.0e-10` for periodic boundaries.

## `2D_02_deformation_reversal`

### Purpose

Verify interface sharpness under strong stretching and recovery.  This is the
primary 2-D VOF/sharp-interface test before shock interaction.

### Domain

- `x in [0,1]`
- `y in [0,1]`
- default grid: `N=128 x 128`
- target grid: `N=256 x 256`
- boundary: periodic

### Initial Condition

Circular patch:

```text
alpha1 = 1 - eps inside radius R=0.15 centered at (0.5,0.75)
alpha1 = eps outside
eps = 1e-6
p = 1.0e5 Pa
T1 = T2 = 300 K
```

### Prescribed Velocity Field

Use the divergence-free single-vortex deformation field:

```text
ux(x,y,t) =  sin(pi*x)^2 * sin(2*pi*y) * cos(pi*t/T)
uy(x,y,t) = -sin(pi*y)^2 * sin(2*pi*x) * cos(pi*t/T)
```

Recommended:

```text
T = 8.0
t_end = T
```

At `t=T/2`, the interface is maximally stretched.  At `t=T`, the exact alpha
field returns to the initial field.

### Exact Solution

No pointwise closed-form field is required during deformation.  At final time,
the exact solution is the initial alpha field.

### Required Plots

- alpha at `t=0`
- alpha at `t=T/2`
- alpha numerical vs exact at `t=T`
- final alpha absolute error

### PASS Criteria

- Final `L1(alpha-alpha_initial) <= 4.0e-2` on `N=128`.
- Final `L1(alpha-alpha_initial) <= 2.5e-2` on `N=256`.
- Interface must not fragment into detached nonphysical islands.
- `min(alpha1) >= -1e-12`, `max(alpha1) <= 1+1e-12`.
- Phase mass relative error `<= 1.0e-8`.

## `2D_03_pressure_equilibrium_interface_advection`

### Purpose

Verify pressure-equilibrium preservation across a moving material interface with
large density/EOS contrast.  This is the multidimensional analogue of the 1-D
pressure-equilibrium advection cases.

### Domain

- `x in [0,1]`
- `y in [0,1]`
- default grid: `N=128 x 128`
- boundary: periodic

### Initial Condition

Oblique material interface:

```text
phi(x,y) = x + y - 0.9
alpha1 = 1 - eps if phi < 0
alpha1 = eps     otherwise
eps = 1e-6
```

Uniform pressure and velocity:

```text
p  = 1.0e5 Pa
ux = 0.20
uy = 0.10
```

Thermal states may differ by phase:

```text
T1 = 300 K
T2 = 300 K
```

EOS:

- phase 1: NASG water or stiffened-gas water
- phase 2: ideal air

### Final Time

```text
t_end = 1.0
```

The interface translates periodically by `(0.2,0.1)`.

### Exact Solution

The material interface is exactly translated with periodic wrapping.  Pressure
and velocity remain uniform.

### Required Plots

- alpha numerical vs exact
- pressure perturbation
- velocity perturbation
- density numerical vs exact mixture density

### PASS Criteria

- `max(abs(p-p0))/p0 <= 1.0e-6` for NASG/ideal.
- `max(sqrt((ux-ux0)^2+(uy-uy0)^2)) <= 1.0e-7`.
- No pressure spike or dip at the interface.
- `L1(alpha-alpha_exact) <= 2.0e-2` on `N=128`.
- Mixture density error localized only to the smeared interface.

## `2D_04_shock_bubble`

### Purpose

Verify compressible shock-interface interaction, shock speed, baroclinic
vorticity generation, and suppression of nonphysical pressure oscillations.

### Domain

Dimensionless shock-bubble setup:

- `x in [0,0.325]`
- `y in [0,0.089]`
- default grid: `650 x 178` for production
- development grid: `325 x 89`
- boundary:
  - left: post-shock inflow or transmissive after shock insertion
  - right: transmissive
  - top/bottom: slip wall or transmissive, depending on reference setup

### Materials

- phase 1: dense bubble gas, e.g. R22/SF6-like ideal gas
- phase 2: air

Default nondimensional ideal-gas data:

```text
air:        gamma = 1.4,   rho = 1.0,    p = 1.0
bubble gas: gamma = 1.249, rho = 3.1538, p = 1.0
```

### Initial Bubble

```text
center = (0.225, 0.0445)
R      = 0.025
alpha1 = 1 - eps inside bubble
alpha1 = eps outside
eps = 1e-6
```

### Shock

Planar Mach `M_s = 1.22` shock in air, initially at:

```text
x_s = 0.05
```

For `gamma=1.4`, nondimensional pre-shock air:

```text
rho_R = 1.0
p_R   = 1.0
u_R   = 0.0
```

Post-shock left state:

```text
p_L/rho_R = 1 + 2*gamma/(gamma+1)*(M_s^2 - 1)
rho_L/rho_R = ((gamma+1)*M_s^2)/((gamma-1)*M_s^2 + 2)
u_L = 2*c_R/(gamma+1)*(M_s^2 - 1)/M_s
c_R = sqrt(gamma*p_R/rho_R)
```

### Final Time

Run until the transmitted/reflected waves are clearly separated from the bubble:

```text
t_end = 0.20 to 0.30 nondimensional
```

Use the same nondimensionalization as the reference solution.

### Reference Solution

No exact closed-form solution.  Compare against:

- high-resolution same-solver run
- published shock-bubble morphology
- shock position and bubble centroid histories

### Required Plots

- density Schlieren-like gradient
- alpha contour
- pressure field
- vorticity field
- centerline pressure/density cuts

### PASS Criteria

- No nonphysical pressure oscillation at the material interface before shock arrival.
- Shock position error `<= 1.5 dx` against high-resolution reference.
- Bubble centroid error `<= 2.0 dx` against high-resolution reference.
- No negative pressure, density, or temperature.
- Interface remains bounded and no isolated checkerboard alpha cells appear.

## `2D_05_rayleigh_taylor`

### Purpose

Verify gravity source coupling, hydrostatic pressure balance, sharp-interface
large-density-ratio evolution, and long-time nonlinear stability.

### Domain

- `x in [0,1]`
- `y in [0,2]`
- default grid: `128 x 256`
- target grid: `256 x 512`
- boundary:
  - x: periodic
  - y: slip wall or reflective wall

### Initial Condition

Heavy fluid over light fluid:

```text
rho_heavy/rho_light = 3
Atwood = 0.5
g = -1 in y direction
```

Perturbed interface:

```text
y_I(x) = 1.0 + 0.05*cos(2*pi*x)
alpha_heavy = 1 - eps if y > y_I
alpha_heavy = eps     if y < y_I
eps = 1e-6
ux = uy = 0
```

Pressure must be initialized hydrostatically:

```text
dp/dy = -rho*g_y
```

### Final Time

Run to:

```text
t_end = 3.0 to 5.0 nondimensional
```

### Reference Solution

Compare bubble and spike locations against high-resolution reference.  If
available, compare against literature Atwood-0.5 RTI data.

### Required Plots

- alpha contours over time
- density field
- pressure perturbation from hydrostatic state
- bubble/spike position history

### PASS Criteria

- Initial hydrostatic state must not generate spurious velocity above `1e-6`
  before the perturbation grows.
- Bubble and spike positions within `2 dx` of high-resolution reference.
- No pressure checkerboard near the interface.
- No alpha overshoot/undershoot beyond `1e-10`.
- Heavy and light phase masses conserved to `1e-8` in closed domain.

# 3-D Cases

## `3D_01_sphere_advection`

### Purpose

Verify 3-D sharp-interface transport and pressure-equilibrium preservation.

### Domain

- `x,y,z in [0,1]`
- default grid: `64 x 64 x 64`
- development grid: `24 x 24 x 24`
- boundary: periodic in all directions

### Initial Condition

Spherical material patch:

```text
center = (0.35, 0.45, 0.55)
R = 0.18
alpha1 = 1 - eps inside sphere
alpha1 = eps outside
eps = 1e-6
ux = 0.25
uy = 0.15
uz = 0.10
p  = 1.0e5 Pa
T1 = T2 = 300 K
```

### Final Time

```text
t_end = 0.05 for smoke
t_end = 1.0 for longer periodic advection if desired
```

### Exact Solution

Sphere translated by `(ux*t, uy*t, uz*t)` with periodic wrapping.

### Required Plots

- mid-plane alpha numerical vs exact
- mid-plane alpha absolute error
- pressure perturbation slice
- integrated phase mass history

### PASS Criteria

- `L1(alpha-alpha_exact) <= 2.5e-2` on `64^3`.
- Pressure perturbation `max(abs(p-p0))/p0 <= 1e-8` for ideal/ideal smoke.
- Phase mass relative error `<= 1e-10`.
- No checkerboard velocity pattern.

## `3D_02_deformation_reversal`

### Purpose

Verify 3-D interface deformation, mass conservation, and recovery under a
strong time-reversed velocity field.

### Domain

- case A: `x,y,z in [0,1]`
- default grid: `128^3`
- development grid: `48^3` or `64^3`
- boundary: periodic or slip according to velocity-field definition

### Initial Condition

```text
sphere center = (0.5, 0.75, 0.25)
R = 0.15
alpha1 = 1 - eps inside sphere
alpha1 = eps outside
p = constant
T1 = T2 = constant
```

### Prescribed Velocity Field

Use the Liovic/Rider-Kothe 3-D deformation field:

```text
ux = -sin(2*pi*y)*sin(pi*x)^2*cos(pi*t/T)
uy =  sin(2*pi*x)*sin(pi*y)^2*cos(pi*t/T)
r  = sqrt((x-0.5)^2 + (y-0.5)^2)
uz = Umax*(1 - r/0.5)^2*cos(pi*t/T)
```

Use `uz=0` outside `r>0.5` if needed to avoid nonphysical values.

Recommended:

```text
T = 3.0
Umax = 1.0
t_end = T
```

### Exact Solution

At final time, exact alpha returns to initial alpha.

### Required Plots

- alpha isosurface at `t=0`
- alpha isosurface at `t=T/2`
- alpha mid-plane numerical vs exact at `t=T`
- final alpha absolute error slice

### PASS Criteria

- Final `L1(alpha-alpha_initial) <= 4.0e-2` on `64^3`.
- Final `L1(alpha-alpha_initial) <= 2.5e-2` on `128^3`.
- Phase mass relative error `<= 1e-8`.
- No severe filament breakup into disconnected numerical droplets.

## `3D_03_rayleigh_taylor`

### Purpose

Verify 3-D gravity-driven instability growth, hydrostatic balance, and
large-density-ratio interface dynamics.

### Domain

- `x in [0,L]`
- `y in [0,L]`
- `z in [0,4L]`
- default: `L=1`
- development grid: `64 x 64 x 256`
- target grid: `128 x 128 x 512`
- boundary:
  - x/y: periodic
  - z: slip/no-penetration or wall, matching reference

### Initial Condition

Heavy fluid above light fluid:

```text
rho_heavy/rho_light = 3
Atwood = 0.5
g = -1 in z direction
```

Perturbed interface around `z=2L`:

```text
eta(x,y) = 0.05*L*(cos(2*pi*x/L) + cos(2*pi*y/L))
z_I(x,y) = 2L + eta(x,y)
alpha_heavy = 1 - eps if z > z_I
alpha_heavy = eps     if z < z_I
eps = 1e-6
u = v = w = 0
```

Initialize pressure hydrostatically.

### Final Time

```text
t_end = 4.0 nondimensional
```

### Reference Solution

Compare bubble, spike, and saddle positions against high-resolution reference or
literature values.

### Required Plots

- alpha isosurfaces at multiple times
- bubble/spike/saddle position history
- pressure perturbation slices
- phase mass history

### PASS Criteria

- Initial hydrostatic equilibrium should not produce spurious velocity above
  `1e-6` before instability growth.
- Bubble, spike, and saddle positions within `2 dx` to `3 dx` of reference.
- No pressure checkerboard at the interface.
- No negative thermodynamic state.
- Phase mass relative error `<= 1e-8`.

## `3D_04_shock_sphere`

### Purpose

Verify fully 3-D compressible shock-interface interaction, shock deformation,
baroclinic vorticity generation, and robust large-density-ratio behavior.

### Domain

Use a 3-D extension of the 2-D shock-bubble setup:

- `x in [0,0.325]`
- `y in [0,0.089]`
- `z in [0,0.089]`
- development grid: `325 x 89 x 89`
- target grid: at least `650 x 178 x 178` if feasible
- boundary:
  - left: post-shock inflow or transmissive after shock insertion
  - right: transmissive
  - y/z: slip wall or transmissive according to reference

### Initial Condition

Spherical bubble:

```text
center = (0.225, 0.0445, 0.0445)
R = 0.025
alpha_bubble = 1 - eps inside sphere
alpha_bubble = eps outside
eps = 1e-6
```

Air shock:

```text
M_s = 1.22
x_s = 0.05
```

Use the same pre-shock/post-shock formulas as `2D_04_shock_bubble`.

### Final Time

```text
t_end = 0.20 to 0.30 nondimensional
```

### Reference Solution

No exact solution.  Compare against:

- high-resolution same-solver reference
- literature shock-sphere morphology
- shock position and bubble centroid

### Required Plots

- density/pressure slices through the sphere center
- alpha isosurface
- vorticity magnitude slice
- shock position history

### PASS Criteria

- Shock position error `<= 2 dx` against high-resolution reference.
- Bubble centroid error `<= 3 dx`.
- No nonphysical pressure spike at material interface before shock impact.
- No negative pressure, density, or temperature.
- Alpha remains bounded.

## `3D_05_dam_break_obstacle`

### Purpose

Verify air-water large-density-ratio free-surface flow with gravity, wall
boundary conditions, obstacle impact, pressure trend, and long-time robustness.

### Domain

Use the standard dam-break-with-obstacle geometry:

```text
domain: 3.22 m x 1.00 m x 1.00 m
x: streamwise
y: spanwise
z: vertical
```

Default development grid:

```text
Nx x Ny x Nz = 161 x 50 x 50
```

Target/reference grid:

```text
Nx x Ny x Nz ~= 282 x 100 x 83 or finer
```

### Initial Condition

Water column:

```text
0 <= x <= 1.228 m
0 <= z <= 0.55 m
alpha_water = 1 - eps
else alpha_water = eps
eps = 1e-6
u = v = w = 0
p = hydrostatic water/air pressure
T_water = T_air = 300 K
```

Obstacle:

```text
size = 0.16 m x 0.40 m x 0.16 m
attached to floor
spanwise centered
streamwise location chosen to match reference probe layout
```

Gravity:

```text
g_z = -9.81 m/s^2
```

### Boundary Conditions

- bottom and obstacle: no-penetration wall
- side walls: slip wall
- top: atmospheric pressure outlet or open boundary
- downstream wall: wall or outlet according to reference setup

### Final Time

```text
t_end = 6.0 s
```

### Reference Solution

Compare to experimental/high-resolution reference using:

- water-height probes
- pressure probes on/near the obstacle
- free-surface snapshots

Recommended probes:

```text
H2 near x = 2.22 m, y = 0.5 m
H4 near x = 0.56 m, y = 0.5 m
P3 before obstacle
P7 above/on obstacle
```

Exact analytical solution is not expected.

### Required Plots

- free-surface snapshots
- water height at H2/H4 vs reference
- pressure at P3/P7 vs reference
- phase mass history

### PASS Criteria

- No negative pressure, density, or temperature.
- Alpha remains bounded.
- Water mass loss due only to open-boundary flux if any.
- Arrival time at obstacle within `5%` of reference.
- First pressure peak time within `5%` of reference.
- Pressure peak magnitude trend qualitatively consistent; quantitative target
  should be set against a chosen high-resolution reference.

# Promotion Criteria

A 2-D/3-D case can be promoted from development to official validation when:

- the setup script is deterministic;
- `diff_vs_exact.png` or reference-comparison plot is always overwritten;
- numerical metrics are written to a machine-readable summary;
- at least one low-resolution CI-safe version exists;
- at least one high-resolution reference run exists for non-exact cases;
- the same solver method is used across the promoted cases unless the case is
  explicitly designed to isolate a source term.
