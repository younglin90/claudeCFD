"""2D flow past cylinder benchmark.

Geometry : periodic-x + periodic-y + interior cylinder (voxelized).
Drive : constant body force F_x. Steady state at low Re.

Validation : drag coefficient C_d (Henderson 1995) :
    Re=20 :  Cd ≈ 2.05
    Re=40 :  Cd ≈ 1.54
"""

import numpy as np
from lbm_voxel import VoxelCase, build_cylinder_mask


def make_cylinder_case(N=64, D=12, Re=20, U_target=0.05, calibrate=True):
    """Single cylinder voxel case with adaptive body-force calibration.

    Without calibration : F0 = U_target · nu · (2π/N)²  (Poiseuille-approx, often off by 2-5x)
    With calibration : pre-runs baseline LBM, adjusts F0 to achieve target Re.

    Returns case with F0 such that converged U_mean·D/nu ≈ Re (within ~10%).
    """
    nu = U_target * D / Re
    F0_init = U_target * nu * (2 * np.pi / N) ** 2
    chi = build_cylinder_mask(N, cx=N // 2, cy=N // 2, radius=D // 2)
    case = VoxelCase(chi, nu=nu, F0=F0_init, kf=0)
    case._cyl_D = D
    case._cyl_U_target = U_target
    case._cyl_Re = Re

    if not calibrate:
        return case

    # Iterative damped calibration : 3 stages, dampen scale by 0.5 each iter
    from solver_scmk import solve_baseline_periodic
    cur_case = case
    for stage in range(3):
        f, _ = solve_baseline_periodic(cur_case, max_steps=3000, tol=1e-3,
                                         check_every=300, verbose=False)
        rho, ux, _ = cur_case.macro(f)
        fluid_mask = cur_case.chi > 0.5
        U_meas = float(np.mean(np.abs(ux[fluid_mask])))
        if U_meas < 1e-10 or not np.isfinite(U_meas):
            break
        # damped update : geometric mean
        raw_scale = U_target / U_meas
        damped_scale = raw_scale ** 0.5  # square-root damping
        new_F0 = cur_case.F0 * damped_scale
        if not np.isfinite(new_F0) or new_F0 <= 0:
            break
        cur_case = VoxelCase(chi, nu=nu, F0=new_F0, kf=0)
    cur_case._cyl_D = D
    cur_case._cyl_U_target = U_target
    cur_case._cyl_Re = Re
    cur_case._cyl_calibrated = True
    return cur_case


def compute_drag(case, f):
    """Estimate drag coefficient from steady velocity field.

    Cd = 2 · F_drag / (rho · U² · D)
    F_drag estimated from body-force balance : F0 · A_fluid = drag (in steady).
    Mean velocity from u_x average over fluid cells.
    """
    rho, ux, uy = case.macro(f)
    fluid_mask = case.chi > 0.5
    U_mean = float(np.mean(ux[fluid_mask]))
    rho_mean = float(np.mean(rho[fluid_mask]))
    # Drag from force balance in steady periodic flow :
    # F0 · A_fluid = F_drag  (total body force balanced by cylinder drag)
    A_fluid = float(np.sum(fluid_mask))
    A_solid = float(np.sum(~fluid_mask))
    F_drag = case.F0 * A_fluid
    Cd = 2.0 * F_drag / (rho_mean * U_mean ** 2 * case._cyl_D + 1e-30)
    return Cd, U_mean, U_mean * case._cyl_D / case.nu  # Cd, U_avg, Re_actual
