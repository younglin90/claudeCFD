"""Factory list for force-free benchmark cases."""

from __future__ import annotations

from lbm_core import LBMCavity
from lbm_couette import CouetteCase

from no_force_suite.no_force_lb_core import (
    NoForceChannelCase,
    NoForceMaskedCase,
    NoForcePoiseuilleRectCase,
    NoForceTJunctionRectCase,
)
from no_force_suite.no_force_masks import (
    make_backward_step_mask,
    make_cylinder_wake_mask,
    make_multi_cylinder_mask,
    make_t_junction_mask,
    make_t_junction_rect_mask,
)


SUPPORTED_CASES = {
    "channel_n32": ("Plane Poiseuille (periodic x, wall y, force-free)", 32, 1.0e-7, 0),
    "channel_poiseuille_rect": ("Plane Poiseuille inlet/outlet rectangular channel", 32, 1.0e-7, 0),
    "couette_n32": ("Couette flow N=32", 32, 1.0e-7, 0),
    "cavity_re100_n33": ("Lid-driven cavity Re=100 N=33", 33, 5.0e-7, 0),
    "cavity_re400_n49": ("Lid-driven cavity Re=400 N=49", 49, 5.0e-7, 0),
    "cavity_re1000_n129": ("Lid-driven cavity Re=1000 N=129", 129, 5.0e-7, 0),
    "multi_cylinder_n32": ("Multi-cylinder masked flow", 32, 1.0e-7, 0),
    "backward_step_n64": ("Backward-facing step", 64, 1.0e-7, 0),
    "cylinder_wake_n64": ("Cylinder wake analogue", 64, 1.0e-7, 0),
    "t_junction_n64": ("T-junction", 64, 1.0e-7, 0),
    "t_junction_rect": ("True inlet/outlet T-junction rectangular", 128, 1.0e-7, 0),
}


def make_case(case_id: str):
    if case_id == "channel_n32":
        # Force-free periodic-x channel case.
        # Note: no inflow/outflow BC in x; initial condition controls the target momentum.
        return NoForceChannelCase(32, nu=0.05, U_in=0.05, x_bc="periodic", initial_profile="poiseuille")

    if case_id == "channel_poiseuille_rect":
        return NoForcePoiseuilleRectCase(Ny=32, Nx=192, nu=0.05, U_in=0.05, initial_profile="poiseuille")

    if case_id == "couette_n32":
        return CouetteCase(32, nu=0.05, U_wall=0.05)

    if case_id == "cavity_re100_n33":
        return LBMCavity(N=33, Re=100, U_wall=0.1)

    if case_id == "cavity_re400_n49":
        return LBMCavity(N=49, Re=400, U_wall=0.1)

    if case_id == "cavity_re1000_n129":
        return LBMCavity(N=129, Re=1000, U_wall=0.1)

    if case_id == "multi_cylinder_n32":
        chi = make_multi_cylinder_mask(32)
        return NoForceMaskedCase(chi, nu=0.05, U_in=0.05)

    if case_id == "backward_step_n64":
        chi = make_backward_step_mask(64)
        return NoForceMaskedCase(chi, nu=0.05, U_in=0.05)

    if case_id == "cylinder_wake_n64":
        chi = make_cylinder_wake_mask(64)
        return NoForceMaskedCase(chi, nu=0.04, U_in=0.05)

    if case_id == "t_junction_n64":
        chi = make_t_junction_mask(64)
        return NoForceMaskedCase(chi, nu=0.05, U_in=0.05)

    if case_id == "t_junction_rect":
        chi = make_t_junction_rect_mask(128, 192, 32)
        return NoForceTJunctionRectCase(chi, nu=0.05, U_in=0.04)

    raise ValueError(f"unsupported case_id: {case_id}")


def supported_case_ids():
    return tuple(SUPPORTED_CASES.keys())
