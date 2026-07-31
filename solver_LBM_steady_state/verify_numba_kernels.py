"""Correctness check: njit lbe_step vs original numpy lbe_step for each Case."""

import numpy as np

from lbm_channel import ChannelCase
from lbm_core import LBMCavity
from lbm_couette import CouetteCase
from lbm_periodic import KolmogorovCase
from lbm_plbe_cavity import PLBECavity
from lbm_voxel import VoxelCase, build_cylinder_mask

import numba_kernels as nk


def _diff(a, b):
    return float(np.max(np.abs(a - b)))


def main():
    cases = []
    cases.append(("KolmogorovCase", KolmogorovCase(N=16, nu=0.05, F0=1e-4, kf=1), nk.kolmogorov_step))
    cases.append(("ChannelCase",    ChannelCase(N=16, nu=0.05, F0=1e-5),         nk.channel_step))
    cases.append(("CouetteCase",    CouetteCase(N=16, nu=0.05, U_wall=0.05),     nk.couette_step))
    cases.append(("LBMCavity",      LBMCavity(N=17, Re=100, U_wall=0.1),         nk.cavity_step))
    chi = np.ones((16, 16))
    chi *= build_cylinder_mask(16, 8, 8, 3)
    cases.append(("VoxelCase",      VoxelCase(chi, nu=0.05, F0=1e-5, kf=0),      nk.voxel_step))
    cases.append(("PLBECavity",     PLBECavity(N=17, Re=1000, U_wall=0.01, gamma=0.25), nk.plbe_step))

    for name, case, njit_step in cases:
        orig_step = type(case).lbe_step  # current binding (still original; patch not applied)
        f0 = case.initial_field()
        f0 = f0 + 1e-3 * np.random.RandomState(0).standard_normal(f0.shape)
        # original numpy step
        f_orig = orig_step(case, f0.copy())
        # njit step via bound method form
        f_jit = njit_step(case, f0.copy())
        d = _diff(f_orig, f_jit)
        flag = "PASS" if d < 1e-10 else f"DIFF={d:.2e}"
        print(f"  {name:18s} max|orig - njit| = {d:.3e}  {flag}")


if __name__ == "__main__":
    main()
