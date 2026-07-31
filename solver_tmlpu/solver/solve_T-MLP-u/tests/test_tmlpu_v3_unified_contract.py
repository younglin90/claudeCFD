from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
bench = importlib.import_module("test_2d_tmlpu_paper_benchmarks")


@pytest.fixture(autouse=True)
def _clean_recon_env(monkeypatch):
    for key in (
        "TMLPU_COMMON_RECON_KEY",
        "TMLPU_INCLUDE_DIAGNOSTIC_BASELINES",
    ):
        monkeypatch.delenv(key, raising=False)


def test_official_comparison_is_mlp_u1_vs_single_tmlpu_key():
    for case in ("leveque", "double_mach", "mach3_step"):
        assert bench._comparison_specs(case) == [
            ("MLP-u1", "mlp_u1"),
            ("T-MLP-u ON", "tmlpu_v3_unified_on"),
        ]


def test_legacy_official_keys_alias_to_unified_reconstruction():
    canonical = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    assert canonical.name == "t_mlp_u_unified"
    for key in (
        "tmlpu_leveque_on",
        "tmlpu_double_mach_on",
        "tmlpu_mach3_step_on",
    ):
        recon = bench._reconstruction_from_key(key)
        assert recon.name == canonical.name
        assert type(recon.scalar_recon) is type(canonical.scalar_recon)
        assert type(recon.euler_recon) is type(canonical.euler_recon)


def test_v4_candidate_key_is_available_without_changing_official_aliases():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v4 = bench._reconstruction_from_key("tmlpu_v4_unified_on")
    assert v4.name == "t_mlp_u_unified"
    assert type(v4.scalar_recon) is type(v3.scalar_recon)
    assert type(v4.euler_recon) is type(v3.euler_recon)
    assert v4.euler_recon is not v3.euler_recon
    assert v4.euler_recon.tmlpu_bound_tvd_separate is True
    assert v4.euler_recon.euler_face_positivity_limiter is True
    assert v4.euler_recon.face_gradient_correction == "beta_shock_shear"


def test_v4_1_candidate_key_relaxes_tangential_contact_branch_only():
    v4 = bench._reconstruction_from_key("tmlpu_v4_unified_on")
    v41 = bench._reconstruction_from_key("tmlpu_v4_1_unified_on")
    assert v41.name == "t_mlp_u_unified"
    assert type(v41.scalar_recon) is type(v4.scalar_recon)
    assert type(v41.euler_recon) is type(v4.euler_recon)
    assert v41.euler_recon.euler_tangential_velocity_tvd == (
        "shear_superbee_root_blend")
    assert v41.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v41.euler_recon.tmlpu_bound_tvd_separate is True
    assert v41.euler_recon.euler_face_positivity_limiter is True


def test_v4_2_candidate_key_enables_density_contact_weak_face_only():
    v41 = bench._reconstruction_from_key("tmlpu_v4_1_unified_on")
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    assert v42.name == "t_mlp_u_unified"
    assert type(v42.scalar_recon) is type(v41.scalar_recon)
    assert type(v42.euler_recon) is type(v41.euler_recon)
    assert v42.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v42.euler_recon.weak_face_mlp is False
    assert v42.euler_recon.euler_tangential_velocity_tvd == (
        v41.euler_recon.euler_tangential_velocity_tvd)
    assert v42.euler_recon.euler_face_positivity_limiter is True
    assert v42.euler_recon.face_gradient_shock_damping in ("", "off", None)
    assert (
        v42.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is False)
    assert (
        v42.euler_recon
        .euler_density_contact_weak_face_entropy_accept is False)


def test_v4_3_candidate_caps_density_weak_face_and_strengthens_shock_damping():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v43 = bench._reconstruction_from_key("tmlpu_v4_3_unified_on")
    assert v43.name == "t_mlp_u_unified"
    assert type(v43.scalar_recon) is type(v42.scalar_recon)
    assert type(v43.euler_recon) is type(v42.euler_recon)
    assert v43.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v43.euler_recon.euler_density_contact_weak_face_mlp_cap == 0.55
    assert v43.euler_recon.euler_density_contact_weak_face_shock_power == 2.0
    assert v43.euler_recon.face_gradient_shock_damping == "density_strong"
    assert v43.euler_recon.euler_tangential_velocity_tvd == (
        v42.euler_recon.euler_tangential_velocity_tvd)


def test_v5_candidate_enables_capped_face_local_density_contact_bvd():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v5 = bench._reconstruction_from_key("tmlpu_v5_unified_on")
    assert v5.name == "t_mlp_u_unified"
    assert type(v5.scalar_recon) is type(v42.scalar_recon)
    assert type(v5.euler_recon) is type(v42.euler_recon)
    assert v5.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v5.euler_recon.euler_density_contact_bvd is True
    assert v5.euler_recon.euler_density_contact_bvd_cap == 0.30
    assert v5.euler_recon.euler_density_contact_cell_bvd is False
    assert v5.euler_recon.face_gradient_shock_damping == (
        v42.euler_recon.face_gradient_shock_damping)


def test_v6_candidate_boosts_clean_density_contact_transport_without_bvd():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v6 = bench._reconstruction_from_key("tmlpu_v6_unified_on")
    assert v6.name == "t_mlp_u_unified"
    assert type(v6.scalar_recon) is type(v42.scalar_recon)
    assert type(v6.euler_recon) is type(v42.euler_recon)
    assert v6.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v6.euler_recon.euler_density_contact_bvd is False
    assert v6.euler_recon.euler_density_contact_cell_bvd is False
    assert v6.euler_recon.euler_density_contact_hancock_boost == 0.14
    assert v6.euler_recon.euler_density_contact_hancock_boost_cap == 0.85
    assert v6.euler_recon.euler_tangential_velocity_tvd == (
        v42.euler_recon.euler_tangential_velocity_tvd)


def test_v7_candidate_relaxes_density_lsq_on_clean_contacts_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v7 = bench._reconstruction_from_key("tmlpu_v7_unified_on")
    assert v7.name == "t_mlp_u_unified"
    assert type(v7.scalar_recon) is type(v42.scalar_recon)
    assert type(v7.euler_recon) is type(v42.euler_recon)
    assert v7.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v7.euler_recon.euler_density_contact_bvd is False
    assert v7.euler_recon.euler_density_contact_cell_bvd is False
    assert v7.euler_recon.euler_density_contact_hancock_boost == 0.0
    assert v7.euler_recon.euler_density_contact_lsq_root_blend == 0.45
    assert v7.euler_recon.euler_density_contact_lsq_root_blend_cap == 0.80
    assert v7.euler_recon.euler_tangential_velocity_tvd == (
        v42.euler_recon.euler_tangential_velocity_tvd)


def test_v8_candidate_boosts_guarded_density_weak_face_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v8 = bench._reconstruction_from_key("tmlpu_v8_unified_on")
    assert v8.name == "t_mlp_u_unified"
    assert type(v8.scalar_recon) is type(v42.scalar_recon)
    assert type(v8.euler_recon) is type(v42.euler_recon)
    assert v8.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v8.euler_recon.euler_density_contact_bvd is False
    assert v8.euler_recon.euler_density_contact_cell_bvd is False
    assert v8.euler_recon.euler_density_contact_hancock_boost == 0.0
    assert v8.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v8.euler_recon.euler_density_contact_weak_face_mlp_cap == 0.75
    assert v8.euler_recon.euler_density_contact_weak_face_root_blend == 0.18
    assert v8.euler_recon.euler_tangential_velocity_tvd == (
        v42.euler_recon.euler_tangential_velocity_tvd)


def test_v9_candidate_adds_guarded_density_lsq_shear_floor_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v9 = bench._reconstruction_from_key("tmlpu_v9_unified_on")
    assert v9.name == "t_mlp_u_unified"
    assert type(v9.scalar_recon) is type(v42.scalar_recon)
    assert type(v9.euler_recon) is type(v42.euler_recon)
    assert v9.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v9.euler_recon.euler_density_contact_bvd is False
    assert v9.euler_recon.euler_density_contact_cell_bvd is False
    assert v9.euler_recon.euler_density_contact_hancock_boost == 0.0
    assert v9.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v9.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v9.euler_recon.euler_density_contact_lsq_shear_floor == 0.22
    assert v9.euler_recon.euler_density_contact_lsq_shear_floor_cap == 0.35
    assert v9.euler_recon.euler_tangential_velocity_tvd == (
        v42.euler_recon.euler_tangential_velocity_tvd)


def test_v10_candidate_adds_swirl_gated_density_weak_face_extra_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v10 = bench._reconstruction_from_key("tmlpu_v10_unified_on")
    assert v10.name == "t_mlp_u_unified"
    assert type(v10.scalar_recon) is type(v42.scalar_recon)
    assert type(v10.euler_recon) is type(v42.euler_recon)
    assert v10.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v10.euler_recon.euler_density_contact_bvd is False
    assert v10.euler_recon.euler_density_contact_cell_bvd is False
    assert v10.euler_recon.euler_density_contact_hancock_boost == 0.0
    assert v10.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v10.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v10.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v10.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.10
    assert v10.euler_recon.euler_tangential_velocity_tvd == (
        v42.euler_recon.euler_tangential_velocity_tvd)


def test_v11_candidate_adds_tangential_micro_preservation_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v11 = bench._reconstruction_from_key("tmlpu_v11_unified_on")
    assert v11.name == "t_mlp_u_unified"
    assert type(v11.scalar_recon) is type(v42.scalar_recon)
    assert type(v11.euler_recon) is type(v42.euler_recon)
    assert v11.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v11.euler_recon.euler_density_contact_bvd is False
    assert v11.euler_recon.euler_density_contact_cell_bvd is False
    assert v11.euler_recon.euler_density_contact_hancock_boost == 0.0
    assert v11.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v11.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v11.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v11.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v11.euler_recon.euler_tangential_velocity_tvd == (
        "shear_superbee_root_micro")
    assert v11.euler_recon._tangential_velocity_tvd_name == (
        "shear_superbee_root_micro")
    assert v11.euler_recon.euler_tangential_shear_micro_blend == 0.06
    assert v11.euler_recon.euler_tangential_shear_micro_cap == 0.18


def test_v12_candidate_adds_bounded_tangential_mood_guard_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v12 = bench._reconstruction_from_key("tmlpu_v12_unified_on")
    assert v12.name == "t_mlp_u_unified"
    assert type(v12.scalar_recon) is type(v42.scalar_recon)
    assert type(v12.euler_recon) is type(v42.euler_recon)
    assert v12.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v12.euler_recon.euler_density_contact_bvd is False
    assert v12.euler_recon.euler_density_contact_cell_bvd is False
    assert v12.euler_recon.euler_density_contact_hancock_boost == 0.0
    assert v12.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v12.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v12.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v12.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v12.euler_recon.euler_tangential_velocity_tvd == (
        "shear_superbee_root_mood")
    assert v12.euler_recon._tangential_velocity_tvd_name == (
        "shear_superbee_root_mood")
    assert v12.euler_recon.euler_tangential_shear_micro_blend == 0.025
    assert v12.euler_recon.euler_tangential_shear_micro_cap == 0.06
    assert v12.euler_recon.euler_tangential_mood_wavespeed_growth_cap == 0.015
    assert v12.euler_recon.euler_tangential_mood_jump_growth_cap == 0.05


def test_v13_candidate_adds_density_weak_face_admissibility_damping_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v13 = bench._reconstruction_from_key("tmlpu_v13_unified_on")
    assert v13.name == "t_mlp_u_unified"
    assert type(v13.scalar_recon) is type(v42.scalar_recon)
    assert type(v13.euler_recon) is type(v42.euler_recon)
    assert v13.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v13.euler_recon.euler_density_contact_bvd is False
    assert v13.euler_recon.euler_density_contact_cell_bvd is False
    assert v13.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v13.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v13.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v13.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v13.euler_recon.euler_tangential_velocity_tvd == (
        v42.euler_recon.euler_tangential_velocity_tvd)
    assert (
        v13.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is True)
    assert v13.euler_recon.euler_density_contact_weak_face_rho_floor == 0.65
    assert v13.euler_recon.euler_density_contact_weak_face_p_floor == 0.80
    assert (
        v13.euler_recon
        .euler_density_contact_weak_face_admissibility_strength == 1.0)
    assert (
        v13.euler_recon
        .euler_density_contact_weak_face_entropy_accept is False)


def test_v14_candidate_adds_density_weak_face_entropy_accept_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v14 = bench._reconstruction_from_key("tmlpu_v14_unified_on")
    assert v14.name == "t_mlp_u_unified"
    assert type(v14.scalar_recon) is type(v42.scalar_recon)
    assert type(v14.euler_recon) is type(v42.euler_recon)
    assert v14.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v14.euler_recon.euler_density_contact_bvd is False
    assert v14.euler_recon.euler_density_contact_cell_bvd is False
    assert v14.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v14.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v14.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v14.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v14.euler_recon.euler_tangential_velocity_tvd == (
        v42.euler_recon.euler_tangential_velocity_tvd)
    assert (
        v14.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is False)
    assert v14.euler_recon.euler_density_contact_weak_face_rho_floor == 0.0
    assert v14.euler_recon.euler_density_contact_weak_face_p_floor == 0.0
    assert (
        v14.euler_recon
        .euler_density_contact_weak_face_entropy_accept is True)
    assert (
        v14.euler_recon
        .euler_density_contact_weak_face_entropy_accept_eps == 0.05)
    assert (
        v14.euler_recon
        .euler_density_contact_weak_face_entropy_reject_scale == 0.35)


def test_v15_candidate_applies_shock_gate_weak_face_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v13 = bench._reconstruction_from_key("tmlpu_v13_unified_on")
    v14 = bench._reconstruction_from_key("tmlpu_v14_unified_on")
    v15 = bench._reconstruction_from_key("tmlpu_v15_unified_on")
    assert v15.name == "t_mlp_u_unified"
    assert type(v15.scalar_recon) is type(v42.scalar_recon)
    assert type(v15.euler_recon) is type(v42.euler_recon)
    assert v15.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v15.euler_recon.euler_density_contact_bvd is False
    assert v15.euler_recon.euler_density_contact_cell_bvd is False
    assert v15.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v15.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v15.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v15.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v15.euler_recon.euler_tangential_velocity_tvd == (
        v42.euler_recon.euler_tangential_velocity_tvd)
    assert v15.euler_recon.euler_density_contact_weak_face_shock_gate is True
    assert v15.euler_recon.euler_density_contact_weak_face_shock_gate_mode == 'wide'
    assert v15.euler_recon.euler_density_contact_weak_face_admissibility_damp is False
    assert v15.euler_recon.euler_density_contact_weak_face_entropy_accept is False
    assert (
        v15.euler_recon
        .euler_density_contact_weak_face_shock_gate_strength == 0.65)
    assert (
        v15.euler_recon.euler_density_contact_weak_face_shock_gate_floor == 0.35)
    assert (
        v15.euler_recon
        .euler_density_contact_weak_face_shock_gate_p_threshold == 0.06)
    assert (
        v15.euler_recon
        .euler_density_contact_weak_face_shock_gate_p_width == 0.24)
    assert (
        v15.euler_recon.euler_density_contact_weak_face_shock_gate_compression_threshold == 0.015)
    assert (
        v15.euler_recon
        .euler_density_contact_weak_face_shock_gate_compression_width == 0.12)
    assert (
        v15.euler_recon
        .euler_density_contact_weak_face_shock_gate_normality_threshold == 0.45)
    assert (
        v15.euler_recon
        .euler_density_contact_weak_face_shock_gate_normality_width == 0.35)
    assert (
        v15.euler_recon
        .euler_density_contact_weak_face_shock_gate_shear_threshold == 0.65)
    assert (
        v15.euler_recon.euler_density_contact_weak_face_shock_gate_shear_width == 0.25)
    assert (
        v15.euler_recon
        .euler_density_contact_weak_face_shock_gate_contact_threshold == 0.25)
    assert (
        v15.euler_recon
        .euler_density_contact_weak_face_shock_gate_contact_width == 0.45)

    assert v13.euler_recon.euler_density_contact_weak_face_shock_gate is False
    assert v14.euler_recon.euler_density_contact_weak_face_shock_gate is False


def test_v16_candidate_applies_core_shock_gate_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v15 = bench._reconstruction_from_key("tmlpu_v15_unified_on")
    v16 = bench._reconstruction_from_key("tmlpu_v16_unified_on")
    assert v16.name == "t_mlp_u_unified"
    assert type(v16.scalar_recon) is type(v42.scalar_recon)
    assert type(v16.euler_recon) is type(v42.euler_recon)
    assert v16.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v16.euler_recon.euler_density_contact_bvd is False
    assert v16.euler_recon.euler_density_contact_cell_bvd is False
    assert v16.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v16.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v16.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v16.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v16.euler_recon.euler_density_contact_weak_face_shock_gate is True
    assert v16.euler_recon.euler_density_contact_weak_face_shock_gate_mode == 'core'
    assert (
        v16.euler_recon
        .euler_density_contact_weak_face_shock_gate_strength == 0.55)
    assert (
        v16.euler_recon.euler_density_contact_weak_face_shock_gate_floor == 0.55)
    assert (
        v16.euler_recon.euler_density_contact_weak_face_shock_gate_p_threshold == 0.08)
    assert (
        v16.euler_recon.euler_density_contact_weak_face_shock_gate_p_width == 0.22)
    assert (
        v16.euler_recon
        .euler_density_contact_weak_face_shock_gate_compression_threshold == 0.025)
    assert (
        v16.euler_recon
        .euler_density_contact_weak_face_shock_gate_compression_width == 0.12)
    assert (
        v16.euler_recon
        .euler_density_contact_weak_face_shock_gate_normality_threshold == 0.55)
    assert (
        v16.euler_recon
        .euler_density_contact_weak_face_shock_gate_normality_width == 0.30)
    assert (
        v16.euler_recon.euler_density_contact_weak_face_shock_gate_shear_threshold == 0.55)
    assert (
        v16.euler_recon.euler_density_contact_weak_face_shock_gate_shear_width == 0.25)
    assert (
        v16.euler_recon
        .euler_density_contact_weak_face_shock_gate_contact_threshold == 0.20)
    assert (
        v16.euler_recon
        .euler_density_contact_weak_face_shock_gate_contact_width == 0.40)
    assert v15.euler_recon.euler_density_contact_weak_face_shock_gate_mode == 'wide'
    assert v15.euler_recon.euler_density_contact_weak_face_shock_gate_strength == 0.65


def test_v17_candidate_applies_density_weak_face_value_scaling_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v16 = bench._reconstruction_from_key("tmlpu_v16_unified_on")
    v17 = bench._reconstruction_from_key("tmlpu_v17_unified_on")
    assert v17.name == "t_mlp_u_unified"
    assert type(v17.scalar_recon) is type(v42.scalar_recon)
    assert type(v17.euler_recon) is type(v42.euler_recon)
    assert v17.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v17.euler_recon.euler_density_contact_bvd is False
    assert v17.euler_recon.euler_density_contact_cell_bvd is False
    assert v17.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v17.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v17.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v17.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v17.euler_recon.euler_density_contact_weak_face_shock_gate is False
    assert v17.euler_recon.euler_density_contact_weak_face_admissibility_damp is False
    assert (
        v17.euler_recon
        .euler_density_contact_weak_face_entropy_accept is False)
    assert v17.euler_recon.euler_density_contact_weak_face_value_scaling is True
    assert (
        v17.euler_recon.euler_density_contact_weak_face_rho_floor_factor == 0.88)
    assert v17.euler_recon.euler_density_contact_weak_face_theta_floor == 0.0
    assert (
        v17.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode == 'global_floor')

    assert v16.euler_recon.euler_density_contact_weak_face_value_scaling is False


def test_v18_candidate_adds_pressure_entropy_blend_only():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v17 = bench._reconstruction_from_key("tmlpu_v17_unified_on")
    v18 = bench._reconstruction_from_key("tmlpu_v18_unified_on")
    assert v18.name == "t_mlp_u_unified"
    assert type(v18.scalar_recon) is type(v42.scalar_recon)
    assert type(v18.euler_recon) is type(v42.euler_recon)
    assert v18.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v18.euler_recon.euler_density_contact_bvd is False
    assert v18.euler_recon.euler_density_contact_cell_bvd is False
    assert v18.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v18.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v18.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v18.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert (
        v18.euler_recon
        .euler_density_contact_weak_face_value_scaling is False)
    assert (
        v18.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is False)
    assert (
        v18.euler_recon
        .euler_density_contact_weak_face_entropy_accept is False)
    assert v18.euler_recon.euler_density_contact_weak_face_shock_gate is False
    assert v18.euler_recon.euler_pressure_contact_entropy_blend is True
    assert v18.euler_recon.euler_pressure_contact_entropy_beta == 0.18
    assert v18.euler_recon.euler_pressure_contact_entropy_cap == 0.18
    assert v18.euler_recon.euler_pressure_contact_entropy_downscale == 0.25
    assert (
        v18.euler_recon
        .euler_pressure_contact_entropy_p_jump_threshold == 0.04)
    assert v18.euler_recon.euler_pressure_contact_entropy_p_jump_width == 0.08
    assert (
        v18.euler_recon
        .euler_pressure_contact_entropy_compression_threshold == 0.01)
    assert (
        v18.euler_recon
        .euler_pressure_contact_entropy_compression_width == 0.07)
    assert (
        v18.euler_recon
        .euler_pressure_contact_entropy_normality_threshold == 0.45)
    assert (
        v18.euler_recon
        .euler_pressure_contact_entropy_normality_width == 0.30)

    assert v17.euler_recon.euler_pressure_contact_entropy_blend is False


def test_v19_candidate_localizes_value_scaling_to_shocklike_low_density_pocket_faces():
    v42 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v17 = bench._reconstruction_from_key("tmlpu_v17_unified_on")
    v19 = bench._reconstruction_from_key("tmlpu_v19_unified_on")
    assert v19.name == "t_mlp_u_unified"
    assert type(v19.scalar_recon) is type(v42.scalar_recon)
    assert type(v19.euler_recon) is type(v42.euler_recon)
    assert v19.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v19.euler_recon.euler_density_contact_bvd is False
    assert v19.euler_recon.euler_density_contact_cell_bvd is False
    assert v19.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v19.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v19.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v19.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v19.euler_recon.euler_density_contact_weak_face_value_scaling is True
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode == 'local_pocket_shock')
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_strength == 1.0)
    assert v19.euler_recon.euler_density_contact_weak_face_rho_floor_factor == 0.86
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_p_floor_factor == 0.90)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_risk_width == 0.08)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_p_threshold == 0.06)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_p_width == 0.10)
    assert (
        v19.euler_recon.euler_density_contact_weak_face_value_scaling_compression_threshold
        == 0.015)
    assert (
        v19.euler_recon.euler_density_contact_weak_face_value_scaling_compression_width
        == 0.065)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_normality_threshold == 0.35)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_normality_width == 0.35)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_contact_threshold == 0.25)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_contact_width == 0.35)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_threshold == 0.60)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_width == 0.25)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_pressure_clean_threshold == 0.04)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_pressure_clean_width == 0.06)
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_hard_protect_cutoff == 0.65)
    assert v19.euler_recon.euler_density_contact_weak_face_admissibility_damp is False
    assert v19.euler_recon.euler_density_contact_weak_face_entropy_accept is False
    assert v19.euler_recon.euler_density_contact_weak_face_shock_gate is False
    assert v19.euler_recon.euler_pressure_contact_entropy_blend is False

    assert v17.euler_recon.euler_density_contact_weak_face_value_scaling is True
    assert (
        v17.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode == 'global_floor')


def test_v20_candidate_blends_v4_2_and_v19_value_scaling_candidates():
    v17 = bench._reconstruction_from_key("tmlpu_v17_unified_on")
    v19 = bench._reconstruction_from_key("tmlpu_v19_unified_on")
    v20 = bench._reconstruction_from_key("tmlpu_v20_unified_on")

    assert v20.name == "t_mlp_u_unified"
    assert type(v20.scalar_recon) is type(v19.scalar_recon)
    assert type(v20.euler_recon) is type(v19.euler_recon)
    assert v20.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v20.euler_recon.euler_density_contact_bvd is False
    assert v20.euler_recon.euler_density_contact_cell_bvd is False
    assert v20.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v20.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v20.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v20.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v20.euler_recon.euler_density_contact_weak_face_value_scaling is True
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode == 'local_pocket_shock')
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_strength == 0.35)
    assert v20.euler_recon.euler_density_contact_weak_face_rho_floor_factor == 0.86
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_p_floor_factor == 0.90)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_risk_width == 0.08)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_p_threshold == 0.06)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_p_width == 0.10)
    assert (
        v20.euler_recon.euler_density_contact_weak_face_value_scaling_compression_threshold
        == 0.015)
    assert (
        v20.euler_recon.euler_density_contact_weak_face_value_scaling_compression_width
        == 0.065)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_normality_threshold == 0.35)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_normality_width == 0.35)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_contact_threshold == 0.25)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_contact_width == 0.35)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_threshold == 0.60)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_width == 0.25)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_pressure_clean_threshold == 0.04)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_pressure_clean_width == 0.06)
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_hard_protect_cutoff == 0.65)
    assert v20.euler_recon.euler_density_contact_weak_face_admissibility_damp is False
    assert v20.euler_recon.euler_density_contact_weak_face_entropy_accept is False
    assert v20.euler_recon.euler_density_contact_weak_face_shock_gate is False
    assert v20.euler_recon.euler_pressure_contact_entropy_blend is False

    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_strength == 1.0)
    assert (
        v17.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode == 'global_floor')


def test_v21_candidate_split_hard_vs_quality_value_scaling():
    v19 = bench._reconstruction_from_key("tmlpu_v19_unified_on")
    v20 = bench._reconstruction_from_key("tmlpu_v20_unified_on")
    v21 = bench._reconstruction_from_key("tmlpu_v21_unified_on")

    assert v21.name == "t_mlp_u_unified"
    assert type(v21.scalar_recon) is type(v20.scalar_recon)
    assert type(v21.euler_recon) is type(v20.euler_recon)
    assert v21.euler_recon.euler_density_contact_weak_face_value_scaling is True
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode == 'local_pocket_shock')
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_value_scaling_strength == 0.25)
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_hard_rho_floor_factor == 0.82)
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_hard_p_floor_factor == 0.84)
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_rho_floor_factor == 0.88)
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_value_scaling_p_floor_factor == 0.90)
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_value_scaling_risk_width == 0.08)
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_value_scaling_hard_protect_cutoff == 0.65)
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is False)
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_entropy_accept is False)
    assert (
        v21.euler_recon
        .euler_density_contact_weak_face_shock_gate is False)
    assert v21.euler_recon.euler_pressure_contact_entropy_blend is False

    # v20 remains quality-strength blend of quality-candidate on top of v19 base.
    assert (
        v20.euler_recon
        .euler_density_contact_weak_face_value_scaling_strength == 0.35)
    # v19 remains pure quality candidate scaling (strength=1).
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_strength == 1.0)


def test_v22_candidate_restores_tangential_rollup_with_v19_density_anchor():
    v19 = bench._reconstruction_from_key("tmlpu_v19_unified_on")
    v20 = bench._reconstruction_from_key("tmlpu_v20_unified_on")
    v21 = bench._reconstruction_from_key("tmlpu_v21_unified_on")
    v22 = bench._reconstruction_from_key("tmlpu_v22_unified_on")

    assert v22.name == "t_mlp_u_unified"
    assert type(v22.scalar_recon) is type(v19.scalar_recon)
    assert type(v22.euler_recon) is type(v19.euler_recon)

    # v22 keeps v19 stable density value-scaling anchor.
    assert (
        v22.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        v19.euler_recon.euler_density_contact_weak_face_value_scaling_mode)
    assert (
        v22.euler_recon
        .euler_density_contact_weak_face_value_scaling_strength ==
        v19.euler_recon.euler_density_contact_weak_face_value_scaling_strength)
    assert (
        v22.euler_recon.euler_density_contact_weak_face_rho_floor_factor ==
        v19.euler_recon.euler_density_contact_weak_face_rho_floor_factor)
    assert (
        v22.euler_recon
        .euler_density_contact_weak_face_value_scaling_p_floor_factor ==
        v19.euler_recon.euler_density_contact_weak_face_value_scaling_p_floor_factor)

    # v22 enables tangential roll-up recovery via mood micro path.
    assert (
        v22.euler_recon
        .euler_tangential_velocity_tvd
        == 'shear_superbee_root_mood')
    assert (
        v22.euler_recon
        .euler_tangential_shear_micro_blend == 0.06)
    assert (
        v22.euler_recon.euler_tangential_shear_micro_cap == 0.16)
    assert (
        v22.euler_recon
        .euler_tangential_mood_wavespeed_growth_cap == 0.02)
    assert (
        v22.euler_recon
        .euler_tangential_mood_jump_growth_cap == 0.05)

    # Ensure prior v19/v20/v21 anchors stay untouched.
    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_strength == 1.0)
    assert v20.euler_recon.euler_density_contact_weak_face_value_scaling_strength == 0.35
    assert v21.euler_recon.euler_density_contact_weak_face_value_scaling_strength == 0.25


def test_v23_candidate_uses_shock_safer_tangential_rollup_defaults():
    v19 = bench._reconstruction_from_key("tmlpu_v19_unified_on")
    v22 = bench._reconstruction_from_key("tmlpu_v22_unified_on")
    v23 = bench._reconstruction_from_key("tmlpu_v23_unified_on")

    assert v23.name == "t_mlp_u_unified"
    assert type(v23.scalar_recon) is type(v19.scalar_recon)
    assert type(v23.euler_recon) is type(v19.euler_recon)

    assert (
        v23.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        v19.euler_recon.euler_density_contact_weak_face_value_scaling_mode)
    assert (
        v23.euler_recon
        .euler_density_contact_weak_face_value_scaling_strength ==
        v19.euler_recon.euler_density_contact_weak_face_value_scaling_strength)
    assert (
        v23.euler_recon.euler_density_contact_weak_face_rho_floor_factor ==
        v19.euler_recon.euler_density_contact_weak_face_rho_floor_factor)
    assert (
        v23.euler_recon
        .euler_density_contact_weak_face_value_scaling_p_floor_factor ==
        v19.euler_recon.euler_density_contact_weak_face_value_scaling_p_floor_factor)

    assert (
        v23.euler_recon
        .euler_tangential_velocity_tvd
        == 'shear_superbee_root_mood')
    assert (
        v23.euler_recon
        .euler_tangential_shear_micro_blend == 0.035)
    assert (
        v23.euler_recon.euler_tangential_shear_micro_cap == 0.10)
    assert (
        v23.euler_recon
        .euler_tangential_mood_wavespeed_growth_cap == 0.012)
    assert (
        v23.euler_recon
        .euler_tangential_mood_jump_growth_cap == 0.030)

    assert (
        v23.euler_recon.euler_tangential_shear_micro_blend <
        v22.euler_recon.euler_tangential_shear_micro_blend)
    assert (
        v23.euler_recon.euler_tangential_shear_micro_cap <
        v22.euler_recon.euler_tangential_shear_micro_cap)


def test_v24_candidate_keeps_v22_rollup_strength_with_stricter_mood_reject():
    v19 = bench._reconstruction_from_key("tmlpu_v19_unified_on")
    v22 = bench._reconstruction_from_key("tmlpu_v22_unified_on")
    v23 = bench._reconstruction_from_key("tmlpu_v23_unified_on")
    v24 = bench._reconstruction_from_key("tmlpu_v24_unified_on")

    assert v24.name == "t_mlp_u_unified"
    assert type(v24.scalar_recon) is type(v19.scalar_recon)
    assert type(v24.euler_recon) is type(v19.euler_recon)

    assert (
        v24.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        v19.euler_recon.euler_density_contact_weak_face_value_scaling_mode)
    assert (
        v24.euler_recon
        .euler_density_contact_weak_face_value_scaling_strength ==
        v19.euler_recon.euler_density_contact_weak_face_value_scaling_strength)
    assert (
        v24.euler_recon
        .euler_tangential_velocity_tvd
        == 'shear_superbee_root_mood')
    assert (
        v24.euler_recon.euler_tangential_shear_micro_blend ==
        v22.euler_recon.euler_tangential_shear_micro_blend)
    assert (
        v24.euler_recon.euler_tangential_shear_micro_cap ==
        v22.euler_recon.euler_tangential_shear_micro_cap)
    assert (
        v24.euler_recon
        .euler_tangential_mood_wavespeed_growth_cap == 0.010)
    assert (
        v24.euler_recon
        .euler_tangential_mood_jump_growth_cap == 0.025)
    assert (
        v24.euler_recon.euler_tangential_mood_wavespeed_growth_cap <
        v23.euler_recon.euler_tangential_mood_wavespeed_growth_cap)
    assert (
        v24.euler_recon.euler_tangential_mood_jump_growth_cap <
        v23.euler_recon.euler_tangential_mood_jump_growth_cap)


def test_v25_candidate_uses_v13_anchor_with_narrow_admissibility_lift():
    v13 = bench._reconstruction_from_key("tmlpu_v13_unified_on")
    v25 = bench._reconstruction_from_key("tmlpu_v25_unified_on")

    assert v25.name == "t_mlp_u_unified"
    assert type(v25.scalar_recon) is type(v13.scalar_recon)
    assert type(v25.euler_recon) is type(v13.euler_recon)

    assert (
        v25.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is True)
    assert (
        v25.euler_recon
        .euler_density_contact_weak_face_admissibility_strength == 1.0)
    assert (
        v25.euler_recon
        .euler_density_contact_weak_face_rho_floor == 0.655)
    assert (
        v25.euler_recon
        .euler_density_contact_weak_face_p_floor == 0.875)
    assert (
        v25.euler_recon
        .euler_density_contact_weak_face_value_scaling is False)
    assert (
        v25.euler_recon
        .euler_density_contact_weak_face_entropy_accept is False)
    assert (
        v25.euler_recon.euler_pressure_contact_entropy_blend is False)
    assert (
        v25.euler_recon.euler_tangential_velocity_tvd ==
        v13.euler_recon.euler_tangential_velocity_tvd)

    assert (
        v13.euler_recon
        .euler_density_contact_weak_face_rho_floor == 0.65)
    assert (
        v13.euler_recon
        .euler_density_contact_weak_face_p_floor == 0.80)


def test_v26_candidate_protects_clean_shear_contacts_on_v13_anchor():
    v13 = bench._reconstruction_from_key("tmlpu_v13_unified_on")
    v25 = bench._reconstruction_from_key("tmlpu_v25_unified_on")
    v26 = bench._reconstruction_from_key("tmlpu_v26_unified_on")

    assert v26.name == "t_mlp_u_unified"
    assert type(v26.scalar_recon) is type(v13.scalar_recon)
    assert type(v26.euler_recon) is type(v13.euler_recon)

    assert (
        v26.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is True)
    assert (
        v26.euler_recon
        .euler_density_contact_weak_face_admissibility_strength ==
        v25.euler_recon.euler_density_contact_weak_face_admissibility_strength)
    assert (
        v26.euler_recon
        .euler_density_contact_weak_face_rho_floor ==
        v25.euler_recon.euler_density_contact_weak_face_rho_floor)
    assert (
        v26.euler_recon
        .euler_density_contact_weak_face_p_floor ==
        v25.euler_recon.euler_density_contact_weak_face_p_floor)
    assert (
        v26.euler_recon
        .euler_density_contact_weak_face_admissibility_shear_protect is True)
    assert (
        v25.euler_recon
        .euler_density_contact_weak_face_admissibility_shear_protect is False)
    assert (
        v26.euler_recon
        .euler_density_contact_weak_face_value_scaling is False)


def test_v27_candidate_uses_v13_anchor_with_late_face_abs_floor():
    v13 = bench._reconstruction_from_key("tmlpu_v13_unified_on")
    v27 = bench._reconstruction_from_key("tmlpu_v27_unified_on")

    assert v27.name == "t_mlp_u_unified"
    assert type(v27.scalar_recon) is type(v13.scalar_recon)
    assert type(v27.euler_recon) is type(v13.euler_recon)

    assert (
        v27.euler_recon
        .euler_density_contact_weak_face_admissibility_damp ==
        v13.euler_recon.euler_density_contact_weak_face_admissibility_damp)
    assert (
        v27.euler_recon
        .euler_density_contact_weak_face_rho_floor ==
        v13.euler_recon.euler_density_contact_weak_face_rho_floor)
    assert (
        v27.euler_recon
        .euler_density_contact_weak_face_p_floor ==
        v13.euler_recon.euler_density_contact_weak_face_p_floor)
    assert v27.euler_recon.euler_face_positivity_limiter is True
    assert v27.euler_recon.euler_face_rho_abs_floor == 0.645
    assert v27.euler_recon.euler_face_p_abs_floor == 0.860
    assert v13.euler_recon.euler_face_rho_abs_floor == 0.0
    assert v13.euler_recon.euler_face_p_abs_floor == 0.0


def test_v28_candidate_adds_density_swirl_preservation_on_v13_anchor():
    v13 = bench._reconstruction_from_key("tmlpu_v13_unified_on")
    v28 = bench._reconstruction_from_key("tmlpu_v28_unified_on")

    assert v28.name == "t_mlp_u_unified"
    assert type(v28.scalar_recon) is type(v13.scalar_recon)
    assert type(v28.euler_recon) is type(v13.euler_recon)

    assert (
        v28.euler_recon
        .euler_density_contact_weak_face_admissibility_damp ==
        v13.euler_recon.euler_density_contact_weak_face_admissibility_damp)
    assert (
        v28.euler_recon
        .euler_density_contact_weak_face_rho_floor ==
        v13.euler_recon.euler_density_contact_weak_face_rho_floor)
    assert (
        v28.euler_recon
        .euler_density_contact_weak_face_p_floor ==
        v13.euler_recon.euler_density_contact_weak_face_p_floor)
    assert (
        v28.euler_recon
        .euler_density_contact_weak_face_swirl_extra == 0.18)
    assert (
        v13.euler_recon
        .euler_density_contact_weak_face_swirl_extra == 0.0)
    assert v28.euler_recon.euler_face_rho_abs_floor == 0.0
    assert v28.euler_recon.euler_face_p_abs_floor == 0.0


def test_v29_candidate_blends_v17_density_floor_only_on_clean_shear():
    v13 = bench._reconstruction_from_key("tmlpu_v13_unified_on")
    v29 = bench._reconstruction_from_key("tmlpu_v29_unified_on")

    assert v29.name == "t_mlp_u_unified"
    assert type(v29.scalar_recon) is type(v13.scalar_recon)
    assert type(v29.euler_recon) is type(v13.euler_recon)

    assert (
        v29.euler_recon
        .euler_density_contact_weak_face_admissibility_damp ==
        v13.euler_recon.euler_density_contact_weak_face_admissibility_damp)
    assert (
        v29.euler_recon
        .euler_density_contact_weak_face_value_scaling is True)
    assert (
        v29.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        'shear_floor_blend')
    assert (
        v29.euler_recon
        .euler_density_contact_weak_face_rho_floor_factor == 0.82)
    assert (
        v29.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha ==
        0.12)
    assert (
        v29.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad ==
        0.02)
    assert (
        v29.euler_recon
        .euler_density_contact_weak_face_swirl_extra == 0.0)
    assert v29.euler_recon.euler_face_rho_abs_floor == 0.0
    assert v29.euler_recon.euler_face_p_abs_floor == 0.0


def test_v30_candidate_micro_restores_density_only_on_clean_shear_contact():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v4_2 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v19 = bench._reconstruction_from_key("tmlpu_v19_unified_on")
    v25 = bench._reconstruction_from_key("tmlpu_v25_unified_on")
    v27 = bench._reconstruction_from_key("tmlpu_v27_unified_on")
    v30 = bench._reconstruction_from_key("tmlpu_v30_unified_on")

    assert v30.name == "t_mlp_u_unified"
    assert type(v30.scalar_recon) is type(v3.scalar_recon)
    assert type(v30.euler_recon) is type(v4_2.euler_recon)

    assert v30.euler_recon.euler_density_contact_weak_face_mlp is False
    assert v4_2.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v30.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v30.euler_recon.euler_face_positivity_limiter is True
    assert v30.euler_recon.tmlpu_bound_tvd_separate is True

    assert (
        v30.euler_recon
        .euler_density_contact_weak_face_value_scaling is True)
    assert (
        v30.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        'clean_shear_micro_restore')
    assert (
        v30.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha ==
        0.08)
    assert (
        v30.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad ==
        0.012)

    assert v30.euler_recon.euler_density_contact_bvd is False
    assert v30.euler_recon.euler_density_contact_cell_bvd is False
    assert (
        v30.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is False)
    assert (
        v30.euler_recon
        .euler_density_contact_weak_face_shock_gate is False)
    assert v30.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v30.euler_recon.euler_face_rho_abs_floor == 0.0
    assert v30.euler_recon.euler_face_p_abs_floor == 0.0
    assert v30.euler_recon.euler_pressure_contact_entropy_blend is False

    assert (
        v19.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        'local_pocket_shock')
    assert (
        v25.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is True)
    assert v27.euler_recon.euler_face_rho_abs_floor == 0.645


def test_v31_candidate_refines_v30_with_coherent_shear_micro_restore():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v4_2 = bench._reconstruction_from_key("tmlpu_v4_2_unified_on")
    v30 = bench._reconstruction_from_key("tmlpu_v30_unified_on")
    v31 = bench._reconstruction_from_key("tmlpu_v31_unified_on")

    assert v31.name == "t_mlp_u_unified"
    assert type(v31.scalar_recon) is type(v3.scalar_recon)
    assert type(v31.euler_recon) is type(v4_2.euler_recon)

    assert v31.euler_recon.euler_density_contact_weak_face_mlp is False
    assert v31.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v31.euler_recon.euler_face_positivity_limiter is True
    assert v31.euler_recon.tmlpu_bound_tvd_separate is True

    assert (
        v31.euler_recon
        .euler_density_contact_weak_face_value_scaling is True)
    assert (
        v31.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        'coherent_shear_micro_restore')
    assert (
        v31.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha ==
        0.045)
    assert (
        v31.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad ==
        0.008)
    assert (
        v31.euler_recon
        .euler_density_contact_weak_face_value_scaling_require_coherent_shear
        is True)
    assert (
        v31.euler_recon
        .euler_density_contact_weak_face_value_scaling_artifact_reject is True)

    assert (
        v31.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is False)
    assert (
        v31.euler_recon
        .euler_density_contact_weak_face_entropy_accept is False)
    assert (
        v31.euler_recon
        .euler_density_contact_weak_face_shock_gate is False)
    assert v31.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v31.euler_recon.euler_pressure_contact_entropy_blend is False
    assert v31.euler_recon.euler_face_rho_abs_floor == 0.0
    assert v31.euler_recon.euler_face_p_abs_floor == 0.0
    assert v31.euler_recon.euler_density_contact_bvd is False
    assert v31.euler_recon.euler_density_contact_cell_bvd is False

    assert (
        v30.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        'clean_shear_micro_restore')


def test_v32_candidate_adds_tangential_pair_restore():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v31 = bench._reconstruction_from_key("tmlpu_v31_unified_on")
    v32 = bench._reconstruction_from_key("tmlpu_v32_unified_on")

    assert v32.name == "t_mlp_u_unified"
    assert type(v32.scalar_recon) is type(v3.scalar_recon)
    assert type(v32.euler_recon) is type(v31.euler_recon)

    assert v32.euler_recon.euler_tangential_pair_restore_on is True
    assert v32.euler_recon.euler_tangential_pair_restore_alpha == 0.045
    assert v32.euler_recon.euler_tangential_pair_restore_cap == 0.075
    assert v32.euler_recon.euler_tangential_pair_restore_wave_cap == 0.010

    assert (
        v32.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        'coherent_shear_micro_restore')
    assert (
        v32.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha ==
        v31.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha)
    assert (
        v32.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad ==
        v31.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad)
    assert v32.euler_recon.euler_tangential_contact_relax_flatten is True

    assert v32.euler_recon.euler_tangential_pair_restore_on != \
        v31.euler_recon.euler_tangential_pair_restore_on


def test_v33_candidate_adds_streamwise_downstream_propagation():
    v31 = bench._reconstruction_from_key("tmlpu_v31_unified_on")
    v32 = bench._reconstruction_from_key("tmlpu_v32_unified_on")
    v33 = bench._reconstruction_from_key("tmlpu_v33_unified_on")

    assert v33.name == "t_mlp_u_unified"
    assert type(v33.scalar_recon) is type(v31.scalar_recon)
    assert type(v33.euler_recon) is type(v32.euler_recon)

    assert (
        v33.euler_recon.euler_density_contact_weak_face_value_scaling_mode ==
        'coherent_shear_micro_restore')
    assert (
        v33.euler_recon.euler_density_contact_weak_face_value_scaling_shear_blend_alpha ==
        0.070)
    assert (
        v33.euler_recon.euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad ==
        0.010)

    assert v33.euler_recon.euler_tangential_pair_restore_alpha == 0.030
    assert v33.euler_recon.euler_tangential_pair_restore_cap == 0.055
    assert v33.euler_recon.euler_tangential_pair_restore_wave_cap == 0.008

    assert (
        v33.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is True)
    assert (
        v33.euler_recon
        .euler_tangential_pair_restore_stream_coherence_min == 0.20)
    assert (
        v33.euler_recon
        .euler_tangential_pair_restore_stream_coherence_full == 0.60)

    assert (
        v33.euler_recon
        .euler_density_contact_weak_face_stream_coherence_on is True)
    assert (
        v33.euler_recon
        .euler_density_contact_weak_face_stream_coherence_min == 0.20)
    assert (
        v33.euler_recon
        .euler_density_contact_weak_face_stream_coherence_full == 0.60)

    assert (
        v33.euler_recon
        .euler_density_contact_weak_face_downstream_rho_beta == 0.035)
    assert (
        v33.euler_recon
        .euler_density_contact_weak_face_downstream_tangential_beta == 0.020)
    assert (
        v33.euler_recon
        .euler_density_contact_weak_face_downstream_rho_cap == 0.006)
    assert (
        v33.euler_recon
        .euler_density_contact_weak_face_downstream_tangential_cap == 0.030)
    assert (
        v33.euler_recon
        .euler_density_contact_weak_face_downstream_rho_wave_cap == 0.004)
    assert (
        v33.euler_recon
        .euler_density_contact_weak_face_downstream_tangential_wave_cap == 0.004)
    assert v33.euler_recon.euler_tangential_contact_relax_flatten is True

    assert (
        v33.euler_recon.euler_tangential_pair_restore_alpha !=
        v32.euler_recon.euler_tangential_pair_restore_alpha)

    assert (
        v33.euler_recon
        .euler_density_contact_weak_face_stream_coherence_on is True)
    assert (
        v32.euler_recon
        .euler_density_contact_weak_face_stream_coherence_on is False)


def test_v34_candidate_adds_contour_continuity_micro_restore():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v32 = bench._reconstruction_from_key("tmlpu_v32_unified_on")
    v33 = bench._reconstruction_from_key("tmlpu_v33_unified_on")
    v34 = bench._reconstruction_from_key("tmlpu_v34_unified_on")

    assert v34.name == "t_mlp_u_unified"
    assert type(v34.scalar_recon) is type(v3.scalar_recon)
    assert type(v34.euler_recon) is type(v32.euler_recon)

    assert v34.euler_recon.euler_density_contact_weak_face_mlp is False
    assert v34.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v34.euler_recon.euler_face_positivity_limiter is True
    assert v34.euler_recon.tmlpu_bound_tvd_separate is True

    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_value_scaling is True)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        'contour_continuity_micro_restore')
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha ==
        0.075)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad ==
        0.010)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_contour_continuity_on is True)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_contour_continuity_min == 0.55)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_contour_continuity_full == 0.85)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_density_increment_cap == 0.008)

    assert v34.euler_recon.euler_tangential_pair_restore_on is True
    assert v34.euler_recon.euler_tangential_pair_restore_alpha == 0.030
    assert v34.euler_recon.euler_tangential_pair_restore_cap == 0.055
    assert v34.euler_recon.euler_tangential_pair_restore_wave_cap == 0.008

    assert (
        v34.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is False)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_stream_coherence_on is False)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_downstream_rho_beta == 0.0)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_downstream_tangential_beta == 0.0)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_downstream_rho_cap == 0.0)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_downstream_tangential_cap == 0.0)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_downstream_rho_wave_cap == 0.0)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_downstream_tangential_wave_cap == 0.0)

    assert (
        v33.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is True)
    assert (
        v32.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is False)


def test_v35_candidate_adds_flow_aligned_pair_extent_continuation():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v32 = bench._reconstruction_from_key("tmlpu_v32_unified_on")
    v33 = bench._reconstruction_from_key("tmlpu_v33_unified_on")
    v34 = bench._reconstruction_from_key("tmlpu_v34_unified_on")
    v35 = bench._reconstruction_from_key("tmlpu_v35_unified_on")

    assert v35.name == "t_mlp_u_unified"
    assert type(v35.scalar_recon) is type(v3.scalar_recon)
    assert type(v35.euler_recon) is type(v32.euler_recon)

    assert v35.euler_recon.euler_density_contact_weak_face_mlp is False
    assert v35.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v35.euler_recon.euler_face_positivity_limiter is True
    assert v35.euler_recon.tmlpu_bound_tvd_separate is True

    assert (
        v35.euler_recon
        .euler_density_contact_weak_face_value_scaling is True)
    assert (
        v35.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        'coherent_shear_micro_restore')
    assert (
        v35.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha ==
        0.070)
    assert (
        v35.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad ==
        0.010)

    assert v35.euler_recon.euler_tangential_pair_restore_on is True
    assert v35.euler_recon.euler_tangential_pair_restore_alpha == 0.030
    assert v35.euler_recon.euler_tangential_pair_restore_cap == 0.055
    assert v35.euler_recon.euler_tangential_pair_restore_wave_cap == 0.008

    assert v35.euler_recon.euler_tangential_pair_extend_on is True
    assert v35.euler_recon.euler_tangential_pair_extend_beta == 0.018
    assert v35.euler_recon.euler_tangential_pair_extend_cap == 0.025
    assert v35.euler_recon.euler_tangential_pair_extend_wave_cap == 0.0035
    assert (
        v35.euler_recon
        .euler_tangential_pair_extend_alignment_min == 0.65)
    assert (
        v35.euler_recon
        .euler_tangential_pair_extend_alignment_full == 0.90)

    assert (
        v35.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is False)
    assert (
        v35.euler_recon
        .euler_density_contact_weak_face_stream_coherence_on is False)
    assert (
        v35.euler_recon
        .euler_density_contact_weak_face_downstream_rho_beta == 0.0)
    assert (
        v35.euler_recon
        .euler_density_contact_weak_face_downstream_tangential_beta == 0.0)
    assert (
        v35.euler_recon
        .euler_density_contact_weak_face_contour_continuity_on is False)
    assert (
        v35.euler_recon
        .euler_density_contact_weak_face_density_increment_cap == 0.0)
    assert v35.euler_recon.euler_pressure_contact_entropy_blend is False

    assert (
        v33.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is True)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_contour_continuity_on is True)


def test_v36_candidate_widens_existing_pair_gate_only():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v32 = bench._reconstruction_from_key("tmlpu_v32_unified_on")
    v33 = bench._reconstruction_from_key("tmlpu_v33_unified_on")
    v34 = bench._reconstruction_from_key("tmlpu_v34_unified_on")
    v35 = bench._reconstruction_from_key("tmlpu_v35_unified_on")
    v36 = bench._reconstruction_from_key("tmlpu_v36_unified_on")

    assert v36.name == "t_mlp_u_unified"
    assert type(v36.scalar_recon) is type(v3.scalar_recon)
    assert type(v36.euler_recon) is type(v32.euler_recon)

    assert v36.euler_recon.euler_density_contact_weak_face_mlp is False
    assert v36.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v36.euler_recon.euler_face_positivity_limiter is True
    assert v36.euler_recon.tmlpu_bound_tvd_separate is True

    assert (
        v36.euler_recon
        .euler_density_contact_weak_face_value_scaling is True)
    assert (
        v36.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode ==
        'coherent_shear_micro_restore')
    assert (
        v36.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha ==
        0.070)
    assert (
        v36.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad ==
        0.010)

    assert v36.euler_recon.euler_tangential_pair_restore_on is True
    assert v36.euler_recon.euler_tangential_pair_restore_alpha == 0.030
    assert v36.euler_recon.euler_tangential_pair_restore_cap == 0.055
    assert v36.euler_recon.euler_tangential_pair_restore_wave_cap == 0.008

    assert v36.euler_recon.euler_tangential_pair_gate_contact_min == 0.24
    assert v36.euler_recon.euler_tangential_pair_gate_contact_full == 0.58
    assert v36.euler_recon.euler_tangential_pair_gate_shear_min == 0.62
    assert v36.euler_recon.euler_tangential_pair_gate_shear_full == 0.88
    assert (
        v36.euler_recon
        .euler_tangential_pair_gate_density_support_min == 0.018)
    assert (
        v36.euler_recon
        .euler_tangential_pair_gate_density_support_full == 0.075)
    assert (
        v36.euler_recon
        .euler_tangential_pair_gate_shock_reject_keep_v32 is True)

    assert v36.euler_recon.euler_tangential_pair_extend_on is False
    assert v36.euler_recon.euler_tangential_pair_extend_beta == 0.0
    assert (
        v36.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is False)
    assert (
        v36.euler_recon
        .euler_density_contact_weak_face_stream_coherence_on is False)
    assert (
        v36.euler_recon
        .euler_density_contact_weak_face_downstream_rho_beta == 0.0)
    assert (
        v36.euler_recon
        .euler_density_contact_weak_face_downstream_tangential_beta == 0.0)
    assert (
        v36.euler_recon
        .euler_density_contact_weak_face_contour_continuity_on is False)
    assert (
        v36.euler_recon
        .euler_density_contact_weak_face_density_increment_cap == 0.0)
    assert v36.euler_recon.euler_pressure_contact_entropy_blend is False

    assert v32.euler_recon.euler_tangential_pair_gate_contact_min == 0.30
    assert v32.euler_recon.euler_tangential_pair_gate_shear_min == 0.70
    assert (
        v33.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is True)
    assert (
        v34.euler_recon
        .euler_density_contact_weak_face_contour_continuity_on is True)
    assert v35.euler_recon.euler_tangential_pair_extend_on is True


def test_v37_candidate_uses_legacy_density_weak_face_order_only():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v13 = bench._reconstruction_from_key("tmlpu_v13_unified_on")
    v36 = bench._reconstruction_from_key("tmlpu_v36_unified_on")
    v37 = bench._reconstruction_from_key("tmlpu_v37_unified_on")

    assert v37.name == "t_mlp_u_unified"
    assert type(v37.scalar_recon) is type(v3.scalar_recon)
    assert type(v37.euler_recon) is type(v13.euler_recon)

    assert v37.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v37.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v37.euler_recon.euler_face_positivity_limiter is True
    assert v37.euler_recon.tmlpu_bound_tvd_separate is True

    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_legacy_order is True)
    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_legacy_relax is True)
    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_legacy_relax_cap == 1.0)
    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_legacy_tvd_after_weak is True)

    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is False)
    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_entropy_accept is False)
    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_shock_gate is False)
    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_value_scaling is False)
    assert v37.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v37.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v37.euler_recon.euler_density_contact_weak_face_shock_power == 1.0
    assert v37.euler_recon.euler_density_contact_weak_face_mlp_cap == 1.0

    assert v37.euler_recon.euler_density_contact_bvd is False
    assert v37.euler_recon.euler_density_contact_cell_bvd is False
    assert v37.euler_recon.euler_density_contact_hancock_boost == 0.0
    assert v37.euler_recon.euler_density_contact_lsq_root_blend == 0.0
    assert v37.euler_recon.euler_density_contact_lsq_shear_floor == 0.0
    assert v37.euler_recon.euler_pressure_contact_entropy_blend is False
    assert v37.euler_recon.euler_tangential_pair_restore_on is False
    assert v37.euler_recon.euler_tangential_pair_extend_on is False
    assert (
        v37.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is False)
    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_stream_coherence_on is False)

    assert (
        v13.euler_recon
        .euler_density_contact_weak_face_legacy_order is False)
    assert (
        v13.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is True)
    assert (
        v36.euler_recon
        .euler_density_contact_weak_face_value_scaling is True)


def test_v38_candidate_uses_head_generic_density_weak_face_only():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v37 = bench._reconstruction_from_key("tmlpu_v37_unified_on")
    v38 = bench._reconstruction_from_key("tmlpu_v38_unified_on")

    assert v38.name == "t_mlp_u_unified"
    assert type(v38.scalar_recon) is type(v3.scalar_recon)
    assert type(v38.euler_recon) is type(v37.euler_recon)

    assert v38.euler_recon.euler_density_contact_weak_face_mlp is True
    assert v38.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v38.euler_recon.tmlpu_bound_tvd_separate is True
    assert v38.euler_recon.euler_face_positivity_limiter is True

    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_head_generic is True)
    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_disable_specialized_relax is True)
    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_head_generic_blend_cap == 1.0)
    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_legacy_order is False)
    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_legacy_relax is False)
    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_legacy_tvd_after_weak is False)

    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_admissibility_damp is False)
    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_entropy_accept is False)
    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_shock_gate is False)
    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_value_scaling is False)
    assert v38.euler_recon.euler_density_contact_weak_face_root_blend == 0.0
    assert v38.euler_recon.euler_density_contact_weak_face_swirl_extra == 0.0
    assert v38.euler_recon.euler_density_contact_weak_face_shock_power == 1.0
    assert v38.euler_recon.euler_density_contact_weak_face_mlp_cap == 1.0

    assert v38.euler_recon.euler_pressure_contact_entropy_blend is False
    assert v38.euler_recon.euler_tangential_pair_restore_on is False
    assert v38.euler_recon.euler_tangential_pair_extend_on is False
    assert (
        v38.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is False)
    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_stream_coherence_on is False)

    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_legacy_order is True)
    assert (
        v37.euler_recon
        .euler_density_contact_weak_face_head_generic is False)


def test_v38_candidate_reads_generic_weak_face_blend_cap(monkeypatch):
    monkeypatch.setenv('TMLPU_V38_GENERIC_WEAK_FACE_BLEND_CAP', '0.85')
    v38 = bench._reconstruction_from_key("tmlpu_v38_unified_on")

    assert (
        v38.euler_recon
        .euler_density_contact_weak_face_head_generic_blend_cap == 0.85)


def test_v39_candidate_uses_contact_characteristic_postpass_only():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v32 = bench._reconstruction_from_key("tmlpu_v32_unified_on")
    v38 = bench._reconstruction_from_key("tmlpu_v38_unified_on")
    v39 = bench._reconstruction_from_key("tmlpu_v39_unified_on")

    assert v39.name == "t_mlp_u_unified"
    assert type(v39.scalar_recon) is type(v3.scalar_recon)
    assert type(v39.euler_recon) is type(v32.euler_recon)

    assert v39.euler_recon.euler_density_contact_weak_face_mlp is False
    assert v39.euler_recon.euler_density_contact_weak_face_value_scaling is True
    assert (
        v39.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode
        == 'coherent_shear_micro_restore')
    assert (
        v39.euler_recon
        .euler_density_contact_weak_face_value_scaling_require_coherent_shear
        is True)
    assert (
        v39.euler_recon
        .euler_density_contact_weak_face_value_scaling_artifact_reject
        is True)
    assert (
        v39.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha
        == 0.070)
    assert (
        v39.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad
        == 0.010)
    assert v39.euler_recon.euler_tangential_pair_restore_on is True
    assert v39.euler_recon.euler_tangential_pair_restore_alpha == 0.030
    assert v39.euler_recon.euler_tangential_pair_restore_cap == 0.055
    assert v39.euler_recon.euler_tangential_pair_restore_wave_cap == 0.008
    assert v39.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v39.euler_recon.tmlpu_bound_tvd_separate is True
    assert v39.euler_recon.euler_face_positivity_limiter is True

    assert v39.euler_recon.euler_contact_characteristic_postpass_on is True
    assert v39.euler_recon.euler_contact_characteristic_entropy_alpha == 0.035
    assert v39.euler_recon.euler_contact_characteristic_tangential_alpha == 0.020
    assert v39.euler_recon.euler_contact_characteristic_entropy_cap == 0.010
    assert v39.euler_recon.euler_contact_characteristic_tangential_cap == 0.025
    assert (
        v39.euler_recon
        .euler_contact_characteristic_tangential_wave_cap == 0.0035)
    assert v39.euler_recon.euler_contact_characteristic_pressure_alpha == 0.0
    assert v39.euler_recon.euler_contact_characteristic_normal_alpha == 0.0
    assert v39.euler_recon.euler_contact_characteristic_mood_fallback_on is True

    assert v39.euler_recon.euler_pressure_contact_entropy_blend is False
    assert v39.euler_recon.euler_tangential_pair_extend_on is False
    assert (
        v39.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is False)
    assert (
        v39.euler_recon
        .euler_density_contact_weak_face_stream_coherence_on is False)
    assert v39.euler_recon.euler_density_contact_weak_face_head_generic is False
    assert (
        v39.euler_recon
        .euler_density_contact_weak_face_disable_specialized_relax is False)

    assert v38.euler_recon.euler_density_contact_weak_face_head_generic is True
    for key in (
        "tmlpu_leveque_on",
        "tmlpu_double_mach_on",
        "tmlpu_mach3_step_on",
    ):
        alias = bench._reconstruction_from_key(key)
        assert type(alias.scalar_recon) is type(v3.scalar_recon)
        assert type(alias.euler_recon) is type(v3.euler_recon)


def test_v40_candidate_uses_patch_contact_shear_postpass_only():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v32 = bench._reconstruction_from_key("tmlpu_v32_unified_on")
    v39 = bench._reconstruction_from_key("tmlpu_v39_unified_on")
    v40 = bench._reconstruction_from_key("tmlpu_v40_unified_on")

    assert v40.name == "t_mlp_u_unified"
    assert type(v40.scalar_recon) is type(v3.scalar_recon)
    assert type(v40.euler_recon) is type(v32.euler_recon)

    assert v40.euler_recon.euler_density_contact_weak_face_mlp is False
    assert v40.euler_recon.euler_density_contact_weak_face_value_scaling is True
    assert (
        v40.euler_recon
        .euler_density_contact_weak_face_value_scaling_mode
        == 'coherent_shear_micro_restore')
    assert (
        v40.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha
        == 0.070)
    assert (
        v40.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad
        == 0.010)
    assert v40.euler_recon.euler_tangential_pair_restore_on is True
    assert v40.euler_recon.euler_tangential_pair_restore_alpha == 0.030
    assert v40.euler_recon.euler_tangential_pair_restore_cap == 0.055
    assert v40.euler_recon.euler_tangential_pair_restore_wave_cap == 0.008
    assert v40.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v40.euler_recon.tmlpu_bound_tvd_separate is True
    assert v40.euler_recon.euler_face_positivity_limiter is True

    assert v40.euler_recon.euler_patch_contact_shear_postpass_on is True
    assert v40.euler_recon.euler_patch_contact_shear_neighbor_blend == 0.30
    assert v40.euler_recon.euler_patch_contact_shear_entropy_alpha == 0.030
    assert v40.euler_recon.euler_patch_contact_shear_tangential_alpha == 0.018
    assert v40.euler_recon.euler_patch_contact_shear_entropy_cap == 0.008
    assert v40.euler_recon.euler_patch_contact_shear_tangential_cap == 0.020
    assert (
        v40.euler_recon
        .euler_patch_contact_shear_tangential_wave_cap == 0.003)
    assert v40.euler_recon.euler_patch_contact_shear_min_valid_neighbours == 2

    assert v40.euler_recon.euler_contact_characteristic_postpass_on is False
    assert v39.euler_recon.euler_contact_characteristic_postpass_on is True
    assert v40.euler_recon.euler_pressure_contact_entropy_blend is False
    assert v40.euler_recon.euler_tangential_pair_extend_on is False
    assert (
        v40.euler_recon
        .euler_tangential_pair_restore_stream_coherence_on is False)
    assert (
        v40.euler_recon
        .euler_density_contact_weak_face_stream_coherence_on is False)
    assert v40.euler_recon.euler_density_contact_weak_face_head_generic is False
    assert (
        v40.euler_recon
        .euler_density_contact_weak_face_disable_specialized_relax is False)

    for key in (
        "tmlpu_leveque_on",
        "tmlpu_double_mach_on",
        "tmlpu_mach3_step_on",
    ):
        alias = bench._reconstruction_from_key(key)
        assert type(alias.scalar_recon) is type(v3.scalar_recon)
        assert type(alias.euler_recon) is type(v3.euler_recon)


def test_v41_candidate_adds_pair_spacing_to_v40_anchor_only():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v40 = bench._reconstruction_from_key("tmlpu_v40_unified_on")
    v41 = bench._reconstruction_from_key("tmlpu_v41_unified_on")

    assert v41.name == "t_mlp_u_unified"
    assert type(v41.scalar_recon) is type(v3.scalar_recon)
    assert type(v41.euler_recon) is type(v40.euler_recon)

    assert v41.euler_recon.euler_density_contact_weak_face_mlp is False
    assert v41.euler_recon.euler_density_contact_weak_face_value_scaling is True
    assert (
        v41.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha
        == 0.070)
    assert (
        v41.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad
        == 0.010)
    assert v41.euler_recon.euler_tangential_pair_restore_on is True
    assert v41.euler_recon.euler_tangential_pair_restore_alpha == 0.030
    assert v41.euler_recon.euler_tangential_pair_restore_cap == 0.055
    assert v41.euler_recon.euler_tangential_pair_restore_wave_cap == 0.008
    assert v41.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v41.euler_recon.tmlpu_bound_tvd_separate is True
    assert v41.euler_recon.euler_face_positivity_limiter is True

    assert v41.euler_recon.euler_patch_contact_shear_postpass_on is True
    assert v41.euler_recon.euler_patch_contact_shear_neighbor_blend == 0.30
    assert v41.euler_recon.euler_patch_contact_shear_entropy_alpha == 0.030
    assert v41.euler_recon.euler_patch_contact_shear_tangential_alpha == 0.018
    assert v41.euler_recon.euler_patch_contact_shear_entropy_cap == 0.008
    assert v41.euler_recon.euler_patch_contact_shear_tangential_cap == 0.020
    assert (
        v41.euler_recon
        .euler_patch_contact_shear_tangential_wave_cap == 0.003)
    assert v41.euler_recon.euler_patch_contact_shear_min_valid_neighbours == 2
    assert v41.euler_recon.euler_patch_contact_shear_pair_spacing_on is True
    assert v41.euler_recon.euler_patch_contact_shear_pair_spacing_beta == 0.35
    assert v41.euler_recon.euler_patch_contact_shear_gate_cap == 1.0
    assert (
        v41.euler_recon
        .euler_patch_contact_shear_pressure_floor_factor == 0.86)
    assert v41.euler_recon.euler_patch_contact_shear_pressure_margin_on is True

    assert v40.euler_recon.euler_patch_contact_shear_pair_spacing_on is False
    assert v40.euler_recon.euler_patch_contact_shear_pressure_floor_factor == 0.80
    assert v41.euler_recon.euler_contact_characteristic_postpass_on is False
    assert v41.euler_recon.euler_pressure_contact_entropy_blend is False

    for key in (
        "tmlpu_leveque_on",
        "tmlpu_double_mach_on",
        "tmlpu_mach3_step_on",
    ):
        alias = bench._reconstruction_from_key(key)
        assert type(alias.scalar_recon) is type(v3.scalar_recon)
        assert type(alias.euler_recon) is type(v3.euler_recon)


def test_v42_candidate_adds_late_pressure_rollback_to_v40_anchor_only():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    v40 = bench._reconstruction_from_key("tmlpu_v40_unified_on")
    v42 = bench._reconstruction_from_key("tmlpu_v42_unified_on")

    assert v42.name == "t_mlp_u_unified"
    assert type(v42.scalar_recon) is type(v3.scalar_recon)
    assert type(v42.euler_recon) is type(v40.euler_recon)

    assert v42.euler_recon.euler_density_contact_weak_face_mlp is False
    assert v42.euler_recon.euler_density_contact_weak_face_value_scaling is True
    assert (
        v42.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_alpha
        == 0.070)
    assert (
        v42.euler_recon
        .euler_density_contact_weak_face_value_scaling_shear_blend_bound_pad
        == 0.010)
    assert v42.euler_recon.euler_tangential_pair_restore_on is True
    assert v42.euler_recon.euler_tangential_pair_restore_alpha == 0.030
    assert v42.euler_recon.euler_tangential_pair_restore_cap == 0.055
    assert v42.euler_recon.euler_tangential_pair_restore_wave_cap == 0.008
    assert v42.euler_recon.euler_tangential_contact_relax_flatten is True
    assert v42.euler_recon.tmlpu_bound_tvd_separate is True
    assert v42.euler_recon.euler_face_positivity_limiter is True

    assert v42.euler_recon.euler_patch_contact_shear_postpass_on is True
    assert v42.euler_recon.euler_patch_contact_shear_neighbor_blend == 0.30
    assert v42.euler_recon.euler_patch_contact_shear_entropy_alpha == 0.030
    assert v42.euler_recon.euler_patch_contact_shear_tangential_alpha == 0.018
    assert v42.euler_recon.euler_patch_contact_shear_entropy_cap == 0.008
    assert v42.euler_recon.euler_patch_contact_shear_tangential_cap == 0.020
    assert (
        v42.euler_recon
        .euler_patch_contact_shear_tangential_wave_cap == 0.003)
    assert v42.euler_recon.euler_patch_contact_shear_min_valid_neighbours == 2

    assert v42.euler_recon.euler_patch_contact_shear_pair_spacing_on is False
    assert v42.euler_recon.euler_patch_contact_shear_pair_spacing_beta == 0.0
    assert v42.euler_recon.euler_patch_contact_shear_late_pressure_rollback_on is True
    assert v42.euler_recon.euler_patch_contact_shear_p_floor_abs == 0.925
    assert v42.euler_recon.euler_patch_contact_shear_rho_floor_abs == 0.700
    assert v42.euler_recon.euler_patch_contact_shear_pressure_floor_factor == 0.86
    assert v42.euler_recon.euler_patch_contact_shear_rho_floor_factor == 0.72
    assert (
        v42.euler_recon
        .euler_patch_contact_shear_tangential_rollback_theta == 0.50)
    assert v42.euler_recon.euler_patch_contact_shear_pressure_margin_on is False
    assert v42.euler_recon.euler_contact_characteristic_postpass_on is False
    assert v42.euler_recon.euler_pressure_contact_entropy_blend is False

    assert v40.euler_recon.euler_patch_contact_shear_late_pressure_rollback_on is False
    for key in (
        "tmlpu_leveque_on",
        "tmlpu_double_mach_on",
        "tmlpu_mach3_step_on",
    ):
        alias = bench._reconstruction_from_key(key)
        assert type(alias.scalar_recon) is type(v3.scalar_recon)
        assert type(alias.euler_recon) is type(v3.euler_recon)


def test_official_aliases_remain_unified_for_v32():
    v3 = bench._reconstruction_from_key("tmlpu_v3_unified_on")
    for key in (
        "tmlpu_leveque_on",
        "tmlpu_double_mach_on",
        "tmlpu_mach3_step_on",
    ):
        recon = bench._reconstruction_from_key(key)
        assert recon.name == v3.name
        assert type(recon.scalar_recon) is type(v3.scalar_recon)
        assert type(recon.euler_recon) is type(v3.euler_recon)


def test_validation_grid_contract_defaults():
    assert bench.LEVEQUE_QUICK_N == 100
    assert bench.LEVEQUE_PAPER_N == 100
    assert bench.DOUBLE_MACH_QUICK_GRID == (480, 120)
    assert bench.DOUBLE_MACH_PAPER_GRID == (960, 240)
    assert bench.MACH3_STEP_QUICK_GRID == (200, 80)
    assert bench.MACH3_STEP_PAPER_GRID == (480, 160)


def test_v93_diagnostics_are_noop_for_v92_face_states(monkeypatch, tmp_path):
    mesh = bench._tri_mesh(4, 3, 1.0, 1.0)
    eq = bench.Euler2D(gamma=1.4)
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    W_cell = np.vstack([
        1.0 + 0.08 * np.sin(2.0 * np.pi * x) + 0.03 * y,
        2.0 + 0.15 * y - 0.04 * x,
        0.08 * np.sin(np.pi * x) - 0.12 * y,
        1.0 + 0.05 * x + 0.02 * np.cos(np.pi * y),
    ])

    monkeypatch.delenv('TMLPU_V93_GATE_DIAGNOSTICS', raising=False)
    v92 = bench._reconstruction_from_key('tmlpu_v92_unified_on')
    W92_L, W92_R = v92.reconstruct(mesh, W_cell, eq)

    monkeypatch.setenv('TMLPU_V93_GATE_DIAGNOSTICS', '1')
    monkeypatch.setenv(
        'TMLPU_V93_GATE_DIAGNOSTICS_PATH',
        str(tmp_path / 'v93_gate_diagnostics.json'))
    v93 = bench._reconstruction_from_key('tmlpu_v93_diag_unified_on')
    W93_L, W93_R = v93.reconstruct(mesh, W_cell, eq)

    assert np.max(np.abs(W93_L - W92_L)) <= 1.0e-13
    assert np.max(np.abs(W93_R - W92_R)) <= 1.0e-13


def test_v95_legacy_pair_target_is_opt_in_from_v92_base(monkeypatch):
    monkeypatch.delenv('TMLPU_V115_TAIL_DIAGNOSTICS', raising=False)
    monkeypatch.delenv('TMLPU_V117_FEATURE_DIAGNOSTICS', raising=False)
    v92 = bench._reconstruction_from_key('tmlpu_v92_unified_on')
    v95 = bench._reconstruction_from_key('tmlpu_v95_legacy_pair_target_on')
    v96 = bench._reconstruction_from_key(
        'tmlpu_v96_legacy_pair_target_half_on')
    v97 = bench._reconstruction_from_key(
        'tmlpu_v97_legacy_pair_target_075_on')
    v98 = bench._reconstruction_from_key(
        'tmlpu_v98_legacy_pair_target_0875_on')
    v99 = bench._reconstruction_from_key(
        'tmlpu_v99_split_legacy_target_on')
    v100 = bench._reconstruction_from_key(
        'tmlpu_v100_safe_legacy_target_on')
    v101 = bench._reconstruction_from_key(
        'tmlpu_v101_safe_legacy_pressure014_on')
    v102 = bench._reconstruction_from_key(
        'tmlpu_v102_safe_legacy_contact035_on')
    v103 = bench._reconstruction_from_key(
        'tmlpu_v103_safe_legacy_shear072_on')
    v104 = bench._reconstruction_from_key(
        'tmlpu_v104_safe_legacy_shear072_norm020_on')
    v105 = bench._reconstruction_from_key(
        'tmlpu_v105_safe_legacy_coherence_on')
    v106 = bench._reconstruction_from_key(
        'tmlpu_v106_safe_legacy_qcurv_on')
    v107 = bench._reconstruction_from_key(
        'tmlpu_v107_safe_legacy_capboost_on')
    v108 = bench._reconstruction_from_key(
        'tmlpu_v108_signed_density_support018_on')
    v109 = bench._reconstruction_from_key(
        'tmlpu_v109_tail_density_support_mid_on')
    v110 = bench._reconstruction_from_key(
        'tmlpu_v110_tail_density_shockdamp_on')
    v111 = bench._reconstruction_from_key(
        'tmlpu_v111_tail_density_min015_on')
    v112 = bench._reconstruction_from_key(
        'tmlpu_v112_tail_density_min0145_on')
    v113 = bench._reconstruction_from_key(
        'tmlpu_v113_tail_density_min015_full055_on')
    v114 = bench._reconstruction_from_key(
        'tmlpu_v114_tail_density_beta115_on')
    v115 = bench._reconstruction_from_key(
        'tmlpu_v115_v111_taildiag_on')
    v116 = bench._reconstruction_from_key(
        'tmlpu_v116_tail_safe_floor_on')
    v117 = bench._reconstruction_from_key(
        'tmlpu_v117_v111_featurediag_on')
    v118 = bench._reconstruction_from_key(
        'tmlpu_v118_shear_contact_relief_on')
    v119 = bench._reconstruction_from_key(
        'tmlpu_v119_shear_contact_relief_floor04_on')
    v120 = bench._reconstruction_from_key(
        'tmlpu_v120_shear_contact_relief_floor02_on')
    v121 = bench._reconstruction_from_key(
        'tmlpu_v121_shear_contact_relief_p006_on')
    v122 = bench._reconstruction_from_key(
        'tmlpu_v122_shear_contact_relief_c0010_on')
    v123 = bench._reconstruction_from_key(
        'tmlpu_v123_shear_contact_relief_floor03_on')
    v124 = bench._reconstruction_from_key(
        'tmlpu_v124_shear_contact_relief_floor035_on')
    v125 = bench._reconstruction_from_key(
        'tmlpu_v125_shear_contact_relief_floor0375_on')
    v126 = bench._reconstruction_from_key(
        'tmlpu_v126_curve_only_relief_floor04_on')
    v127 = bench._reconstruction_from_key(
        'tmlpu_v127_signed_only_relief_floor04_on')
    v128 = bench._reconstruction_from_key(
        'tmlpu_v128_asym_relief_signed04_curve02_on')
    v129 = bench._reconstruction_from_key(
        'tmlpu_v129_signed_relief_density014_on')
    v130 = bench._reconstruction_from_key(
        'tmlpu_v130_signed_only_relief_floor06_on')
    v131 = bench._reconstruction_from_key(
        'tmlpu_v131_signed_gate_decay_relief_on')
    v132 = bench._reconstruction_from_key(
        'tmlpu_v132_signed_gate_decay_floor07_on')
    v133 = bench._reconstruction_from_key(
        'tmlpu_v133_signed_decay_floor10_capboost_on')
    v134 = bench._reconstruction_from_key(
        'tmlpu_v134_signed_postrollback_preserve_on')
    v135 = bench._reconstruction_from_key(
        'tmlpu_v135_signed_anchored_curve_assist_on')
    v136 = bench._reconstruction_from_key(
        'tmlpu_v136_signed_aligned_curve_assist_on')
    v137 = bench._reconstruction_from_key(
        'tmlpu_v137_signed_anchor_curve_floor06_on')
    v138 = bench._reconstruction_from_key(
        'tmlpu_v138_signed_anchor_curve_keep_signed_on')
    v139 = bench._reconstruction_from_key(
        'tmlpu_v139_signed_anchor_density_trace_on')
    v140 = bench._reconstruction_from_key(
        'tmlpu_v140_v131_signed_anchor_curve_gate_diag_on')
    v141 = bench._reconstruction_from_key(
        'tmlpu_v141_anchor_curve_diag_epsfix_on')
    v142 = bench._reconstruction_from_key(
        'tmlpu_v142_highsafe_raw_curve_microassist_on')
    v143 = bench._reconstruction_from_key(
        'tmlpu_v143_signed_sidecar_decay_on')
    v144 = bench._reconstruction_from_key(
        'tmlpu_v144_signed_sidecar_decay_blend015_on')
    v145 = bench._reconstruction_from_key(
        'tmlpu_v145_signed_decay_floor12_on')

    assert type(v95.scalar_recon) is type(v92.scalar_recon)
    assert type(v95.euler_recon) is type(v92.euler_recon)
    assert type(v96.scalar_recon) is type(v92.scalar_recon)
    assert type(v96.euler_recon) is type(v92.euler_recon)
    assert type(v97.scalar_recon) is type(v92.scalar_recon)
    assert type(v97.euler_recon) is type(v92.euler_recon)
    assert type(v98.scalar_recon) is type(v92.scalar_recon)
    assert type(v98.euler_recon) is type(v92.euler_recon)
    assert type(v99.scalar_recon) is type(v92.scalar_recon)
    assert type(v99.euler_recon) is type(v92.euler_recon)
    assert type(v100.scalar_recon) is type(v92.scalar_recon)
    assert type(v100.euler_recon) is type(v92.euler_recon)
    assert type(v101.scalar_recon) is type(v92.scalar_recon)
    assert type(v101.euler_recon) is type(v92.euler_recon)
    assert type(v102.scalar_recon) is type(v92.scalar_recon)
    assert type(v102.euler_recon) is type(v92.euler_recon)
    assert type(v103.scalar_recon) is type(v92.scalar_recon)
    assert type(v103.euler_recon) is type(v92.euler_recon)
    assert type(v104.scalar_recon) is type(v92.scalar_recon)
    assert type(v104.euler_recon) is type(v92.euler_recon)
    assert type(v105.scalar_recon) is type(v92.scalar_recon)
    assert type(v105.euler_recon) is type(v92.euler_recon)
    assert type(v106.scalar_recon) is type(v92.scalar_recon)
    assert type(v106.euler_recon) is type(v92.euler_recon)
    assert type(v107.scalar_recon) is type(v92.scalar_recon)
    assert type(v107.euler_recon) is type(v92.euler_recon)
    assert type(v108.scalar_recon) is type(v92.scalar_recon)
    assert type(v108.euler_recon) is type(v92.euler_recon)
    assert type(v109.scalar_recon) is type(v92.scalar_recon)
    assert type(v109.euler_recon) is type(v92.euler_recon)
    assert type(v110.scalar_recon) is type(v92.scalar_recon)
    assert type(v110.euler_recon) is type(v92.euler_recon)
    assert type(v111.scalar_recon) is type(v92.scalar_recon)
    assert type(v111.euler_recon) is type(v92.euler_recon)
    assert type(v112.scalar_recon) is type(v92.scalar_recon)
    assert type(v112.euler_recon) is type(v92.euler_recon)
    assert type(v113.scalar_recon) is type(v92.scalar_recon)
    assert type(v113.euler_recon) is type(v92.euler_recon)
    assert type(v114.scalar_recon) is type(v92.scalar_recon)
    assert type(v114.euler_recon) is type(v92.euler_recon)
    assert type(v115.scalar_recon) is type(v92.scalar_recon)
    assert type(v115.euler_recon) is type(v92.euler_recon)
    assert type(v116.scalar_recon) is type(v92.scalar_recon)
    assert type(v116.euler_recon) is type(v92.euler_recon)
    assert type(v117.scalar_recon) is type(v92.scalar_recon)
    assert type(v117.euler_recon) is type(v92.euler_recon)
    assert type(v118.scalar_recon) is type(v92.scalar_recon)
    assert type(v118.euler_recon) is type(v92.euler_recon)
    assert type(v119.scalar_recon) is type(v92.scalar_recon)
    assert type(v119.euler_recon) is type(v92.euler_recon)
    assert type(v120.scalar_recon) is type(v92.scalar_recon)
    assert type(v120.euler_recon) is type(v92.euler_recon)
    assert type(v121.scalar_recon) is type(v92.scalar_recon)
    assert type(v121.euler_recon) is type(v92.euler_recon)
    assert type(v122.scalar_recon) is type(v92.scalar_recon)
    assert type(v122.euler_recon) is type(v92.euler_recon)
    assert type(v123.scalar_recon) is type(v92.scalar_recon)
    assert type(v123.euler_recon) is type(v92.euler_recon)
    assert type(v124.scalar_recon) is type(v92.scalar_recon)
    assert type(v124.euler_recon) is type(v92.euler_recon)
    assert type(v125.scalar_recon) is type(v92.scalar_recon)
    assert type(v125.euler_recon) is type(v92.euler_recon)
    assert type(v126.scalar_recon) is type(v92.scalar_recon)
    assert type(v126.euler_recon) is type(v92.euler_recon)
    assert type(v127.scalar_recon) is type(v92.scalar_recon)
    assert type(v127.euler_recon) is type(v92.euler_recon)
    assert type(v128.scalar_recon) is type(v92.scalar_recon)
    assert type(v128.euler_recon) is type(v92.euler_recon)
    assert type(v129.scalar_recon) is type(v92.scalar_recon)
    assert type(v129.euler_recon) is type(v92.euler_recon)
    assert type(v130.scalar_recon) is type(v92.scalar_recon)
    assert type(v130.euler_recon) is type(v92.euler_recon)
    assert type(v131.scalar_recon) is type(v92.scalar_recon)
    assert type(v131.euler_recon) is type(v92.euler_recon)
    assert type(v132.scalar_recon) is type(v92.scalar_recon)
    assert type(v132.euler_recon) is type(v92.euler_recon)
    assert type(v133.scalar_recon) is type(v92.scalar_recon)
    assert type(v133.euler_recon) is type(v92.euler_recon)
    assert type(v134.scalar_recon) is type(v92.scalar_recon)
    assert type(v134.euler_recon) is type(v92.euler_recon)
    assert type(v135.scalar_recon) is type(v92.scalar_recon)
    assert type(v135.euler_recon) is type(v92.euler_recon)
    assert type(v136.scalar_recon) is type(v92.scalar_recon)
    assert type(v136.euler_recon) is type(v92.euler_recon)
    assert type(v137.scalar_recon) is type(v92.scalar_recon)
    assert type(v137.euler_recon) is type(v92.euler_recon)
    assert type(v138.scalar_recon) is type(v92.scalar_recon)
    assert type(v138.euler_recon) is type(v92.euler_recon)
    assert type(v139.scalar_recon) is type(v92.scalar_recon)
    assert type(v139.euler_recon) is type(v92.euler_recon)
    assert type(v140.scalar_recon) is type(v92.scalar_recon)
    assert type(v140.euler_recon) is type(v92.euler_recon)
    assert type(v141.scalar_recon) is type(v92.scalar_recon)
    assert type(v141.euler_recon) is type(v92.euler_recon)
    assert type(v142.scalar_recon) is type(v92.scalar_recon)
    assert type(v142.euler_recon) is type(v92.euler_recon)
    assert type(v143.scalar_recon) is type(v92.scalar_recon)
    assert type(v143.euler_recon) is type(v92.euler_recon)
    assert type(v144.scalar_recon) is type(v92.scalar_recon)
    assert type(v144.euler_recon) is type(v92.euler_recon)
    assert type(v145.scalar_recon) is type(v92.scalar_recon)
    assert type(v145.euler_recon) is type(v92.euler_recon)
    assert v92.euler_recon.euler_tangential_legacy_pair_target_on is False
    assert v92.euler_recon.euler_tangential_legacy_pair_target_blend == 0.0
    assert (
        v92.euler_recon.euler_tangential_signed_pair_legacy_target_blend
        == -1.0)
    assert (
        v92.euler_recon.euler_tangential_density_curve_legacy_target_blend
        == -1.0)
    assert v92.euler_recon.euler_tangential_safe_legacy_gate_on is False
    assert v95.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v95.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v96.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v96.euler_recon.euler_tangential_legacy_pair_target_blend == 0.50
    assert v97.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v97.euler_recon.euler_tangential_legacy_pair_target_blend == 0.75
    assert v98.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v98.euler_recon.euler_tangential_legacy_pair_target_blend == 0.875
    assert v99.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v99.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert (
        v99.euler_recon.euler_tangential_signed_pair_legacy_target_blend
        == 0.75)
    assert (
        v99.euler_recon.euler_tangential_density_curve_legacy_target_blend
        == 1.0)
    assert v100.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v100.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v100.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v100.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v100.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v100.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v100.euler_recon.euler_tangential_safe_legacy_shear_min == 0.82
    assert v100.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v101.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v101.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v101.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v101.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.014
    assert v101.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v101.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v101.euler_recon.euler_tangential_safe_legacy_shear_min == 0.82
    assert v101.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v102.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v102.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v102.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v102.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v102.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v102.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v102.euler_recon.euler_tangential_safe_legacy_shear_min == 0.82
    assert v102.euler_recon.euler_tangential_safe_legacy_contact_min == 0.35
    assert v103.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v103.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v103.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v103.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v103.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v103.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v103.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v103.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v104.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v104.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v104.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v104.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v104.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v104.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.20
    assert v104.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v104.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v105.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v105.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v105.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v105.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v105.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v105.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v105.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v105.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v105.euler_recon.euler_tangential_safe_legacy_coherence_on is True
    assert v105.euler_recon.euler_tangential_safe_legacy_coherence_beta == 0.25
    assert v105.euler_recon.euler_tangential_safe_legacy_coherence_floor == 0.08
    assert v105.euler_recon.euler_tangential_safe_legacy_coherence_cap == 0.35
    assert v106.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v106.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v106.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v106.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v106.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v106.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v106.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v106.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v106.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v106.euler_recon.euler_tangential_safe_legacy_qcurv_on is True
    assert v106.euler_recon.euler_tangential_safe_legacy_qcurv_beta == 0.18
    assert v106.euler_recon.euler_tangential_safe_legacy_qcurv_q_min == 0.012
    assert v106.euler_recon.euler_tangential_safe_legacy_qcurv_q_full == 0.045
    assert v106.euler_recon.euler_tangential_safe_legacy_qcurv_curve_min == 0.20
    assert v106.euler_recon.euler_tangential_safe_legacy_qcurv_curve_full == 0.50
    assert v107.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v107.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v107.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v107.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v107.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v107.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v107.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v107.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v107.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v107.euler_recon.euler_tangential_safe_legacy_qcurv_on is False
    assert v107.euler_recon.euler_tangential_signed_pair_tail_cap == 0.045
    assert v107.euler_recon.euler_tangential_signed_pair_tail_wave_cap == 0.0060
    assert v107.euler_recon.euler_tangential_density_curve_pair_tail_cap == 0.040
    assert (
        v107.euler_recon.euler_tangential_density_curve_pair_tail_wave_cap
        == 0.0055)
    assert v108.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v108.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v108.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v108.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v108.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v108.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v108.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v108.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v108.euler_recon.euler_tangential_pair_gate_density_support_min
        == v103.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v108.euler_recon.euler_tangential_pair_gate_density_support_full
        == v103.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert v108.euler_recon.euler_tangential_tail_density_support_min == 0.014
    assert v108.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v108.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v108.euler_recon.euler_tangential_safe_legacy_qcurv_on is False
    assert v109.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v109.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v109.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v109.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v109.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v109.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v109.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v109.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v109.euler_recon.euler_tangential_pair_gate_density_support_min
        == v103.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v109.euler_recon.euler_tangential_pair_gate_density_support_full
        == v103.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert v109.euler_recon.euler_tangential_tail_density_support_min == 0.017
    assert v109.euler_recon.euler_tangential_tail_density_support_full == 0.070
    assert v109.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v109.euler_recon.euler_tangential_safe_legacy_qcurv_on is False
    assert v110.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v110.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v110.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v110.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v110.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v110.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v110.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v110.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v110.euler_recon.euler_tangential_pair_gate_density_support_min
        == v103.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v110.euler_recon.euler_tangential_pair_gate_density_support_full
        == v103.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert v110.euler_recon.euler_tangential_tail_density_support_min == 0.014
    assert v110.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v110.euler_recon.euler_tangential_tail_density_shock_damp_on is True
    assert v110.euler_recon.euler_tangential_tail_density_shock_damp_theta == 0.65
    assert (
        v110.euler_recon
        .euler_tangential_tail_density_shock_damp_pressure_min == 0.010)
    assert (
        v110.euler_recon
        .euler_tangential_tail_density_shock_damp_compression_min == 0.002)
    assert (
        v110.euler_recon
        .euler_tangential_tail_density_shock_damp_normality_min == 0.16)
    assert v110.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v110.euler_recon.euler_tangential_safe_legacy_qcurv_on is False
    assert v111.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v111.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v111.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v111.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v111.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v111.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v111.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v111.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v111.euler_recon.euler_tangential_pair_gate_density_support_min
        == v103.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v111.euler_recon.euler_tangential_pair_gate_density_support_full
        == v103.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert v111.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v111.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v111.euler_recon.euler_tangential_tail_density_shock_damp_on is False
    assert v111.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v111.euler_recon.euler_tangential_safe_legacy_qcurv_on is False
    assert v112.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v112.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v112.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v112.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v112.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v112.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v112.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v112.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v112.euler_recon.euler_tangential_pair_gate_density_support_min
        == v103.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v112.euler_recon.euler_tangential_pair_gate_density_support_full
        == v103.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert v112.euler_recon.euler_tangential_tail_density_support_min == 0.0145
    assert v112.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v112.euler_recon.euler_tangential_tail_density_shock_damp_on is False
    assert v112.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v112.euler_recon.euler_tangential_safe_legacy_qcurv_on is False
    assert v113.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v113.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v113.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v113.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v113.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v113.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v113.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v113.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v113.euler_recon.euler_tangential_pair_gate_density_support_min
        == v103.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v113.euler_recon.euler_tangential_pair_gate_density_support_full
        == v103.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert v113.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v113.euler_recon.euler_tangential_tail_density_support_full == 0.055
    assert v113.euler_recon.euler_tangential_tail_density_shock_damp_on is False
    assert v113.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v113.euler_recon.euler_tangential_safe_legacy_qcurv_on is False
    assert v114.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v114.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v114.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v114.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v114.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v114.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v114.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v114.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v114.euler_recon.euler_tangential_pair_gate_density_support_min
        == v103.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v114.euler_recon.euler_tangential_pair_gate_density_support_full
        == v103.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert v114.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v114.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v114.euler_recon.euler_tangential_tail_density_shock_damp_on is False
    assert v114.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v114.euler_recon.euler_tangential_safe_legacy_qcurv_on is False
    assert (
        v114.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 1.15
        * v103.euler_recon.euler_tangential_density_curve_pair_tail_beta)
    assert (
        v114.euler_recon.euler_tangential_signed_pair_tail_beta
        == v103.euler_recon.euler_tangential_signed_pair_tail_beta)
    assert (
        v114.euler_recon.euler_tangential_signed_pair_tail_cap
        == v103.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert v115.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v115.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v115.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v115.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v115.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v115.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v115.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v115.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v115.euler_recon.euler_tangential_pair_gate_density_support_min
        == v111.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v115.euler_recon.euler_tangential_pair_gate_density_support_full
        == v111.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert (
        v115.euler_recon.euler_tangential_tail_density_support_min
        == v111.euler_recon.euler_tangential_tail_density_support_min)
    assert (
        v115.euler_recon.euler_tangential_tail_density_support_full
        == v111.euler_recon.euler_tangential_tail_density_support_full)
    assert (
        v115.euler_recon.euler_tangential_tail_density_shock_damp_on
        == v111.euler_recon.euler_tangential_tail_density_shock_damp_on)
    assert (
        v115.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == v111.euler_recon.euler_tangential_density_curve_pair_tail_beta)
    assert (
        v115.euler_recon.euler_tangential_signed_pair_tail_beta
        == v111.euler_recon.euler_tangential_signed_pair_tail_beta)
    assert v116.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v116.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v116.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v116.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v116.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v116.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v116.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v116.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v116.euler_recon.euler_tangential_pair_gate_density_support_min
        == v111.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v116.euler_recon.euler_tangential_pair_gate_density_support_full
        == v111.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert v116.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v116.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v116.euler_recon.euler_tangential_tail_density_shock_damp_on is False
    assert v116.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v116.euler_recon.euler_tangential_safe_legacy_qcurv_on is False
    assert v116.euler_recon.euler_tangential_tail_safe_floor_on is True
    assert v116.euler_recon.euler_tangential_tail_safe_floor == 0.18
    assert v117.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v117.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v117.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v117.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v117.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v117.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v117.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v117.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v117.euler_recon.euler_tangential_pair_gate_density_support_min
        == v111.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v117.euler_recon.euler_tangential_pair_gate_density_support_full
        == v111.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert (
        v117.euler_recon.euler_tangential_tail_density_support_min
        == v111.euler_recon.euler_tangential_tail_density_support_min)
    assert (
        v117.euler_recon.euler_tangential_tail_density_support_full
        == v111.euler_recon.euler_tangential_tail_density_support_full)
    assert (
        v117.euler_recon.euler_tangential_tail_density_shock_damp_on
        == v111.euler_recon.euler_tangential_tail_density_shock_damp_on)
    assert (
        v117.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == v111.euler_recon.euler_tangential_density_curve_pair_tail_beta)
    assert (
        v117.euler_recon.euler_tangential_signed_pair_tail_beta
        == v111.euler_recon.euler_tangential_signed_pair_tail_beta)
    assert os.environ.get('TMLPU_V117_FEATURE_DIAGNOSTICS') is None
    assert v118.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v118.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v118.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v118.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v118.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v118.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v118.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v118.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert (
        v118.euler_recon.euler_tangential_pair_gate_density_support_min
        == v111.euler_recon.euler_tangential_pair_gate_density_support_min)
    assert (
        v118.euler_recon.euler_tangential_pair_gate_density_support_full
        == v111.euler_recon.euler_tangential_pair_gate_density_support_full)
    assert v118.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v118.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v118.euler_recon.euler_tangential_tail_density_shock_damp_on is False
    assert v118.euler_recon.euler_tangential_safe_legacy_coherence_on is False
    assert v118.euler_recon.euler_tangential_safe_legacy_qcurv_on is False
    assert v118.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v118.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.08
    assert v118.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert (
        v118.euler_recon.euler_tangential_tail_shear_contact_normality_max
        == 0.08)
    assert (
        v118.euler_recon.euler_tangential_tail_shear_contact_pressure_max
        == 0.008)
    assert (
        v118.euler_recon.euler_tangential_tail_shear_contact_compression_max
        == 0.0015)
    assert v119.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v119.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v119.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v119.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v119.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v119.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v119.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v119.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v119.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v119.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v119.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v119.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert (
        v119.euler_recon.euler_tangential_tail_shear_contact_shear_min
        == v118.euler_recon.euler_tangential_tail_shear_contact_shear_min)
    assert (
        v119.euler_recon.euler_tangential_tail_shear_contact_normality_max
        == v118.euler_recon.euler_tangential_tail_shear_contact_normality_max)
    assert (
        v119.euler_recon.euler_tangential_tail_shear_contact_pressure_max
        == v118.euler_recon.euler_tangential_tail_shear_contact_pressure_max)
    assert (
        v119.euler_recon.euler_tangential_tail_shear_contact_compression_max
        == v118.euler_recon.euler_tangential_tail_shear_contact_compression_max)
    assert v120.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v120.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v120.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v120.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v120.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v120.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v120.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v120.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v120.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v120.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v120.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v120.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.02
    assert (
        v120.euler_recon.euler_tangential_tail_shear_contact_shear_min
        == v118.euler_recon.euler_tangential_tail_shear_contact_shear_min)
    assert (
        v120.euler_recon.euler_tangential_tail_shear_contact_normality_max
        == v118.euler_recon.euler_tangential_tail_shear_contact_normality_max)
    assert (
        v120.euler_recon.euler_tangential_tail_shear_contact_pressure_max
        == v118.euler_recon.euler_tangential_tail_shear_contact_pressure_max)
    assert (
        v120.euler_recon.euler_tangential_tail_shear_contact_compression_max
        == v118.euler_recon.euler_tangential_tail_shear_contact_compression_max)
    assert v121.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v121.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v121.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v121.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v121.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v121.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v121.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v121.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v121.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v121.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v121.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v121.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert (
        v121.euler_recon.euler_tangential_tail_shear_contact_shear_min
        == v118.euler_recon.euler_tangential_tail_shear_contact_shear_min)
    assert (
        v121.euler_recon.euler_tangential_tail_shear_contact_normality_max
        == v118.euler_recon.euler_tangential_tail_shear_contact_normality_max)
    assert v121.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.006
    assert (
        v121.euler_recon.euler_tangential_tail_shear_contact_compression_max
        == v118.euler_recon.euler_tangential_tail_shear_contact_compression_max)
    assert v122.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v122.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v122.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v122.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v122.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v122.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v122.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v122.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v122.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v122.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v122.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v122.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert (
        v122.euler_recon.euler_tangential_tail_shear_contact_shear_min
        == v118.euler_recon.euler_tangential_tail_shear_contact_shear_min)
    assert (
        v122.euler_recon.euler_tangential_tail_shear_contact_normality_max
        == v118.euler_recon.euler_tangential_tail_shear_contact_normality_max)
    assert v122.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v122.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0010
    assert v123.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v123.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v123.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v123.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v123.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v123.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v123.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v123.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v123.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v123.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v123.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v123.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.03
    assert v123.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v123.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v123.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v123.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert v124.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v124.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v124.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v124.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v124.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v124.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v124.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v124.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v124.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v124.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v124.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v124.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.035
    assert v124.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v124.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v124.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v124.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert v125.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v125.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v125.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v125.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v125.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v125.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v125.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v125.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v125.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v125.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v125.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v125.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.0375
    assert v125.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v125.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v125.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v125.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    for recon in (v118, v119, v120, v121, v122, v123, v124, v125):
        assert (
            recon.euler_recon
            .euler_tangential_tail_shear_contact_relief_apply_signed is True)
        assert (
            recon.euler_recon
            .euler_tangential_tail_shear_contact_relief_apply_curve is True)
    assert v126.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v126.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v126.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v126.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v126.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v126.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v126.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v126.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v126.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v126.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v126.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v126.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v126.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v126.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v126.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v126.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v126.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is False)
    assert (
        v126.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is True)
    assert v127.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v127.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v127.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v127.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v127.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v127.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v127.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v127.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v127.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v127.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v127.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v127.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v127.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v127.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v127.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v127.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v127.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v127.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    for recon in (v118, v119, v120, v121, v122, v123, v124, v125, v126, v127):
        assert (
            recon.euler_recon
            .euler_tangential_tail_shear_contact_signed_floor == -1.0)
        assert (
            recon.euler_recon
            .euler_tangential_tail_shear_contact_curve_floor == -1.0)
    assert v128.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v128.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v128.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v128.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v128.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v128.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v128.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v128.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v128.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v128.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v128.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v128.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v128.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v128.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v128.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v128.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v128.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is True)
    assert v128.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v128.euler_recon.euler_tangential_tail_shear_contact_curve_floor == 0.02
    for recon in (
            v118, v119, v120, v121, v122, v123, v124, v125, v126, v127,
            v128):
        assert (
            recon.euler_recon.euler_tangential_signed_tail_density_support_min
            == -1.0)
        assert (
            recon.euler_recon.euler_tangential_signed_tail_density_support_full
            == -1.0)
    assert v129.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v129.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v129.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v129.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v129.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v129.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v129.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v129.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v129.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v129.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v129.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v129.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v129.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v129.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v129.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v129.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v129.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v129.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v129.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v129.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v129.euler_recon.euler_tangential_signed_tail_density_support_min == 0.014
    assert v129.euler_recon.euler_tangential_signed_tail_density_support_full == 0.060
    assert v130.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v130.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v130.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v130.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v130.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v130.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v130.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v130.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v130.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v130.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v130.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v130.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.06
    assert v130.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v130.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v130.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v130.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v130.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v130.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v130.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.06
    assert v130.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v130.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v130.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    for recon in (
            v118, v119, v120, v121, v122, v123, v124, v125, v126, v127,
            v128, v129, v130):
        assert (
            recon.euler_recon
            .euler_tangential_signed_tail_safe_decay_relief_on is False)
        assert recon.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert v131.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v131.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v131.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v131.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v131.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v131.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v131.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v131.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v131.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v131.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v131.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v131.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v131.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v131.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v131.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v131.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v131.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v131.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v131.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v131.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v131.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v131.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v131.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v131.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert v132.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v132.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v132.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v132.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v132.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v132.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v132.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v132.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v132.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v132.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v132.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v132.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v132.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v132.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v132.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v132.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v132.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v132.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v132.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v132.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v132.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v132.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v132.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v132.euler_recon.euler_tangential_signed_tail_safe_floor == 0.07
    assert v133.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v133.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v133.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v133.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v133.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v133.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v133.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v133.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v133.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v133.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v133.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v133.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v133.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v133.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v133.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v133.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v133.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v133.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v133.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v133.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v133.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v133.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v133.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v133.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert v133.euler_recon.euler_tangential_signed_pair_tail_cap == 0.060
    assert (
        v133.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert v134.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v134.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v134.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v134.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v134.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v134.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v134.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v134.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v134.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v134.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v134.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v134.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v134.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v134.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v134.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v134.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v134.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v134.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v134.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v134.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v134.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v134.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v134.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v134.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v134.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v134.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v134.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is True)
    assert (
        v134.euler_recon
        .euler_tangential_signed_tail_postrollback_theta == 0.35)
    assert v135.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v135.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v135.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v135.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v135.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v135.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v135.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v135.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v135.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v135.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v135.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v135.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v135.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v135.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v135.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v135.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v135.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v135.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v135.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v135.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v135.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v135.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v135.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v135.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v135.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v135.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v135.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v135.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is True)
    assert (
        v135.euler_recon
        .euler_tangential_tail_signed_anchored_curve_floor == 0.04)
    assert v136.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v136.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v136.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v136.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v136.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v136.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v136.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v136.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v136.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v136.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v136.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v136.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v136.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v136.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v136.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v136.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v136.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v136.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v136.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v136.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v136.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v136.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v136.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v136.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v136.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v136.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v136.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v136.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is True)
    assert (
        v136.euler_recon
        .euler_tangential_tail_signed_anchored_curve_floor == 0.04)
    assert (
        v136.euler_recon
        .euler_tangential_tail_signed_anchored_curve_align_on is True)
    assert v137.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v137.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v137.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v137.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v137.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v137.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v137.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v137.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v137.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v137.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v137.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v137.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v137.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v137.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v137.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v137.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v137.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v137.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v137.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v137.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v137.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v137.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v137.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v137.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v137.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v137.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v137.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v137.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is True)
    assert (
        v137.euler_recon
        .euler_tangential_tail_signed_anchored_curve_floor == 0.06)
    assert (
        v137.euler_recon
        .euler_tangential_tail_signed_anchored_curve_align_on is False)
    assert v138.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v138.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v138.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v138.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v138.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v138.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v138.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v138.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v138.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v138.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v138.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v138.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v138.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v138.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v138.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v138.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v138.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v138.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v138.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v138.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v138.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v138.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v138.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v138.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v138.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v138.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v138.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v138.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is True)
    assert (
        v138.euler_recon
        .euler_tangential_tail_signed_anchored_curve_floor == 0.04)
    assert (
        v138.euler_recon
        .euler_tangential_tail_signed_anchored_curve_align_on is False)
    assert (
        v138.euler_recon
        .euler_tangential_tail_signed_anchored_curve_preserve_signed_on is True)
    assert v139.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v139.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v139.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v139.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v139.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v139.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v139.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v139.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v139.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v139.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v139.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v139.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v139.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v139.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v139.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v139.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v139.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v139.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v139.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v139.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v139.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v139.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v139.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v139.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v139.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v139.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v139.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v139.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is False)
    assert (
        v139.euler_recon
        .euler_tangential_tail_signed_anchored_curve_align_on is False)
    assert (
        v139.euler_recon
        .euler_tangential_tail_signed_anchored_curve_preserve_signed_on is False)
    assert v139.euler_recon.euler_density_signed_tail_trace_on is True
    assert v139.euler_recon.euler_density_signed_tail_trace_beta == 0.15
    assert v139.euler_recon.euler_density_signed_tail_trace_cap == 0.004
    assert v139.euler_recon.euler_density_signed_tail_trace_wave_cap == 0.0015
    assert v140.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v140.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v140.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v140.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v140.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v140.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v140.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v140.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v140.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v140.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v140.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v140.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v140.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v140.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v140.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v140.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v140.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v140.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v140.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v140.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v140.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v140.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v140.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v140.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v140.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v140.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v140.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v140.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is True)
    assert (
        v140.euler_recon
        .euler_tangential_tail_signed_anchored_curve_floor == 0.04)
    assert (
        v140.euler_recon
        .euler_tangential_tail_signed_anchored_curve_align_on is False)
    assert (
        v140.euler_recon
        .euler_tangential_tail_signed_anchored_curve_preserve_signed_on is False)
    assert v140.euler_recon.euler_density_signed_tail_trace_on is False
    assert v141.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v141.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v141.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v141.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v141.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v141.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v141.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v141.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v141.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v141.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v141.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v141.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v141.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v141.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v141.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v141.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v141.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v141.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v141.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v141.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v141.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v141.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v141.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v141.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v141.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v141.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v141.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v141.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is True)
    assert (
        v141.euler_recon
        .euler_tangential_tail_signed_anchored_curve_floor == 0.04)
    assert (
        v141.euler_recon
        .euler_tangential_tail_signed_anchored_curve_align_on is False)
    assert (
        v141.euler_recon
        .euler_tangential_tail_signed_anchored_curve_preserve_signed_on is False)
    assert v141.euler_recon.euler_density_signed_tail_trace_on is False
    assert v142.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v142.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v142.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v142.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v142.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v142.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v142.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v142.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v142.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v142.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v142.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v142.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v142.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v142.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v142.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v142.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v142.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v142.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v142.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v142.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v142.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v142.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v142.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v142.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v142.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v142.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v142.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v142.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is True)
    assert (
        v142.euler_recon
        .euler_tangential_tail_signed_anchored_curve_floor == 0.04)
    assert (
        v142.euler_recon
        .euler_tangential_tail_signed_anchored_curve_align_on is False)
    assert (
        v142.euler_recon
        .euler_tangential_tail_signed_anchored_curve_preserve_signed_on is False)
    assert v142.euler_recon.euler_density_signed_tail_trace_on is False
    assert (
        v142.euler_recon
        .euler_tangential_highsafe_raw_curve_microassist_on is True)
    assert (
        v142.euler_recon
        .euler_tangential_highsafe_raw_curve_microassist_floor == 0.015)
    assert (
        v142.euler_recon
        .euler_tangential_highsafe_raw_curve_microassist_cap == 0.020)
    assert (
        v142.euler_recon
        .euler_tangential_highsafe_raw_curve_microassist_wave_cap == 0.0025)
    assert (
        v142.euler_recon
        .euler_tangential_highsafe_raw_curve_safe_min == 0.40)
    assert (
        v142.euler_recon
        .euler_tangential_highsafe_raw_curve_shear_min == 0.94)
    assert (
        v142.euler_recon
        .euler_tangential_highsafe_raw_curve_normality_max == 0.08)
    assert (
        v142.euler_recon
        .euler_tangential_highsafe_raw_curve_pressure_max == 0.008)
    assert (
        v142.euler_recon
        .euler_tangential_highsafe_raw_curve_compression_max == 0.0015)
    assert v143.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v143.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v143.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v143.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v143.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v143.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v143.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v143.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v143.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v143.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v143.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v143.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v143.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v143.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v143.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v143.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v143.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v143.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v143.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v143.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v143.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v143.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v143.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is False
    assert v143.euler_recon.euler_tangential_signed_tail_sidecar_decay_on is True
    assert v143.euler_recon.euler_tangential_signed_tail_sidecar_safe_floor == 0.10
    assert v143.euler_recon.euler_tangential_signed_tail_sidecar_blend == 0.35
    assert (
        v143.euler_recon.euler_tangential_signed_pair_tail_cap
        == v127.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v143.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v127.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v143.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v143.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is False)
    assert (
        v143.euler_recon
        .euler_tangential_highsafe_raw_curve_microassist_on is False)
    assert v143.euler_recon.euler_density_signed_tail_trace_on is False
    assert v144.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v144.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v144.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v144.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v144.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v144.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v144.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v144.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v144.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v144.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v144.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v144.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v144.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v144.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v144.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v144.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v144.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v144.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v144.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v144.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v144.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v144.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v144.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is False
    assert v144.euler_recon.euler_tangential_signed_tail_sidecar_decay_on is True
    assert v144.euler_recon.euler_tangential_signed_tail_sidecar_safe_floor == 0.10
    assert v144.euler_recon.euler_tangential_signed_tail_sidecar_blend == 0.15
    assert (
        v144.euler_recon.euler_tangential_signed_pair_tail_cap
        == v127.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v144.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v127.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v144.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v144.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is False)
    assert (
        v144.euler_recon
        .euler_tangential_highsafe_raw_curve_microassist_on is False)
    assert v144.euler_recon.euler_density_signed_tail_trace_on is False
    assert v145.euler_recon.euler_tangential_legacy_pair_target_on is True
    assert v145.euler_recon.euler_tangential_legacy_pair_target_blend == 1.0
    assert v145.euler_recon.euler_tangential_safe_legacy_gate_on is True
    assert v145.euler_recon.euler_tangential_safe_legacy_pressure_hi == 0.010
    assert v145.euler_recon.euler_tangential_safe_legacy_compression_hi == 0.002
    assert v145.euler_recon.euler_tangential_safe_legacy_normality_hi == 0.14
    assert v145.euler_recon.euler_tangential_safe_legacy_shear_min == 0.72
    assert v145.euler_recon.euler_tangential_safe_legacy_contact_min == 0.45
    assert v145.euler_recon.euler_tangential_tail_density_support_min == 0.015
    assert v145.euler_recon.euler_tangential_tail_density_support_full == 0.060
    assert v145.euler_recon.euler_tangential_tail_shear_contact_relief_on is True
    assert v145.euler_recon.euler_tangential_tail_shear_contact_relief_floor == 0.04
    assert v145.euler_recon.euler_tangential_tail_shear_contact_shear_min == 0.94
    assert v145.euler_recon.euler_tangential_tail_shear_contact_normality_max == 0.08
    assert v145.euler_recon.euler_tangential_tail_shear_contact_pressure_max == 0.008
    assert v145.euler_recon.euler_tangential_tail_shear_contact_compression_max == 0.0015
    assert (
        v145.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v145.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)
    assert v145.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v145.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v145.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v145.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert v145.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on is True
    assert v145.euler_recon.euler_tangential_signed_tail_safe_floor == 0.12
    assert (
        v145.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v145.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v145.euler_recon
        .euler_tangential_signed_tail_postrollback_preserve_on is False)
    assert (
        v145.euler_recon
        .euler_tangential_tail_signed_anchored_curve_assist_on is False)
    assert (
        v145.euler_recon
        .euler_tangential_highsafe_raw_curve_microassist_on is False)
    assert v145.euler_recon.euler_density_signed_tail_trace_on is False
    assert (
        v145.euler_recon
        .euler_tangential_signed_tail_sidecar_decay_on is False)
    assert os.environ.get('TMLPU_V115_TAIL_DIAGNOSTICS') is None
    assert v95.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v95.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v95.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v95.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v96.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v96.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v96.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v96.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v97.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v97.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v97.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v97.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v98.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v98.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v98.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v98.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v99.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v99.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v99.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v99.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v100.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v100.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v100.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v100.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v101.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v101.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v101.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v101.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v102.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v102.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v102.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v102.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v103.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v103.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v103.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v103.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v104.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v104.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v104.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v104.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v105.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v105.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v105.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v105.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v106.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v106.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v106.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v106.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v107.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v107.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v107.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v107.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v108.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v108.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v108.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v108.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v109.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v109.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v109.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v109.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v110.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v110.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v110.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v110.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v111.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v111.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v111.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v111.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v112.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v112.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v112.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v112.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v113.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v113.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v113.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v113.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v114.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v114.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v114.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v114.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v115.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v115.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v115.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v115.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v116.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v116.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v116.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v116.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v117.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v117.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v117.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v117.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v118.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v118.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v118.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v118.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v119.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v119.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v119.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v119.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v120.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v120.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v120.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v120.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v121.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v121.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v121.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v121.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v122.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v122.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v122.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v122.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v123.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v123.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v123.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v123.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v124.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v124.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v124.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v124.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v125.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v125.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v125.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v125.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v126.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v126.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v126.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v126.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v127.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v127.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v127.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v127.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v128.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v128.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v128.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v128.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v129.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v129.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v129.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v129.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v130.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v130.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v130.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v130.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v131.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v131.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v131.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v131.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v132.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v132.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v132.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v132.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v133.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v133.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v133.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v133.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v134.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v134.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v134.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v134.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v135.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v135.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v135.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v135.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v136.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v136.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v136.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v136.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v137.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v137.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v137.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v137.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v138.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v138.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v138.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v138.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v139.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v139.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v139.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v139.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v140.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v140.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v140.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v140.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v141.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v141.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v141.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v141.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v142.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v142.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v142.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v142.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v143.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v143.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v143.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v143.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v144.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v144.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v144.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v144.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)
    assert v145.euler_recon.euler_density_contact_weak_face_stream_coherence_on is False
    assert v145.euler_recon.euler_density_contact_weak_face_downstream_rho_beta == 0.0
    assert v145.euler_recon.euler_density_contact_weak_face_downstream_rho_cap == 0.0
    assert (
        v145.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0)


def test_v146_signed_gate_shadow_diag_is_v131_behavior_preserving():
    v131 = bench._reconstruction_from_key(
        "tmlpu_v131_signed_gate_decay_relief_on")
    v146 = bench._reconstruction_from_key(
        "tmlpu_v146_signed_gate_shadow_diag_on")

    assert type(v146.scalar_recon) is type(v131.scalar_recon)
    assert type(v146.euler_recon) is type(v131.euler_recon)
    assert (
        v146.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v146.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v146.euler_recon.euler_tangential_tail_shear_contact_relief_apply_signed
        is True)
    assert (
        v146.euler_recon.euler_tangential_tail_shear_contact_relief_apply_curve
        is False)
    assert v146.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v146.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v146.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v146.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert (
        v146.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v146.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v146.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert (
        v146.euler_recon.euler_tangential_highsafe_raw_curve_microassist_on
        is False)
    assert v146.euler_recon.euler_density_signed_tail_trace_on is False
    assert (
        v146.euler_recon.euler_tangential_signed_tail_postrollback_preserve_on
        is False)
    assert (
        v146.euler_recon.euler_tangential_signed_tail_sidecar_decay_on
        is False)
    assert (
        v146.euler_recon.euler_density_contact_weak_face_stream_coherence_on
        is False)
    assert (
        v146.euler_recon.euler_density_contact_weak_face_downstream_rho_beta
        == 0.0)
    assert (
        v146.euler_recon.euler_density_contact_weak_face_downstream_rho_cap
        == 0.0)


def test_v147_signed_beta044_keeps_v131_gate_settings():
    v131 = bench._reconstruction_from_key(
        "tmlpu_v131_signed_gate_decay_relief_on")
    v147 = bench._reconstruction_from_key(
        "tmlpu_v147_signed_beta044_on")

    assert type(v147.scalar_recon) is type(v131.scalar_recon)
    assert type(v147.euler_recon) is type(v131.euler_recon)
    assert v147.euler_recon.euler_tangential_signed_pair_tail_beta == 0.044
    assert (
        v147.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v147.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v147.euler_recon.euler_tangential_tail_shear_contact_relief_apply_signed
        is True)
    assert (
        v147.euler_recon.euler_tangential_tail_shear_contact_relief_apply_curve
        is False)
    assert v147.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v147.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v147.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v147.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert (
        v147.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v147.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v147.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert (
        v147.euler_recon.euler_tangential_highsafe_raw_curve_microassist_on
        is False)
    assert v147.euler_recon.euler_density_signed_tail_trace_on is False
    assert (
        v147.euler_recon.euler_tangential_signed_tail_postrollback_preserve_on
        is False)
    assert (
        v147.euler_recon.euler_tangential_signed_tail_sidecar_decay_on
        is False)
    assert (
        v147.euler_recon.euler_density_contact_weak_face_stream_coherence_on
        is False)


def test_v148_signed_beta038_keeps_v131_gate_settings():
    v131 = bench._reconstruction_from_key(
        "tmlpu_v131_signed_gate_decay_relief_on")
    v148 = bench._reconstruction_from_key(
        "tmlpu_v148_signed_beta038_on")

    assert type(v148.scalar_recon) is type(v131.scalar_recon)
    assert type(v148.euler_recon) is type(v131.euler_recon)
    assert v148.euler_recon.euler_tangential_signed_pair_tail_beta == 0.038
    assert (
        v148.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v148.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v148.euler_recon.euler_tangential_tail_shear_contact_relief_apply_signed
        is True)
    assert (
        v148.euler_recon.euler_tangential_tail_shear_contact_relief_apply_curve
        is False)
    assert v148.euler_recon.euler_tangential_tail_shear_contact_signed_floor == 0.04
    assert v148.euler_recon.euler_tangential_tail_shear_contact_curve_floor == -1.0
    assert v148.euler_recon.euler_tangential_signed_tail_density_support_min == -1.0
    assert v148.euler_recon.euler_tangential_signed_tail_density_support_full == -1.0
    assert (
        v148.euler_recon.euler_tangential_signed_pair_tail_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_cap)
    assert (
        v148.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == v131.euler_recon.euler_tangential_signed_pair_tail_wave_cap)
    assert (
        v148.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert (
        v148.euler_recon.euler_tangential_highsafe_raw_curve_microassist_on
        is False)
    assert v148.euler_recon.euler_density_signed_tail_trace_on is False
    assert (
        v148.euler_recon.euler_tangential_signed_tail_postrollback_preserve_on
        is False)
    assert (
        v148.euler_recon.euler_tangential_signed_tail_sidecar_decay_on
        is False)
    assert (
        v148.euler_recon.euler_density_contact_weak_face_stream_coherence_on
        is False)


def test_v149_v135_downstream_density_micro_keeps_safety_gates():
    v135 = bench._reconstruction_from_key(
        "tmlpu_v135_signed_anchored_curve_assist_on")
    v149 = bench._reconstruction_from_key(
        "tmlpu_v149_v135_downstream_density_micro_on")

    assert type(v149.scalar_recon) is type(v135.scalar_recon)
    assert type(v149.euler_recon) is type(v135.euler_recon)
    assert (
        v149.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v149.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v149.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is True)
    assert (
        v149.euler_recon.euler_tangential_tail_signed_anchored_curve_floor
        == 0.04)
    assert (
        v149.euler_recon.euler_density_contact_weak_face_stream_coherence_on
        is True)
    assert (
        v149.euler_recon.euler_density_contact_weak_face_stream_coherence_min
        == 0.20)
    assert (
        v149.euler_recon.euler_density_contact_weak_face_stream_coherence_full
        == 0.60)
    assert (
        v149.euler_recon.euler_density_contact_weak_face_downstream_rho_beta
        == 0.018)
    assert (
        v149.euler_recon.euler_density_contact_weak_face_downstream_rho_cap
        == 0.003)
    assert (
        v149.euler_recon.euler_density_contact_weak_face_downstream_rho_wave_cap
        == 0.0015)
    assert (
        v149.euler_recon
        .euler_density_contact_weak_face_downstream_tangential_beta == 0.0)
    assert (
        v149.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_signed is True)
    assert (
        v149.euler_recon
        .euler_tangential_tail_shear_contact_relief_apply_curve is False)


def test_v150_v135_pair_extend_micro_keeps_safety_gates():
    v135 = bench._reconstruction_from_key(
        "tmlpu_v135_signed_anchored_curve_assist_on")
    v150 = bench._reconstruction_from_key(
        "tmlpu_v150_v135_pair_extend_micro_on")

    assert type(v150.scalar_recon) is type(v135.scalar_recon)
    assert type(v150.euler_recon) is type(v135.euler_recon)
    assert (
        v150.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v150.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v150.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is True)
    assert (
        v150.euler_recon.euler_tangential_tail_signed_anchored_curve_floor
        == 0.04)
    assert v150.euler_recon.euler_tangential_pair_extend_on is True
    assert v150.euler_recon.euler_tangential_pair_extend_beta == 0.015
    assert v150.euler_recon.euler_tangential_pair_extend_cap == 0.015
    assert v150.euler_recon.euler_tangential_pair_extend_wave_cap == 0.002
    assert (
        v150.euler_recon.euler_tangential_pair_extend_alignment_min == 0.20)
    assert (
        v150.euler_recon.euler_tangential_pair_extend_alignment_full == 0.60)
    assert v150.euler_recon.euler_tangential_pair_extend_shock_exclude is True
    assert (
        v150.euler_recon.euler_density_contact_weak_face_stream_coherence_on
        is False)


def test_v153_v131_pair_extend_micro_keeps_v131_safety_gates():
    v131 = bench._reconstruction_from_key(
        "tmlpu_v131_signed_gate_decay_relief_on")
    v153 = bench._reconstruction_from_key(
        "tmlpu_v153_v131_pair_extend_micro_on")

    assert type(v153.scalar_recon) is type(v131.scalar_recon)
    assert type(v153.euler_recon) is type(v131.euler_recon)
    assert (
        v153.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v153.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v153.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert v153.euler_recon.euler_tangential_pair_extend_on is True
    assert v153.euler_recon.euler_tangential_pair_extend_beta == 0.012
    assert v153.euler_recon.euler_tangential_pair_extend_cap == 0.010
    assert v153.euler_recon.euler_tangential_pair_extend_wave_cap == 0.0015
    assert (
        v153.euler_recon.euler_tangential_pair_extend_alignment_min == 0.25)
    assert (
        v153.euler_recon.euler_tangential_pair_extend_alignment_full == 0.65)
    assert v153.euler_recon.euler_tangential_pair_extend_shock_exclude is True
    assert (
        v153.euler_recon.euler_density_contact_weak_face_stream_coherence_on
        is False)


def test_v155_v131_reduced_signed_tail_keeps_safety_decay():
    v131 = bench._reconstruction_from_key(
        "tmlpu_v131_signed_gate_decay_relief_on")
    v155 = bench._reconstruction_from_key(
        "tmlpu_v155_v131_reduced_signed_tail_on")

    assert type(v155.scalar_recon) is type(v131.scalar_recon)
    assert type(v155.euler_recon) is type(v131.euler_recon)
    assert (
        v155.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v155.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert v155.euler_recon.euler_tangential_signed_pair_tail_beta == 0.032
    assert v155.euler_recon.euler_tangential_signed_pair_tail_cap == 0.026
    assert (
        v155.euler_recon.euler_tangential_signed_pair_tail_wave_cap == 0.0032)
    assert (
        v155.euler_recon.euler_tangential_tail_shear_contact_relief_apply_signed
        is True)
    assert (
        v155.euler_recon.euler_tangential_tail_shear_contact_relief_apply_curve
        is False)
    assert (
        v155.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert v155.euler_recon.euler_tangential_pair_extend_on is False


def test_v157_v131_antisheet_keeps_v131_safety_gates():
    v131 = bench._reconstruction_from_key(
        "tmlpu_v131_signed_gate_decay_relief_on")
    v157 = bench._reconstruction_from_key(
        "tmlpu_v157_v131_antisheet_on")

    assert type(v157.scalar_recon) is type(v131.scalar_recon)
    assert type(v157.euler_recon) is type(v131.euler_recon)
    assert (
        v157.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v157.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v157.euler_recon.euler_tangential_signed_tail_antisheet_on is True)
    assert (
        v157.euler_recon.euler_tangential_signed_tail_antisheet_strength
        == 0.45)
    assert (
        v157.euler_recon.euler_tangential_signed_tail_antisheet_min_factor
        == 0.55)
    assert (
        v157.euler_recon.euler_tangential_signed_tail_antisheet_q_hi == 0.070)
    assert (
        v157.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert v157.euler_recon.euler_tangential_pair_extend_on is False


def test_v158_v131_strong_antisheet_keeps_v131_safety_gates():
    v158 = bench._reconstruction_from_key(
        "tmlpu_v158_v131_strong_antisheet_on")

    assert (
        v158.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v158.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v158.euler_recon.euler_tangential_signed_tail_antisheet_on is True)
    assert (
        v158.euler_recon.euler_tangential_signed_tail_antisheet_strength
        == 0.90)
    assert (
        v158.euler_recon.euler_tangential_signed_tail_antisheet_min_factor
        == 0.20)
    assert (
        v158.euler_recon.euler_tangential_signed_tail_antisheet_q_hi == 0.18)
    assert (
        v158.euler_recon.euler_tangential_signed_tail_antisheet_contact_min
        == 0.10)
    assert (
        v158.euler_recon.euler_tangential_signed_tail_antisheet_contact_full
        == 0.45)
    assert v158.euler_recon.euler_tangential_pair_extend_on is False


def test_v159_v131_qcore_gate_keeps_v131_safety_decay():
    v159 = bench._reconstruction_from_key(
        "tmlpu_v159_v131_qcore_gate_on")

    assert (
        v159.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v159.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v159.euler_recon.euler_tangential_signed_pair_tail_q_min == 0.050)
    assert (
        v159.euler_recon.euler_tangential_signed_pair_tail_q_full == 0.140)
    assert (
        v159.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert v159.euler_recon.euler_tangential_pair_extend_on is False
    assert (
        v159.euler_recon.euler_tangential_signed_tail_antisheet_on is False)


def test_v160_v131_signed_tail_hffilter_keeps_v131_safety_decay():
    v160 = bench._reconstruction_from_key(
        "tmlpu_v160_v131_signed_tail_hffilter_on")

    assert (
        v160.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v160.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v160.euler_recon.euler_tangential_signed_tail_hf_filter_on is True)
    assert (
        v160.euler_recon.euler_tangential_signed_tail_hf_filter_strength
        == 0.35)
    assert (
        v160.euler_recon.euler_tangential_signed_tail_hf_filter_shock_exclude
        is True)
    assert (
        v160.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert v160.euler_recon.euler_tangential_pair_extend_on is False
    assert (
        v160.euler_recon.euler_tangential_signed_tail_antisheet_on is False)


def test_v161_v131_bridge_cut_keeps_v131_safety_decay():
    v161 = bench._reconstruction_from_key(
        "tmlpu_v161_v131_bridge_cut_on")

    assert (
        v161.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v161.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v161.euler_recon.euler_tangential_signed_tail_bridge_cut_on is True)
    assert (
        v161.euler_recon.euler_tangential_signed_tail_bridge_cut_strength
        == 0.55)
    assert (
        v161.euler_recon.euler_tangential_signed_tail_bridge_cut_min_factor
        == 0.25)
    assert (
        v161.euler_recon.euler_tangential_signed_tail_bridge_cut_q_min
        == 0.08)
    assert (
        v161.euler_recon.euler_tangential_signed_tail_bridge_cut_q_full
        == 0.22)
    assert (
        v161.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert v161.euler_recon.euler_tangential_pair_extend_on is False
    assert (
        v161.euler_recon.euler_tangential_signed_tail_antisheet_on is False)


def test_v162_v131_conservative_bridge_cut_is_weaker_than_v161():
    v161 = bench._reconstruction_from_key(
        "tmlpu_v161_v131_bridge_cut_on")
    v162 = bench._reconstruction_from_key(
        "tmlpu_v162_v131_conservative_bridge_cut_on")

    assert (
        v162.euler_recon.euler_tangential_signed_tail_bridge_cut_on is True)
    assert (
        v162.euler_recon.euler_tangential_signed_tail_bridge_cut_strength
        < v161.euler_recon.euler_tangential_signed_tail_bridge_cut_strength)
    assert (
        v162.euler_recon.euler_tangential_signed_tail_bridge_cut_min_factor
        > v161.euler_recon.euler_tangential_signed_tail_bridge_cut_min_factor)
    assert (
        v162.euler_recon.euler_tangential_signed_tail_bridge_cut_q_min
        == 0.12)
    assert (
        v162.euler_recon.euler_tangential_signed_tail_bridge_cut_q_full
        == 0.28)
    assert (
        v162.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert v162.euler_recon.euler_tangential_pair_extend_on is False


def test_v163_v131_shock_ridge_guard_uses_single_feature_guard():
    v163 = bench._reconstruction_from_key(
        "tmlpu_v163_v131_shock_ridge_guard_on")

    assert (
        v163.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is True)
    assert v163.euler_recon.euler_tangential_signed_tail_safe_floor == 0.10
    assert (
        v163.euler_recon.euler_tangential_signed_tail_shock_ridge_clean_on
        is True)
    assert (
        v163.euler_recon.euler_tangential_signed_tail_shock_ridge_strength
        == 0.50)
    assert (
        v163.euler_recon.euler_tangential_signed_tail_shock_ridge_min_factor
        == 0.60)
    assert (
        v163.euler_recon.euler_tangential_signed_tail_shock_ridge_density_min
        == 0.35)
    assert (
        v163.euler_recon.euler_tangential_signed_tail_shock_ridge_density_full
        == 0.85)
    assert (
        v163.euler_recon.euler_tangential_signed_tail_shock_ridge_q_keep_min
        == 0.10)
    assert (
        v163.euler_recon.euler_tangential_signed_tail_shock_ridge_q_keep_full
        == 0.25)
    assert (
        v163.euler_recon.euler_tangential_signed_tail_bridge_cut_on is False)
    assert (
        v163.euler_recon.euler_tangential_signed_tail_hf_filter_on is False)
    assert (
        v163.euler_recon.euler_tangential_tail_signed_anchored_curve_assist_on
        is False)
    assert v163.euler_recon.euler_tangential_pair_extend_on is False


def test_v164_v131_density_support_damp_is_stronger_than_v163():
    v163 = bench._reconstruction_from_key(
        "tmlpu_v163_v131_shock_ridge_guard_on")
    v164 = bench._reconstruction_from_key(
        "tmlpu_v164_v131_density_support_damp_on")

    assert (
        v164.euler_recon.euler_tangential_signed_tail_shock_ridge_clean_on
        is True)
    assert (
        v164.euler_recon.euler_tangential_signed_tail_shock_ridge_strength
        > v163.euler_recon.euler_tangential_signed_tail_shock_ridge_strength)
    assert (
        v164.euler_recon.euler_tangential_signed_tail_shock_ridge_min_factor
        < v163.euler_recon.euler_tangential_signed_tail_shock_ridge_min_factor)
    assert (
        v164.euler_recon.euler_tangential_signed_tail_shock_ridge_density_min
        == 0.20)
    assert (
        v164.euler_recon.euler_tangential_signed_tail_shock_ridge_density_full
        == 0.70)
    assert (
        v164.euler_recon.euler_tangential_signed_tail_shock_ridge_q_keep_min
        == 2.0)
    assert (
        v164.euler_recon.euler_tangential_signed_tail_shock_ridge_q_keep_full
        == 3.0)
    assert (
        v164.euler_recon.euler_tangential_signed_tail_bridge_cut_on is False)
    assert (
        v164.euler_recon.euler_tangential_signed_tail_hf_filter_on is False)


def test_v165_v131_signed_tail_ablation_disables_tail_paths():
    v165 = bench._reconstruction_from_key(
        "tmlpu_v165_v131_signed_tail_ablation_on")

    assert v165.euler_recon.euler_tangential_signed_pair_tail_on is False
    assert v165.euler_recon.euler_tangential_signed_pair_tail_beta == 0.0
    assert (
        v165.euler_recon.euler_tangential_density_curve_pair_tail_on is False)
    assert (
        v165.euler_recon.euler_tangential_density_curve_pair_tail_beta == 0.0)
    assert (
        v165.euler_recon.euler_tangential_tail_shear_contact_relief_apply_signed
        is False)
    assert (
        v165.euler_recon.euler_tangential_tail_shear_contact_relief_apply_curve
        is False)
    assert (
        v165.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is False)
    assert (
        v165.euler_recon.euler_tangential_signed_tail_bridge_cut_on is False)
    assert (
        v165.euler_recon.euler_tangential_signed_tail_hf_filter_on is False)
    assert (
        v165.euler_recon.euler_tangential_signed_tail_shock_ridge_clean_on
        is False)
    assert v165.euler_recon.euler_tangential_pair_extend_on is False
    assert v165.euler_recon.euler_density_signed_tail_trace_on is False


def test_v166_v165_micro_signed_restore_only_reenables_small_signed_tail():
    v166 = bench._reconstruction_from_key(
        "tmlpu_v166_v165_micro_signed_restore_on")

    assert v166.euler_recon.euler_tangential_signed_pair_tail_on is True
    assert v166.euler_recon.euler_tangential_signed_pair_tail_beta == 0.016
    assert v166.euler_recon.euler_tangential_signed_pair_tail_cap == 0.013
    assert (
        v166.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == 0.0016)
    assert (
        v166.euler_recon.euler_tangential_tail_shear_contact_relief_apply_signed
        is True)
    assert (
        v166.euler_recon.euler_tangential_tail_shear_contact_signed_floor
        == 0.02)
    assert (
        v166.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is False)
    assert (
        v166.euler_recon.euler_tangential_density_curve_pair_tail_on is False)
    assert (
        v166.euler_recon.euler_tangential_signed_tail_bridge_cut_on is False)
    assert (
        v166.euler_recon.euler_tangential_signed_tail_hf_filter_on is False)
    assert v166.euler_recon.euler_tangential_pair_extend_on is False
    assert v166.euler_recon.euler_density_signed_tail_trace_on is False


def test_v167_v165_curve_restore_adds_tiny_density_curve_tail():
    v167 = bench._reconstruction_from_key(
        "tmlpu_v167_v165_curve_restore_on")

    assert v167.euler_recon.euler_tangential_signed_pair_tail_on is True
    assert v167.euler_recon.euler_tangential_signed_pair_tail_beta == 0.020
    assert v167.euler_recon.euler_tangential_signed_pair_tail_cap == 0.016
    assert (
        v167.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == 0.0020)
    assert (
        v167.euler_recon.euler_tangential_density_curve_pair_tail_on is True)
    assert (
        v167.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.012)
    assert (
        v167.euler_recon.euler_tangential_density_curve_pair_tail_cap
        == 0.010)
    assert (
        v167.euler_recon.euler_tangential_density_curve_pair_tail_wave_cap
        == 0.0012)
    assert (
        v167.euler_recon.euler_tangential_tail_shear_contact_relief_apply_curve
        is True)
    assert (
        v167.euler_recon.euler_tangential_tail_shear_contact_curve_floor
        == 0.015)
    assert (
        v167.euler_recon.euler_tangential_signed_tail_safe_decay_relief_on
        is False)
    assert (
        v167.euler_recon.euler_tangential_signed_tail_bridge_cut_on is False)
    assert (
        v167.euler_recon.euler_tangential_signed_tail_hf_filter_on is False)
    assert v167.euler_recon.euler_tangential_pair_extend_on is False
    assert v167.euler_recon.euler_density_signed_tail_trace_on is False


def test_v168_v167_curve_hffilter_filters_signed_and_curve_tails():
    v168 = bench._reconstruction_from_key(
        "tmlpu_v168_v167_curve_hffilter_on")

    assert v168.euler_recon.euler_tangential_signed_pair_tail_on is True
    assert v168.euler_recon.euler_tangential_density_curve_pair_tail_on is True
    assert v168.euler_recon.euler_tangential_signed_tail_hf_filter_on is True
    assert (
        v168.euler_recon.euler_tangential_signed_tail_hf_filter_strength
        == 0.20)
    assert (
        v168.euler_recon.euler_tangential_signed_tail_hf_filter_min_weight
        == 1e-10)
    assert (
        v168.euler_recon.euler_tangential_signed_tail_hf_filter_shock_exclude
        is True)
    assert (
        v168.euler_recon.euler_tangential_density_curve_tail_hf_filter_on
        is True)
    assert (
        v168.euler_recon.euler_tangential_density_curve_tail_hf_filter_strength
        == 0.35)
    assert (
        v168.euler_recon.euler_tangential_density_curve_tail_hf_filter_min_weight
        == 1e-10)
    assert (
        v168.euler_recon.euler_tangential_density_curve_tail_hf_filter_shock_exclude
        is True)
    assert v168.euler_recon.euler_tangential_pair_extend_on is False
    assert v168.euler_recon.euler_density_signed_tail_trace_on is False


def test_v174_v168_roi_strength_increases_tail_strength_under_hffilter():
    v174 = bench._reconstruction_from_key(
        "tmlpu_v174_v168_roi_strength_on")

    assert v174.euler_recon.euler_tangential_signed_pair_tail_on is True
    assert v174.euler_recon.euler_tangential_density_curve_pair_tail_on is True
    assert v174.euler_recon.euler_tangential_signed_tail_hf_filter_on is True
    assert v174.euler_recon.euler_tangential_density_curve_tail_hf_filter_on is True
    assert v174.euler_recon.euler_tangential_signed_pair_tail_beta == 0.030
    assert v174.euler_recon.euler_tangential_signed_pair_tail_cap == 0.024
    assert (
        v174.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == 0.0030)
    assert (
        v174.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.020)
    assert (
        v174.euler_recon.euler_tangential_density_curve_pair_tail_cap
        == 0.016)
    assert (
        v174.euler_recon.euler_tangential_density_curve_pair_tail_wave_cap
        == 0.0020)
    assert (
        v174.euler_recon.euler_tangential_signed_tail_hf_filter_strength
        == 0.16)
    assert (
        v174.euler_recon.euler_tangential_density_curve_tail_hf_filter_strength
        == 0.26)
    assert v174.euler_recon.euler_tangential_pair_extend_on is False
    assert v174.euler_recon.euler_density_signed_tail_trace_on is False


def test_v175_v174_stronger_filtered_roi_raises_tail_and_filter():
    v175 = bench._reconstruction_from_key(
        "tmlpu_v175_v174_stronger_filtered_roi_on")

    assert v175.euler_recon.euler_tangential_signed_pair_tail_on is True
    assert v175.euler_recon.euler_tangential_density_curve_pair_tail_on is True
    assert v175.euler_recon.euler_tangential_signed_tail_hf_filter_on is True
    assert v175.euler_recon.euler_tangential_density_curve_tail_hf_filter_on is True
    assert v175.euler_recon.euler_tangential_signed_pair_tail_beta == 0.045
    assert v175.euler_recon.euler_tangential_signed_pair_tail_cap == 0.034
    assert (
        v175.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == 0.0045)
    assert (
        v175.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.030)
    assert (
        v175.euler_recon.euler_tangential_density_curve_pair_tail_cap
        == 0.024)
    assert (
        v175.euler_recon.euler_tangential_density_curve_pair_tail_wave_cap
        == 0.0030)
    assert (
        v175.euler_recon.euler_tangential_signed_tail_hf_filter_strength
        == 0.24)
    assert (
        v175.euler_recon.euler_tangential_density_curve_tail_hf_filter_strength
        == 0.42)
    assert v175.euler_recon.euler_tangential_pair_extend_on is False
    assert v175.euler_recon.euler_density_signed_tail_trace_on is False


def test_v176_v174_pair_extend_roi_enables_weak_pair_extension():
    v176 = bench._reconstruction_from_key(
        "tmlpu_v176_v174_pair_extend_roi_on")

    assert v176.euler_recon.euler_tangential_signed_pair_tail_beta == 0.030
    assert (
        v176.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.020)
    assert v176.euler_recon.euler_tangential_pair_extend_on is True
    assert v176.euler_recon.euler_tangential_pair_extend_beta == 0.010
    assert v176.euler_recon.euler_tangential_pair_extend_cap == 0.008
    assert v176.euler_recon.euler_tangential_pair_extend_wave_cap == 0.0012
    assert v176.euler_recon.euler_tangential_pair_extend_alignment_min == 0.30
    assert (
        v176.euler_recon.euler_tangential_pair_extend_alignment_full
        == 0.70)
    assert v176.euler_recon.euler_tangential_pair_extend_shock_exclude is True
    assert v176.euler_recon.euler_density_signed_tail_trace_on is False


def test_v177_v174_mid_strength_roi_sits_between_v174_and_v175():
    v177 = bench._reconstruction_from_key(
        "tmlpu_v177_v174_mid_strength_roi_on")

    assert v177.euler_recon.euler_tangential_signed_pair_tail_beta == 0.036
    assert v177.euler_recon.euler_tangential_signed_pair_tail_cap == 0.028
    assert (
        v177.euler_recon.euler_tangential_signed_pair_tail_wave_cap
        == 0.0036)
    assert (
        v177.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.024)
    assert (
        v177.euler_recon.euler_tangential_density_curve_pair_tail_cap
        == 0.019)
    assert (
        v177.euler_recon.euler_tangential_density_curve_pair_tail_wave_cap
        == 0.0024)
    assert (
        v177.euler_recon.euler_tangential_signed_tail_hf_filter_strength
        == 0.18)
    assert (
        v177.euler_recon.euler_tangential_density_curve_tail_hf_filter_strength
        == 0.30)
    assert v177.euler_recon.euler_tangential_pair_extend_on is False
    assert v177.euler_recon.euler_density_signed_tail_trace_on is False


def test_v178_v174_dual_bridge_cut_enables_signed_and_curve_cuts():
    v178 = bench._reconstruction_from_key(
        "tmlpu_v178_v174_dual_bridge_cut_on")

    assert v178.euler_recon.euler_tangential_signed_pair_tail_beta == 0.030
    assert (
        v178.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.020)
    assert v178.euler_recon.euler_tangential_signed_tail_bridge_cut_on is True
    assert (
        v178.euler_recon.euler_tangential_density_curve_tail_bridge_cut_on
        is True)
    assert (
        v178.euler_recon.euler_tangential_signed_tail_bridge_cut_strength
        == 0.14)
    assert (
        v178.euler_recon.euler_tangential_signed_tail_bridge_cut_min_factor
        == 0.78)
    assert (
        v178.euler_recon
        .euler_tangential_density_curve_tail_bridge_cut_strength
        == 0.16)
    assert (
        v178.euler_recon
        .euler_tangential_density_curve_tail_bridge_cut_min_factor
        == 0.76)
    assert v178.euler_recon.euler_tangential_pair_extend_on is False
    assert v178.euler_recon.euler_density_signed_tail_trace_on is False


def test_v179_v174_antisheet_damps_weak_q_contact_sheet():
    v179 = bench._reconstruction_from_key(
        "tmlpu_v179_v174_antisheet_on")

    assert v179.euler_recon.euler_tangential_signed_pair_tail_beta == 0.030
    assert (
        v179.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.020)
    assert v179.euler_recon.euler_tangential_signed_tail_antisheet_on is True
    assert (
        v179.euler_recon.euler_tangential_signed_tail_antisheet_strength
        == 0.35)
    assert (
        v179.euler_recon.euler_tangential_signed_tail_antisheet_min_factor
        == 0.60)
    assert (
        v179.euler_recon.euler_tangential_signed_tail_antisheet_q_hi
        == 0.055)
    assert (
        v179.euler_recon.euler_tangential_signed_tail_antisheet_contact_min
        == 0.28)
    assert (
        v179.euler_recon.euler_tangential_signed_tail_antisheet_contact_full
        == 0.66)
    assert v179.euler_recon.euler_tangential_pair_extend_on is False
    assert v179.euler_recon.euler_density_signed_tail_trace_on is False


def test_v180_v174_swirl_core_enables_q_dominant_tail():
    v180 = bench._reconstruction_from_key(
        "tmlpu_v180_v174_swirl_core_on")

    assert v180.euler_recon.euler_tangential_signed_pair_tail_beta == 0.030
    assert (
        v180.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.020)
    assert v180.euler_recon.euler_tangential_swirl_tail_on is True
    assert v180.euler_recon.euler_tangential_swirl_tail_beta == 0.018
    assert v180.euler_recon.euler_tangential_swirl_tail_cap == 0.014
    assert v180.euler_recon.euler_tangential_swirl_tail_wave_cap == 0.0018
    assert v180.euler_recon.euler_tangential_swirl_tail_q_min == 0.010
    assert v180.euler_recon.euler_tangential_swirl_tail_q_full == 0.036
    assert v180.euler_recon.euler_tangential_swirl_tail_pressure_hi == 0.018
    assert (
        v180.euler_recon.euler_tangential_swirl_tail_compression_hi
        == 0.004)
    assert v180.euler_recon.euler_tangential_swirl_tail_normality_hi == 0.18
    assert v180.euler_recon.euler_tangential_pair_extend_on is False
    assert v180.euler_recon.euler_density_signed_tail_trace_on is False


def test_v181_v174_qbridge_cut_damps_mid_q_bridges_only():
    v181 = bench._reconstruction_from_key(
        "tmlpu_v181_v174_qbridge_cut_on")

    assert v181.scalar_recon is not v181.euler_recon
    assert v181.euler_recon.euler_tangential_signed_pair_tail_beta == 0.030
    assert (
        v181.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.020)
    assert v181.euler_recon.euler_tangential_signed_tail_qbridge_cut_on is True
    assert (
        v181.euler_recon.euler_tangential_signed_tail_qbridge_cut_strength
        == 0.42)
    assert (
        v181.euler_recon.euler_tangential_signed_tail_qbridge_cut_min_factor
        == 0.62)
    assert (
        v181.euler_recon.euler_tangential_signed_tail_qbridge_cut_q_lo_pct
        == 28.0)
    assert (
        v181.euler_recon.euler_tangential_signed_tail_qbridge_cut_q_mid_pct
        == 60.0)
    assert (
        v181.euler_recon.euler_tangential_signed_tail_qbridge_cut_q_core_pct
        == 82.0)
    assert (
        v181.euler_recon.euler_tangential_signed_tail_qbridge_cut_q_top_pct
        == 96.0)
    assert (
        v181.euler_recon.euler_tangential_density_curve_tail_qbridge_cut_on
        is True)
    assert (
        v181.euler_recon.euler_tangential_density_curve_tail_qbridge_cut_strength
        == 0.50)
    assert (
        v181.euler_recon.euler_tangential_density_curve_tail_qbridge_cut_min_factor
        == 0.56)
    assert v181.euler_recon.euler_tangential_swirl_tail_on is False
    assert v181.euler_recon.euler_tangential_pair_extend_on is False
    assert v181.euler_recon.euler_density_signed_tail_trace_on is False


def test_v182_v174_total_qbridge_damp_targets_full_tangential_update():
    v182 = bench._reconstruction_from_key(
        "tmlpu_v182_v174_total_qbridge_damp_on")

    assert v182.scalar_recon is not v182.euler_recon
    assert v182.euler_recon.euler_tangential_signed_pair_tail_beta == 0.030
    assert (
        v182.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.020)
    assert v182.euler_recon.euler_tangential_total_qbridge_damp_on is True
    assert (
        v182.euler_recon.euler_tangential_total_qbridge_damp_strength
        == 0.30)
    assert (
        v182.euler_recon.euler_tangential_total_qbridge_damp_min_factor
        == 0.70)
    assert (
        v182.euler_recon.euler_tangential_total_qbridge_damp_q_lo_pct
        == 25.0)
    assert (
        v182.euler_recon.euler_tangential_total_qbridge_damp_q_mid_pct
        == 58.0)
    assert (
        v182.euler_recon.euler_tangential_total_qbridge_damp_q_core_pct
        == 80.0)
    assert (
        v182.euler_recon.euler_tangential_total_qbridge_damp_q_top_pct
        == 95.0)
    assert v182.euler_recon.euler_tangential_swirl_tail_on is False
    assert v182.euler_recon.euler_tangential_pair_extend_on is False
    assert v182.euler_recon.euler_density_signed_tail_trace_on is False


def test_v183_v174_micro_pair_extend_is_weaker_than_v176():
    v183 = bench._reconstruction_from_key(
        "tmlpu_v183_v174_micro_pair_extend_on")
    v176 = bench._reconstruction_from_key(
        "tmlpu_v176_v174_pair_extend_roi_on")

    assert v183.scalar_recon is not v183.euler_recon
    assert v183.euler_recon.euler_tangential_pair_extend_on is True
    assert v183.euler_recon.euler_tangential_pair_extend_beta == 0.0035
    assert v183.euler_recon.euler_tangential_pair_extend_cap == 0.0028
    assert (
        v183.euler_recon.euler_tangential_pair_extend_wave_cap
        == 0.00045)
    assert (
        v183.euler_recon.euler_tangential_pair_extend_beta
        < v176.euler_recon.euler_tangential_pair_extend_beta)
    assert (
        v183.euler_recon.euler_tangential_pair_extend_cap
        < v176.euler_recon.euler_tangential_pair_extend_cap)
    assert v183.euler_recon.euler_tangential_pair_extend_shock_exclude is True
    assert v183.euler_recon.euler_tangential_swirl_tail_on is False
    assert v183.euler_recon.euler_density_signed_tail_trace_on is False


def test_v184_v174_midq_cell_blend_enables_direct_bridge_damping():
    v184 = bench._reconstruction_from_key(
        "tmlpu_v184_v174_midq_cell_blend_on")

    assert v184.scalar_recon is not v184.euler_recon
    assert v184.euler_recon.euler_tangential_midq_cell_blend_on is True
    assert v184.euler_recon.euler_tangential_midq_cell_blend_strength == 0.32
    assert v184.euler_recon.euler_tangential_midq_cell_blend_q_lo_pct == 12.0
    assert v184.euler_recon.euler_tangential_midq_cell_blend_q_mid_pct == 48.0
    assert (
        v184.euler_recon.euler_tangential_midq_cell_blend_q_core_pct
        == 78.0)
    assert v184.euler_recon.euler_tangential_midq_cell_blend_q_top_pct == 95.0
    assert (
        v184.euler_recon.euler_tangential_midq_cell_blend_contact_min
        == 0.18)
    assert (
        v184.euler_recon.euler_tangential_midq_cell_blend_contact_full
        == 0.52)
    assert v184.euler_recon.euler_tangential_pair_extend_on is False
    assert v184.euler_recon.euler_tangential_swirl_tail_on is False
    assert v184.euler_recon.euler_density_signed_tail_trace_on is False


def test_v185_v174_soft_midq_cell_blend_is_weaker_and_narrower_than_v184():
    v184 = bench._reconstruction_from_key(
        "tmlpu_v184_v174_midq_cell_blend_on")
    v185 = bench._reconstruction_from_key(
        "tmlpu_v185_v174_soft_midq_cell_blend_on")

    assert v185.scalar_recon is not v185.euler_recon
    assert v185.euler_recon.euler_tangential_midq_cell_blend_on is True
    assert v185.euler_recon.euler_tangential_midq_cell_blend_strength == 0.16
    assert (
        v185.euler_recon.euler_tangential_midq_cell_blend_strength
        < v184.euler_recon.euler_tangential_midq_cell_blend_strength)
    assert v185.euler_recon.euler_tangential_midq_cell_blend_q_lo_pct == 20.0
    assert v185.euler_recon.euler_tangential_midq_cell_blend_q_mid_pct == 58.0
    assert (
        v185.euler_recon.euler_tangential_midq_cell_blend_q_core_pct
        == 86.0)
    assert v185.euler_recon.euler_tangential_midq_cell_blend_q_top_pct == 97.0
    assert (
        v185.euler_recon.euler_tangential_midq_cell_blend_contact_min
        == 0.22)
    assert (
        v185.euler_recon.euler_tangential_midq_cell_blend_contact_full
        == 0.58)
    assert v185.euler_recon.euler_tangential_pair_extend_on is False
    assert v185.euler_recon.euler_tangential_swirl_tail_on is False


def test_v169_v167_qtight_core_raises_rotation_gates_without_hffilter():
    v169 = bench._reconstruction_from_key(
        "tmlpu_v169_v167_qtight_core_on")

    assert v169.euler_recon.euler_tangential_signed_pair_tail_on is True
    assert v169.euler_recon.euler_tangential_density_curve_pair_tail_on is True
    assert v169.euler_recon.euler_tangential_signed_pair_tail_beta == 0.020
    assert (
        v169.euler_recon.euler_tangential_density_curve_pair_tail_beta
        == 0.012)
    assert v169.euler_recon.euler_tangential_signed_pair_tail_q_min == 0.022
    assert v169.euler_recon.euler_tangential_signed_pair_tail_q_full == 0.070
    assert (
        v169.euler_recon.euler_tangential_density_curve_pair_tail_q_min
        == 0.020)
    assert (
        v169.euler_recon.euler_tangential_density_curve_pair_tail_q_full
        == 0.065)
    assert v169.euler_recon.euler_tangential_signed_tail_hf_filter_on is False
    assert (
        v169.euler_recon.euler_tangential_density_curve_tail_hf_filter_on
        is False)
    assert v169.euler_recon.euler_tangential_pair_extend_on is False
    assert v169.euler_recon.euler_density_signed_tail_trace_on is False


def test_v170_v167_pressure_entropy_enables_contact_pressure_smoothing():
    v170 = bench._reconstruction_from_key(
        "tmlpu_v170_v167_pressure_entropy_on")

    assert v170.euler_recon.euler_tangential_signed_pair_tail_on is True
    assert v170.euler_recon.euler_tangential_density_curve_pair_tail_on is True
    assert v170.euler_recon.euler_pressure_contact_entropy_blend is True
    assert v170.euler_recon.euler_pressure_contact_entropy_beta == 0.12
    assert v170.euler_recon.euler_pressure_contact_entropy_cap == 0.08
    assert v170.euler_recon.euler_pressure_contact_entropy_downscale == 1.0
    assert (
        v170.euler_recon.euler_pressure_contact_entropy_p_jump_threshold
        == 0.025)
    assert v170.euler_recon.euler_pressure_contact_entropy_p_jump_width == 0.070
    assert (
        v170.euler_recon.euler_pressure_contact_entropy_compression_threshold
        == 0.008)
    assert (
        v170.euler_recon.euler_pressure_contact_entropy_compression_width
        == 0.060)
    assert (
        v170.euler_recon.euler_pressure_contact_entropy_normality_threshold
        == 0.40)
    assert (
        v170.euler_recon.euler_pressure_contact_entropy_normality_width
        == 0.30)
    assert v170.euler_recon.euler_tangential_pair_extend_on is False
    assert v170.euler_recon.euler_density_signed_tail_trace_on is False


def test_v171_v167_pressure_jump_limit_enables_pressure_jump_limiter():
    v171 = bench._reconstruction_from_key(
        "tmlpu_v171_v167_pressure_jump_limit_on")

    assert v171.euler_recon.euler_tangential_signed_pair_tail_on is True
    assert v171.euler_recon.euler_tangential_density_curve_pair_tail_on is True
    assert v171.euler_recon.euler_pressure_contact_entropy_blend is False
    assert v171.euler_recon.euler_pressure_face_jump_limiter_on is True
    assert v171.euler_recon.euler_pressure_face_jump_limiter_strength == 0.85
    assert v171.euler_recon.euler_pressure_face_jump_limiter_growth_cap == 0.10
    assert v171.euler_recon.euler_pressure_face_jump_limiter_abs_floor == 1e-10
    assert (
        v171.euler_recon.euler_pressure_face_jump_limiter_p_jump_threshold
        == 0.025)
    assert v171.euler_recon.euler_pressure_face_jump_limiter_p_jump_width == 0.070
    assert (
        v171.euler_recon
        .euler_pressure_face_jump_limiter_compression_threshold
        == 0.008)
    assert (
        v171.euler_recon.euler_pressure_face_jump_limiter_compression_width
        == 0.060)
    assert (
        v171.euler_recon.euler_pressure_face_jump_limiter_normality_threshold
        == 0.40)
    assert (
        v171.euler_recon.euler_pressure_face_jump_limiter_normality_width
        == 0.30)
    assert v171.euler_recon.euler_tangential_pair_extend_on is False
    assert v171.euler_recon.euler_density_signed_tail_trace_on is False


def test_v172_v167_soft_bridge_cut_enables_weak_bridge_attenuation():
    v172 = bench._reconstruction_from_key(
        "tmlpu_v172_v167_soft_bridge_cut_on")

    assert v172.euler_recon.euler_tangential_signed_pair_tail_on is True
    assert v172.euler_recon.euler_tangential_density_curve_pair_tail_on is True
    assert v172.euler_recon.euler_pressure_contact_entropy_blend is False
    assert v172.euler_recon.euler_pressure_face_jump_limiter_on is False
    assert v172.euler_recon.euler_tangential_signed_tail_bridge_cut_on is True
    assert v172.euler_recon.euler_tangential_signed_tail_bridge_cut_strength == 0.12
    assert (
        v172.euler_recon.euler_tangential_signed_tail_bridge_cut_min_factor
        == 0.82)
    assert v172.euler_recon.euler_tangential_signed_tail_bridge_cut_q_min == 0.16
    assert v172.euler_recon.euler_tangential_signed_tail_bridge_cut_q_full == 0.34
    assert (
        v172.euler_recon.euler_tangential_signed_tail_bridge_cut_contact_min
        == 0.35)
    assert (
        v172.euler_recon.euler_tangential_signed_tail_bridge_cut_contact_full
        == 0.72)
    assert (
        v172.euler_recon.euler_tangential_signed_tail_bridge_cut_omega_lo_pct
        == 78.0)
    assert (
        v172.euler_recon.euler_tangential_signed_tail_bridge_cut_omega_hi_pct
        == 97.0)
    assert v172.euler_recon.euler_tangential_pair_extend_on is False
    assert v172.euler_recon.euler_density_signed_tail_trace_on is False


def test_v173_v167_curve_bridge_cut_extends_bridge_attenuation_to_curve_tail():
    v173 = bench._reconstruction_from_key(
        "tmlpu_v173_v167_curve_bridge_cut_on")

    assert v173.euler_recon.euler_tangential_signed_tail_bridge_cut_on is True
    assert (
        v173.euler_recon.euler_tangential_density_curve_tail_bridge_cut_on
        is True)
    assert (
        v173.euler_recon
        .euler_tangential_density_curve_tail_bridge_cut_strength
        == 0.18)
    assert (
        v173.euler_recon
        .euler_tangential_density_curve_tail_bridge_cut_min_factor
        == 0.78)
    assert (
        v173.euler_recon.euler_tangential_density_curve_tail_bridge_cut_q_min
        == 0.14)
    assert (
        v173.euler_recon.euler_tangential_density_curve_tail_bridge_cut_q_full
        == 0.32)
    assert (
        v173.euler_recon
        .euler_tangential_density_curve_tail_bridge_cut_contact_min
        == 0.30)
    assert (
        v173.euler_recon
        .euler_tangential_density_curve_tail_bridge_cut_contact_full
        == 0.68)
    assert (
        v173.euler_recon
        .euler_tangential_density_curve_tail_bridge_cut_omega_lo_pct
        == 75.0)
    assert (
        v173.euler_recon
        .euler_tangential_density_curve_tail_bridge_cut_omega_hi_pct
        == 96.0)
    assert v173.euler_recon.euler_pressure_face_jump_limiter_on is False
