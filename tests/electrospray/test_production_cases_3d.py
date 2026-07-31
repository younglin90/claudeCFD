from __future__ import annotations

from dataclasses import replace

import numpy as np

from production_cases_3d import (
    DEFECT_BENT,
    DEFECT_BUMP,
    DEFECT_NORMAL,
    DEFECT_OXIDIZED,
    DEFECT_SPLIT,
    build_production_case_setup,
    config_with_ramped_voltage,
    default_production_case_matrix,
    production_observables_3d,
    production_phase_pair,
)


def _small_case(case_id: str):
    case = next(case for case in default_production_case_matrix() if case.case_id == case_id)
    return replace(case, grid=(7, 7, 12), dt=1.0e-3)


def test_default_production_case_matrix_contains_manuscript_cases() -> None:
    cases = default_production_case_matrix()
    ids = {case.case_id for case in cases}
    kinds = {case.defect.kind for case in cases}

    assert len(cases) == 9
    assert {"P0_normal", "P1_bent_10deg", "P2_bump_010Do", "P3_split_010Do", "P4_oxidized_severe"} <= ids
    assert {DEFECT_NORMAL, DEFECT_BENT, DEFECT_BUMP, DEFECT_SPLIT, DEFECT_OXIDIZED} <= kinds


def test_production_setup_uses_voltage_ramp_and_open_boundaries() -> None:
    case = _small_case("P0_normal")
    setup = build_production_case_setup(case, time=0.0)

    assert setup.voltage == 0.0
    assert setup.config.potential_boundary_values["z_min"] == 0.0
    assert setup.config.boundary_conditions["z_min"].kind == "inlet"
    assert setup.config.boundary_conditions["z_max"].kind == "outlet"
    assert setup.config.boundary_conditions["x_min"].kind == "open"
    assert setup.config.boundary_conditions["y_max"].kind == "open"
    assert np.all((setup.state.alpha_liquid >= 0.0) & (setup.state.alpha_liquid <= 1.0))
    assert setup.state.velocity.shape == (setup.mesh.cell_count, 3)
    assert setup.state.charge_density.shape == (setup.mesh.cell_count,)

    ramped = config_with_ramped_voltage(case, setup.config, time=case.ramp_time)
    assert ramped.potential_boundary_values["z_min"] == case.emitter_voltage
    assert ramped.boundary_conditions["z_min"].potential == case.emitter_voltage


def test_defect_initial_conditions_change_geometry_or_material() -> None:
    normal = _small_case("P0_normal")
    bent = _small_case("P1_bent_10deg")
    bump = _small_case("P2_bump_010Do")
    split = _small_case("P3_split_010Do")
    oxidized = _small_case("P4_oxidized_severe")

    normal_setup = build_production_case_setup(normal, time=normal.ramp_time)
    bent_setup = build_production_case_setup(bent, time=bent.ramp_time)
    bump_setup = build_production_case_setup(bump, time=bump.ramp_time)
    split_setup = build_production_case_setup(split, time=split.ramp_time)

    normal_mass = float(np.sum(normal_setup.state.alpha_liquid))
    bent_centroid_x = float(
        np.sum(bent_setup.mesh.cell_centers[:, 0] * bent_setup.state.alpha_liquid)
        / max(np.sum(bent_setup.state.alpha_liquid), 1.0e-300)
    )
    normal_centroid_x = float(
        np.sum(normal_setup.mesh.cell_centers[:, 0] * normal_setup.state.alpha_liquid)
        / max(np.sum(normal_setup.state.alpha_liquid), 1.0e-300)
    )

    assert bent_centroid_x > normal_centroid_x
    assert float(np.sum(bump_setup.state.alpha_liquid)) >= normal_mass
    assert np.any(split_setup.state.alpha_liquid > 0.5)
    assert production_phase_pair(oxidized).liquid.conductivity < production_phase_pair(normal).liquid.conductivity


def test_production_observables_are_finite() -> None:
    case = _small_case("P0_normal")
    setup = build_production_case_setup(case, time=case.ramp_time)
    observables = production_observables_3d(setup.mesh, setup.state, case)

    assert np.isfinite(observables.cone_angle_degrees)
    assert np.isfinite(observables.jet_radius)
    assert np.isfinite(observables.whipping_offset)
    assert np.isfinite(observables.mean_charge_to_mass_ratio)
    assert observables.cone_angle_degrees > 0.0
    assert observables.jet_radius > 0.0
    assert observables.droplet_count >= 1
