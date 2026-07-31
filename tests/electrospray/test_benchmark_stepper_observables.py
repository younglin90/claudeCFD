import pytest

from benchmark_stepper_observables import (
    AXISYMMETRIC_CONEJET_SOLVER_PATH,
    COMMON_SCHEME_ID,
    COUPLED_DROPLET_SOLVER_PATH,
    FULL_CFD_AXISYMMETRIC_CONEJET_SOLVER_PATH,
    FULL_CFD_DROPLET_SOLVER_PATH,
    das_saintillan_coupled_droplet_predictions,
    das_saintillan_full_cfd_droplet_predictions,
    huh_wirz_axisymmetric_conejet_predictions,
    huh_wirz_full_cfd_breakup_predictions,
    huh_wirz_full_cfd_axisymmetric_conejet_predictions,
    prediction_mapping,
    prediction_scheme_ids,
    prediction_solver_paths,
    run_axisymmetric_conejet_grid_refinement,
    run_axisymmetric_conejet_observable_stepper,
    run_axisymmetric_conejet_transient_stepper,
    run_full_cfd_axisymmetric_conejet_observable_stepper,
    run_full_cfd_droplet_benchmark_stepper,
)
from full_cfd_solver import FULL_CFD_SCHEME_ID


def test_huh_wirz_predictions_are_tied_to_axisymmetric_solver_path() -> None:
    predictions = huh_wirz_axisymmetric_conejet_predictions()

    assert prediction_solver_paths(predictions) == {AXISYMMETRIC_CONEJET_SOLVER_PATH}
    assert prediction_scheme_ids(predictions) == {COMMON_SCHEME_ID}
    tbp = run_axisymmetric_conejet_observable_stepper("huh_wirz_tbp_high_conductivity")
    mapping = prediction_mapping(predictions)
    assert mapping[("huh_wirz_tbp_high_conductivity", "jet_diameter")] == pytest.approx(tbp.jet_diameter)
    assert mapping[("huh_wirz_tbp_high_conductivity", "total_current")] == pytest.approx(tbp.total_current)
    assert mapping[("huh_wirz_tbp_minimum_flow_cone_to_jet", "cone_to_jet_length")] == pytest.approx(
        6.477765754166168e-05
    )
    assert tbp.jet_diameter == pytest.approx(6.20166475902742e-6)


def test_huh_wirz_nonbreakup_predictions_are_tied_to_full_cfd_timestep_path() -> None:
    predictions = huh_wirz_full_cfd_axisymmetric_conejet_predictions()

    assert prediction_solver_paths(predictions) == {FULL_CFD_AXISYMMETRIC_CONEJET_SOLVER_PATH}
    assert prediction_scheme_ids(predictions) == {FULL_CFD_SCHEME_ID}
    mapping = prediction_mapping(predictions)
    tbp = run_full_cfd_axisymmetric_conejet_observable_stepper("huh_wirz_tbp_high_conductivity")
    minimum_flow = run_full_cfd_axisymmetric_conejet_observable_stepper("huh_wirz_tbp_minimum_flow_cone_to_jet")
    assert tbp.all_required_operators_active is True
    assert tbp.jet_diameter == pytest.approx(mapping[("huh_wirz_tbp_high_conductivity", "jet_diameter")])
    assert tbp.total_current == pytest.approx(mapping[("huh_wirz_tbp_high_conductivity", "total_current")])
    assert tbp.charge_to_mass_ratio == pytest.approx(
        mapping[("huh_wirz_tbp_high_conductivity", "charge_to_mass_ratio")]
    )
    assert minimum_flow.cone_to_jet_length == pytest.approx(
        mapping[("huh_wirz_tbp_minimum_flow_cone_to_jet", "cone_to_jet_length")]
    )
    assert tbp.gas_charge_leakage_fraction <= 1.0e-12
    assert tbp.divergence_reduction_ratio < 0.20


def test_huh_wirz_subgrid_breakup_predictions_use_full_cfd_timestep_outputs() -> None:
    predictions = huh_wirz_full_cfd_breakup_predictions()

    assert prediction_solver_paths(predictions) == {FULL_CFD_AXISYMMETRIC_CONEJET_SOLVER_PATH}
    assert prediction_scheme_ids(predictions) == {FULL_CFD_SCHEME_ID}
    mapping = prediction_mapping(predictions)
    assert mapping[("huh_wirz_heptane_moderate_conductivity", "droplet_diameter")] == pytest.approx(
        3.4168229257766155e-05
    )
    assert mapping[("huh_wirz_tbp_high_conductivity", "droplet_diameter")] == pytest.approx(1.0711695407254123e-05)
    assert mapping[("huh_wirz_tbp_high_conductivity", "charge_to_mass_ratio")] == pytest.approx(0.62)


def test_das_saintillan_predictions_are_tied_to_coupled_droplet_solver_path() -> None:
    predictions = das_saintillan_coupled_droplet_predictions()

    assert prediction_solver_paths(predictions) == {COUPLED_DROPLET_SOLVER_PATH}
    assert prediction_scheme_ids(predictions) == {COMMON_SCHEME_ID}
    assert prediction_mapping(predictions)[("das_saintillan_prolate_system_3", "deformation_parameter")] == pytest.approx(
        0.28383354887261714
    )


def test_das_saintillan_full_cfd_predictions_use_full_timestep_path() -> None:
    predictions = das_saintillan_full_cfd_droplet_predictions()

    assert prediction_solver_paths(predictions) == {FULL_CFD_DROPLET_SOLVER_PATH}
    assert prediction_scheme_ids(predictions) == {FULL_CFD_SCHEME_ID}
    mapping = prediction_mapping(predictions)
    assert mapping[("das_saintillan_transient_system_1b", "deformation_parameter")] == pytest.approx(
        -0.07522166670107548
    )
    assert mapping[("das_saintillan_transient_system_1c", "deformation_parameter")] == pytest.approx(
        -0.13450768203170077
    )
    assert mapping[("das_saintillan_prolate_system_3", "deformation_parameter")] == pytest.approx(
        0.2825750210312717
    )
    run = run_full_cfd_droplet_benchmark_stepper("y", 6)
    assert run.scheme_id == FULL_CFD_SCHEME_ID
    assert run.vof_area_relative_error <= 1.0e-12
    assert run.gas_charge_leakage_fraction <= 1.0e-12
    assert run.alpha_bounds_violation <= 1.0e-12


def test_axisymmetric_conejet_interface_focuses_over_pseudo_time() -> None:
    transient = run_axisymmetric_conejet_transient_stepper("huh_wirz_tbp_high_conductivity")

    assert len(transient.jet_diameter_history) == 8
    assert transient.monotone_focusing is True
    assert transient.jet_diameter_history[0] > transient.jet_diameter_history[-1]
    assert transient.final_to_initial_ratio < 0.08


def test_axisymmetric_conejet_observables_have_grid_refinement_evidence() -> None:
    refinement = run_axisymmetric_conejet_grid_refinement("huh_wirz_tbp_high_conductivity")

    assert refinement.medium_fine_relative_change < 0.05
    assert refinement.coarse_medium_relative_change > refinement.medium_fine_relative_change
    assert refinement.fine.jet_diameter == pytest.approx(6.2015371410032736e-6)
    assert refinement.fine.total_current == pytest.approx(3.092884917036496e-8)


def test_prediction_mapping_rejects_duplicate_case_observable_pairs() -> None:
    prediction = huh_wirz_axisymmetric_conejet_predictions()[0]

    with pytest.raises(ValueError, match="duplicate solver prediction"):
        prediction_mapping((prediction, prediction))
