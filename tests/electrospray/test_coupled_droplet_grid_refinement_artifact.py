from pathlib import Path

from validation_cases_coupled import coupled_ehd_2d_droplet_deformation_observable


def test_coupled_droplet_grid_refinement_artifact_matches_executable_case() -> None:
    artifact = Path("docs/electrospray/coupled_droplet_grid_refinement_table.md")
    text = artifact.read_text(encoding="utf-8")

    final_deformations = []
    for grid_size in (16, 20, 24):
        final, initial, diagnostics, _state = coupled_ehd_2d_droplet_deformation_observable(grid_size)
        max_violation = max(item.max_violation for item in diagnostics)
        max_gas_leakage = max(item.gas_charge_leakage_fraction for item in diagnostics)
        max_charge_relaxation = max(abs(item.free_charge_loss_fraction) for item in diagnostics)
        max_projection_ratio = max(item.projected_divergence_norm / item.predictor_divergence_norm for item in diagnostics)
        max_pressure_correction_norm = max(item.pressure_correction_norm for item in diagnostics)
        final_deformations.append(final)

        assert f"| {grid_size} |" in text
        assert f"{initial:.6e}" in text
        assert f"{final:.6e}" in text
        assert f"{(final - initial):.6e}" in text
        assert f"{max_violation:.6e}" in text
        assert f"{max_gas_leakage:.6e}" in text
        assert f"{max_charge_relaxation:.6e}" in text
        assert f"{max_projection_ratio:.6e}" in text
        assert f"{max_pressure_correction_norm:.6e}" in text

    coarse, medium, fine = final_deformations
    refinement_ratio = abs(fine - medium) / abs(medium - coarse)
    assert coarse < medium < fine
    assert refinement_ratio <= 0.6
    assert f"{refinement_ratio:.6e}" in text
    assert "non-zero pressure correction" in text
    assert "Charge loss is reported separately" in text
