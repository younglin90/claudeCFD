from pathlib import Path

from manuscript_tables import (
    cone_jet_error_budget_markdown,
    coupled_droplet_grid_refinement_markdown,
    dielectric_maxwell_droplet_history_markdown,
    huh_wirz_same_path_grid_refinement_markdown,
    taylor_cone_voltage_ramp_markdown,
)


def test_generated_manuscript_tables_match_committed_artifacts() -> None:
    expected = {
        Path("docs/electrospray/cone_jet_error_budget_table.md"): cone_jet_error_budget_markdown(),
        Path("docs/electrospray/taylor_cone_voltage_ramp_balance_table.md"): taylor_cone_voltage_ramp_markdown(),
        Path("docs/electrospray/coupled_droplet_grid_refinement_table.md"): coupled_droplet_grid_refinement_markdown(),
        Path("docs/electrospray/dielectric_maxwell_droplet_history_table.md"): dielectric_maxwell_droplet_history_markdown(),
        Path("docs/electrospray/huh_wirz_same_path_grid_refinement_table.md"): huh_wirz_same_path_grid_refinement_markdown(),
    }

    for path, generated in expected.items():
        assert path.read_text(encoding="utf-8") == generated


def test_dielectric_maxwell_droplet_table_reports_voltage_squared_scaling() -> None:
    table = dielectric_maxwell_droplet_history_markdown()

    assert "Voltage-Squared Scaling Check" in table
    assert "| Final deformation D |" in table
    assert "| Max Maxwell acceleration |" in table
    assert "4.000000e+00" in table


def test_dielectric_maxwell_droplet_table_reports_grid_refinement() -> None:
    table = dielectric_maxwell_droplet_history_markdown()

    assert "Grid-Refinement Check" in table
    assert "| Grid | Initial D | Final D | Increment from previous grid | Status |" in table
    assert "medium-to-fine deformation increment" in table


def test_dielectric_maxwell_droplet_table_reports_timestep_refinement() -> None:
    table = dielectric_maxwell_droplet_history_markdown()

    assert "Timestep-Refinement Check" in table
    assert "| dt | Steps | Final D | Relative delta | Status |" in table
    assert "1.000000e-04" in table


def test_huh_wirz_same_path_table_reports_iteration_refinement() -> None:
    table = huh_wirz_same_path_grid_refinement_markdown()

    assert "Pseudo-Time Iteration Refinement" in table
    assert "| Steps | Current | Jet diameter | Droplet diameter | q/m | Status |" in table
    assert "maximum 24-to-48-step relative change" in table
