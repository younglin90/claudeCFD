from __future__ import annotations

from pathlib import Path

from cone_jet_metrics import ConeJetObservableSet, cone_jet_error_budget_rows


def test_cone_jet_error_budget_table_artifact_matches_executable_rows() -> None:
    text = Path("docs/electrospray/cone_jet_error_budget_table.md").read_text()
    reference = ConeJetObservableSet(
        current=8.0e-8,
        jet_diameter=1.2e-6,
        droplet_diameter=3.0e-6,
        charge_to_mass=2.5e5,
    )
    prediction = ConeJetObservableSet(
        current=8.6e-8,
        jet_diameter=1.1e-6,
        droplet_diameter=3.35e-6,
        charge_to_mass=2.25e5,
    )
    tolerances = {
        "current": 0.20,
        "jet_diameter": 0.20,
        "droplet_diameter": 0.25,
        "charge_to_mass": 0.20,
    }

    for row in cone_jet_error_budget_rows(prediction, reference, tolerances):
        assert row.observable in text
        assert f"{row.prediction:.6e}" in text
        assert f"{row.reference:.6e}" in text
        assert f"{row.relative_error:.6e}" in text
        assert f"{row.tolerance:.6e}" in text
    assert "Acceptance requires every observable to pass" in text
    assert "Reference provenance: internal reduced verification reference" in text
    assert "Huh-Wirz external metadata remains metadata_only" in text
    assert "digitized external cone-jet reference values" in text
