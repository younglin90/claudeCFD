from pathlib import Path


def test_figure_manifest_lists_generated_manuscript_figures() -> None:
    text = Path("docs/electrospray/figure_manifest.md").read_text(encoding="utf-8")

    for figure in (
        "docs/electrospray/figures/cone_jet_error_budget.png",
        "docs/electrospray/figures/taylor_cone_voltage_ramp.png",
        "docs/electrospray/figures/coupled_droplet_grid_refinement.png",
    ):
        assert figure in text


def test_figure_manifest_keeps_external_benchmark_gap_explicit() -> None:
    text = Path("docs/electrospray/figure_manifest.md").read_text(encoding="utf-8")

    assert "deterministic validation artifacts" in text
    assert "do not replace final external benchmark plots" in text
