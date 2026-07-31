from pathlib import Path

from manuscript_figures import write_manuscript_figures


def test_write_manuscript_figures_creates_nonempty_pngs(tmp_path) -> None:
    paths = write_manuscript_figures(tmp_path)

    assert [path.name for path in paths] == [
        "cone_jet_error_budget.png",
        "taylor_cone_voltage_ramp.png",
        "coupled_droplet_grid_refinement.png",
        "external_benchmark_numeric_comparison.png",
    ]
    for path in paths:
        assert path.is_file()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1000
        assert path.parent == tmp_path / "docs" / "electrospray" / "figures"
