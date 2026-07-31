from pathlib import Path


def test_huh_wirz_benchmark_requirements_capture_primary_source_and_outputs() -> None:
    text = Path("docs/electrospray/huh_wirz_conejet_benchmark_requirements.md").read_text(encoding="utf-8")

    for required in (
        "10.1063/5.0120737",
        "arXiv:2111.10383",
        "VOF",
        "leaky-dielectric",
        "charge relaxation",
        "droplet diameter",
        "total current",
        "charge-to-mass",
    ):
        assert required in text


def test_huh_wirz_benchmark_requirements_keep_completion_gate_numeric() -> None:
    text = Path("docs/electrospray/huh_wirz_conejet_benchmark_requirements.md").read_text(encoding="utf-8")

    assert "digitized or tabulated reference values" in text
    assert "machine-readable artifact" in text
    assert "not a resolved two-phase Navier-Stokes cone-jet reproduction" in text
