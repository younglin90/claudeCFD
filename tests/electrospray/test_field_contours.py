from pathlib import Path

from field_contours import (
    FIELD_CONTOUR_CASES,
    FIELD_CONTOUR_FIELDS,
    expected_field_contour_png_paths,
    field_contour_artifacts_are_current,
    field_contour_manifest_markdown,
    write_field_contour_artifacts,
)


def test_write_field_contour_artifacts_creates_all_major_calculation_pngs(tmp_path) -> None:
    paths = write_field_contour_artifacts(tmp_path)

    expected_png_count = len(FIELD_CONTOUR_CASES) * len(FIELD_CONTOUR_FIELDS)
    assert len([path for path in paths if path.suffix == ".png"]) == expected_png_count
    assert all(path.is_file() for path in paths)
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths if path.suffix == ".png")
    assert field_contour_artifacts_are_current(tmp_path / "docs" / "electrospray")


def test_field_contour_manifest_documents_pressure_velocity_temperature_density(tmp_path) -> None:
    write_field_contour_artifacts(tmp_path)
    text = (tmp_path / "docs" / "electrospray" / "field_contour_manifest.md").read_text(encoding="utf-8")

    assert text == field_contour_manifest_markdown()
    assert "pressure.png" in text
    assert "velocity_magnitude.png" in text
    assert "temperature_isothermal_K.png" in text
    assert "density.png" in text
    assert "full_cfd_huh_wirz_tbp_axisymmetric_conejet" in text


def test_expected_field_contour_png_paths_are_stable(tmp_path) -> None:
    paths = expected_field_contour_png_paths(tmp_path)

    assert len(paths) == len(FIELD_CONTOUR_CASES) * len(FIELD_CONTOUR_FIELDS)
    assert paths[0] == tmp_path / "docs" / "electrospray" / "contours" / "full_cfd_timestep_contract" / "alpha_liquid.png"
    assert paths[-1].name == "electric_field_magnitude.png"
    assert len(set(paths)) == len(paths)
