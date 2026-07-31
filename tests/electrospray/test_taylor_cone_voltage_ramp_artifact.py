import math
from pathlib import Path

import numpy as np

from cone_geometry import taylor_cone_voltage_ramp_balance


def test_taylor_cone_voltage_ramp_artifact_matches_executable_case() -> None:
    artifact = Path("docs/electrospray/taylor_cone_voltage_ramp_balance_table.md")
    text = artifact.read_text(encoding="utf-8")

    points = taylor_cone_voltage_ramp_balance(
        radius=0.2,
        half_angle=math.radians(49.292),
        surface_tension=0.072,
        permittivity=2.0,
        extractor_gap=0.25,
        voltage_fractions=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
    )

    residuals = [point.normalized_balance_residual for point in points]
    fields = [point.normal_electric_field for point in points]
    assert np.all(np.diff(fields) > 0.0)
    assert np.all(np.diff(residuals) < 0.0)

    for point in points:
        status = "PASS" if point.normalized_balance_residual <= 1.0e-12 else "RAMP"
        assert f"{point.voltage_fraction:.2f}" in text
        assert f"{point.voltage:.6e}" in text
        assert f"{point.normal_electric_field:.6e}" in text
        assert f"{point.maxwell_pressure:.6e}" in text
        assert f"{point.capillary_pressure:.6e}" in text
        assert f"{point.normalized_balance_residual:.6e}" in text
        assert status in text

    assert "final normalized balance residual below 1e-12" in text
