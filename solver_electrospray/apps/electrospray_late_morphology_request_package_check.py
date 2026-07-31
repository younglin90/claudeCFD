#!/usr/bin/env python3
"""Check that the Candido late-morphology request package is usable."""

from __future__ import annotations

import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "electrospray"

VOLUME_COLUMNS = [
    "case",
    "reference_time_ms",
    "digitized_experimental_volume_di3",
    "source",
    "source_type",
    "extraction_method",
    "not_derived_from_reported_error",
]

CONTOUR_COLUMNS = [
    "case",
    "reference_time_ms",
    "point_id",
    "contour_y_di",
    "contour_radius_di",
    "source",
    "source_type",
    "extraction_method",
    "not_derived_from_reported_error",
]


def _header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def main() -> int:
    request = DOCS / "candido_late_morphology_data_request.md"
    volume = DOCS / "candido_late_morphology_external_volume_template.csv"
    contour = DOCS / "candido_late_morphology_external_contour_template.csv"
    schema = DOCS / "candido_late_morphology_external_schema.csv"

    for path in [request, volume, contour, schema]:
        if not path.exists():
            raise AssertionError(f"missing request-package file: {path}")

    text = request.read_text(encoding="utf-8")
    required_phrases = [
        "0.8 ms",
        "0.9 ms",
        "not_derived_from_reported_error",
        "reported relative errors",
        "python3 apps/electrospray_late_morphology_dataset_check.py --require-valid",
    ]
    for phrase in required_phrases:
        if phrase not in text:
            raise AssertionError(f"request package missing phrase: {phrase}")

    if _header(volume) != VOLUME_COLUMNS:
        raise AssertionError("volume template header mismatch")
    if _header(contour) != CONTOUR_COLUMNS:
        raise AssertionError("contour template header mismatch")
    if len(volume.read_text(encoding="utf-8").splitlines()) != 1:
        raise AssertionError("volume template must not contain placeholder data rows")
    if len(contour.read_text(encoding="utf-8").splitlines()) != 1:
        raise AssertionError("contour template must not contain placeholder data rows")

    print("late morphology request package check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
