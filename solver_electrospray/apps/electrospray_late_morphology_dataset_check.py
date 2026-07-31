#!/usr/bin/env python3
"""Validate an external Candido late-morphology reference dataset.

This checker intentionally does not make the paper-validation metric pass.
It only defines the minimum machine-readable input needed before the remaining
0.8/0.9 ms Candido morphology gate can be closed without inferring reference
volumes from the paper's reported relative-error row.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


VOLUME_REQUIRED_COLUMNS = {
    "case",
    "reference_time_ms",
    "digitized_experimental_volume_di3",
    "source",
    "source_type",
    "extraction_method",
    "not_derived_from_reported_error",
}
CONTOUR_REQUIRED_COLUMNS = {
    "case",
    "reference_time_ms",
    "contour_y_di",
    "contour_radius_di",
    "source",
    "source_type",
    "extraction_method",
    "not_derived_from_reported_error",
}
REQUIRED_COLUMNS = VOLUME_REQUIRED_COLUMNS
REQUIRED_CASE = "long_window_ca025"
REQUIRED_TIMES_MS = (0.8, 0.9)


@dataclass
class CheckResult:
    ok: bool
    status: str
    messages: list[str]
    rows_by_time: dict[float, dict[str, str]]
    input_mode: str = "unknown"


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _has_columns(columns: list[str], required: set[str]) -> bool:
    return required.issubset(set(columns))


def _reject_reported_error(row: dict[str, str], time_ms: float, messages: list[str]) -> None:
    joined = " ".join(
        [
            row.get("source_type", ""),
            row.get("extraction_method", ""),
            row.get("source", ""),
        ]
    ).lower()
    if "reported_error" in joined or "relative_error" in joined:
        messages.append(f"{time_ms} ms row appears derived from reported error, not geometry")
    if not _truthy(row.get("not_derived_from_reported_error", "")):
        messages.append(f"{time_ms} ms not_derived_from_reported_error must be true")


def _validate_metadata(row: dict[str, str], time_ms: float, messages: list[str]) -> None:
    if not row.get("source", "").strip():
        messages.append(f"{time_ms} ms source is empty")
    if not row.get("source_type", "").strip():
        messages.append(f"{time_ms} ms source_type is empty")
    if not row.get("extraction_method", "").strip():
        messages.append(f"{time_ms} ms extraction_method is empty")
    _reject_reported_error(row, time_ms, messages)


def _integrate_axisymmetric_volume_di3(points: list[tuple[float, float]]) -> float:
    """Return pi * integral r^2 dy for y/r coordinates normalized by Di."""
    ordered = sorted(points)
    if len(ordered) < 3:
        return math.nan
    volume = 0.0
    for (y0, r0), (y1, r1) in zip(ordered, ordered[1:]):
        dy = y1 - y0
        if dy <= 0.0:
            return math.nan
        volume += 0.5 * math.pi * (r0 * r0 + r1 * r1) * dy
    return volume


def _validate_volume_rows(
    rows: list[dict[str, str]], columns: list[str]
) -> CheckResult:
    missing_columns = sorted(VOLUME_REQUIRED_COLUMNS.difference(columns))
    if missing_columns:
        return CheckResult(
            ok=False,
            status="INVALID_EXTERNAL_DATASET_SCHEMA",
            messages=["missing volume columns: " + ";".join(missing_columns)],
            rows_by_time={},
            input_mode="volume",
        )

    messages: list[str] = []
    rows_by_time: dict[float, dict[str, str]] = {}
    for required_time in REQUIRED_TIMES_MS:
        matching = [
            row
            for row in rows
            if row.get("case") == REQUIRED_CASE
            and abs(_as_float(row.get("reference_time_ms", "")) - required_time) < 1e-9
        ]
        if not matching:
            messages.append(f"missing {REQUIRED_CASE} row at {required_time} ms")
            continue
        row = dict(matching[0])
        row["input_mode"] = "volume"
        row["contour_point_count"] = ""
        rows_by_time[required_time] = row
        volume = _as_float(row.get("digitized_experimental_volume_di3", ""))
        if not math.isfinite(volume) or volume <= 0.0:
            messages.append(f"{required_time} ms volume is not finite positive")
        _validate_metadata(row, required_time, messages)

    if messages:
        return CheckResult(
            ok=False,
            status="INVALID_EXTERNAL_LATE_MORPHOLOGY_DATASET",
            messages=messages,
            rows_by_time=rows_by_time,
            input_mode="volume",
        )
    return CheckResult(
        ok=True,
        status="VALID_EXTERNAL_LATE_MORPHOLOGY_DATASET",
        messages=["0.8 and 0.9 ms external late morphology references are present"],
        rows_by_time=rows_by_time,
        input_mode="volume",
    )


def _validate_contour_rows(
    rows: list[dict[str, str]], columns: list[str]
) -> CheckResult:
    missing_columns = sorted(CONTOUR_REQUIRED_COLUMNS.difference(columns))
    if missing_columns:
        return CheckResult(
            ok=False,
            status="INVALID_EXTERNAL_DATASET_SCHEMA",
            messages=["missing contour columns: " + ";".join(missing_columns)],
            rows_by_time={},
            input_mode="contour",
        )

    messages: list[str] = []
    rows_by_time: dict[float, dict[str, str]] = {}
    for required_time in REQUIRED_TIMES_MS:
        matching = [
            row
            for row in rows
            if row.get("case") == REQUIRED_CASE
            and abs(_as_float(row.get("reference_time_ms", "")) - required_time) < 1e-9
        ]
        if len(matching) < 3:
            messages.append(
                f"{required_time} ms contour needs at least three coordinate rows"
            )
            continue
        first = matching[0]
        _validate_metadata(first, required_time, messages)
        points: list[tuple[float, float]] = []
        for row in matching:
            y_di = _as_float(row.get("contour_y_di", ""))
            r_di = _as_float(row.get("contour_radius_di", ""))
            if not math.isfinite(y_di) or not math.isfinite(r_di) or r_di < 0.0:
                messages.append(f"{required_time} ms contour has invalid y/r coordinate")
                continue
            points.append((y_di, r_di))
        volume = _integrate_axisymmetric_volume_di3(points)
        if not math.isfinite(volume) or volume <= 0.0:
            messages.append(f"{required_time} ms integrated contour volume is invalid")
            continue
        rows_by_time[required_time] = {
            "case": REQUIRED_CASE,
            "reference_time_ms": str(required_time),
            "digitized_experimental_volume_di3": f"{volume:.17g}",
            "source": first.get("source", ""),
            "source_type": first.get("source_type", ""),
            "extraction_method": first.get("extraction_method", ""),
            "not_derived_from_reported_error": first.get(
                "not_derived_from_reported_error", ""
            ),
            "input_mode": "contour",
            "contour_point_count": str(len(points)),
        }

    if messages:
        return CheckResult(
            ok=False,
            status="INVALID_EXTERNAL_LATE_MORPHOLOGY_DATASET",
            messages=messages,
            rows_by_time=rows_by_time,
            input_mode="contour",
        )
    return CheckResult(
        ok=True,
        status="VALID_EXTERNAL_LATE_MORPHOLOGY_DATASET",
        messages=[
            "0.8 and 0.9 ms external contour references are present and integrated"
        ],
        rows_by_time=rows_by_time,
        input_mode="contour",
    )


def validate_dataset(path: Path) -> CheckResult:
    rows, columns = _read_rows(path)
    if not path.exists():
        return CheckResult(
            ok=False,
            status="BLOCKED_EXTERNAL_DATASET_MISSING",
            messages=[f"missing input CSV: {path}"],
            rows_by_time={},
            input_mode="missing",
        )
    if _has_columns(columns, VOLUME_REQUIRED_COLUMNS):
        return _validate_volume_rows(rows, columns)
    if _has_columns(columns, CONTOUR_REQUIRED_COLUMNS):
        return _validate_contour_rows(rows, columns)
    volume_missing = sorted(VOLUME_REQUIRED_COLUMNS.difference(columns))
    contour_missing = sorted(CONTOUR_REQUIRED_COLUMNS.difference(columns))
    if volume_missing or contour_missing:
        return CheckResult(
            ok=False,
            status="INVALID_EXTERNAL_DATASET_SCHEMA",
            messages=[
                "missing volume columns: " + ";".join(volume_missing),
                "missing contour columns: " + ";".join(contour_missing),
            ],
            rows_by_time={},
            input_mode="unknown",
        )
    return CheckResult(
        ok=False,
        status="INVALID_EXTERNAL_DATASET_SCHEMA",
        messages=["unrecognized external dataset schema"],
        rows_by_time={},
        input_mode="unknown",
    )


def write_report(path: Path, input_path: Path, result: CheckResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "required_time_ms",
                "input_path",
                "observed_reference_time_ms",
                "digitized_experimental_volume_di3",
                "input_mode",
                "contour_point_count",
                "source",
                "source_type",
                "extraction_method",
                "not_derived_from_reported_error",
                "status",
                "message",
            ],
        )
        writer.writeheader()
        for required_time in REQUIRED_TIMES_MS:
            row = result.rows_by_time.get(required_time, {})
            writer.writerow(
                {
                    "case": REQUIRED_CASE,
                    "required_time_ms": required_time,
                    "input_path": str(input_path),
                    "observed_reference_time_ms": row.get("reference_time_ms", ""),
                    "digitized_experimental_volume_di3": row.get(
                        "digitized_experimental_volume_di3", ""
                    ),
                    "input_mode": row.get("input_mode", result.input_mode),
                    "contour_point_count": row.get("contour_point_count", ""),
                    "source": row.get("source", ""),
                    "source_type": row.get("source_type", ""),
                    "extraction_method": row.get("extraction_method", ""),
                    "not_derived_from_reported_error": row.get(
                        "not_derived_from_reported_error", ""
                    ),
                    "status": result.status,
                    "message": "; ".join(result.messages),
                }
            )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _write_contour_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        missing = validate_dataset(root / "missing.csv")
        assert not missing.ok
        assert missing.status == "BLOCKED_EXTERNAL_DATASET_MISSING"

        invalid_schema = root / "invalid_schema.csv"
        invalid_schema.write_text("case,reference_time_ms\nx,0.8\n", encoding="utf-8")
        result = validate_dataset(invalid_schema)
        assert not result.ok
        assert result.status == "INVALID_EXTERNAL_DATASET_SCHEMA"

        derived = root / "derived.csv"
        _write_csv(
            derived,
            [
                {
                    "case": REQUIRED_CASE,
                    "reference_time_ms": str(t),
                    "digitized_experimental_volume_di3": "1.0",
                    "source": "candido reported_error row",
                    "source_type": "reported_error_backsolve",
                    "extraction_method": "relative_error inversion",
                    "not_derived_from_reported_error": "0",
                }
                for t in REQUIRED_TIMES_MS
            ],
        )
        result = validate_dataset(derived)
        assert not result.ok
        assert result.status == "INVALID_EXTERNAL_LATE_MORPHOLOGY_DATASET"

        valid = root / "valid.csv"
        _write_csv(
            valid,
            [
                {
                    "case": REQUIRED_CASE,
                    "reference_time_ms": str(t),
                    "digitized_experimental_volume_di3": str(1.0 + t),
                    "source": "external digitized Candido Fig3b contour",
                    "source_type": "external_digitized_contour",
                    "extraction_method": "manual contour digitization",
                    "not_derived_from_reported_error": "1",
                }
                for t in REQUIRED_TIMES_MS
            ],
        )
        result = validate_dataset(valid)
        assert result.ok
        assert result.status == "VALID_EXTERNAL_LATE_MORPHOLOGY_DATASET"
        report = root / "report.csv"
        write_report(report, valid, result)
        report_rows, _ = _read_rows(report)
        assert len(report_rows) == 2

        contour = root / "contour.csv"
        contour_rows: list[dict[str, str]] = []
        for t in REQUIRED_TIMES_MS:
            for point_id, (y_di, r_di) in enumerate(
                [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]
            ):
                contour_rows.append(
                    {
                        "case": REQUIRED_CASE,
                        "reference_time_ms": str(t),
                        "point_id": str(point_id),
                        "contour_y_di": str(y_di),
                        "contour_radius_di": str(r_di),
                        "source": "external digitized Candido Fig3b contour",
                        "source_type": "external_digitized_contour",
                        "extraction_method": "manual contour digitization",
                        "not_derived_from_reported_error": "1",
                    }
                )
        _write_contour_csv(contour, contour_rows)
        result = validate_dataset(contour)
        assert result.ok
        assert result.input_mode == "contour"
        for row in result.rows_by_time.values():
            volume = _as_float(row["digitized_experimental_volume_di3"])
            assert abs(volume - 0.125 * math.pi) < 1e-12
            assert row["contour_point_count"] == "3"
    print("late morphology dataset checker self-test passed")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input",
        type=Path,
        default=repo / "docs" / "electrospray" / "candido_late_morphology_external_dataset.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo
        / "build"
        / "benchmark_logs"
        / "candido_late_morphology_external_dataset_check3d.csv",
    )
    parser.add_argument("--require-valid", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    result = validate_dataset(args.input)
    write_report(args.report, args.input, result)
    print(result.status)
    for message in result.messages:
        print(message)
    return 0 if result.ok or not args.require_valid else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
