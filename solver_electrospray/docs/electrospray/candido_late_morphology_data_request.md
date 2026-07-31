# Candido Late Morphology Data Request Package

This package defines the exact external data needed to close the remaining
Candido Fig. 3(b) late-morphology validation gate without back-solving from the
paper's reported relative errors.

## Requested Data

Target case: `long_window_ca025`.

Required late times:

- `0.8 ms`
- `0.9 ms`

Accepted input mode A, direct volume:

- Fill `docs/electrospray/candido_late_morphology_external_volume_template.csv`.
- Provide `digitized_experimental_volume_di3`, the experimental morphology
  volume normalized by inlet-diameter cubed.

Accepted input mode B, free-surface contour:

- Fill `docs/electrospray/candido_late_morphology_external_contour_template.csv`.
- Provide at least three `(contour_y_di, contour_radius_di)` points per required
  time, normalized by inlet diameter.
- The validator computes `V/Di^3 = pi * integral(contour_radius_di^2 d contour_y_di)`.

Hard provenance rule:

- `not_derived_from_reported_error` must be `1`.
- Values inferred from the published relative-error row are not acceptable.
- The source should be a digitized experimental/numerical contour, original
  data from the corresponding author, or equivalent independent geometry.

## Validation Command

Copy the completed external CSV to:

```bash
docs/electrospray/candido_late_morphology_external_dataset.csv
```

Then run:

```bash
python3 apps/electrospray_late_morphology_dataset_check.py --require-valid
```

The validator writes:

```bash
build/benchmark_logs/candido_late_morphology_external_dataset_check3d.csv
```

The paper-level morphology gate remains blocked until the validator reports
`VALID_EXTERNAL_LATE_MORPHOLOGY_DATASET` and the benchmark comparison is rerun
against that independent data.

## Request Text

Subject: Request for Candido Fig. 3(b) late morphology reference data

Dear corresponding author,

I am independently validating a 3D EHD Taylor cone-jet solver against the
published Candido and Pascoa Phys. Fluids 35, 052110 (2023) case. The public
article and figure assets expose the 0.0, 0.4, and 0.7 ms Fig. 3(b) panels, and
the paper reports relative morphology errors at 0.8 and 0.9 ms, but I could not
find the underlying late-time contour coordinates or experimental morphology
volumes.

Could you provide either the Fig. 3(b) experimental/numerical morphology volumes
at 0.8 and 0.9 ms, normalized by inlet-diameter cubed, or the corresponding
free-surface contour coordinates normalized by inlet diameter?

I will use the data only as an external validation reference and will keep the
published relative-error row separate from the reference extraction.

Thank you.
