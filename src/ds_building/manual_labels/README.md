# Manual Labels Mini-Pipeline

This pipeline builds vibration-to-road-quality datasets by joining manually labeled
road-quality points with nearby street sensor measurements.

## Inputs

- Manual labels: `data/street_data/labels/molewa_labels.csv`
- Street measurements: `data/street_data/raw/molewa_street - bearbeitet.csv`

Both files are loaded with `utils.load_data`, so they must include a `timestamp`
column parseable by pandas using ISO8601 format. The matching steps also expect
`lat`, `lon`, `vehicleType`, `speed`, `vibration_x`, `vibration_y`, and
`vibration_z` columns where relevant.

## Pipeline Modes

The entrypoint is `run_manual_ds_build.py`. It supports two mapping directions:

- `labels_first`: for each manual label, find nearby street measurements and use
  the manual label as the dataset label.
- `street_first`: for each street measurement, find nearby manual labels and use
  the most frequent nearby manual label.

The mode is controlled by `ManualLabelsConfig.mapping_type` in
`run_manual_ds_build.py`.

## Key Settings

`run_manual_ds_build.py` exposes the pipeline settings through
`ManualLabelsConfig`:

- `labels_path`: manual labels input CSV.
- `street_path`: street measurements input CSV.
- `output_dir`: root output directory.
- `mapping_type`: `labels_first` or `street_first`.
- `mapping_procedure`: `single` or `average` for `labels_first`;
  `most_frequent` for `street_first`.
- `vehicle_type`: vehicle type to include in the final dataset.
- `lon_threshold` and `lat_threshold`: coarse coordinate filters before distance
  calculation.
- `speed_threshold`: minimum street-measurement speed.
- `radius`: maximum geodesic distance in metres for candidate matches.
- `time_threshold`: labels-first only; keeps street measurements within this many
  days of the latest matched timestamp for a label.

## Running

Run from the repository root:

```bash
python src/ds_building/manual_labels/run_manual_ds_build.py
```

The script currently iterates over the configured mapping combinations:

- labels input: `data/street_data/labels/molewa_labels.csv`
- street input: `data/street_data/raw/molewa_street - bearbeitet.csv`
- output directory: `data/street_data/datasets`
- `labels_first` with `single`
- `labels_first` with `average`
- `street_first` with `most_frequent`
- vehicle type: `Car`

To change a run, edit the config object created in `run_manual_ds_build.py`:

```python
config = ManualLabelsConfig(
    mapping_type="street_first",
    mapping_procedure="most_frequent",
)
```

Supported mapping values:

- `mapping_type="labels_first"` supports `mapping_procedure="single"` and
  `mapping_procedure="average"`.
- `mapping_type="street_first"` supports `mapping_procedure="most_frequent"`.

## Outputs

The script writes both pickle and CSV outputs under:

- `data/street_data/datasets/labels_first/`
- `data/street_data/datasets/street_first/`

Output filenames encode the selected radius, mapping procedure, time threshold
where applicable, and vehicle type.

Each output row uses the same flat schema across all supported mapping
procedures:

- `vibration_x`, `vibration_y`, `vibration_z`: sensor vibration values.
- `speed`: street-measurement speed.
- `label`: manual road-quality label.
- `lon`, `lat`: street-measurement location.
- `timestamp`: street-measurement timestamp.

For `labels_first` with `single`, each matching street measurement becomes one
output row, so `speed`, `lon`, `lat`, and `timestamp` are copied directly from
that street row.

For `labels_first` with `average`, each manual label produces one output row.
The vibration axes, speed, longitude, and latitude are arithmetic means of the
matched street rows. The timestamp is the mean timestamp of the matched street
rows.

For `street_first` with `most_frequent`, each matching street measurement
becomes one output row. Sensor values, speed, location, and timestamp are copied
from that street row, while `label` is the most frequent nearby manual label.

## Files

- `run_manual_ds_build.py`: configures paths, thresholds, mode selection, and
  output writing.
- `labels_first.py`: implements manual-label-first matching and dataset creation.
- `street_first.py`: implements street-measurement-first matching and dataset
  creation.
