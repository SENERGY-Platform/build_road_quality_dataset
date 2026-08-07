# Model Building Pipeline

This folder contains the first model-building orchestration step: loading labelled road-quality parquet datasets, preparing a consistent feature table, and optionally saving reusable feature datasets.

## Entry Point

Run the pipeline from the repository root:

```bash
python src/model_building/run_model_pipeline.py
```

The entry point builds a `DataConfig` and calls `data_loader.load_feature_ds`.

## Inputs

The pipeline expects two labelled dataset folders:

- `data/open_street_map/datasets`
- `data/molewa/datasets`

Each input parquet file must include:

- `timestamp`
- `label`
- `vibration_x`
- `vibration_y`
- `vibration_z`
- `speed`
- `longitude` and `latitude`, or `lon` and `lat`

Labels may be numeric or one of:

- `good`
- `medium`
- `bad`

## Outputs

When `feature_ds_dir` is configured, processed feature datasets are saved under:

- `data/molewa/model_building/feature_ds/osm`
- `data/molewa/model_building/feature_ds/manual`

Saved feature files are reused when `skip_feature_build_if_exists=True` and both output folders contain parquet files.

## Feature Columns

The prepared output keeps the required numeric input columns and adds three road damage score columns:

- `score_mild`
- `score_standard`
- `score_strict`

The score formula removes gravity from vibration magnitude and normalizes by speed with different exponents.
