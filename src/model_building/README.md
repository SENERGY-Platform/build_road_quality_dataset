# Model Building Pipeline

This folder contains the pipeline for building road-quality models from labelled manual and OpenStreetMap datasets. 
The pipeline loads or creates feature datasets, builds experiment test cases, splits training and test data, trains configured models, and evaluates them on held-out manual labels.

## Entry Point

Run the pipeline from the repository root:

```bash
python src/model_building/run_model_pipeline.py
```

The entry point is `run_model_pipeline.py`. It currently defines the data paths, experiment settings, selected features, model list, and train/test split percentages directly in code.

## Input Data

The pipeline expects two labelled parquet dataset folders:

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

Labels may be numeric or text:

- `good` maps to `0.0`
- `medium` maps to `1.0`
- `bad` maps to `2.0`


## Feature Preparation

Feature loading is handled by `data/data_loader.py`.

The loader either reads saved feature datasets or builds them from the raw labelled parquet files. Saved features are reused when `skip_feature_build_if_exists=True` and both of these folders contain parquet files:

- `data/molewa/model_building/feature_ds/osm`
- `data/molewa/model_building/feature_ds/manual`

If reusable feature datasets are not found, the loader:

- normalizes `lon` and `lat` to `longitude` and `latitude`
- parses numeric columns and timestamps
- converts text labels to numeric labels
- adds road-damage score features
- optionally writes the processed feature datasets to `feature_ds_dir`

The prepared feature datasets keep the configured input columns and can include these generated score columns:

- `vibration_magnitude`
- `score_mild`
- `score_standard`
- `score_strict`

The score formula removes gravity from vibration magnitude and normalizes by speed with different exponents.

## Experiment Cases

Experiment cases are created by `data/data_test_cases.py` from the loaded manual and OSM feature datasets.

The configured cases are:

- Case A: manual data only
- Case B: manual data plus OSM data
- Case C: OSM data only for training, with manual data still used for testing

Each generated case records the manual dataset id, OSM dataset id when present, dataset version, and parsed dataset parameters. Case B and Case C are generated for every manual/OSM dataset pairing.

## Data Splitting

Splitting is handled by `data/data_split.py`.

The held-out test set always comes from the manual labelled dataset:

- Case A trains on the manual train split and tests on the manual test split.
- Case B trains on the manual train split plus sampled OSM training data, then tests on the manual test split.
- Case C trains on sampled OSM training data, then tests on the manual test split.

Manual data is split with `train_test_split`, using `test_set_percentage` from `ExperimentConfig`. The split is stratified by discrete label category and uses `random_state=42`.

For Case B and Case C, OSM training rows are sampled to match the manual training label distribution. By default, the OSM training sample size matches the manual training size. Set `case_b_all_osm_data=True` or `case_c_all_osm_data=True` to request all available OSM rows for that case type, subject to the available class distribution.

Models that need validation data split validation rows from the training data only. The held-out manual test set is not used for validation.

## Model Training

Model setup and training are handled by `models/model_build.py`.

Supported model names in `ExperimentConfig.models` are:

- `ANN`
- `Linear`
- `XGBoost`

Current training behavior:

- `Linear` uses `sklearn.linear_model.Ridge`.
- `XGBoost` uses `xgboost.XGBClassifier`.
- `ANN` uses `TwoPhaseANNModel`, which is currently a placeholder.

Training labels for `Linear` and `XGBoost` are converted from numeric labels to discrete classes before fitting:

- values `< 0.5` become `0`, good
- values `>= 0.5` and `< 1.5` become `1`, medium
- values `>= 1.5` become `2`, bad

## Model Evaluation

Evaluation is implemented in `models/model_build.py` by `evaluate_model`.

For each trained model, the pipeline calls:

```python
metrics = evaluate_model(trained_model, model_data)
```

The evaluator predicts on `model_data.test_x`, where the test rows come from the held-out manual dataset. It returns:

- `mae`: mean absolute error between `model_data.test_y` and the raw numeric predictions.
- `f1_macro`: macro-average F1 across the three road-quality classes.
- `f1_good`: F1 for class `0`, good.
- `f1_medium`: F1 for class `1`, medium.
- `f1_bad`: F1 for class `2`, bad.

For F1 metrics, both the true test labels and predictions are converted to discrete classes with `label_discrete_from_continuous`:

- values `< 0.5` become `0`, good
- values `>= 0.5` and `< 1.5` become `1`, medium
- values `>= 1.5` become `2`, bad

Class-wise F1 uses `zero_division=0`, so a missing or unpredicted class receives an F1 score of `0` instead of raising a warning.

## Logging

Pipeline logging is configured in `pipeline_logging.py` and writes structured event messages to stdout.

The current logs include:

- `testcase_started`: test-case id plus manual and OSM row counts
- `model_data_used`: train, validation, and test row counts for each model/test-case pair

## Current Outputs

The pipeline can write reusable feature datasets to:

- `data/molewa/model_building/feature_ds/osm`
- `data/molewa/model_building/feature_ds/manual`

Evaluation metrics are currently returned from `evaluate_model`, but `run_model_pipeline.py` does not yet persist or log them.

## Known Limitations

- `TwoPhaseANNModel` is still a placeholder and does not implement `predict`, so evaluation will fail for `ANN` until the model is implemented.
- Model metrics are calculated but not saved to disk.
- Experiment configuration is currently hard-coded in `run_model_pipeline.py`.
