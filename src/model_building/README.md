# Model Building Pipeline

This folder contains the pipeline for building road-quality models from labelled manual and OpenStreetMap datasets.
The pipeline loads or creates feature datasets, builds experiment test cases, splits training and test data, trains configured models, and evaluates them on held-out manual labels.

## Entry Point

Run the pipeline from the repository root:

```bash
python src/model_building/model_evaluation_pipeline.py
```

The entry point is `model_evaluation_pipeline.py`. Its `example()` function defines a `DataConfig` and an `ExperimentConfig`, then calls `run_experiment`.

You can define experiments in another file by importing `run_experiment` and passing a `DataConfig` plus an `ExperimentConfig`:

```python
from src.model_building.data.data_loader import DataConfig
from src.model_building.model_evaluation_pipeline import run_experiment
from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.config.model_config import XGBoostModelConfig

data_config = DataConfig(
    osm_ds_dir="data/open_street_map/datasets",
    manual_ds_dir="data/molewa/datasets",
    feature_ds_dir="data/molewa/model_building/feature_ds",
    skip_feature_build_if_exists=True,
)

experiment_config = ExperimentConfig(
    experiment_name="my_experiment",
    case_type="B",
    all_osm_data=False,
    cross_validation_k=10,
    ds_version="v1.0",
    feature_set_name="raw_features",
    features=["vibration_x", "vibration_y", "vibration_z", "speed"],
    model="XGBoost",
    test_set_percentage=0.3,
    xgb_model_config=XGBoostModelConfig(),
)

run_experiment(data_config, experiment_config)
```

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

Each `ExperimentConfig` runs one configured case:

- Case A: manual data only
- Case B: manual data plus OSM data
- Case C: OSM data only for training, with manual data still used for testing

Each generated dataset combination records the manual dataset id, OSM dataset id when present, dataset version, and parsed dataset parameters. Case B and Case C are generated for every manual/OSM dataset pairing within the selected case.

## Data Splitting

Splitting is handled by `data/data_split.py`.

The held-out test set always comes from the manual labelled dataset:

- Case A trains on the manual train split and tests on the manual test split.
- Case B trains on the manual train split plus sampled OSM training data, then tests on the manual test split.
- Case C trains on sampled OSM training data, then tests on the manual test split.

Manual data is split repeatedly with `train_test_split`, using `test_set_percentage` from `ExperimentConfig`. Each split is stratified by discrete label category and uses a different `random_state`.

For Case B and Case C, OSM training rows are sampled to match the manual training label distribution. By default, the OSM training sample size matches the manual training size. Set `all_osm_data=True` to request all available OSM rows for the configured case type, subject to the available class distribution. For Case A, set `all_osm_data=None`.

Models that need validation data split validation rows from the training data only. The held-out manual test set is not used for validation.

## Model Training

Model setup and training are handled by `models/model_build.py`.

Supported model names in `ExperimentConfig.model` are:

- `ANN`
- `Linear`
- `XGBoost`

Current training behavior:

- `Linear` uses `sklearn.preprocessing.StandardScaler` followed by `sklearn.linear_model.Ridge`; set its regularization strength with `LinearModelConfig(alpha=...)`.
- `XGBoost` uses `xgboost.XGBRegressor`.
- `ANN` uses `TwoPhaseANNModel`.

Evaluation labels and predictions are converted from numeric labels to discrete classes for F1 scoring:

- values `< 0.5` become `0`, good
- values `>= 0.5` and `< 1.5` become `1`, medium
- values `>= 1.5` become `2`, bad

## Model Evaluation

Evaluation is implemented in `models/model_build.py` by `evaluate_model`.
The evaluation pipeline uses `crossvalidate_model` to train and evaluate one configured model across repeated stratified splits for one test case.

For each split, the pipeline creates a fresh model with `setup_model`, prepares validation data when needed with `update_model_data_for_validation`, and then calls:

```python
trained_model, performance = evaluate_model(model, model_data)
```

`evaluate_model` now owns the fit-and-score step: it trains the supplied model with `train_model`, measures training time, predicts on `model_data.test_x`, measures inference time, and returns both the fitted model and its `ModelPerformance`. The test rows come from the held-out manual dataset.

`ModelPerformance` contains:

- `mae`: mean absolute error between `model_data.test_y` and the raw numeric predictions.
- `f1_macro`: macro-average F1 across the three road-quality classes.
- `f1_good`: F1 for class `0`, good.
- `f1_medium`: F1 for class `1`, medium.
- `f1_bad`: F1 for class `2`, bad.
- `train_time_s`: wall-clock seconds spent fitting one model split, measured around `train_model` inside `evaluate_model`.
- `inference_time_ms_per_sample`: wall-clock prediction time per test sample, measured around `model.predict` in `evaluate_model`.
- `inference_n_samples`: number of test samples used for the inference timing measurement.

For F1 metrics, both the true test labels and predictions are converted to discrete classes with `label_discrete_from_continuous`:

- values `< 0.5` become `0`, good
- values `>= 0.5` and `< 1.5` become `1`, medium
- values `>= 1.5` become `2`, bad

Class-wise F1 uses `zero_division=0`, so a missing or unpredicted class receives an F1 score of `0` instead of raising a warning.

`CrossValidationPerformance.get_final_performance()` returns two objects: the mean `ModelPerformance` and a `ModelPerformanceStd` with standard deviations across repeated splits. `inference_time_ms_per_sample` is averaged with `inference_n_samples` as weights, and the final `inference_n_samples` is the total number of timed predictions across all splits.

Cross-validation also caches the median split's model, dataset, and performance on `median_model`, `median_data_set`, and `median_performance`. The median split is selected by macro F1 first and MAE second, while keeping all three cached objects from the same split index.

## Logging

Pipeline logging is configured in `pipeline_logging.py` and writes structured event messages to stdout.

The current logs include (if not commented out):

- `testcase_started`: test-case id plus manual and OSM row counts
- `model_data_used`: train, validation, and test row counts for each model/test-case pair
- `model_evaluated`: per-split evaluation metrics for each model/test-case pair
- `cross_val_performance`: averaged evaluation metrics plus metric standard deviations across all repeated splits for each model/test-case pair

`logging/mlflow_logging.py` contains the MLflow integration. It can create a parent run for one model/dataset-case group and nested trial runs for hyperparameter configurations. Trial runs currently log dataset parameters, feature configuration, model parameters, source-aware train/test/validation sizes, mean metrics, and metric standard deviations. Parent runs record the selected best trial's run id, trial name, mean metrics, and model parameters.

## Current Outputs

The pipeline can write reusable feature datasets to:

- `data/molewa/model_building/feature_ds/osm`
- `data/molewa/model_building/feature_ds/manual`

Evaluation metrics are returned from `evaluate_model` and aggregated by `model_evaluation_pipeline.py`. When an `MlflowLogger` is supplied, trial-level parameters, source-aware dataset sizes, mean metrics, and metric standard deviations are persisted to MLflow.

## Known Limitations

- Best-model artifact logging is not implemented yet.
- The default runnable example is currently defined in `model_evaluation_pipeline.py`.
