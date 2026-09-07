# Experiments

This folder contains runnable experiment scripts that configure the model-building pipeline for specific optimisation runs.

## Ridge Hyperparameter Optimisation

Run from the repository root:

```bash
python src/experiments/hp_optimisation_linear.py
```

`hp_optimisation_linear.py` evaluates the `Linear` model, which is implemented as:

- `sklearn.preprocessing.StandardScaler`
- `sklearn.linear_model.Ridge`

The script tests 30 `alpha` values from `1e-5` to `1e5` with `np.logspace(-5, 5, 30)`. Each value is passed through `LinearModelConfig(alpha=...)`, into `ExperimentConfig.linear_model_config`, and then into the inner `Ridge` estimator.

The number of tested parameter sets defaults to `RIDGE_N_PARAMETER_SETS` in `global_config.py` and can be overridden with `run_ridge_optimisation(..., n_parameter_sets=...)`.

The script runs one experiment per case/OSM-size setting: `A` with no OSM data, `B` with all and limited OSM data, and `C` with all and limited OSM data. Each experiment uses five repeated stratified splits and these input features:

- `vibration_x`
- `vibration_y`
- `vibration_z`
- `speed`
- `vibration_magnitude`
- `score_mild`
- `score_standard`
- `score_strict`

During the run, `run_model_optimisation` starts the shared MLflow parent/trial workflow and the model-building pipeline logs averaged cross-validation metrics for each dataset/test-case and alpha. At the end, `log_model_optimisation_summary` logs:

- number of unique dataset/test-cases tested
- number of alpha-by-dataset model runs
- number of alpha values tested
- best MAE, including testcase id, alpha, and full metrics
- best macro-F1, including testcase id, alpha, and full metrics
- mean MAE and mean macro-F1 across all alpha-by-dataset runs

Metrics are logged to stdout and persisted to MLflow.

## XGBoost Hyperparameter Optimisation

Run from the repository root:

```bash
python src/experiments/hp_optimisation_xgboost.py
```

`hp_optimisation_xgboost.py` evaluates the `XGBoost` model with `xgboost.XGBRegressor`. The script defines a discrete search space and samples the configured number of unique parameter combinations without replacement using `np.random.default_rng`.

The number of sampled parameter sets defaults to `XGB_N_PARAMETER_SETS` in `global_config.py` and can be overridden with `run_xgb_optimisation(..., n_parameter_sets=...)`.

The search space is:

- `n_estimators`: `100`, `200`, `400`, `800`
- `learning_rate`: `0.03`, `0.05`, `0.1`, `0.2`
- `max_depth`: `3`, `4`, `5`, `6`, `8`
- `min_child_weight`: `1`, `3`, `5`, `10`, `20`
- `subsample`: `0.7`, `0.85`, `1.0`
- `colsample_bytree`: `0.7`, `0.85`, `1.0`
- `reg_lambda`: `0.5`, `1.0`, `5.0`, `10.0`

Each sampled combination is passed through `XGBoostModelConfig(...)`, into `ExperimentConfig.xgb_model_config`, and then into `XGBRegressor`.

The configured experiments use the same case/OSM-size settings, cross-validation setup, and feature list as the ridge optimisation. During the run, `run_model_optimisation` starts the shared MLflow parent/trial workflow and the model-building pipeline logs averaged cross-validation metrics for each dataset/test-case and parameter set. At the end, `log_model_optimisation_summary` logs:

- number of unique dataset/test-cases tested
- number of parameter-set-by-dataset model runs
- number of parameter sets tested
- best MAE, including testcase id, parameter set id, parameter values, and full metrics
- best macro-F1, including testcase id, parameter set id, parameter values, and full metrics
- mean MAE and mean macro-F1 across all parameter-set-by-dataset runs
