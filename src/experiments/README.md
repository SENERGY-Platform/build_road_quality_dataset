# Experiments

This folder contains runnable experiment scripts that configure the model-building pipeline for specific optimisation runs.

## Ridge Hyperparameter Optimisation

Run from the repository root:

```bash
python src/experiments/hp_optimisation_linear.py
```

`linear_hp_optimisation.py` evaluates the `Linear` model, which is implemented as:

- `sklearn.preprocessing.StandardScaler`
- `sklearn.linear_model.Ridge`

The script tests 30 `alpha` values from `1e-5` to `1e5` with `np.logspace(-5, 5, 30)`. Each value is passed through `LinearModelConfig(alpha=...)`, into `ExperimentConfig.linear_model_config`, and then into the inner `Ridge` estimator.

The configured experiment uses cases `A`, `B`, and `C`, five repeated stratified splits, and these input features:

- `vibration_x`
- `vibration_y`
- `vibration_z`
- `speed`
- `vibration_magnitude`
- `score_mild`
- `score_standard`
- `score_strict`

During the run, the shared model-building pipeline logs averaged cross-validation metrics for each dataset/test-case and alpha. At the end, `log_ridge_optimisation_summary` logs:

- number of unique dataset/test-cases tested
- number of alpha-by-dataset model runs
- number of alpha values tested
- best MAE, including testcase id, alpha, and full metrics
- best macro-F1, including testcase id, alpha, and full metrics
- mean MAE and mean macro-F1 across all alpha-by-dataset runs

Metrics are logged to stdout but are not persisted to disk.

## XGBoost Hyperparameter Optimisation

Run from the repository root:

```bash
python src/experiments/hp_optimisation_xgboost.py
```

`xgb_hp_optimisation.py` evaluates the `XGBoost` model with `xgboost.XGBRegressor`. The script defines a discrete search space and samples 30 unique parameter combinations without replacement using `np.random.default_rng`.

The search space is:

- `n_estimators`: `100`, `200`, `400`, `800`
- `learning_rate`: `0.03`, `0.05`, `0.1`, `0.2`
- `max_depth`: `3`, `4`, `5`, `6`, `8`
- `min_child_weight`: `1`, `3`, `5`, `10`, `20`
- `subsample`: `0.7`, `0.85`, `1.0`
- `colsample_bytree`: `0.7`, `0.85`, `1.0`
- `reg_lambda`: `0.5`, `1.0`, `5.0`, `10.0`

Each sampled combination is passed through `XGBoostModelConfig(...)`, into `ExperimentConfig.xgb_model_config`, and then into `XGBRegressor`.

The configured experiment uses the same cases, cross-validation setup, and feature list as the ridge optimisation. During the run, the shared model-building pipeline logs averaged cross-validation metrics for each dataset/test-case and parameter set. At the end, `log_xgb_optimisation_summary` logs:

- number of unique dataset/test-cases tested
- number of parameter-set-by-dataset model runs
- number of parameter sets tested
- best MAE, including testcase id, parameter set id, parameter values, and full metrics
- best macro-F1, including testcase id, parameter set id, parameter values, and full metrics
- mean MAE and mean macro-F1 across all parameter-set-by-dataset runs
