from __future__ import annotations

from dataclasses import asdict, dataclass
import logging

import numpy as np

from src.experiments.experiment_config_defaults import (
    CROSS_VALIDATION_K,
    DS_VERSION,
    EXPERIMENT_NAME,
    FEATURE_DS_DIR,
    FEATURES,
    FEATURE_SET_NAME,
    MANUAL_DS_DIR,
    OSM_DS_DIR,
    SKIP_FEATURE_BUILD_IF_EXISTS,
    TEST_SET_PERCENTAGE,
)
from src.experiments.hp_optimisation.result import OptimisationResult, ParameterSet
from src.model_building.config.experiment_config import ExperimentConfig, ModelConfig
from src.model_building.data.data_loader import DataConfig
from src.model_building.logging.pipeline_logging import configure_pipeline_logging
from src.model_building.model_evaluation_pipeline import run_experiment


@dataclass(frozen=True)
class ModelParameterRun:
    """Model-specific hyperparameter config for one optimisation run."""

    parameter_set_id: int
    parameters: ParameterSet
    model_config: ModelConfig


def setup_data_config(
    osm_ds_dir: str = OSM_DS_DIR,
    manual_ds_dir: str = MANUAL_DS_DIR,
    feature_ds_dir: str = FEATURE_DS_DIR,
    skip_feature_build_if_exists: bool = SKIP_FEATURE_BUILD_IF_EXISTS,
) -> DataConfig:
    """Return the shared data-loading config for model optimisation runs."""
    return DataConfig(
        osm_ds_dir=osm_ds_dir,
        manual_ds_dir=manual_ds_dir,
        feature_ds_dir=feature_ds_dir,
        skip_feature_build_if_exists=skip_feature_build_if_exists,
    )


def setup_experiment_config(
    model_name: str,
    model_config: ModelConfig,
    test_case: str,
    all_osm: bool | None,
) -> ExperimentConfig:
    """Return the shared experiment config with the selected model config attached."""
    return ExperimentConfig(
        experiment_name=EXPERIMENT_NAME,
        case_type=test_case,
        all_osm_data=all_osm,
        cross_validation_k=CROSS_VALIDATION_K,
        ds_version=DS_VERSION,
        feature_set_name=FEATURE_SET_NAME,
        features=FEATURES,
        model=model_name,
        test_set_percentage=TEST_SET_PERCENTAGE,
    ).set_model_config(model_config)


def _log_parameter_selected(logger: logging.Logger, model_name: str, parameter_run: ModelParameterRun) -> None:
    """Log the selected parameter set using the established model-specific event names."""
    event_name_by_model = {
        "ANN": "ann_parameter_selected",
        "Linear": "ridge_parameter_selected",
        "XGBoost": "xgb_parameter_selected",
    }
    event_name = event_name_by_model.get(model_name, "model_parameter_selected")
    logger.info(
        "event=%s parameter_set_id=%s parameters=%s",
        event_name,
        parameter_run.parameter_set_id,
        parameter_run.parameters,
    )


def log_model_optimisation_summary(
    logger: logging.Logger,
    model_name: str,
    results: list[OptimisationResult],
) -> None:
    """Log aggregate performance stats for one model optimisation run."""
    event_name_by_model = {
        "ANN": "ann_optimisation_summary",
        "Linear": "ridge_optimisation_summary",
        "XGBoost": "xgb_optimisation_summary",
    }
    event_name = event_name_by_model.get(model_name, "model_optimisation_summary")

    if not results:
        logger.info("event=%s datasets_tested=0 model_runs_tested=0", event_name)
        return

    best_mae_result = min(results, key=lambda result: result.performance.mae)
    best_f1_result = max(results, key=lambda result: result.performance.f1_macro)
    mae_scores = [result.performance.mae for result in results]
    f1_scores = [result.performance.f1_macro for result in results]
    parameter_set_ids = {result.parameter_set_id for result in results}
    testcase_ids = {result.testcase_id for result in results}

    logger.info(
        (
            "event=%s model=%s "
            "datasets_tested=%s model_runs_tested=%s parameter_sets_tested=%s "
            "best_mae=%s best_mae_testcase_id=%s best_mae_parameter_set_id=%s "
            "best_mae_parameters=%s best_mae_metrics=%s "
            "best_f1_macro=%s best_f1_testcase_id=%s best_f1_parameter_set_id=%s "
            "best_f1_parameters=%s best_f1_metrics=%s "
            "mean_mae=%s mean_f1_macro=%s"
        ),
        event_name,
        model_name,
        len(testcase_ids),
        len(results),
        len(parameter_set_ids),
        best_mae_result.performance.mae,
        best_mae_result.testcase_id,
        best_mae_result.parameter_set_id,
        best_mae_result.parameters,
        asdict(best_mae_result.performance),
        best_f1_result.performance.f1_macro,
        best_f1_result.testcase_id,
        best_f1_result.parameter_set_id,
        best_f1_result.parameters,
        asdict(best_f1_result.performance),
        float(np.mean(mae_scores)),
        float(np.mean(f1_scores)),
    )


def run_model_optimisation(
    model_name: str,
    test_case: str,
    use_all_osm: bool | None,
    parameter_runs: list[ModelParameterRun],
    data_config: DataConfig | None = None,
    log_to_mlflow: bool = True,
) -> list[OptimisationResult]:
    """Run one model's hyperparameter optimisation with the shared MLflow workflow."""
    if data_config is None:
        data_config = setup_data_config()
    local_logger = configure_pipeline_logging()

    if not parameter_runs:
        log_model_optimisation_summary(local_logger, model_name, [])
        return []

    mlflow_logger = None
    if log_to_mlflow:
        base_experiment_config = setup_experiment_config(model_name, parameter_runs[0].model_config, test_case, use_all_osm)
        from src.experiments.mlflow_secret import MLFLOW_TRACKING_URI
        from src.model_building.logging.mlflow_logging import MlflowLogger

        mlflow_logger = MlflowLogger(
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=base_experiment_config.experiment_name,
            model=base_experiment_config.model,
            dataset_case_group=base_experiment_config.get_ds_case_group(),
        ).start_parent_run()

    run_results: list[OptimisationResult] = []
    for parameter_run in parameter_runs:
        _log_parameter_selected(local_logger, model_name, parameter_run)
        experiment_config = setup_experiment_config(
            model_name,
            parameter_run.model_config,
            test_case,
            use_all_osm,
        )

        experiment_results = run_experiment(
            data_config,
            experiment_config,
            local_logger,
            mlflow_logger,
            parameter_run.parameter_set_id,
        )
        run_results.extend(
            OptimisationResult(
                model_name=model_name,
                parameter_set_id=parameter_run.parameter_set_id,
                parameters=parameter_run.parameters,
                testcase_id=data_test_case_id,
                performance=performance,
                performance_std=performance_std,
            )
            for data_test_case_id, cross_val_performance in experiment_results.items()
            for performance, performance_std in [cross_val_performance.get_final_performance()]
        )

    if mlflow_logger is not None:
        mlflow_logger.end_parent_run()
    log_model_optimisation_summary(local_logger, model_name, run_results)
    return run_results
