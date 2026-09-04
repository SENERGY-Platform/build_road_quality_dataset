from __future__ import annotations

import logging
from src.model_building.logging.mlflow_logging import MlflowLogger

from src.model_building.config.model_config import XGBoostModelConfig, LinearModelConfig, ANNModelConfig
from src.model_building.data import data_loader
from src.model_building.data.data_loader import DataConfig, load_feature_ds
from src.model_building.models.metrics import CrossValidationPerformance
from src.model_building.models.model_build import setup_model, update_model_data_for_validation, evaluate_model
from src.model_building.data.data_test_cases import get_test_dataset_configurations, DataTestCase
from src.model_building.data.data_split import build_stratified_shuffle_split_datasets
from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.logging.pipeline_logging import (
    configure_pipeline_logging,
    log_cross_val_performance,
    log_testcase_started,
)


def _run_model_parameter_trial(test_cases: list[DataTestCase],
                               experiment_config: ExperimentConfig,
                               logger: logging.Logger,
                               mlflow_logger: MlflowLogger | None,
                               parameter_set_id: int | None) -> dict[str, CrossValidationPerformance]:
    """Run one hyperparameter configuration across all dataset test cases."""
    # connect to experiment parent run -> model + dataset_case_group
    # for each dataset combination in group + experiment hyperparameters run one trial
    performance_by_test_case: dict[str, CrossValidationPerformance] = {}
    for test_case in test_cases:
        log_testcase_started(logger, test_case)
        if mlflow_logger is not None:
            if parameter_set_id is None:
                raise ValueError("parameter_set_id is required when mlflow_logger is provided.")
            mlflow_logger.start_trial(test_case, experiment_config, parameter_set_id)

        cross_val_performance = crossvalidate_model(test_case, experiment_config, logger)
        performance_by_test_case[test_case.case_id] = cross_val_performance

        if mlflow_logger is not None:
            mlflow_logger.end_trial(cross_val_performance)
    return performance_by_test_case


def run_experiment(
        data_config: DataConfig,
        experiment_config: ExperimentConfig,
        logger: logging.Logger | None = None,
        mlflow_logger: MlflowLogger | None = None,
        parameter_set_id: int | None = None,
) -> dict[str, CrossValidationPerformance]:
    """Load feature datasets and run one model evaluation experiment."""
    if logger is None:
        logger = configure_pipeline_logging()

    feature_datasets = load_feature_ds(data_config)
    test_cases = get_test_dataset_configurations(feature_datasets, experiment_config)

    model_performance_by_test_case = _run_model_parameter_trial(
        test_cases,
        experiment_config,
        logger,
        mlflow_logger,
        parameter_set_id,
    )
    return model_performance_by_test_case


def crossvalidate_model(test_case: DataTestCase, experiment_config: ExperimentConfig,
                        logger: logging.Logger) -> CrossValidationPerformance:
    """Train and evaluate a fresh model across each stratified split.

    Each split builds model-specific training/validation data, calls
    `evaluate_model` to fit and score the model, then adds the resulting
    quality and runtime metrics to the cross-validation aggregate.
    """
    stratified_shuffle_split_datasets = build_stratified_shuffle_split_datasets(test_case, experiment_config)
    cross_val_performance = CrossValidationPerformance()
    for test_case_model_data in stratified_shuffle_split_datasets:
        model = setup_model(experiment_config)
        model_data = update_model_data_for_validation(model, test_case_model_data, experiment_config)
        trained_model, run_performance = evaluate_model(model, model_data)

        cross_val_performance.add_performance(run_performance)
        cross_val_performance.add_model(trained_model)
        cross_val_performance.add_model_data(model_data)
    cross_val_performance.calculate_median_model()
    log_cross_val_performance(logger, test_case.case_id, experiment_config.model, cross_val_performance)
    return cross_val_performance


def example() -> None:
    """Run the default example model evaluation experiment."""
    data_config = data_loader.DataConfig(
        osm_ds_dir="data/open_street_map/datasets",
        manual_ds_dir="data/molewa/datasets",
        feature_ds_dir="data/molewa/model_building/feature_ds",
        skip_feature_build_if_exists=True,
    )
    # testcases:
    #  A: manual only
    #  B: combination
    #  C: only osm (but manual as test set)
    experiment_config = ExperimentConfig(
        experiment_name="Testrun",
        case_type='A',
        all_osm_data=None,
        cross_validation_k=10,
        ds_version="v1.0",
        feature_set_name="raw_features",
        features=["vibration_x", "vibration_y", "vibration_z", "speed"],
        # vibration_magnitude, score_mild, score_standard, score_strict
        model='ANN',
        test_set_percentage=0.3,
        ann_model_config=ANNModelConfig(
            val_set_percentage=0.2,
            layer_num_first_round=10,
            layer_num_second_round=3,
        ),
        # linear_model_config=LinearModelConfig(alpha=1.0),
        # xgb_model_config=XGBoostModelConfig()
    )
    run_experiment(data_config, experiment_config)


if __name__ == "__main__":
    example()
