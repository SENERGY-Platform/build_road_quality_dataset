from dataclasses import asdict
import logging

from src.model_building.config.model_config import XGBoostModelConfig, LinearModelConfig, ANNModelConfig
from src.model_building.data import data_loader
from src.model_building.data.data_loader import DataConfig, load_feature_ds
from src.model_building.models.metrics import CrossValidationPerformance
from src.model_building.models.model_build import setup_model, train_model, update_model_data_for_validation, evaluate_model
from src.model_building.data.data_test_cases import get_test_dataset_configurations, DataTestCase
from src.model_building.data.data_split import build_stratified_shuffle_split_datasets
from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.pipeline_logging import (
    configure_pipeline_logging,
    log_cross_val_performance,
    log_model_data_used,
    log_model_metrics,
    log_testcase_started,
)


def run_experiment(
    data_config: DataConfig,
    experiment_config: ExperimentConfig,
    logger: logging.Logger | None = None,
) -> dict[str, dict[str, CrossValidationPerformance]]:
    """Load feature datasets and run one model evaluation experiment."""
    if logger is None:
        logger = configure_pipeline_logging()

    feature_datasets = load_feature_ds(data_config)
    test_cases = get_test_dataset_configurations(
        feature_datasets,
        experiment_config.test_cases,
        experiment_config.ds_version,
    )
    performance_by_model_test_case: dict[str, dict[str, CrossValidationPerformance]] = {}
    for model_str in experiment_config.models:
        if model_str not in performance_by_model_test_case:
            performance_by_model_test_case[model_str] = {}

        for test_case in test_cases:
            log_testcase_started(logger, test_case)
            cross_val_performance = crossvalidate_model(model_str, test_case, experiment_config, logger)
            performance_by_model_test_case[model_str][test_case.case_id] = cross_val_performance
    return performance_by_model_test_case

def crossvalidate_model(model_str:str, test_case:DataTestCase, experiment_config:ExperimentConfig, logger: logging.Logger) -> CrossValidationPerformance:
    """Train and evaluate one model across repeated stratified splits for one test case."""
    stratified_shuffle_split_datasets = build_stratified_shuffle_split_datasets(test_case, experiment_config)
    cross_val_performance = CrossValidationPerformance()
    for test_case_model_data in stratified_shuffle_split_datasets:
        model = setup_model(model_str, experiment_config)
        model_data = update_model_data_for_validation(model, test_case_model_data, experiment_config)
        #log_model_data_used(logger, test_case.case_id, model_str, model_data)
        trained_model = train_model(model, model_data)
        run_performance = evaluate_model(trained_model, model_data)
        #log_model_metrics(logger, test_case.case_id, model_str, asdict(run_performance))
        cross_val_performance.add_performance(run_performance)
    log_cross_val_performance(logger, test_case.case_id, model_str, cross_val_performance)
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
        test_cases=['A','B', 'C'],
        case_b_all_osm_data=False, # True uses all available data vs False equals osm train data to available manual data
        case_c_all_osm_data=False,
        cross_validation_k=10,
        ds_version="v1.0",
        features=["vibration_x", "vibration_y", "vibration_z", "speed"], # vibration_magnitude, score_mild, score_standard, score_strict
        models=['Linear','XGBoost'],
        test_set_percentage=0.3,
        ann_model_config=ANNModelConfig(
            val_set_percentage=0.2,
            layer_num_first_round=10,
            layer_num_second_round=3,
        ),
        linear_model_config=LinearModelConfig(alpha=1.0),
        xgb_model_config=XGBoostModelConfig()
    )

    run_experiment(data_config, experiment_config)

if __name__ == "__main__":
    example()
