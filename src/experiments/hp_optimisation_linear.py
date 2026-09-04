import numpy as np

from src.experiments.result_types import RidgeOptimisationResult
from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.config.model_config import LinearModelConfig
from src.model_building.data.data_loader import DataConfig
from src.model_building.model_evaluation_pipeline import run_experiment
from src.model_building.logging.pipeline_logging import configure_pipeline_logging, log_ridge_optimisation_summary
from src.experiments.global_config import (EXPERIMENT_NAME, DS_VERSION, FEATURE_SET_NAME, FEATURES, CROSS_VALIDATION_K,
                                           TEST_SET_PERCENTAGE)


def setup_data_config() -> DataConfig:
    """Return the shared data-loading config for ridge optimisation runs."""
    return DataConfig(
        osm_ds_dir="data/open_street_map/datasets",
        manual_ds_dir="data/molewa/datasets",
        feature_ds_dir="data/molewa/model_building/feature_ds",
        skip_feature_build_if_exists=True,
    )

def setup_experiment_config(linear_model_config: LinearModelConfig, test_case:str, all_osm:bool|None) -> ExperimentConfig:
    """Return an experiment config for one ridge hyperparameter set."""
    return ExperimentConfig(
        experiment_name=EXPERIMENT_NAME,
        case_type=test_case,
        all_osm_data=all_osm, # True uses all available data vs False equals osm train data to available manual data
        cross_validation_k=CROSS_VALIDATION_K,
        ds_version=DS_VERSION,
        feature_set_name=FEATURE_SET_NAME,
        features=FEATURES,
        model='Linear',
        test_set_percentage=TEST_SET_PERCENTAGE,
        linear_model_config=linear_model_config,
    )

def run_ridge_optimisation(test_case: str, use_all_osm: bool|None) -> None:
    """Run the default example model evaluation experiment."""
    data_config = setup_data_config()
    logger = configure_pipeline_logging()
    run_results: list[RidgeOptimisationResult] = []

    alpha_space = np.logspace(-5, 5, 30)
    for alpha in alpha_space:
        model_config = LinearModelConfig(alpha=float(alpha))
        logger.info("event=ridge_parameter_selected alpha=%s", model_config.alpha)
        experiment_config = setup_experiment_config(model_config, test_case, use_all_osm)
        experiment_results = run_experiment(data_config, experiment_config, logger)
        run_results.extend(
            RidgeOptimisationResult(
                alpha=float(alpha),
                testcase_id=data_test_case_id,
                performance=performance,
                performance_std=performance_std,
            )
            for data_test_case_id, cross_val_performance in experiment_results.items()
            for performance, performance_std in [cross_val_performance.get_final_performance()]
        )

    log_ridge_optimisation_summary(logger, run_results)

if __name__ == "__main__":
    run_ridge_optimisation("A", None)
    run_ridge_optimisation("B", True)
    run_ridge_optimisation("B", False)
    run_ridge_optimisation("C", True)
    run_ridge_optimisation("C", False)
