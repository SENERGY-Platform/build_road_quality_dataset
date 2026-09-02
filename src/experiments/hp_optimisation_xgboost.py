from itertools import product

import numpy as np

from src.experiments.result_types import XGBoostOptimisationResult
from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.config.model_config import XGBoostModelConfig
from src.model_building.data.data_loader import DataConfig
from src.model_building.model_evaluation_pipeline import run_experiment
from src.model_building.logging.pipeline_logging import configure_pipeline_logging, log_xgb_optimisation_summary
from src.experiments.global_config import (EXPERIMENT_NAME, DS_VERSION, FEATURE_SET_NAME, FEATURES, CROSS_VALIDATION_K,
                                           TEST_SET_PERCENTAGE)


def setup_data_config() -> DataConfig:
    return DataConfig(
        osm_ds_dir="data/open_street_map/datasets",
        manual_ds_dir="data/molewa/datasets",
        feature_ds_dir="data/molewa/model_building/feature_ds",
        skip_feature_build_if_exists=True,
    )


def setup_experiment_config(xgb_model_config: XGBoostModelConfig, test_case: str,
                            all_osm: bool | None) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=EXPERIMENT_NAME,
        test_case=test_case,
        all_osm_data=all_osm, # True uses all available data vs False equals osm train data to available manual data
        cross_validation_k=CROSS_VALIDATION_K,
        ds_version=DS_VERSION,
        feature_set_name=FEATURE_SET_NAME,
        features=FEATURES,
        models=['XGBoost'],
        test_set_percentage=TEST_SET_PERCENTAGE,
        xgb_model_config=xgb_model_config,
    )


def sample_parameter_combinations(
        parameter_space: dict[str, list[int | float]],
        n_combinations: int,
        random_state: int = 42,
) -> list[dict[str, int | float]]:
    """Draw random unique parameter combinations from a discrete search space."""
    parameter_names = list(parameter_space)
    all_combinations = [
        dict(zip(parameter_names, values))
        for values in product(*(parameter_space[name] for name in parameter_names))
    ]
    if n_combinations > len(all_combinations):
        raise ValueError(
            f"Requested {n_combinations} parameter combinations, but only "
            f"{len(all_combinations)} unique combinations exist."
        )

    rng = np.random.default_rng(random_state)
    selected_indices = rng.choice(len(all_combinations), size=n_combinations, replace=False)
    return [all_combinations[index] for index in selected_indices]


def run_xgb_optimisation(test_case: str, use_all_osm: bool | None) -> None:
    """Run randomised XGBoost hyperparameter optimisation across configured datasets."""
    data_config = setup_data_config()
    local_logger = configure_pipeline_logging()
    run_results: list[XGBoostOptimisationResult] = []

    # model hyper parameter exploration config
    parameter_space = {
        "n_estimators": [100, 200, 400, 800],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 10, 20],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [0.5, 1.0, 5.0, 10.0],
    }
    n_random_parameter_sets = 30

    parameter_test_cases = sample_parameter_combinations(parameter_space, n_random_parameter_sets)
    for parameter_set_id, parameters in enumerate(parameter_test_cases):
        model_config = XGBoostModelConfig(**parameters)
        local_logger.info(
            "event=xgb_parameter_selected parameter_set_id=%s parameters=%s",
            parameter_set_id,
            parameters,
        )
        experiment_config = setup_experiment_config(model_config, test_case, use_all_osm)
        experiment_results = run_experiment(data_config, experiment_config, local_logger)['XGBoost']

        run_results.extend(
            XGBoostOptimisationResult(
                parameter_set_id=parameter_set_id,
                parameters=parameters,
                testcase_id=data_test_case_id,
                performance=performance,
                performance_std=performance_std,
            )
            for data_test_case_id, cross_val_performance in experiment_results.items()
            for performance, performance_std in [cross_val_performance.get_final_performance()]
        )

    log_xgb_optimisation_summary(local_logger, run_results)


if __name__ == '__main__':
    run_xgb_optimisation("A", None)
    run_xgb_optimisation("B", True)
    run_xgb_optimisation("B", False)
    run_xgb_optimisation("C", True)
    run_xgb_optimisation("C", False)
