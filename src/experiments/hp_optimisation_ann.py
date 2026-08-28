from itertools import product

import numpy as np

from src.experiments.result_types import ANNOptimisationResult
from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.config.model_config import ANNModelConfig
from src.model_building.data.data_loader import DataConfig
from src.model_building.model_evaluation_pipeline import run_experiment
from src.model_building.logging.pipeline_logging import configure_pipeline_logging, log_ann_optimisation_summary
from src.experiments.global_config import (EXPERIMENT_NAME, DS_VERSION, FEATURE_SET_NAME, FEATURES, CROSS_VALIDATION_K,
                                           TEST_SET_PERCENTAGE)


def setup_data_config() -> DataConfig:
    return DataConfig(
        osm_ds_dir="data/open_street_map/datasets",
        manual_ds_dir="data/molewa/datasets",
        feature_ds_dir="data/molewa/model_building/feature_ds",
        skip_feature_build_if_exists=True,
    )


def setup_experiment_config(ann_model_config: ANNModelConfig, test_cases:list[str], all_osm:bool) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=EXPERIMENT_NAME,
        test_cases=test_cases,
        # True uses all available data vs False equals osm train data to available manual data
        case_b_all_osm_data=all_osm,
        case_c_all_osm_data=all_osm,
        cross_validation_k=CROSS_VALIDATION_K,
        ds_version=DS_VERSION,
        feature_set_name=FEATURE_SET_NAME,
        features=FEATURES,
        models=['ANN'],
        test_set_percentage=TEST_SET_PERCENTAGE,
        ann_model_config=ann_model_config,
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


def run_ann_optimisation(test_cases: list[str], use_all_osm:bool) -> None:
    """Run randomized ANN hyperparameter optimisation across configured datasets."""
    data_config = setup_data_config()
    logger = configure_pipeline_logging()
    run_results: list[ANNOptimisationResult] = []

    # model hyper parameter exploration config
    val_set_percentage = 0.2
    parameter_space = {
        "layer_num_first_round": [2, 4, 6, 8, 10],
        "layer_num_second_round": [1, 2, 3, 4],
        "pretrain_learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "finetune_learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "batch_size": [32, 64, 128],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "weight_decay": [0.0, 0.00001, 0.0001, 0.001],
    }
    n_random_parameter_sets = 30

    parameter_test_cases = sample_parameter_combinations(parameter_space, n_random_parameter_sets)
    for parameter_set_id, parameters in enumerate(parameter_test_cases):
        model_config = ANNModelConfig(val_set_percentage=val_set_percentage, **parameters)
        logger.info(
            "event=ann_parameter_selected parameter_set_id=%s parameters=%s",
            parameter_set_id,
            parameters,
        )
        experiment_config = setup_experiment_config(model_config, test_cases, use_all_osm)
        experiment_results = run_experiment(data_config, experiment_config, logger)['ANN']

        run_results.extend(
            ANNOptimisationResult(
                parameter_set_id=parameter_set_id,
                parameters=parameters,
                testcase_id=data_test_case_id,
                performance=cross_val_performance.get_final_performance(),
            )
            for data_test_case_id, cross_val_performance in experiment_results.items()
        )

    log_ann_optimisation_summary(logger, run_results)


if __name__ == '__main__':
    run_ann_optimisation(['A', 'B', 'C'], True)
    run_ann_optimisation(['B', 'C'], False)
