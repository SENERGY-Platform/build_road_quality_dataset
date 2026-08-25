import numpy as np

from src.experiments.result_types import RidgeOptimisationResult
from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.config.model_config import LinearModelConfig
from src.model_building.data.data_loader import DataConfig
from src.model_building.model_evaluation_pipeline import run_experiment
from src.model_building.pipeline_logging import configure_pipeline_logging, log_ridge_optimisation_summary


def setup_data_config() -> DataConfig:
    return DataConfig(
        osm_ds_dir="data/open_street_map/datasets",
        manual_ds_dir="data/molewa/datasets",
        feature_ds_dir="data/molewa/model_building/feature_ds",
        skip_feature_build_if_exists=True,
    )

def setup_experiment_config(linear_model_config: LinearModelConfig) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="ridge_optimisation_abc",
        test_cases=['A', 'B', 'C'],
        # True uses all available data vs False equals osm train data to available manual data
        case_b_all_osm_data=False,
        case_c_all_osm_data=False,
        cross_validation_k=5,
        ds_version="v1.0",
        features=["vibration_x", "vibration_y", "vibration_z", "speed", "vibration_magnitude",
                  "score_mild", "score_standard", "score_strict"],
        models=['Linear'],
        test_set_percentage=0.3,
        linear_model_config=linear_model_config,
    )

def run_ridge_optimisation() -> None:
    """Run the default example model evaluation experiment."""
    data_config = setup_data_config()
    logger = configure_pipeline_logging()
    run_results: list[RidgeOptimisationResult] = []

    alpha_space = np.logspace(-5, 5, 30)
    for alpha in alpha_space:
        model_config = LinearModelConfig(alpha=float(alpha))
        experiment_config = setup_experiment_config(model_config)
        experiment_results = run_experiment(data_config, experiment_config, logger)['Linear']
        run_results.extend(
            RidgeOptimisationResult(
                alpha=float(alpha),
                testcase_id=data_test_case_id,
                performance=cross_val_performance.get_final_performance(),
            )
            for data_test_case_id, cross_val_performance in experiment_results.items()
        )

    log_ridge_optimisation_summary(logger, run_results)

if __name__ == "__main__":
    run_ridge_optimisation()