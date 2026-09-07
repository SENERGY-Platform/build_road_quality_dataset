from itertools import product

import numpy as np

from src.experiments.model_optimisation_pipeline import ModelParameterRun, ParameterSet, run_model_optimisation
from src.experiments.result_types import OptimisationResult
from src.experiments.global_config import LOG_TO_MLFLOW, XGB_N_PARAMETER_SETS
from src.model_building.config.model_config import XGBoostModelConfig


def sample_parameter_combinations(
        parameter_space: dict[str, list[int | float]],
        n_combinations: int,
        random_state: int = 42,
) -> list[ParameterSet]:
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


def run_xgb_optimisation(
    test_case: str,
    use_all_osm: bool | None,
    n_parameter_sets: int = XGB_N_PARAMETER_SETS,
    log_to_mlflow: bool = LOG_TO_MLFLOW,
) -> list[OptimisationResult]:
    """Run randomised XGBoost hyperparameter optimisation across configured datasets."""
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
    parameter_test_cases = sample_parameter_combinations(parameter_space, n_parameter_sets)
    parameter_runs = [
        ModelParameterRun(
            parameter_set_id=parameter_set_id,
            parameters=parameters,
            model_config=XGBoostModelConfig(**parameters),
        )
        for parameter_set_id, parameters in enumerate(parameter_test_cases)
    ]

    return run_model_optimisation(
        model_name="XGBoost",
        test_case=test_case,
        use_all_osm=use_all_osm,
        parameter_runs=parameter_runs,
        log_to_mlflow=log_to_mlflow,
    )


if __name__ == '__main__':
    run_xgb_optimisation("A", None)
    run_xgb_optimisation("B", True)
    run_xgb_optimisation("B", False)
    run_xgb_optimisation("C", True)
    run_xgb_optimisation("C", False)
