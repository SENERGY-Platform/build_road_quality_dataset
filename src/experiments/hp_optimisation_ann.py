from itertools import product

import numpy as np

from src.experiments.model_optimisation_pipeline import ModelParameterRun, ParameterSet, run_model_optimisation
from src.experiments.result_types import OptimisationResult
from src.experiments.global_config import ANN_N_PARAMETER_SETS, LOG_TO_MLFLOW
from src.model_building.config.model_config import ANNModelConfig


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


def run_ann_optimisation(
    test_case: str,
    use_all_osm: bool | None,
    n_parameter_sets: int = ANN_N_PARAMETER_SETS,
    log_to_mlflow: bool = LOG_TO_MLFLOW,
) -> list[OptimisationResult]:
    """Run randomised ANN hyperparameter optimisation across configured datasets."""
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
    parameter_test_cases = sample_parameter_combinations(parameter_space, n_parameter_sets)
    parameter_runs = [
        ModelParameterRun(
            parameter_set_id=parameter_set_id,
            parameters=parameters,
            model_config=ANNModelConfig(val_set_percentage=val_set_percentage, **parameters),
        )
        for parameter_set_id, parameters in enumerate(parameter_test_cases)
    ]

    return run_model_optimisation(
        model_name="ANN",
        test_case=test_case,
        use_all_osm=use_all_osm,
        parameter_runs=parameter_runs,
        log_to_mlflow=log_to_mlflow,
    )


if __name__ == '__main__':
    run_ann_optimisation('A', None)
    run_ann_optimisation('B', True)
    run_ann_optimisation('B', False)
    run_ann_optimisation('C', True)
    run_ann_optimisation('C', False)
