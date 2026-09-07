from itertools import product

import numpy as np

from src.experiments.hp_optimisation.result import ParameterSet, ParameterValue


def sample_parameter_combinations(
    parameter_space: dict[str, list[ParameterValue]],
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
