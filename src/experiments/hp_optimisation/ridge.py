import numpy as np

from src.experiments.hp_optimisation.pipeline import ModelParameterRun, run_model_optimisation
from src.experiments.hp_optimisation.result import OptimisationResult
from src.experiments.experiment_config_defaults import LOG_TO_MLFLOW, RIDGE_N_PARAMETER_SETS
from src.model_building.config.model_config import LinearModelConfig


def run_ridge_optimisation(
    test_case: str,
    use_all_osm: bool | None,
    n_parameter_sets: int = RIDGE_N_PARAMETER_SETS,
    log_to_mlflow: bool = LOG_TO_MLFLOW,
) -> list[OptimisationResult]:
    """Run the default example model evaluation experiment."""
    parameter_runs = [
        ModelParameterRun(
            parameter_set_id=parameter_set_id,
            parameters={"alpha": float(alpha)},
            model_config=LinearModelConfig(alpha=float(alpha)),
        )
        for parameter_set_id, alpha in enumerate(np.logspace(-5, 5, n_parameter_sets))
    ]

    return run_model_optimisation(
        model_name="Linear",
        test_case=test_case,
        use_all_osm=use_all_osm,
        parameter_runs=parameter_runs,
        log_to_mlflow=log_to_mlflow,
    )


if __name__ == "__main__":
    run_ridge_optimisation("A", None)
    run_ridge_optimisation("B", True)
    run_ridge_optimisation("B", False)
    run_ridge_optimisation("C", True)
    run_ridge_optimisation("C", False)
