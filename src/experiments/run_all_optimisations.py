import json
from pathlib import Path

from src.experiments.experiment_config_defaults import (
    ANN_N_PARAMETER_SETS,
    LOG_TO_MLFLOW,
    RIDGE_N_PARAMETER_SETS,
    XGB_N_PARAMETER_SETS, RUN_ON_RAY,
)
from src.experiments.mlflow_secret import RAY_ADDRESS
from src.experiments.hp_optimisation.ann import run_ann_optimisation
from src.experiments.hp_optimisation.ridge import run_ridge_optimisation
from src.experiments.hp_optimisation.xgboost import run_xgb_optimisation


RAY_RUNTIME_ENV_FILE = Path(__file__).resolve().parents[2] / "ray_runtime_env.json"

def load_ray_runtime_env() -> dict:
    """Load the Ray runtime environment used by worker tasks."""
    with RAY_RUNTIME_ENV_FILE.open(encoding="utf-8") as runtime_env_file:
        return json.load(runtime_env_file)


def run_all_optimisations() -> None:
    """Run all configured optimisation experiments serially."""
    for test_case, use_all_osm in [
        ("A", None),
        ("B", False),
        ("B", True),
        ("C", False),
        ("C", True),
    ]:
        run_ridge_optimisation(
            test_case,
            use_all_osm,
            n_parameter_sets=RIDGE_N_PARAMETER_SETS,
            log_to_mlflow=LOG_TO_MLFLOW,
        )
        run_xgb_optimisation(
            test_case,
            use_all_osm,
            n_parameter_sets=XGB_N_PARAMETER_SETS,
            log_to_mlflow=LOG_TO_MLFLOW,
        )
        run_ann_optimisation(
            test_case,
            use_all_osm,
            n_parameter_sets=ANN_N_PARAMETER_SETS,
            log_to_mlflow=LOG_TO_MLFLOW,
        )


def run_all_optimisations_on_ray() -> None:
    """Run the complete serial optimisation pipeline inside one Ray worker."""
    import ray

    @ray.remote(num_gpus=1)
    def _run_all_optimisations_remote() -> None:
        run_all_optimisations()

    started_ray = False
    if not ray.is_initialized():
        ray.init(address=RAY_ADDRESS, runtime_env=load_ray_runtime_env())
        started_ray = True

    try:
        ray.get(_run_all_optimisations_remote.remote())
    finally:
        if started_ray:
            ray.shutdown()


if __name__ == "__main__":
    if RUN_ON_RAY:
        run_all_optimisations_on_ray()
    else:
        run_all_optimisations()