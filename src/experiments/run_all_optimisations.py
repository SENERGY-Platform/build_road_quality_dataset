import os
from pathlib import Path

from src.experiments.experiment_config_defaults import (
    ANN_N_PARAMETER_SETS,
    LOG_TO_MLFLOW,
    RIDGE_N_PARAMETER_SETS,
    RUN_ON_RAY,
    XGB_N_PARAMETER_SETS,
)
from src.experiments.mlflow_secret import MLFLOW_TRACKING_URI, RAY_ADDRESS
from src.experiments.hp_optimisation.ann import run_ann_optimisation
from src.experiments.hp_optimisation.ridge import run_ridge_optimisation
from src.experiments.hp_optimisation.xgboost import run_xgb_optimisation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    # uv may need to download large CUDA wheels when Ray creates the worker
    # environment for the first time.
    os.environ.setdefault("RAY_CLIENT_MAX_CONNECTION_TIMEOUT_S", "600")
    os.environ.setdefault("RAY_CLIENT_SERVER_CHECK_CHANNEL_TIMEOUT_S", "600")

    import ray

    @ray.remote(num_gpus=1)
    def _run_all_optimisations_remote() -> None:
        run_all_optimisations()

    started_ray = False
    if not ray.is_initialized():
        ray.init(
            address=RAY_ADDRESS,
            runtime_env={
                "working_dir": str(PROJECT_ROOT),
                "py_executable": "uv run --locked python",
                "excludes": ["/.venv", "/data", "/.uv-cache-*"],
                "env_vars": {"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
            },
        )
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