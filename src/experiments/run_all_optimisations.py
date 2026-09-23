import os
from pathlib import Path
import tempfile

from src.experiments.experiment_config_defaults import (
    ANN_N_PARAMETER_SETS,
    FEATURE_DS_DIR,
    LOG_TO_MLFLOW,
    RIDGE_N_PARAMETER_SETS,
    RUN_ON_RAY,
    XGB_N_PARAMETER_SETS,
)
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
    ray_address = os.environ.get("RAY_ADDRESS")
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if ray_address is None or mlflow_tracking_uri is None:
        from src.experiments.mlflow_secret import MLFLOW_TRACKING_URI, RAY_ADDRESS

        ray_address = ray_address or RAY_ADDRESS
        mlflow_tracking_uri = mlflow_tracking_uri or MLFLOW_TRACKING_URI

    # uv may need to download large CUDA wheels when Ray creates the worker
    # environment for the first time.
    os.environ.setdefault("RAY_CLIENT_MAX_CONNECTION_TIMEOUT_S", "600")
    os.environ.setdefault("RAY_CLIENT_SERVER_CHECK_CHANNEL_TIMEOUT_S", "600")

    import ray

    @ray.remote(num_gpus=1)
    def run_all_optimisations_remote(feature_dataset_files: dict[str, bytes]) -> None:
        previous_working_dir = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="road-quality-data-") as temporary_dir:
            feature_ds_dir = Path(temporary_dir) / FEATURE_DS_DIR
            for relative_path, contents in feature_dataset_files.items():
                output_path = feature_ds_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(contents)

            os.chdir(temporary_dir)
            try:
                run_all_optimisations()
            finally:
                os.chdir(previous_working_dir)

    ray.init(
        address=ray_address,
        runtime_env={
            "working_dir": str(PROJECT_ROOT),
            "py_executable": "uv run --locked python",
            "excludes": ["/.venv", "/data", "/.uv-cache-*"],
            "env_vars": {"MLFLOW_TRACKING_URI": mlflow_tracking_uri},
        },
    )

    try:
        feature_ds_dir = PROJECT_ROOT / FEATURE_DS_DIR
        feature_dataset_files = {
            file_path.relative_to(feature_ds_dir).as_posix(): file_path.read_bytes()
            for group in ("manual", "osm")
            for file_path in (feature_ds_dir / group).glob("*.parquet")
        }
        feature_dataset_files_ref = ray.put(feature_dataset_files)
        ray.get(run_all_optimisations_remote.remote(feature_dataset_files_ref))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    if RUN_ON_RAY:
        run_all_optimisations_on_ray()
    else:
        run_all_optimisations()