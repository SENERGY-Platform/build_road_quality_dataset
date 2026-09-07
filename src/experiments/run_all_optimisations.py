from src.experiments.experiment_config_defaults import (
    ANN_N_PARAMETER_SETS,
    LOG_TO_MLFLOW,
    RIDGE_N_PARAMETER_SETS,
    XGB_N_PARAMETER_SETS,
)
from src.experiments.hp_optimisation.ann import run_ann_optimisation
from src.experiments.hp_optimisation.ridge import run_ridge_optimisation
from src.experiments.hp_optimisation.xgboost import run_xgb_optimisation


if __name__ == "__main__":
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