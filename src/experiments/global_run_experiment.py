from src.experiments.hp_optimisation_ann import run_ann_optimisation
from src.experiments.hp_optimisation_linear import run_ridge_optimisation
from src.experiments.hp_optimisation_xgboost import run_xgb_optimisation


if __name__ == "__main__":
    for test_case, use_all_osm in [
        ("A", None),
        ("B", True),
        ("B", False),
        ("C", True),
        ("C", False),
    ]:
        run_ridge_optimisation(test_case, use_all_osm)
        run_xgb_optimisation(test_case, use_all_osm)
        run_ann_optimisation(test_case, use_all_osm)
