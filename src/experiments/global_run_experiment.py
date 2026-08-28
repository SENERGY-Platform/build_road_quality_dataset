from src.experiments.hp_optimisation_ann import run_ann_optimisation
from src.experiments.hp_optimisation_linear import run_ridge_optimisation
from src.experiments.hp_optimisation_xgboost import run_xgb_optimisation


if __name__ == "__main__":
    run_ridge_optimisation(['A', 'B', 'C'], True)
    run_ridge_optimisation(['B', 'C'], False)
    run_xgb_optimisation(['A', 'B', 'C'], True)
    run_xgb_optimisation(['B', 'C'], False)
    run_ann_optimisation(['A', 'B', 'C'], True)
    run_ann_optimisation(['B', 'C'], False)
