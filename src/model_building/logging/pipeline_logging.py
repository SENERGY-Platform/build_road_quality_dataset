from dataclasses import asdict
import logging
import sys

import numpy as np

from src.experiments.result_types import (
    OptimisationResult,
)
from src.model_building.data.data_test_cases import DataTestCase
from src.model_building.data.model_data import ModelData
from src.model_building.models.metrics import CrossValidationPerformance

LOGGER_NAME = "model_building.pipeline"


def configure_pipeline_logging() -> logging.Logger:
    """Configure a lightweight stdout logger for pipeline events."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def log_testcase_started(logger: logging.Logger, test_case: DataTestCase) -> None:
    """Log the raw data size for a testcase once before model-specific processing."""
    manual_size = 0 if test_case.manual_ds is None else len(test_case.manual_ds.index)
    osm_size = 0 if test_case.osm_ds is None else len(test_case.osm_ds.index)

    logger.info(
        "event=testcase_started testcase_id=%s manual_rows=%s osm_rows=%s",
        test_case.case_id,
        manual_size,
        osm_size,
    )


def log_model_data_used(logger: logging.Logger, testcase_id: str, model_name: str, model_data: ModelData) -> None:
    """Log the data rows available to a model after model-specific data preparation."""
    logger.info(
        (
            "event=model_data_used testcase_id=%s model=%s "
            "case_origin=%s random_state=%s "
            "manual_train_rows=%s osm_train_rows=%s manual_val_rows=%s osm_val_rows=%s test_rows=%s"
        ),
        testcase_id,
        model_name,
        model_data.test_case_id,
        model_data.random_state,
        len(model_data.manual_train_y.index),
        len(model_data.osm_train_y.index),
        len(model_data.manual_val_y.index),
        len(model_data.osm_val_y.index),
        len(model_data.test_y.index),
    )


def log_model_metrics(logger: logging.Logger, testcase_id: str, model_name: str, metrics: dict[str, float]) -> None:
    """Log evaluation metrics for one model and testcase."""
    logger.info(
        "event=model_evaluated testcase_id=%s model=%s metrics=%s",
        testcase_id,
        model_name,
        metrics,
    )


def log_cross_val_performance(
    logger: logging.Logger,
    testcase_id: str,
    model_name: str,
    cross_val_performance: CrossValidationPerformance,
) -> None:
    """Log the final averaged performance across repeated stratified splits."""
    performance, performance_std = cross_val_performance.get_final_performance()
    logger.info(
        "event=cross_val_performance testcase_id=%s model=%s metrics=%s metrics_std=%s",
        testcase_id,
        model_name,
        asdict(performance),
        asdict(performance_std),
    )


def log_model_optimisation_summary(
    logger: logging.Logger,
    model_name: str,
    results: list[OptimisationResult],
) -> None:
    """Log aggregate performance stats for one model optimisation run."""
    event_name_by_model = {
        "ANN": "ann_optimisation_summary",
        "Linear": "ridge_optimisation_summary",
        "XGBoost": "xgb_optimisation_summary",
    }
    event_name = event_name_by_model.get(model_name, "model_optimisation_summary")

    if not results:
        logger.info("event=%s datasets_tested=0 model_runs_tested=0", event_name)
        return

    best_mae_result = min(results, key=lambda result: result.performance.mae)
    best_f1_result = max(results, key=lambda result: result.performance.f1_macro)
    mae_scores = [result.performance.mae for result in results]
    f1_scores = [result.performance.f1_macro for result in results]
    parameter_set_ids = {result.parameter_set_id for result in results}
    testcase_ids = {result.testcase_id for result in results}

    logger.info(
        (
            "event=%s model=%s "
            "datasets_tested=%s model_runs_tested=%s parameter_sets_tested=%s "
            "best_mae=%s best_mae_testcase_id=%s best_mae_parameter_set_id=%s "
            "best_mae_parameters=%s best_mae_metrics=%s "
            "best_f1_macro=%s best_f1_testcase_id=%s best_f1_parameter_set_id=%s "
            "best_f1_parameters=%s best_f1_metrics=%s "
            "mean_mae=%s mean_f1_macro=%s"
        ),
        event_name,
        model_name,
        len(testcase_ids),
        len(results),
        len(parameter_set_ids),
        best_mae_result.performance.mae,
        best_mae_result.testcase_id,
        best_mae_result.parameter_set_id,
        best_mae_result.parameters,
        asdict(best_mae_result.performance),
        best_f1_result.performance.f1_macro,
        best_f1_result.testcase_id,
        best_f1_result.parameter_set_id,
        best_f1_result.parameters,
        asdict(best_f1_result.performance),
        float(np.mean(mae_scores)),
        float(np.mean(f1_scores)),
    )
