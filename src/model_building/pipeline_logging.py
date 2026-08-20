import logging
import sys

from src.model_building.data.data_test_cases import DataTestCase
from src.model_building.data.model_data import ModelData

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
            "manual_train_rows=%s osm_train_rows=%s manual_val_rows=%s osm_val_rows=%s test_rows=%s"
        ),
        testcase_id,
        model_name,
        len(model_data.manual_train_y.index),
        len(model_data.osm_train_y.index),
        len(model_data.manual_val_y.index),
        len(model_data.osm_val_y.index),
        len(model_data.test_y.index),
    )
