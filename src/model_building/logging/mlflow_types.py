from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Optional

from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.config.model_config import ANNModelConfig, LinearModelConfig, XGBoostModelConfig
from src.model_building.data.data_test_cases import DataTestCase, DataTestCaseManualParameters, DataTestCaseOsmParameters
from src.model_building.data.model_data import ModelData, ModelDataSizeSummary
from src.model_building.models.metrics import ModelPerformance, ModelPerformanceStd


def _asdict(value: Any) -> dict[str, Any]:
    """Return a dictionary representation for dataclass and dict values."""
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected dataclass or dict, got {type(value).__name__}.")


def _prefix_params(prefix: str, params: dict[str, Any]) -> dict[str, Any]:
    """Prefix parameter keys for stable Mlflow namespaces."""
    return {
        f"{prefix}_{key}": value
        for key, value in params.items()
    }


@dataclass(init=False)
class ParentRunConfig:
    """
    Configuration for one mlflow parent/grouping run.
    Example parent:
        xgboost__case_b_osm_allpoints

    Represents one:
        model x dataset-combination

    The parent groups multiple hyperparameter trials and,
    after all trials finish, can store information about the
    winning trial and the fitted winning model.
    """

    model_name: str             # i.e. xgboost, ridge, ann
    dataset_case_group: str     # i.e. "case_a_manual_only", "case_b_osm_allpoints", "case_b_osm_limitedpoints"
    run_name: str               # i.e. {model}__{dataset_case_group}
    run_type: str               # parent

    mlflow_experiment_id: Optional[str]     # gets initialised once mlflow run is started and id was created
    mlflow_run_id: Optional[str]            # gets initialised once mlflow run is started and id was created

    # ------------------------------------------------------------------
    # Best-trial summary
    # These are initially None and get populated AFTER all trials have been evaluated.

    best_trial_run_id: str | None
    best_trial_name: str | None
    best_trial_number: int | None
    best_trial_params: dict[str, Any]
    best_trial_model_params: dict[str, Any]
    best_trial_metrics: dict[str, float]
    best_trial_model: Any | None
    best_trial_model_data: ModelData | None

    def __init__(self, model_name: str, dataset_case_group: str):
        """Initialise parent-run metadata for a model and dataset-case group."""
        self.model_name = model_name
        self.dataset_case_group = dataset_case_group
        self.run_name = f"{model_name}__{dataset_case_group}"

        # parameters that get filled once the run is started and later finished
        self.mlflow_experiment_id = None
        self.mlflow_run_id = None
        self.run_type = "parent"
        self.best_trial_run_id = None
        self.best_trial_name = None
        self.best_trial_number = None
        self.best_trial_params = {}
        self.best_trial_model_params = {}
        self.best_trial_metrics = {}
        self.best_trial_model = None
        self.best_trial_model_data = None

    def get_start_tags(self, run_started_at_utc: str) -> dict[str, str]:
        """Return the stable mlflow tags for a newly started parent run."""
        return {
            "model": self.model_name,
            "dataset_case_group": self.dataset_case_group,
            "run_type": self.run_type,
            "run_started_at_utc": run_started_at_utc,
        }

    def get_finish_tags(self, run_finished_at_utc: str) -> dict[str, str]:
        """Return final parent-run tags, including best-trial identity if known."""
        tags = {"run_finished_at_utc": run_finished_at_utc}
        if self.best_trial_run_id is not None:
            tags["best_trial_run_id"] = self.best_trial_run_id
        if self.best_trial_name is not None:
            tags["best_trial_name"] = self.best_trial_name
        if self.best_trial_number is not None:
            tags["best_trial_number"] = str(self.best_trial_number)
        return tags

    def get_best_params(self) -> dict[str, Any]:
        """Return parent-level parameters from the winning trial."""
        return self.best_trial_params

    def get_best_metrics(self) -> dict[str, Any]:
        """Return parent-level metrics that describe the winning trial."""
        return self.best_trial_metrics

    def set_mlflow_ids(self, experiment_id: str, run_id: str) -> None:
        """Store MLflow identifiers once the parent run has started."""
        self.mlflow_experiment_id = experiment_id
        self.mlflow_run_id = run_id

    def record_best_trial(
        self,
        trial_config: TrialRunConfig,
        model: Any,
        model_data: ModelData,
        metrics: dict[str, float],
        model_params: dict[str, Any],
    ) -> None:
        """Persist the current best-trial summary on the parent config."""
        self.best_trial_run_id = trial_config.mlflow_run_id
        self.best_trial_name = trial_config.trial_name
        self.best_trial_number = trial_config.trial_number
        self.best_trial_params = {
            **trial_config.get_dataset_params(),
            **trial_config.get_feature_params(),
            **trial_config.get_model_params(),
        }
        if trial_config.dataset_sizes is not None:
            self.best_trial_params.update(trial_config.get_dataset_size_params())
        self.best_trial_model_params = model_params
        self.best_trial_metrics = metrics
        self.best_trial_model = model
        self.best_trial_model_data = model_data


@dataclass(init=False)
class TrialRunConfig:
    """
    Configuration describing one concrete training/evaluation run.
    Example:
        hp_007_

    This should contain enough information to understand and reproduce
    the run independently of the parent hierarchy.
    """

    # trial meta data
    model_name: str             # i.e. xgboost, ridge, ann
    dataset_case_group: str     # i.e. "case_a_manual_only", "case_b_osm_allpoints", "case_b_osm_limitedpoints"
    test_case_id: str           # i.e. {model}__{dataset_case_group}
    parameter_set_id: int       # specific counter id of which model hp combination it is
    trial_number: int           # parent-local sequence counter for this started trial
    trial_name: str             # i.e. f"hp_{parameter_set_id:03d}__{case_group}__{manual_ds_id}__{osm_ds_id}"
    run_type: str               # trial

    # mlflow meta data
    mlflow_parent_run_id: str
    mlflow_run_id: Optional[str]

    # mlflow parameters
    manual_ds_params: DataTestCaseManualParameters
    osm_ds_params: DataTestCaseOsmParameters
    osm_all_datapoints: Optional[bool]
    dataset_sizes: ModelDataSizeSummary | None

    feature_set_name: str
    features: list[str]

    model_params: ANNModelConfig | XGBoostModelConfig | LinearModelConfig

    # mlflow metrics
    metrics: ModelPerformance | None
    metrics_std: ModelPerformanceStd | None

    def __init__(
        self,
        test_case: DataTestCase,
        experiment_config: ExperimentConfig,
        parameter_set_id: int,
        trial_number: int,
        parent_config: ParentRunConfig,
    ):
        """Initialise trial metadata from a dataset case, experiment, and parent run."""
        if parent_config.mlflow_run_id is None:
            raise ValueError("Cannot initialise a trial config without a parent mlflow run ID.")

        self.model_name = parent_config.model_name
        self.dataset_case_group = parent_config.dataset_case_group
        self.test_case_id = test_case.case_id
        self.parameter_set_id = parameter_set_id
        self.trial_number = trial_number
        self.trial_name = f"hp_{parameter_set_id:03d}__{test_case.case_id}"
        self.run_type = "trial"

        self.mlflow_parent_run_id = parent_config.mlflow_run_id
        self.manual_ds_params = test_case.manual_parameters
        self.osm_ds_params = test_case.osm_parameters
        self.osm_all_datapoints = experiment_config.all_osm_data
        self.feature_set_name = experiment_config.feature_set_name
        self.features = experiment_config.features
        self.model_params = experiment_config.get_model_params()

        # get filled once the run starts or later finishes
        self.dataset_sizes = None
        self.metrics = None
        self.metrics_std = None
        self.mlflow_run_id = None
        self.extra_tags = {}

    def get_start_tags(self, run_started_at_utc: str) -> dict[str, str]:
        """Return the stable mlflow tags for a newly started trial run."""
        return {
            "model": self.model_name,
            "dataset_case_group": self.dataset_case_group,
            "test_case_id": self.test_case_id,
            "parameter_set_id": str(self.parameter_set_id),
            "trial_number": str(self.trial_number),
            "parent_run_id": self.mlflow_parent_run_id,
            "run_type": self.run_type,
            "run_started_at_utc": run_started_at_utc,
            **self.extra_tags,
        }

    def get_dataset_params(self) -> dict[str, Any]:
        """Return trial parameters that describe dataset construction."""
        return {
            **_prefix_params("manual_ds", _asdict(self.manual_ds_params)),
            **_prefix_params("osm_ds", _asdict(self.osm_ds_params)),
            "osm_all_datapoints": self.osm_all_datapoints,
        }

    def get_feature_params(self) -> dict[str, Any]:
        """Return trial parameters that describe the resolved feature set."""
        return {
            "feature_set_name": self.feature_set_name,
            "features": self.features,
            "feature_count": len(self.features),
        }

    def get_model_params_without_prefix(self) -> dict[str, Any]:
        """Return raw model hyperparameters without the standard mlflow key prefix."""
        return _asdict(self.model_params)

    def get_model_params(self) -> dict[str, Any]:
        """Return trial model parameters with the standard mlflow key prefix."""
        return _prefix_params("model", self.get_model_params_without_prefix())

    def get_dataset_size_params(self) -> dict[str, Any]:
        """Return dataset-size parameters after they have been recorded."""
        return _asdict(self.dataset_sizes)

    def record_dataset_sizes(self, dataset_sizes: ModelDataSizeSummary) -> None:
        """Store source-aware dataset sizes from the selected CV fold."""
        self.dataset_sizes = dataset_sizes

    def record_metrics(
        self,
        metrics: ModelPerformance,
        metrics_std: ModelPerformanceStd,
    ) -> None:
        """Store mean and standard-deviation metrics for a completed trial."""
        self.metrics = metrics
        self.metrics_std = metrics_std

    def get_metrics(self) -> dict[str, float]:
        """Return recorded mean metrics as MLflow-compatible floats."""
        if self.metrics is None:
            return {}
        return {
            key: float(value)
            for key, value in _asdict(self.metrics).items()
        }

    def get_metrics_with_std(self) -> dict[str, float]:
        """Return recorded mean metrics plus `_std` suffixed standard deviations."""
        metrics = self.get_metrics()
        if self.metrics_std is None:
            return metrics
        metrics.update({
            f"{key}_std": float(value)
            for key, value in _asdict(self.metrics_std).items()
        })
        return metrics
