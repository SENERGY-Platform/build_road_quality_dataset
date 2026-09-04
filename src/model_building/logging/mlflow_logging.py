from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import mlflow

from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.data.data_test_cases import DataTestCase
from src.model_building.logging.mlflow_types import ParentRunConfig, TrialRunConfig
from src.model_building.models.metrics import CrossValidationPerformance


class MlflowLogger:
    """
    Project-specific wrapper around mlflow.

    Responsibilities
    ----------------
    - Configure the mlflow experiment.
    - Create the <model>__<dataset_combination> parent run.
    - Create one trial run per hyperparameter configuration.
    - Apply our project's standard tags.
    - Log:
        * dataset-building parameters
        * model hyperparameters
        * dataset sizes
        * evaluation metrics
        * timings
    - Track which trial is currently the best within a parent.
    - Log the winner summary to the parent run.

    Non-responsibilities
    --------------------
    - Creating datasets
    - Training models
    - Calculating metrics
    - Choosing Ray search spaces
    - Scheduling/parallelisation
    - Deciding which hyperparameter combinations exist

    The logger RECEIVES these things from the pipeline.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        model: str,
        dataset_case_group: str
    ):
        """Initialise an MLflow logger for one model and dataset-case group."""
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.parent_run_config: ParentRunConfig = ParentRunConfig(
            model_name=model,
            dataset_case_group=dataset_case_group,
        )

        self.active_parent_run = None
        self.active_trial_run = None
        self.current_trial_config: TrialRunConfig | None = None
        self.trial_run_configs: list[TrialRunConfig] = []
        self.current_best_trial_run: TrialRunConfig | None = None
        self.total_trial_runs: int = 0


    def start_parent_run(self) -> MlflowLogger:
        """
        Start one grouping run. Connects to the configured mlflow tracking service, resolves or creates
        the experiment, and starts the parent run for this model/dataset group.

        Example
        -------
        xgboost__case_b_osm_allpoints

        Parent tags:
            model = xgboost
            dataset_combination = case_b_osm_allpoints
            run_type = parent


        Return:
            Context-manager/run representation.

        """
        if self.active_parent_run is not None:
            raise RuntimeError("A parent mlflow run is already active.")

        mlflow.set_tracking_uri(self.tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(self.experiment_name)
        self.active_parent_run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=self.parent_run_config.run_name,
            tags=self.parent_run_config.get_start_tags(self._utc_now()),
        )
        self.parent_run_config.set_mlflow_ids(
            experiment_id=experiment_id,
            run_id=self.active_parent_run.info.run_id,
        )

        return self

    def end_parent_run(self) -> None:
        """
        Finalise the grouping run.

        Workflow
        -----------------
        1. Write the winner's summary to the parent when a winning trial exists:
            - best_trial_run_id tag
            - best metrics
            - winning model hyperparameters
        2. Close parent run.

        Important:
        Parent parameters should generally be written only after
        the winning trial is known, because mlflow parameters
        should not be treated as mutable state.
        """
        if self.active_parent_run is None:
            raise RuntimeError("Cannot end an mlflow parent run because no parent run is active.")
        if self.active_trial_run is not None:
            raise RuntimeError("Cannot end an mlflow parent run while a trial run is active.")

        mlflow.set_tags(self.parent_run_config.get_finish_tags(self._utc_now()))
        best_model_params = self.parent_run_config.get_best_model_params()
        if best_model_params:
            mlflow.log_params(self._format_params(best_model_params))
        best_metrics = self.parent_run_config.get_best_metrics()
        if best_metrics:
            mlflow.log_metrics(best_metrics)
        mlflow.end_run(status="FINISHED")
        self.active_parent_run = None


    # ------------------------------------------------------------------
    # Trial / hyperparameter run
    # ------------------------------------------------------------------
    def start_trial(
        self,
        test_case: DataTestCase,
        experiment_config: ExperimentConfig,
        parameter_set_id: int,
    ) -> TrialRunConfig:
        """
        Start one REAL training run underneath the current parent.

        Example:
            hp_001

        Suggested trial tags:
            model = xgboost
            dataset_combination = case_b_osm_allpoints
            run_type = trial

        Even though model and dataset_combination are represented
        by the parent, repeat them on the trial for easy filtering.

        Return the run/run_id so that the pipeline can associate
        everything with this trial.
        """
        if self.active_parent_run is None or self.parent_run_config.mlflow_run_id is None:
            raise RuntimeError("Cannot start an mlflow trial before starting a parent run.")
        if self.active_trial_run is not None:
            raise RuntimeError("A trial mlflow run is already active.")

        self.total_trial_runs += 1

        trial_config = TrialRunConfig(
            parameter_set_id=parameter_set_id,
            test_case=test_case,
            experiment_config=experiment_config,
            parent_config=self.parent_run_config,
            trial_number=self.total_trial_runs,
        )
        self.active_trial_run = mlflow.start_run(
            run_name=trial_config.trial_name,
            nested=True,
            tags=trial_config.get_start_tags(self._utc_now()),
        )
        trial_config.mlflow_run_id = self.active_trial_run.info.run_id
        self.current_trial_config = trial_config

        self._log_trial_setup(trial_config)

        return trial_config


    def end_trial(self, cross_val_performance: CrossValidationPerformance) -> None:
        """Close the currently active mlflow trial run."""
        if self.active_trial_run is None:
            raise RuntimeError("Cannot end an mlflow trial because no trial run is active.")
        if self.current_trial_config is None:
            raise RuntimeError("Cannot end an mlflow trial because no trial config is active.")

        self._record_current_trial_results(cross_val_performance)
        self._close_current_trial()

    def _log_trial_setup(self, trial_config: TrialRunConfig) -> None:
        """Log dataset, feature, and model parameters for a trial."""
        self._log_params(trial_config.get_dataset_params())
        self._log_params(trial_config.get_feature_params())
        self._log_params(trial_config.get_model_params())

    def _record_current_trial_results(
        self,
        cross_val_performance: CrossValidationPerformance,
    ) -> None:
        """Record and log aggregate metrics and dataset sizes for the active trial."""
        if self.current_trial_config is None:
            raise RuntimeError("Cannot record trial results because no trial config is active.")

        performance, performance_std = cross_val_performance.get_final_performance()
        model_data = cross_val_performance.median_data_set
        if model_data is not None:
            dataset_sizes = model_data.get_dataset_sizes()
            self.current_trial_config.record_dataset_sizes(dataset_sizes)
            self._log_params(self.current_trial_config.get_dataset_size_params())

        self.current_trial_config.record_metrics(performance, performance_std)
        mlflow.log_metrics(self.current_trial_config.get_metrics_with_std())

    def _close_current_trial(self) -> None:
        """Close the active trial run and retain its completed trial config."""
        if self.current_trial_config is None:
            raise RuntimeError("Cannot close an mlflow trial because no trial config is active.")

        mlflow.set_tag("run_finished_at_utc", self._utc_now())
        mlflow.end_run(status="FINISHED")
        self.trial_run_configs.append(self.current_trial_config)
        self.active_trial_run = None
        self.current_trial_config = None

    @staticmethod
    def _utc_now() -> str:
        """Return the current UTC timestamp in ISO-8601 format."""
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _log_params(cls, params: dict[str, Any]) -> None:
        """Format and log parameters to the active MLflow run."""
        mlflow.log_params(cls._format_params(params))

    @classmethod
    def _format_params(cls, params: dict[str, Any]) -> dict[str, str | int | float]:
        """Convert parameter values into MLflow-compatible scalar values."""
        return {
            key: cls._format_param_value(value)
            for key, value in params.items()
        }

    @staticmethod
    def _format_param_value(value: Any) -> str | int | float:
        """Convert one parameter value into a stable MLflow representation."""
        if value is None:
            return "not_applicable"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        if isinstance(value, (int, float, str)):
            return value
        return str(value)

    def _find_trial_config(self, trial_run_id: str) -> TrialRunConfig | None:
        """Return the trial config for an active or completed Mlflow run ID."""
        if self.current_trial_config is not None and self.current_trial_config.mlflow_run_id == trial_run_id:
            return self.current_trial_config
        return next(
            (trial_config for trial_config in self.trial_run_configs if trial_config.mlflow_run_id == trial_run_id),
            None,
        )

