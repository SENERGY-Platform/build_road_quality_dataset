from __future__ import annotations

from datetime import datetime, timezone
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.data.model_data import ModelData
from src.model_building.data.data_test_cases import DataTestCase
from src.model_building.logging.mlflow_types import ParentRunConfig, TrialRunConfig
from src.model_building.models.metrics import CrossValidationPerformance
from src.model_building.models.models_ann import TwoPhaseANNModel


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
        best_params = self.parent_run_config.get_best_params()
        if best_params:
            mlflow.log_params(self._format_params(best_params))
        best_metrics = self.parent_run_config.get_best_metrics()
        if best_metrics:
            mlflow.log_metrics(best_metrics)
        self._log_best_trial_artifacts()
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
        self._consider_current_trial_as_best(cross_val_performance)

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

    # ------------------------------------------------------------------
    # Winner selection
    # ------------------------------------------------------------------
    def _consider_current_trial_as_best(
        self,
        cross_val_performance: CrossValidationPerformance,
    ) -> None:
        """Compare the active trial with the current best trial."""
        if self.current_trial_config is None:
            raise RuntimeError("Cannot select a best trial because no trial config is active.")
        if self.current_trial_config.mlflow_run_id is None:
            raise RuntimeError("Cannot select a best trial before the trial has an mlflow run ID.")
        if cross_val_performance.median_model is None:
            raise ValueError("Cannot select a best trial without a median fitted model.")
        if cross_val_performance.median_data_set is None:
            raise ValueError("Cannot select a best trial without median model data.")

        self.consider_as_best(
            model=cross_val_performance.median_model,
            model_data=cross_val_performance.median_data_set,
            trial_run_id=self.current_trial_config.mlflow_run_id,
            metrics=self.current_trial_config.get_metrics_with_std(),
            model_params=self.current_trial_config.get_model_params_without_prefix(),
        )

    def consider_as_best(
        self,
        model: Any,
        model_data: ModelData,
        trial_run_id: str,
        metrics: dict[str, float],
        model_params: dict[str, Any],
    ) -> None:
        """
        Record a trial as the parent winner when it beats the current best trial.

        Ranking is higher macro F1 first, then lower MAE for exact macro F1 ties.
        The model and model data are retained in memory and logged once when
        the parent run is finalised.
        """
        if not self._is_better(metrics, self.parent_run_config.best_trial_metrics):
            return

        trial_config = self._find_trial_config(trial_run_id)
        if trial_config is None:
            raise ValueError(f"No trial config exists for mlflow run ID {trial_run_id}.")
        if trial_config.metrics is None:
            raise ValueError(f"Trial {trial_run_id} has no recorded metrics.")

        self.current_best_trial_run = trial_config
        self.parent_run_config.record_best_trial(
            trial_config=trial_config,
            model=model,
            model_data=model_data,
            metrics=metrics,
            model_params=model_params,
        )

    @staticmethod
    def _is_better(
        candidate_metrics: dict[str, float],
        current_best_metrics: Optional[dict[str, float]],
    ) -> bool:
        """Return whether the candidate metrics beat the current best metrics."""
        if "f1_macro" not in candidate_metrics:
            raise ValueError("Cannot compare mlflow trials without an 'f1_macro' metric.")
        if current_best_metrics is None or not current_best_metrics:
            return True
        if "f1_macro" not in current_best_metrics:
            return True

        candidate_f1 = candidate_metrics["f1_macro"]
        current_best_f1 = current_best_metrics["f1_macro"]
        if candidate_f1 != current_best_f1:
            return candidate_f1 > current_best_f1

        return candidate_metrics.get("mae", float("+inf")) < current_best_metrics.get("mae", float("+inf"))

    # ------------------------------------------------------------------
    # Parent artifacts
    # ------------------------------------------------------------------
    def _log_best_trial_artifacts(self) -> None:
        """Log the best trial's fitted model and median model data to the parent run."""
        best_model = self.parent_run_config.best_trial_model

        if best_model is not None:
            self._log_model_artifact(best_model, artifact_path="best_model")

    def _log_model_artifact(self, model: Any, artifact_path: str) -> None:
        """Log a fitted model using the strongest MLflow representation available."""
        model_params = self._format_params(self.parent_run_config.get_best_params())
        model_metadata = self._get_best_model_artifact_metadata()
        model_data = self.parent_run_config.best_trial_model_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            extra_files = self._write_best_model_extra_files(Path(tmp_dir), model_data)

            if isinstance(model, XGBRegressor):
                mlflow.xgboost.log_model(
                    model,
                    name=artifact_path,
                    params=model_params,
                    metadata=model_metadata,
                    extra_files=extra_files,
                )
                return

            if isinstance(model, Pipeline):
                mlflow.sklearn.log_model(
                    model,
                    name=artifact_path,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                    params=model_params,
                    metadata=model_metadata,
                    extra_files=extra_files,
                )
                return

            if isinstance(model, TwoPhaseANNModel):
                model_path = Path(tmp_dir) / "two_phase_ann_model.pkl"
                with model_path.open("wb") as model_file:
                    pickle.dump(model, model_file)
                mlflow.log_artifact(str(model_path), artifact_path=artifact_path)
                self._log_extra_files(extra_files, artifact_path=artifact_path)
                return

            model_path = Path(tmp_dir) / "model.pkl"
            with model_path.open("wb") as model_file:
                pickle.dump(model, model_file)
            mlflow.log_artifact(str(model_path), artifact_path=artifact_path)
            self._log_extra_files(extra_files, artifact_path=artifact_path)

    @staticmethod
    def _log_extra_files(extra_files: list[str], artifact_path: str) -> None:
        """Log prepared extra files for non-flavor model artifacts."""
        for extra_file in extra_files:
            extra_path = Path(extra_file)
            if extra_path.is_dir():
                mlflow.log_artifacts(str(extra_path), artifact_path=f"{artifact_path}/{extra_path.name}")
            else:
                mlflow.log_artifact(str(extra_path), artifact_path=artifact_path)

    def _write_best_model_extra_files(self, output_dir: Path, model_data: ModelData | None) -> list[str]:
        """Write files that should be included inside the logged model package."""
        reproducibility_path = output_dir / "reproducibility.json"
        with reproducibility_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(self._get_best_model_artifact_metadata(), metadata_file, indent=2)

        extra_files = [str(reproducibility_path)]
        if model_data is not None:
            model_data_dir = output_dir / "model_data"
            model_data_dir.mkdir()
            self._write_model_data_parts(model_data, model_data_dir)
            extra_files.append(str(model_data_dir))
        return extra_files

    @staticmethod
    def _write_model_data_parts(model_data: ModelData, output_dir: Path) -> None:
        """Write model-data parts as parquet plus a small metadata artifact."""
        metadata = {
            "test_case_id": model_data.test_case_id,
            "random_state": model_data.random_state,
            "dataset_sizes": MlflowLogger._format_params(model_data.get_dataset_sizes().__dict__),
        }
        metadata_path = output_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        data_frames = {
            "manual_train_x": model_data.manual_train_x,
            "osm_train_x": model_data.osm_train_x,
            "test_x": model_data.test_x,
            "manual_val_x": model_data.manual_val_x,
            "osm_val_x": model_data.osm_val_x,
        }
        labels = {
            "manual_train_y": model_data.manual_train_y,
            "osm_train_y": model_data.osm_train_y,
            "test_y": model_data.test_y,
            "manual_val_y": model_data.manual_val_y,
            "osm_val_y": model_data.osm_val_y,
        }

        for name, data_frame in data_frames.items():
            data_frame.to_parquet(output_dir / f"{name}.parquet")
        for name, series in labels.items():
            series.rename("label").to_frame().to_parquet(output_dir / f"{name}.parquet")

    def _get_best_model_artifact_metadata(self) -> dict[str, Any]:
        """Return model-level metadata needed to understand and reproduce the best run."""
        return {
            "best_trial_run_id": self.parent_run_config.best_trial_run_id,
            "best_trial_name": self.parent_run_config.best_trial_name,
            "best_trial_number": self.parent_run_config.best_trial_number,
            "params": self._format_params(self.parent_run_config.get_best_params()),
            "model_params": self._format_params(self.parent_run_config.best_trial_model_params),
            "metrics": self.parent_run_config.best_trial_metrics,
        }
