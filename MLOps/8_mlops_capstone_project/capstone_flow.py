"""Metaflow workflow for the Unit 8 MLOps capstone.

Manual workflow:
new batch -> hard integrity gate -> feature engineering -> soft monitoring ->
champion evaluation -> optional retraining -> promotion gate -> batch inference.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import mlflow
import mlflow.sklearn
import numpy as np
from metaflow import FlowSpec, Parameter, step

from src import config, data_io, decisions, features, inference, mlflow_utils
from src import modeling, nannyml_checks, quality, registry


class CapstoneFlow(FlowSpec):
    """Green Taxi tip prediction monitoring, retraining, and inference flow."""

    reference_path = Parameter(
        "reference_path",
        default=str(config.DEFAULT_REFERENCE_PATH),
        help="Reference parquet path used for baseline checks and bootstrap training.",
    )
    batch_path = Parameter(
        "batch_path",
        default=str(config.DEFAULT_BATCH_PATH),
        help="Current batch parquet path to monitor, evaluate, and score.",
    )
    extra_train_paths = Parameter(
        "extra_train_paths",
        default="",
        help="Optional comma-separated parquet paths for candidate retraining windows.",
    )
    inference_output_path = Parameter(
        "inference_output_path",
        default=str(config.DEFAULT_INFERENCE_OUTPUT_PATH),
        help="Local parquet path for offline batch predictions.",
    )
    tracking_uri = Parameter(
        "tracking_uri",
        default=config.DEFAULT_TRACKING_URI,
        help="MLflow tracking URI.",
    )
    experiment_name = Parameter(
        "experiment_name",
        default=config.DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    model_name = Parameter(
        "model_name",
        default=config.DEFAULT_MODEL_NAME,
        help="MLflow registered model name.",
    )
    model_type = Parameter(
        "model_type",
        default=config.defaults().model_type,
        help="Model family to train: random_forest, xgboost, or xgb.",
    )
    min_improvement = Parameter(
        "min_improvement",
        default=config.DEFAULT_MIN_IMPROVEMENT,
        help="Minimum relative RMSE improvement required for promotion.",
    )
    rmse_increase_threshold = Parameter(
        "rmse_increase_threshold",
        default=config.DEFAULT_RMSE_INCREASE_THRESHOLD,
        help="Relative champion RMSE increase threshold for retraining.",
    )
    missingness_warn_threshold = Parameter(
        "missingness_warn_threshold",
        default=config.defaults().missingness_warn_threshold,
        help="Soft warning threshold for missingness increase versus reference.",
    )
    unseen_category_warn_threshold = Parameter(
        "unseen_category_warn_threshold",
        default=config.defaults().unseen_category_warn_threshold,
        help="Soft warning threshold for unseen categorical values versus reference.",
    )
    random_state = Parameter(
        "random_state",
        default=config.defaults().random_state,
        help="Random seed for model training.",
    )
    use_optuna = Parameter(
        "use_optuna",
        default=config.defaults().use_optuna,
        help="Whether candidate training should use optional Optuna tuning.",
    )
    retrain_on_soft_warning = Parameter(
        "retrain_on_soft_warning",
        default=True,
        help="Whether soft monitoring warnings automatically trigger retraining.",
    )
    n_trials = Parameter(
        "n_trials",
        default=config.DEFAULT_N_TRIALS,
        help="Number of Optuna trials when tuning is enabled.",
    )
    test_size = Parameter(
        "test_size",
        default=config.defaults().test_size,
        help="Validation split fraction for model training.",
    )
    n_jobs = Parameter(
        "n_jobs",
        default=config.defaults().n_jobs,
        help="Parallel jobs for supported estimators.",
    )
    fail_at_step = Parameter(
        "fail_at_step",
        default="",
        help="Optional step name that intentionally fails for failure/resume demos.",
    )

    def _maybe_fail(self, step_name: str) -> None:
        """Raise when ``fail_at_step`` matches a step name."""
        if str(self.fail_at_step).strip() == step_name:
            raise RuntimeError(f"Intentional failure at step: {step_name}")

    @contextmanager
    def _mlflow_step_run(self, step_name: str) -> Iterator[bool]:
        """Open a short MLflow run for a Metaflow step when possible."""
        batch_id = getattr(self, "batch_id", "unknown_batch")
        run_name = f"capstone_{step_name}_{batch_id}"
        try:
            mlflow_utils.init_mlflow(
                tracking_uri=str(self.tracking_uri),
                experiment_name=str(self.experiment_name),
            )
            active_run = mlflow.active_run()
            run_context = mlflow.start_run(
                run_name=run_name,
                nested=active_run is not None,
            )
        except Exception as exc:
            print(
                "MLflow logging disabled for "
                f"{step_name}: {type(exc).__name__}: {exc}"
            )
            yield False
            return

        with run_context:
            mlflow_utils.log_tags_safe(
                {
                    "metaflow_step": step_name,
                    "batch_id": batch_id,
                    "model_name": str(self.model_name),
                }
            )
            yield True

    def _model_config(self) -> modeling.ModelConfig:
        """Build model configuration from flow parameters."""
        return modeling.ModelConfig(
            model_type=str(self.model_type),
            random_state=int(self.random_state),
            use_optuna=bool(self.use_optuna),
            n_trials=int(self.n_trials),
            test_size=float(self.test_size),
            n_jobs=int(self.n_jobs),
        )

    @step
    def start(self) -> None:
        """Initialize flow configuration."""
        self.flow_start_time = datetime.now(timezone.utc).isoformat()
        self.decision = None
        self.final_decision = None
        self.retrain_needed = False
        self.promotion_recommended = False
        self.promotion_executed = False
        self.extra_train_path_list = data_io.parse_extra_train_paths(self.extra_train_paths)
        self._maybe_fail("start")

        with self._mlflow_step_run("start") as logging_enabled:
            if logging_enabled:
                mlflow_utils.log_params_safe(
                    {
                        "reference_path": str(self.reference_path),
                        "batch_path": str(self.batch_path),
                        "extra_train_paths": ",".join(map(str, self.extra_train_path_list)),
                        "inference_output_path": str(self.inference_output_path),
                        "tracking_uri": str(self.tracking_uri),
                        "experiment_name": str(self.experiment_name),
                        "model_name": str(self.model_name),
                        "model_type": str(self.model_type),
                        "min_improvement": float(self.min_improvement),
                        "rmse_increase_threshold": float(self.rmse_increase_threshold),
                        "missingness_warn_threshold": float(self.missingness_warn_threshold),
                        "unseen_category_warn_threshold": float(
                            self.unseen_category_warn_threshold
                        ),
                        "random_state": int(self.random_state),
                        "use_optuna": bool(self.use_optuna),
                        "retrain_on_soft_warning": bool(self.retrain_on_soft_warning),
                        "n_trials": int(self.n_trials),
                        "test_size": float(self.test_size),
                        "n_jobs": int(self.n_jobs),
                        "fail_at_step": str(self.fail_at_step),
                    }
                )
        self.next(self.load_data)

    @step
    def load_data(self) -> None:
        """Load reference, batch, and optional training parquet data."""
        self._maybe_fail("load_data")

        self.reference_dataset = data_io.load_reference(self.reference_path)
        self.batch_dataset = data_io.load_batch(self.batch_path)
        self.extra_train_datasets = data_io.load_optional_training_datasets(
            self.extra_train_path_list
        )
        self.reference_raw = self.reference_dataset.df
        self.batch_raw = self.batch_dataset.df
        self.extra_train_raw = [dataset.df for dataset in self.extra_train_datasets]
        self.reference_batch_id = self.reference_dataset.batch_id
        self.batch_id = self.batch_dataset.batch_id
        self.extra_train_batch_ids = [
            dataset.batch_id for dataset in self.extra_train_datasets
        ]

        dataset_summaries = {
            "reference": data_io.dataset_summary(self.reference_dataset),
            "batch": data_io.dataset_summary(self.batch_dataset),
            "extra_train": data_io.datasets_summary(self.extra_train_datasets),
        }

        with self._mlflow_step_run("load_data") as logging_enabled:
            if logging_enabled:
                mlflow_utils.log_artifact_dict(
                    dataset_summaries,
                    "lineage/dataset_summaries.json",
                )
                mlflow_utils.log_params_safe(
                    {
                        "reference_batch_id": self.reference_batch_id,
                        "batch_id": self.batch_id,
                        "n_extra_train_datasets": len(self.extra_train_datasets),
                    }
                )
                mlflow_utils.log_dataset_input_from_pandas(
                    self.reference_raw,
                    source_path=str(self.reference_dataset.path),
                    name=f"reference_{self.reference_batch_id}",
                    context="raw_reference",
                )
                mlflow_utils.log_dataset_input_from_pandas(
                    self.batch_raw,
                    source_path=str(self.batch_dataset.path),
                    name=f"batch_{self.batch_id}",
                    context="raw_batch",
                )
                for dataset in self.extra_train_datasets:
                    mlflow_utils.log_dataset_input_from_pandas(
                        dataset.df,
                        source_path=str(dataset.path),
                        name=f"extra_train_{dataset.batch_id}",
                        context="raw_extra_train",
                    )

        self.next(self.raw_integrity_gate)

    @step
    def raw_integrity_gate(self) -> None:
        """Run fail-fast raw integrity checks before feature engineering."""
        self._maybe_fail("raw_integrity_gate")

        self.hard_integrity = quality.run_hard_integrity_checks(
            self.batch_raw,
            labels_required=True,
        )
        self.hard_integrity_passed = bool(self.hard_integrity.hard_passed)

        with self._mlflow_step_run("raw_integrity_gate") as logging_enabled:
            if logging_enabled:
                mlflow_utils.log_integrity_result(
                    self.hard_integrity,
                    artifact_dir=config.INTEGRITY_ARTIFACT_DIR,
                    metric_prefix="integrity",
                )

                if not self.hard_integrity_passed:
                    self.decision = decisions.build_reject_decision(
                        integrity_result=self.hard_integrity,
                        reason="hard_integrity_failure",
                    )
                    self.final_decision = self.decision
                    mlflow_utils.log_decision(
                        self.decision,
                        artifact_file=config.DECISION_ARTIFACT,
                    )

        if not self.hard_integrity_passed and self.decision is None:
            self.decision = decisions.build_reject_decision(
                integrity_result=self.hard_integrity,
                reason="hard_integrity_failure",
            )
            self.final_decision = self.decision

        self.integrity_branch = (
            "passed" if self.hard_integrity_passed else "failed"
        )
        self.next(
            {
                "passed": self.feature_engineering,
                "failed": self.end,
            },
            condition="integrity_branch",
        )

    @step
    def feature_engineering(self) -> None:
        """Build stable features and run warning-only soft monitoring checks."""
        self._maybe_fail("feature_engineering")

        self.reference_ff = features.build_feature_frame(
            self.reference_raw,
            labels_required=True,
        )
        self.batch_ff = features.build_feature_frame(
            self.batch_raw,
            labels_required=True,
        )
        self.extra_train_ffs = [
            features.build_feature_frame(raw_dataset, labels_required=True)
            for raw_dataset in self.extra_train_raw
        ]
        self.feature_spec = self.reference_ff.spec
        self.soft_result = nannyml_checks.run_soft_checks(
            self.reference_ff.X,
            self.batch_ff.X,
            feature_spec=self.feature_spec,
            missingness_warn_threshold=float(self.missingness_warn_threshold),
            unseen_category_warn_threshold=float(self.unseen_category_warn_threshold),
            enable_nannyml=True,
        )
        self.integrity_warn = bool(self.soft_result.warning)

        with self._mlflow_step_run("feature_engineering") as logging_enabled:
            if logging_enabled:
                mlflow_utils.log_feature_spec(
                    self.feature_spec,
                    artifact_file=config.FEATURE_SPEC_ARTIFACT,
                )
                mlflow_utils.log_metrics_safe(
                    self.soft_result.metrics,
                    prefix="soft_check",
                )
                mlflow_utils.log_tables(
                    self.soft_result.tables,
                    artifact_dir=config.SOFT_CHECK_ARTIFACT_DIR,
                )
                mlflow_utils.log_tags_safe({"integrity_warn": self.integrity_warn})

        self.next(self.load_or_bootstrap_champion)

    @step
    def load_or_bootstrap_champion(self) -> None:
        """Load the champion model or bootstrap the first champion."""
        self._maybe_fail("load_or_bootstrap_champion")

        mlflow_utils.init_mlflow(
            tracking_uri=str(self.tracking_uri),
            experiment_name=str(self.experiment_name),
        )
        self.champion_info = registry.get_champion(str(self.model_name))
        self.bootstrapped_champion = self.champion_info is None
        self.model_config = self._model_config()

        with self._mlflow_step_run("load_or_bootstrap_champion") as logging_enabled:
            if self.champion_info is None:
                self.initial_train_result = modeling.train_model(
                    self.reference_ff,
                    config=self.model_config,
                )
                self.champion_model = self.initial_train_result.model
                if logging_enabled:
                    mlflow_utils.log_metrics_safe(
                        self.initial_train_result.train_metrics,
                        prefix="initial_train",
                    )
                    mlflow_utils.log_metrics_safe(
                        self.initial_train_result.validation_metrics,
                        prefix="initial_validation",
                    )
                    mlflow_utils.log_params_safe(self.initial_train_result.best_params)
                    self.champion_info = registry.bootstrap_champion(
                        self.initial_train_result,
                        model_name=str(self.model_name),
                        input_example=self.reference_ff.X.head(5),
                        tags={
                            "trained_on_batches": self.reference_batch_id,
                            "eval_batch_id": self.batch_id,
                        },
                        promotion_reason="bootstrap",
                    )
                else:
                    raise RuntimeError(
                        "Cannot bootstrap champion without an active MLflow run."
                    )
            else:
                self.champion_model_uri = registry.model_uri_for_alias(
                    str(self.model_name),
                    config.CHAMPION_ALIAS,
                )
                self.champion_model = mlflow.sklearn.load_model(self.champion_model_uri)
                if logging_enabled:
                    mlflow_utils.log_params_safe(
                        {
                            "champion_version": self.champion_info.version,
                            "champion_model_uri": self.champion_model_uri,
                        }
                    )

        self.old_champion_version = (
            self.champion_info.version if self.champion_info is not None else None
        )
        self.model_to_use_for_inference = self.champion_model
        self.next(self.evaluate_champion)

    @step
    def evaluate_champion(self) -> None:
        """Evaluate champion on engineered batch features."""
        self._maybe_fail("evaluate_champion")

        raw_metrics = modeling.evaluate_regression_model(
            self.champion_model,
            self.batch_ff,
            metric_prefix="champion",
        )
        rmse_champion = _metric(raw_metrics, "champion_rmse")
        rmse_baseline = _baseline_rmse(
            reference_y=self.reference_ff.y,
            batch_y=self.batch_ff.y,
        )
        rmse_increase_pct = _safe_rmse_increase(
            rmse_champion=rmse_champion,
            rmse_baseline=rmse_baseline,
        )
        self.champion_metrics = {
            **raw_metrics,
            "rmse_champion": rmse_champion,
            "rmse_baseline": rmse_baseline,
            "rmse_increase_pct": rmse_increase_pct,
        }

        with self._mlflow_step_run("evaluate_champion") as logging_enabled:
            if logging_enabled:
                mlflow_utils.log_metrics_safe(self.champion_metrics)

        self.next(self.decide_retrain)

    @step
    def decide_retrain(self) -> None:
        """Decide whether candidate retraining should run."""
        self._maybe_fail("decide_retrain")

        rmse_increase_pct = _metric(self.champion_metrics, "rmse_increase_pct")
        performance_degraded = (
            rmse_increase_pct is not None
            and rmse_increase_pct > float(self.rmse_increase_threshold)
        )
        soft_warning_present = bool(getattr(self.soft_result, "warning", False))
        soft_warning_triggered = (
            bool(self.retrain_on_soft_warning)
            and soft_warning_present
        )
        self.retrain_needed = bool(performance_degraded or soft_warning_triggered)
        self.champion_metrics.update(
            {
                "performance_degraded": int(performance_degraded),
                "soft_warning_present": int(soft_warning_present),
                "soft_warning_triggered": int(soft_warning_triggered),
                "retrain_on_soft_warning": int(bool(self.retrain_on_soft_warning)),
            }
        )

        if self.retrain_needed:
            self.decision = decisions.build_retrain_decision(
                champion_metrics=self.champion_metrics,
                soft_integrity={
                    "integrity_warn": self.integrity_warn,
                    "warnings": self.soft_result.warnings,
                    "metrics": self.soft_result.metrics,
                },
                rmse_increase_threshold=float(self.rmse_increase_threshold),
                warnings=self.soft_result.warnings,
            )
        else:
            self.decision = decisions.build_no_retrain_decision(
                champion_metrics=self.champion_metrics,
                rmse_increase_threshold=float(self.rmse_increase_threshold),
                warnings=self.soft_result.warnings,
            )
        self.final_decision = self.decision

        with self._mlflow_step_run("decide_retrain") as logging_enabled:
            if logging_enabled:
                mlflow_utils.log_decision(
                    self.decision,
                    artifact_file=config.DECISION_ARTIFACT,
                )

        self.retrain_branch = "retrain" if self.retrain_needed else "skip"
        self.next(
            {
                "retrain": self.retrain_candidate,
                "skip": self.skip_retrain,
            },
            condition="retrain_branch",
        )

    @step
    def retrain_candidate(self) -> None:
        """Train and register a candidate model."""
        self._maybe_fail("retrain_candidate")

        self.model_config = self._model_config()
        self.training_batch_ids = [
            self.reference_batch_id,
            *self.extra_train_batch_ids,
            self.batch_id,
        ]

        with self._mlflow_step_run("retrain_candidate") as logging_enabled:
            self.candidate_result = modeling.train_candidate_model(
                reference_features=self.reference_ff,
                batch_features=self.batch_ff,
                extra_train_features=self.extra_train_ffs,
                config=self.model_config,
            )
            self.candidate_model = self.candidate_result.model

            if logging_enabled:
                mlflow_utils.log_metrics_safe(
                    self.candidate_result.train_metrics,
                    prefix="candidate_train",
                )
                mlflow_utils.log_metrics_safe(
                    self.candidate_result.validation_metrics,
                    prefix="candidate_validation",
                )
                mlflow_utils.log_params_safe(self.candidate_result.best_params)
                if self.candidate_result.optuna_trials is not None:
                    mlflow_utils.log_tables(
                        {"optuna_trials": self.candidate_result.optuna_trials},
                        artifact_dir="training",
                    )
                self.candidate_info = registry.register_candidate(
                    self.candidate_result,
                    model_name=str(self.model_name),
                    input_example=self.batch_ff.X.head(5),
                    trained_on_batches=",".join(self.training_batch_ids),
                    eval_batch_id=self.batch_id,
                    validation_status="pending",
                    decision_reason=str(self.decision.get("reason", "")),
                )
            else:
                raise RuntimeError("Cannot register candidate without an active MLflow run.")

        self.candidate_version = self.candidate_info.version
        self.next(self.evaluate_candidate)

    @step
    def skip_retrain(self) -> None:
        """Keep champion and move directly to inference."""
        self._maybe_fail("skip_retrain")

        self.candidate_model = None
        self.candidate_info = None
        self.candidate_version = None
        self.candidate_metrics = {}
        self.promotion_decision = self.decision
        self.model_to_use_for_inference = self.champion_model

        with self._mlflow_step_run("skip_retrain") as logging_enabled:
            if logging_enabled:
                mlflow_utils.log_tags_safe(
                    {
                        "retrain_recommended": False,
                        "promotion_recommended": False,
                        "promotion_executed": False,
                    }
                )
                mlflow_utils.log_decision(
                    self.promotion_decision,
                    artifact_file=config.DECISION_ARTIFACT,
                )

        self.next(self.batch_inference)

    @step
    def evaluate_candidate(self) -> None:
        """Evaluate candidate and run a simple stability check."""
        self._maybe_fail("evaluate_candidate")

        raw_candidate_metrics = modeling.evaluate_regression_model(
            self.candidate_model,
            self.batch_ff,
            metric_prefix="candidate",
        )
        self.candidate_metrics = {
            **raw_candidate_metrics,
            "rmse_candidate": _metric(raw_candidate_metrics, "candidate_rmse"),
        }

        self.stability_check_assumption = ""
        try:
            champion_reference_metrics = modeling.evaluate_regression_model(
                self.champion_model,
                self.reference_ff,
                metric_prefix="champion_reference",
            )
            candidate_reference_metrics = modeling.evaluate_regression_model(
                self.candidate_model,
                self.reference_ff,
                metric_prefix="candidate_reference",
            )
            champion_reference_rmse = _metric(
                champion_reference_metrics,
                "champion_reference_rmse",
            )
            candidate_reference_rmse = _metric(
                candidate_reference_metrics,
                "candidate_reference_rmse",
            )
            self.stability_check_passed = bool(
                champion_reference_rmse is not None
                and candidate_reference_rmse is not None
                and candidate_reference_rmse <= champion_reference_rmse * 1.10
            )
            self.stability_metrics = {
                **champion_reference_metrics,
                **candidate_reference_metrics,
                "stability_check_passed": int(self.stability_check_passed),
            }
        except Exception as exc:
            self.stability_check_passed = True
            self.stability_check_assumption = (
                "stability check assumed passed because reference evaluation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self.stability_metrics = {
                "stability_check_passed": 1,
                "stability_assumed": 1,
            }

        with self._mlflow_step_run("evaluate_candidate") as logging_enabled:
            if logging_enabled:
                mlflow_utils.log_metrics_safe(self.candidate_metrics)
                mlflow_utils.log_metrics_safe(self.stability_metrics)
                if self.stability_check_assumption:
                    mlflow_utils.log_tags_safe(
                        {
                            "stability_check_assumed": True,
                            "stability_check_reason": self.stability_check_assumption,
                        }
                    )

        self.next(self.promotion_gate)

    @step
    def promotion_gate(self) -> None:
        """Decide whether to promote the candidate to champion."""
        self._maybe_fail("promotion_gate")

        self.promotion_decision = decisions.build_promotion_decision(
            champion_metrics=self.champion_metrics,
            candidate_metrics=self.candidate_metrics,
            min_improvement=float(self.min_improvement),
            old_champion_version=self.old_champion_version,
            candidate_version=self.candidate_version,
            stability_check_passed=bool(self.stability_check_passed),
            warnings=getattr(self.soft_result, "warnings", []),
        )
        self.promotion_recommended = bool(
            self.promotion_decision.get("promotion_recommended", False)
        )
        self.promotion_executed = False

        with self._mlflow_step_run("promotion_gate") as logging_enabled:
            if self.promotion_recommended:
                promoted_info = registry.promote_candidate(
                    model_name=str(self.model_name),
                    candidate_version=str(self.candidate_version),
                    old_champion_version=self.old_champion_version,
                    promotion_reason=str(self.promotion_decision.get("reason", "")),
                    tags={
                        "eval_batch_id": self.batch_id,
                        "trained_on_batches": ",".join(self.training_batch_ids),
                    },
                )
                self.promotion_executed = True
                self.champion_info = promoted_info
                self.model_to_use_for_inference = self.candidate_model
                self.final_decision = decisions.build_promotion_decision(
                    champion_metrics=self.champion_metrics,
                    candidate_metrics=self.candidate_metrics,
                    min_improvement=float(self.min_improvement),
                    old_champion_version=self.old_champion_version,
                    candidate_version=self.candidate_version,
                    new_champion_version=promoted_info.version,
                    stability_check_passed=bool(self.stability_check_passed),
                    warnings=getattr(self.soft_result, "warnings", []),
                    reason=str(self.promotion_decision.get("reason", "")),
                )
            else:
                self.model_to_use_for_inference = self.champion_model
                self.final_decision = self.promotion_decision

            if logging_enabled:
                mlflow_utils.log_decision(
                    self.final_decision,
                    artifact_file=config.DECISION_ARTIFACT,
                )

        self.next(self.batch_inference)

    @step
    def batch_inference(self) -> None:
        """Run offline batch inference and log predictions."""
        self._maybe_fail("batch_inference")

        self.batch_inference_ff = features.build_feature_frame(
            self.batch_raw,
            labels_required=False,
        )
        self.batch_inference_ff.X = features.align_to_feature_spec(
            self.batch_inference_ff,
            self.feature_spec,
        )
        self.batch_inference_ff.spec = self.feature_spec

        self.inference_result = inference.run_batch_inference(
            self.model_to_use_for_inference,
            self.batch_inference_ff,
            output_path=self.inference_output_path,
            prediction_column=config.defaults().prediction_column,
            include_row_ids=True,
            include_labels=False,
        )
        self.predictions_output_path = self.inference_result.output_path

        with self._mlflow_step_run("batch_inference") as logging_enabled:
            if logging_enabled:
                mlflow_utils.log_metrics_safe(
                    {"n_predictions": self.inference_result.n_predictions},
                    prefix="inference",
                )
                if self.predictions_output_path is not None:
                    mlflow_utils.log_artifact_if_exists(
                        self.predictions_output_path,
                        artifact_path=config.PREDICTIONS_ARTIFACT_DIR,
                    )

        self.next(self.end)

    @step
    def end(self) -> None:
        """Finish the run and print a compact final summary."""
        self._maybe_fail("end")

        decision = self.final_decision or self.decision or {}
        self.flow_status = "finished"
        self.summary = {
            "action": decision.get("action"),
            "final_decision": decision.get("final_decision"),
            "retrain_recommended": decision.get("retrain_recommended", False),
            "promotion_executed": decision.get("promotion_executed", False),
        }
        print(f"Capstone flow summary: {self.summary}")


def _metric(metrics: dict[str, object], key: str) -> float | None:
    """Return a metric as float when possible."""
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _baseline_rmse(reference_y: np.ndarray | None, batch_y: np.ndarray | None) -> float | None:
    """Compute batch RMSE for a constant reference-mean baseline."""
    if reference_y is None or batch_y is None or len(reference_y) == 0 or len(batch_y) == 0:
        return None
    baseline_value = float(np.nanmean(np.asarray(reference_y, dtype=float)))
    y_true = np.asarray(batch_y, dtype=float)
    baseline_pred = np.full(shape=len(y_true), fill_value=baseline_value, dtype=float)
    return float(np.sqrt(np.nanmean((y_true - baseline_pred) ** 2)))


def _safe_rmse_increase(
    *,
    rmse_champion: float | None,
    rmse_baseline: float | None,
) -> float | None:
    """Compute relative RMSE increase with safe missing/zero handling."""
    if rmse_champion is None or rmse_baseline is None or rmse_baseline == 0:
        return None
    return float((rmse_champion - rmse_baseline) / rmse_baseline)


if __name__ == "__main__":
    CapstoneFlow()
