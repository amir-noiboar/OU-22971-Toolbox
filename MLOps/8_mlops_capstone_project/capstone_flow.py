"""Metaflow skeleton for the Unit 8 MLOps capstone.

Manual workflow:
new batch -> integrity gate -> feature engineering -> champion evaluation ->
optional retraining -> promotion gate -> batch inference.

TODO: Implement the placeholder calls with the real capstone business logic.
"""

from __future__ import annotations

from metaflow import FlowSpec, Parameter, step

from src import data_io, decisions, features, inference, mlflow_utils, modeling
from src import nannyml_checks, quality, registry


class CapstoneFlow(FlowSpec):
    """Green Taxi tip prediction monitoring and retraining flow skeleton."""

    reference_path = Parameter(
        "reference_path",
        help="Reference parquet path used for baseline checks and bootstrap training.",
        required=True,
    )
    batch_path = Parameter(
        "batch_path",
        help="Current batch parquet path to monitor, evaluate, and score.",
        required=True,
    )
    extra_train_paths = Parameter(
        "extra_train_paths",
        default="",
        help="Optional comma-separated parquet paths for candidate retraining windows.",
    )
    model_name = Parameter(
        "model_name",
        default="green_taxi_tip_model",
        help="MLflow registered model name.",
    )
    experiment_name = Parameter(
        "experiment_name",
        default="8_green_taxi_capstone",
        help="MLflow experiment name.",
    )
    tracking_uri = Parameter(
        "tracking_uri",
        default="http://127.0.0.1:5000",
        help="MLflow tracking URI.",
    )
    min_improvement = Parameter(
        "min_improvement",
        default=0.01,
        help="Minimum relative RMSE improvement required for promotion.",
    )
    rmse_increase_threshold = Parameter(
        "rmse_increase_threshold",
        default=0.10,
        help="Placeholder relative RMSE increase threshold for retrain decisions.",
    )
    use_optuna = Parameter(
        "use_optuna",
        default=False,
        help="Whether candidate training should use optional Optuna tuning.",
    )
    n_trials = Parameter(
        "n_trials",
        default=20,
        help="Number of Optuna trials when tuning is enabled.",
    )
    fail_at_step = Parameter(
        "fail_at_step",
        default="",
        help="Optional step name that intentionally fails for failure/resume demos.",
    )

    def _maybe_fail(self, step_name: str) -> None:
        """Raise when fail_at_step matches a step name.

        TODO: Use this for a Metaflow failure/resume demo.
        """
        if self.fail_at_step == step_name:
            raise RuntimeError(f"Intentional failure at step: {step_name}")

    @step
    def start(self) -> None:
        """Initialize run configuration and MLflow tracking."""
        self._maybe_fail("start")
        self.extra_train_path_list = [
            path.strip() for path in str(self.extra_train_paths).split(",") if path.strip()
        ]

        # TODO: Log flow parameters and common tags to MLflow.
        mlflow_utils.initialize_mlflow(
            tracking_uri=str(self.tracking_uri),
            experiment_name=str(self.experiment_name),
        )
        self.next(self.load_data)

    @step
    def load_data(self) -> None:
        """Load reference and current batch data."""
        self._maybe_fail("load_data")

        # TODO: Preserve dataset lineage in MLflow for reference, batch, and extra train data.
        self.reference_resolved_path = data_io.resolve_path(self.reference_path)
        self.batch_resolved_path = data_io.resolve_path(self.batch_path)
        self.reference_raw = data_io.load_parquet(self.reference_resolved_path)
        self.batch_raw = data_io.load_parquet(self.batch_resolved_path)
        self.extra_train_raw = data_io.load_optional_parquets(self.extra_train_path_list)
        mlflow_utils.log_dataset_lineage(
            {
                "reference_path": str(self.reference_resolved_path),
                "batch_path": str(self.batch_resolved_path),
                "extra_train_paths": self.extra_train_path_list,
            }
        )
        self.next(self.raw_integrity_gate)

    @step
    def raw_integrity_gate(self) -> None:
        """Run fail-fast raw checks and warning-only NannyML checks."""
        self._maybe_fail("raw_integrity_gate")

        # TODO: Log raw integrity artifacts, metrics, and tables.
        self.hard_integrity = quality.run_hard_integrity_checks(
            self.batch_raw,
            labels_required=True,
        )
        self.hard_integrity_passed = bool(self.hard_integrity.get("hard_passed", False))
        mlflow_utils.log_tables("raw_integrity", self.hard_integrity.get("tables", {}))
        mlflow_utils.log_metrics("raw_integrity", self.hard_integrity.get("metrics", {}))

        if not self.hard_integrity_passed:
            # TODO: Log decision.json always, including action="reject_batch".
            self.final_decision = decisions.build_reject_decision(
                reason="hard_integrity_failure",
                evidence=self.hard_integrity,
            )
            mlflow_utils.log_decision_json(self.final_decision)
            self.next(self.end)
        else:
            # TODO: Log NannyML soft-gate artifacts and set integrity_warn=true when needed.
            self.soft_integrity = nannyml_checks.run_soft_monitoring_checks(
                reference_data=self.reference_raw,
                current_data=self.batch_raw,
            )
            mlflow_utils.set_decision_tags(
                {"integrity_warn": str(self.soft_integrity.get("integrity_warn", False)).lower()}
            )
            mlflow_utils.log_tables("nannyml", self.soft_integrity.get("tables", {}))
            self.next(self.feature_engineering)

    @step
    def feature_engineering(self) -> None:
        """Build stable features for reference, batch, and training windows."""
        self._maybe_fail("feature_engineering")

        # TODO: Reuse the same feature engineering for training, evaluation, and inference.
        self.reference_features = features.build_features(self.reference_raw)
        self.batch_features = features.build_features(self.batch_raw)
        self.extra_train_features = features.build_optional_training_features(self.extra_train_raw)
        self.feature_spec = features.build_feature_spec(self.reference_features)

        # TODO: Log feature_spec.json for schema debugging and inference alignment.
        mlflow_utils.log_artifact_dict(self.feature_spec, "feature_spec.json")
        self.next(self.load_or_bootstrap_champion)

    @step
    def load_or_bootstrap_champion(self) -> None:
        """Load the champion model or bootstrap the first champion."""
        self._maybe_fail("load_or_bootstrap_champion")

        # TODO: Try models:/<model_name>@champion.
        self.champion = registry.get_champion_by_alias(str(self.model_name))
        if self.champion is None:
            # TODO: Bootstrap champion if no @champion model exists.
            self.initial_model = modeling.train_initial_model(
                features=self.reference_features,
                use_optuna=bool(self.use_optuna),
                n_trials=int(self.n_trials),
            )
            self.champion = registry.bootstrap_champion(
                model=self.initial_model,
                model_name=str(self.model_name),
                metadata={"promotion_reason": "bootstrap"},
            )
        self.next(self.evaluate_champion)

    @step
    def evaluate_champion(self) -> None:
        """Evaluate champion on engineered batch features."""
        self._maybe_fail("evaluate_champion")

        # TODO: Evaluate champion on engineered batch features and log RMSE/MAE.
        self.champion_metrics = modeling.evaluate_regression(
            model=self.champion,
            features=self.batch_features,
        )
        mlflow_utils.log_metrics("champion", self.champion_metrics)
        self.next(self.decide_retrain)

    @step
    def decide_retrain(self) -> None:
        """Decide whether candidate retraining should run."""
        self._maybe_fail("decide_retrain")

        # TODO: Log decision.json always and set retrain_recommended tag.
        self.retrain_decision = decisions.build_retrain_decision(
            champion_metrics=self.champion_metrics,
            soft_integrity=self.soft_integrity,
            rmse_increase_threshold=float(self.rmse_increase_threshold),
        )
        self.retrain_needed = bool(self.retrain_decision.get("retrain_needed", False))
        mlflow_utils.set_decision_tags(
            {"retrain_recommended": str(self.retrain_needed).lower()}
        )
        mlflow_utils.log_decision_json(self.retrain_decision)

        if self.retrain_needed:
            self.next(self.retrain_candidate)
        else:
            self.next(self.skip_retrain)

    @step
    def retrain_candidate(self) -> None:
        """Train and register a candidate model."""
        self._maybe_fail("retrain_candidate")

        # TODO: Train candidate on the selected rolling or expanding training window.
        self.candidate_model = modeling.train_candidate_model(
            reference_features=self.reference_features,
            batch_features=self.batch_features,
            extra_train_features=self.extra_train_features,
            use_optuna=bool(self.use_optuna),
            n_trials=int(self.n_trials),
        )

        # TODO: Register candidate model with tags and validation_status=pending.
        self.candidate_version = registry.register_candidate(
            model=self.candidate_model,
            model_name=str(self.model_name),
            metadata={"decision_reason": self.retrain_decision.get("reason", "")},
        )
        self.next(self.evaluate_candidate)

    @step
    def skip_retrain(self) -> None:
        """Record the no-retrain path."""
        self._maybe_fail("skip_retrain")

        # TODO: Log final no-retrain decision evidence before batch inference.
        self.candidate_model = None
        self.candidate_version = ""
        self.candidate_metrics = {}
        self.promotion_decision = decisions.build_no_retrain_decision(
            reason=self.retrain_decision.get("reason", "retrain_not_recommended"),
            evidence=self.retrain_decision,
        )
        mlflow_utils.log_decision_json(self.promotion_decision)
        self.next(self.batch_inference)

    @step
    def evaluate_candidate(self) -> None:
        """Evaluate candidate on the same engineered batch features."""
        self._maybe_fail("evaluate_candidate")

        # TODO: Evaluate candidate on the same batch as champion.
        self.candidate_metrics = modeling.evaluate_regression(
            model=self.candidate_model,
            features=self.batch_features,
        )
        mlflow_utils.log_metrics("candidate", self.candidate_metrics)
        self.next(self.promotion_gate)

    @step
    def promotion_gate(self) -> None:
        """Decide whether to promote the candidate to champion."""
        self._maybe_fail("promotion_gate")

        # TODO: Apply promotion gate with min_improvement.
        # TODO: Add stability check to avoid one-batch overfit.
        # TODO: Prevent promotion without evaluation metrics.
        self.promotion_decision = decisions.build_promotion_decision(
            champion_metrics=self.champion_metrics,
            candidate_metrics=self.candidate_metrics,
            min_improvement=float(self.min_improvement),
            evidence={
                "hard_integrity": self.hard_integrity,
                "soft_integrity": self.soft_integrity,
                "candidate_version": self.candidate_version,
            },
        )
        self.promotion_recommended = bool(
            self.promotion_decision.get("promotion_recommended", False)
        )
        mlflow_utils.set_decision_tags(
            {"promotion_recommended": str(self.promotion_recommended).lower()}
        )

        if self.promotion_recommended:
            # TODO: Update @champion alias.
            # TODO: Tag previous champion as previous_champion.
            # TODO: Tag new champion as champion with promoted_at and promotion_reason.
            registry.promote_candidate(
                model_name=str(self.model_name),
                candidate_version=str(self.candidate_version),
                reason=str(self.promotion_decision.get("reason", "")),
            )

        # TODO: Log decision.json always with final promotion or rejection details.
        mlflow_utils.log_decision_json(self.promotion_decision)
        self.next(self.batch_inference)

    @step
    def batch_inference(self) -> None:
        """Run offline batch inference and log predictions."""
        self._maybe_fail("batch_inference")

        # TODO: Use the current champion after any alias flip.
        # TODO: Log predictions.parquet as a batch inference artifact.
        self.predictions = inference.run_batch_inference(
            model=self.champion,
            features=self.batch_features,
        )
        inference.write_predictions_parquet(self.predictions, "predictions.parquet")
        mlflow_utils.log_artifact_file("predictions.parquet")
        self.next(self.end)

    @step
    def end(self) -> None:
        """Finish the run and leave an auditable final state."""
        self._maybe_fail("end")

        # TODO: Ensure decision.json was logged on every terminal path.
        self.flow_status = "finished"


if __name__ == "__main__":
    CapstoneFlow()
