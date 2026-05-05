"""MLflow helper stubs for the Unit 8 capstone."""

from __future__ import annotations


def initialize_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Initialize MLflow tracking for a capstone run.

    TODO: Set tracking URI, experiment, and common run tags.
    """
    raise NotImplementedError("MLflow initialization is not implemented yet.")


def log_params(params: dict[str, object]) -> None:
    """Log run parameters to MLflow.

    TODO: Normalize values before logging.
    """
    raise NotImplementedError("MLflow parameter logging is not implemented yet.")


def set_decision_tags(tags: dict[str, str]) -> None:
    """Set decision-oriented MLflow tags.

    TODO: Include tags such as retrain_recommended and promotion_recommended.
    """
    raise NotImplementedError("MLflow tag logging is not implemented yet.")


def log_metrics(prefix: str, metrics: dict[str, float]) -> None:
    """Log scalar metrics with a namespace prefix.

    TODO: Skip missing or non-finite metrics safely.
    """
    raise NotImplementedError("MLflow metric logging is not implemented yet.")


def log_tables(prefix: str, tables: dict[str, object]) -> None:
    """Log small tables or table-like artifacts to MLflow.

    TODO: Use mlflow.log_table for JSON-friendly tables.
    """
    raise NotImplementedError("MLflow table logging is not implemented yet.")


def log_artifact_dict(payload: dict[str, object], artifact_file: str) -> None:
    """Log a dictionary artifact to MLflow.

    TODO: Use this for feature_spec.json and compact summaries.
    """
    raise NotImplementedError("MLflow dictionary artifact logging is not implemented yet.")


def log_artifact_file(path: str) -> None:
    """Log an existing file artifact to MLflow.

    TODO: Use this for predictions.parquet and report artifacts.
    """
    raise NotImplementedError("MLflow file artifact logging is not implemented yet.")


def log_decision_json(decision: dict[str, object]) -> None:
    """Log decision.json for the current run.

    TODO: Ensure decision.json is logged on every terminal path.
    """
    raise NotImplementedError("Decision artifact logging is not implemented yet.")


def log_dataset_lineage(datasets: dict[str, object]) -> None:
    """Log dataset lineage to MLflow.

    TODO: Use mlflow.log_input with LocalArtifactDatasetSource for each dataset.
    """
    raise NotImplementedError("Dataset lineage logging is not implemented yet.")
