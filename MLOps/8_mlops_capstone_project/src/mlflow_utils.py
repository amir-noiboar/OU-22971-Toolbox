"""MLflow utility stubs for the capstone workflow.

TODO: Centralize tracking setup and lightweight logging helpers.
"""

from __future__ import annotations


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configure MLflow tracking for a capstone run.

    TODO: Set the tracking URI, experiment, and common tags.
    """
    raise NotImplementedError("MLflow configuration is not implemented yet.")


def log_metrics(prefix: str, metrics: dict[str, float]) -> None:
    """Log scalar metrics with an optional namespace prefix.

    TODO: Normalize metric names and skip invalid values safely.
    """
    raise NotImplementedError("Metric logging is not implemented yet.")


def log_dict_artifact(payload: dict[str, object], artifact_path: str) -> None:
    """Log a dictionary payload as an MLflow artifact.

    TODO: Persist JSON artifacts such as feature specs and decisions.
    """
    raise NotImplementedError("Dictionary artifact logging is not implemented yet.")
