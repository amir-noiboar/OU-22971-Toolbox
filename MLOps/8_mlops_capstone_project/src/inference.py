"""Batch inference stubs for the Unit 8 capstone."""

from __future__ import annotations


def run_batch_inference(model: object, features: object) -> object:
    """Generate batch predictions for the current data slice.

    TODO: Load the current champion after any promotion alias flip.
    TODO: Return prediction rows with stable identifiers and metadata.
    """
    raise NotImplementedError("Batch inference is not implemented yet.")


def write_predictions_parquet(predictions: object, output_path: str) -> None:
    """Write batch predictions to a parquet artifact.

    TODO: Write and log predictions.parquet as an MLflow artifact.
    """
    raise NotImplementedError("Prediction parquet writing is not implemented yet.")
