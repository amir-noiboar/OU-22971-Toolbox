"""Offline batch inference utilities for the Unit 8 capstone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class InferenceResult:
    """Result returned by batch inference helpers."""

    predictions: pd.DataFrame
    output_path: Path | None
    n_predictions: int
    prediction_column: str = "predicted_tip_amount"


def predict_batch(
    model: object,
    feature_frame: object,
    *,
    prediction_column: str = "predicted_tip_amount",
    include_row_ids: bool = True,
    include_labels: bool = False,
) -> pd.DataFrame:
    """Run model predictions for a FeatureFrame-like object."""
    if not hasattr(model, "predict"):
        raise TypeError("model must provide a predict method.")

    X = getattr(feature_frame, "X", None)
    if not isinstance(X, pd.DataFrame):
        raise ValueError("feature_frame.X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("feature_frame.X is empty; cannot run batch inference.")

    raw_predictions = model.predict(X.copy())
    predictions = pd.Series(raw_predictions, name=prediction_column).astype(float)
    if len(predictions) != len(X):
        raise RuntimeError(
            f"Prediction length {len(predictions)} does not match feature rows {len(X)}."
        )

    output = pd.DataFrame(
        {
            "prediction_index": range(len(predictions)),
            prediction_column: predictions.to_numpy(dtype=float),
        }
    )

    row_ids = getattr(feature_frame, "row_ids", None)
    if include_row_ids and row_ids is not None:
        row_id_series = pd.Series(row_ids).reset_index(drop=True)
        if len(row_id_series) != len(predictions):
            raise ValueError("feature_frame.row_ids length must match predictions.")
        output.insert(1, "row_id", row_id_series)

    y = getattr(feature_frame, "y", None)
    if include_labels and y is not None:
        label_series = pd.Series(y, name="actual_tip_amount").astype(float).reset_index(drop=True)
        if len(label_series) != len(predictions):
            raise ValueError("feature_frame.y length must match predictions.")
        output["actual_tip_amount"] = label_series

    return output


def write_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write predictions to a parquet file and return the resolved path."""
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")

    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)
    return path


def run_batch_inference(
    model: object,
    feature_frame: object,
    *,
    output_path: str | Path | None = None,
    prediction_column: str = "predicted_tip_amount",
    include_row_ids: bool = True,
    include_labels: bool = False,
) -> InferenceResult:
    """Run batch inference and optionally persist predictions to parquet."""
    predictions = predict_batch(
        model,
        feature_frame,
        prediction_column=prediction_column,
        include_row_ids=include_row_ids,
        include_labels=include_labels,
    )
    written_path = write_predictions(predictions, output_path) if output_path is not None else None
    return InferenceResult(
        predictions=predictions,
        output_path=written_path,
        n_predictions=int(len(predictions)),
        prediction_column=prediction_column,
    )


def batch_predict(
    model: object,
    feature_frame: object,
    **kwargs: object,
) -> pd.DataFrame:
    """Compatibility alias for ``predict_batch``."""
    return predict_batch(model, feature_frame, **kwargs)


def write_predictions_parquet(predictions: object, output_path: str | Path) -> Path:
    """Compatibility wrapper for the current flow skeleton."""
    if isinstance(predictions, InferenceResult):
        predictions = predictions.predictions
    return write_predictions(predictions, output_path)
