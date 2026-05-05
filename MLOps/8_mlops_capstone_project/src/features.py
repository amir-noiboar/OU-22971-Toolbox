"""Feature engineering stubs for Green Taxi tip prediction.

TODO: Build one shared feature pipeline for training, evaluation, and inference.
"""

from __future__ import annotations


def build_features(raw_data: object, feature_spec: dict[str, object] | None = None) -> object:
    """Transform raw taxi data into model-ready features.

    TODO: Add datetime, location, numeric transforms, imputation, and schema alignment.
    """
    raise NotImplementedError("Feature engineering is not implemented yet.")


def infer_feature_spec(features: object) -> dict[str, object]:
    """Create a serializable feature specification from engineered features.

    TODO: Capture feature names, dtypes, ordering, and transformation metadata.
    """
    raise NotImplementedError("Feature spec inference is not implemented yet.")


def validate_feature_spec(features: object, feature_spec: dict[str, object]) -> None:
    """Validate engineered features against a saved feature specification.

    TODO: Enforce stable columns, order, dtypes, and missing-value expectations.
    """
    raise NotImplementedError("Feature spec validation is not implemented yet.")
