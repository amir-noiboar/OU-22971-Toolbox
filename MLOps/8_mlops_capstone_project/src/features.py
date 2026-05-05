"""Feature engineering stubs for Green Taxi tip prediction."""

from __future__ import annotations


def build_features(raw_data: object, feature_spec: dict[str, object] | None = None) -> object:
    """Create model-ready features from raw taxi data.

    TODO: Reuse this function consistently for training, evaluation, and inference.
    TODO: Add pickup hour, pickup day-of-week, and pickup month features.
    TODO: Add duration_min from pickup and dropoff timestamps.
    TODO: Add location features from pickup/dropoff zone IDs or joined zone metadata.
    TODO: Add clipping/log transforms for heavy-tailed numeric fields.
    TODO: Produce and enforce a stable feature schema.
    TODO: Exclude leakage columns such as tip_amount and total_amount.
    TODO: Apply credit-card-only filtering if the final modeling policy uses it.
    """
    raise NotImplementedError("Feature engineering is not implemented yet.")


def build_feature_spec(features: object) -> dict[str, object]:
    """Build a serializable feature schema specification.

    TODO: Store feature names, order, dtypes, target policy, and transform metadata.
    """
    raise NotImplementedError("Feature spec creation is not implemented yet.")


def align_to_feature_spec(features: object, feature_spec: dict[str, object]) -> object:
    """Align features to a saved feature specification.

    TODO: Reorder columns, add missing defaults, reject incompatible dtypes, and log warnings.
    """
    raise NotImplementedError("Feature schema alignment is not implemented yet.")


def build_optional_training_features(raw_datasets: list[object]) -> list[object]:
    """Feature-engineer optional extra training datasets.

    TODO: Apply the same transformations and schema policy used for reference and batch data.
    """
    raise NotImplementedError("Optional training feature engineering is not implemented yet.")
