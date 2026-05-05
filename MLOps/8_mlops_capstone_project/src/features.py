"""Reusable feature engineering for Green Taxi tip prediction.

The functions in this module are deliberately pure: no MLflow logging, no path
resolution, and no model preprocessing. Modeling code should consume the
structured feature metadata here and decide how to impute, scale, or encode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


TARGET_NAME = "tip_amount"
PICKUP_DATETIME_COL = "lpep_pickup_datetime"
DROPOFF_DATETIME_COL = "lpep_dropoff_datetime"

NUMERIC_CANDIDATES = [
    "trip_distance",
    "fare_amount",
    "passenger_count",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "duration_min",
]

CATEGORICAL_CANDIDATES = [
    "VendorID",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "trip_type",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
]

LEAKAGE_COLUMNS = {TARGET_NAME, "total_amount"}
RAW_DATETIME_COLUMNS = {PICKUP_DATETIME_COL, DROPOFF_DATETIME_COL}
FILTER_ONLY_COLUMNS = {"payment_type"}

CLIP_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "trip_distance": (0.0, 200.0),
    "fare_amount": (0.0, 500.0),
    "passenger_count": (0.0, 10.0),
    "extra": (0.0, 100.0),
    "mta_tax": (0.0, 10.0),
    "tolls_amount": (0.0, 200.0),
    "improvement_surcharge": (0.0, 10.0),
    "congestion_surcharge": (0.0, 10.0),
    "duration_min": (0.0, 360.0),
}


@dataclass(frozen=True)
class FeatureSpec:
    """Schema metadata for capstone model features."""

    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    target_name: str = TARGET_NAME
    labels_available: bool = True


@dataclass
class FeatureFrame:
    """Feature matrix, optional labels, schema metadata, and row identifiers."""

    X: pd.DataFrame
    y: Optional[np.ndarray]
    spec: FeatureSpec
    row_ids: Optional[pd.Series] = None


def build_feature_frame(
    df_raw: pd.DataFrame,
    *,
    labels_required: bool = True,
    credit_card_only: bool = True,
) -> FeatureFrame:
    """Build a model-ready feature frame from raw Green Taxi records.

    The output separates continuous numeric columns from categorical/id columns
    so modeling code can use a ``ColumnTransformer`` with imputation and
    ``OneHotEncoder``. Missing numeric values remain ``np.nan``. Missing
    categorical values become the string ``"missing"``.
    """
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("df_raw must be a pandas DataFrame.")

    df = df_raw.copy()
    has_feature_source = _has_any_feature_source(df)
    df = _coerce_datetime_columns(df)
    df = _add_time_features(df)
    df = _add_duration_feature(df)
    df = _filter_credit_card_rows(df, credit_card_only=credit_card_only)

    row_ids = _make_row_ids(df)
    y, labels_available = _extract_target(df, labels_required=labels_required)

    numeric_features = _choose_available_features(df, NUMERIC_CANDIDATES)
    categorical_features = _choose_available_features(df, CATEGORICAL_CANDIDATES)
    feature_names = numeric_features + categorical_features

    if not feature_names or not has_feature_source:
        raise ValueError("No usable numeric or categorical features are available.")

    numeric_frame = _coerce_numeric_features(df, numeric_features)
    numeric_frame = _clip_numeric_features(numeric_frame)
    categorical_frame = _coerce_categorical_features(df, categorical_features)

    X = pd.concat([numeric_frame, categorical_frame], axis=1)
    X = X.loc[:, feature_names].copy()

    forbidden = LEAKAGE_COLUMNS | RAW_DATETIME_COLUMNS | FILTER_ONLY_COLUMNS
    leaked = sorted(set(X.columns) & forbidden)
    if leaked:
        raise RuntimeError(f"Leakage or non-feature columns reached X: {leaked}")

    spec = FeatureSpec(
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_name=TARGET_NAME,
        labels_available=labels_available,
    )
    return FeatureFrame(X=X, y=y, spec=spec, row_ids=row_ids)


def _coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw datetime columns when present."""
    out = df.copy()
    if PICKUP_DATETIME_COL in out.columns:
        out[PICKUP_DATETIME_COL] = pd.to_datetime(
            out[PICKUP_DATETIME_COL],
            errors="coerce",
        )
    if DROPOFF_DATETIME_COL in out.columns:
        out[DROPOFF_DATETIME_COL] = pd.to_datetime(
            out[DROPOFF_DATETIME_COL],
            errors="coerce",
        )
    return out


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pickup calendar features, using missing values when pickup is absent."""
    out = df.copy()
    if PICKUP_DATETIME_COL not in out.columns:
        out["pickup_hour"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["pickup_weekday"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["pickup_month"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        return out

    pickup = pd.to_datetime(out[PICKUP_DATETIME_COL], errors="coerce")
    out["pickup_hour"] = pickup.dt.hour.astype("Int64")
    out["pickup_weekday"] = pickup.dt.dayofweek.astype("Int64")
    out["pickup_month"] = pickup.dt.month.astype("Int64")
    return out


def _add_duration_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add trip duration in minutes, using missing values when datetimes are absent."""
    out = df.copy()
    if PICKUP_DATETIME_COL not in out.columns or DROPOFF_DATETIME_COL not in out.columns:
        out["duration_min"] = np.nan
        return out

    pickup = pd.to_datetime(out[PICKUP_DATETIME_COL], errors="coerce")
    dropoff = pd.to_datetime(out[DROPOFF_DATETIME_COL], errors="coerce")
    out["duration_min"] = (dropoff - pickup).dt.total_seconds() / 60.0
    return out


def _filter_credit_card_rows(df: pd.DataFrame, *, credit_card_only: bool) -> pd.DataFrame:
    """Keep credit-card rows when payment_type is available and filtering is enabled."""
    if not credit_card_only or "payment_type" not in df.columns:
        return df.copy()

    payment_type = pd.to_numeric(df["payment_type"], errors="coerce")
    return df.loc[payment_type == 1].copy()


def _extract_target(
    df: pd.DataFrame,
    *,
    labels_required: bool,
) -> tuple[Optional[np.ndarray], bool]:
    """Return numeric labels when available."""
    if TARGET_NAME not in df.columns:
        if labels_required:
            raise ValueError(f"Required target column is missing: {TARGET_NAME}")
        return None, False

    target = pd.to_numeric(df[TARGET_NAME], errors="coerce").fillna(0.0)
    return target.to_numpy(dtype=float), True


def _choose_available_features(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Select candidate features that are present after derivation."""
    return [column for column in candidates if column in df.columns]


def _has_any_feature_source(df: pd.DataFrame) -> bool:
    """Return whether raw data contains at least one known feature source."""
    raw_numeric = set(NUMERIC_CANDIDATES) - {"duration_min"}
    raw_categorical = set(CATEGORICAL_CANDIDATES) - {
        "pickup_hour",
        "pickup_weekday",
        "pickup_month",
    }
    known_sources = raw_numeric | raw_categorical | RAW_DATETIME_COLUMNS
    return bool(set(df.columns) & known_sources)


def _coerce_numeric_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Coerce numeric features to float while preserving missing values."""
    out = pd.DataFrame(index=df.index)
    for column in feature_names:
        out[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return out


def _clip_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic conservative clipping to numeric features."""
    out = df.copy()
    for column, (lower, upper) in CLIP_BOUNDS.items():
        if column in out.columns:
            out[column] = out[column].clip(lower=lower, upper=upper)
    return out


def _coerce_categorical_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Coerce categorical/id features to object strings with explicit missing values."""
    out = pd.DataFrame(index=df.index)
    for column in feature_names:
        out[column] = _coerce_categorical_series(df[column])
    return out


def _coerce_categorical_series(series: pd.Series) -> pd.Series:
    """Convert a single categorical/id series to stable string values."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing_numeric = numeric.dropna()

    if not non_missing_numeric.empty and np.isclose(
        non_missing_numeric,
        np.round(non_missing_numeric),
    ).all():
        values = numeric.round().astype("Int64").astype("string")
    else:
        values = series.astype("string")

    return values.fillna("missing").astype(object)


def _make_row_ids(df: pd.DataFrame) -> pd.Series:
    """Return the original row index after filtering as a Series."""
    return pd.Series(df.index, index=df.index, name="row_id")


def build_features(
    raw_data: pd.DataFrame,
    feature_spec: FeatureSpec | dict[str, object] | None = None,
) -> FeatureFrame:
    """Compatibility wrapper returning a ``FeatureFrame``.

    TODO: Modeling code can later pass a saved FeatureSpec and call a dedicated
    alignment helper before prediction.
    """
    labels_required = TARGET_NAME in raw_data.columns
    frame = build_feature_frame(raw_data, labels_required=labels_required)
    if feature_spec is None:
        return frame
    return FeatureFrame(
        X=align_to_feature_spec(frame.X, feature_spec),
        y=frame.y,
        spec=_as_feature_spec(feature_spec, labels_available=frame.spec.labels_available),
        row_ids=frame.row_ids,
    )


def build_feature_spec(features: FeatureFrame | pd.DataFrame) -> FeatureSpec:
    """Build or extract a feature schema specification."""
    if isinstance(features, FeatureFrame):
        return features.spec
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a FeatureFrame or pandas DataFrame.")

    numeric_features = [
        column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_features = [column for column in features.columns if column not in numeric_features]
    return FeatureSpec(
        feature_names=numeric_features + categorical_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def align_to_feature_spec(
    features: FeatureFrame | pd.DataFrame,
    feature_spec: FeatureSpec | dict[str, object],
) -> pd.DataFrame:
    """Align a feature matrix to a saved ``FeatureSpec`` column order.

    Missing numeric columns are filled with ``np.nan`` and missing categorical
    columns are filled with ``"missing"``. Encoding and imputation still belong
    in ``modeling.py``.
    """
    spec = _as_feature_spec(feature_spec)
    X = features.X.copy() if isinstance(features, FeatureFrame) else features.copy()

    aligned = pd.DataFrame(index=X.index)
    for column in spec.numeric_features:
        if column in X.columns:
            aligned[column] = pd.to_numeric(X[column], errors="coerce").astype(float)
        else:
            aligned[column] = np.nan

    for column in spec.categorical_features:
        if column in X.columns:
            aligned[column] = _coerce_categorical_series(X[column])
        else:
            aligned[column] = "missing"

    return aligned.loc[:, spec.feature_names]


def build_optional_training_features(raw_datasets: list[pd.DataFrame]) -> list[FeatureFrame]:
    """Feature-engineer optional extra training datasets."""
    return [
        build_feature_frame(raw_dataset, labels_required=True)
        for raw_dataset in raw_datasets
    ]


def _as_feature_spec(
    value: FeatureSpec | dict[str, object],
    *,
    labels_available: bool | None = None,
) -> FeatureSpec:
    """Normalize a FeatureSpec-like object."""
    if isinstance(value, FeatureSpec):
        if labels_available is None or labels_available == value.labels_available:
            return value
        return FeatureSpec(
            feature_names=list(value.feature_names),
            numeric_features=list(value.numeric_features),
            categorical_features=list(value.categorical_features),
            target_name=value.target_name,
            labels_available=labels_available,
        )

    feature_names = list(value.get("feature_names", []))
    numeric_features = list(value.get("numeric_features", []))
    categorical_features = list(value.get("categorical_features", []))
    if not feature_names:
        feature_names = numeric_features + categorical_features

    return FeatureSpec(
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_name=str(value.get("target_name", TARGET_NAME)),
        labels_available=bool(
            value.get(
                "labels_available",
                True if labels_available is None else labels_available,
            )
        ),
    )
