"""Hard raw-data integrity checks for the Unit 8 capstone.

This module is intentionally pure: it does not log to MLflow, write files, or
run NannyML. The flow can use the returned result to log artifacts and decide
whether to reject a batch before feature engineering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


PICKUP_DATETIME_COL = "lpep_pickup_datetime"
DROPOFF_DATETIME_COL = "lpep_dropoff_datetime"
TARGET_COL = "tip_amount"

BASE_REQUIRED_COLUMNS = (
    PICKUP_DATETIME_COL,
    DROPOFF_DATETIME_COL,
    "trip_distance",
    "fare_amount",
    "payment_type",
    "PULocationID",
    "DOLocationID",
)

ALLOWED_PAYMENT_TYPES = (1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class IntegrityResult:
    """Structured output for hard integrity checks."""

    hard_passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | int] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like access for flow code that consumes result objects."""
        return getattr(self, key, default)


def run_hard_integrity_checks(
    df_raw: pd.DataFrame,
    *,
    labels_required: bool = True,
    max_negative_distance_frac: float = 0.005,
    max_negative_duration_frac: float = 0.001,
) -> IntegrityResult:
    """Run fail-fast raw-data checks before feature engineering.

    Hard failures are limited to schema, datetime parseability, impossible
    distance/duration values, and label availability when labels are required.
    The negative distance and duration thresholds default to small tolerances
    for real TLC raw-data noise while still surfacing those issues as warnings.
    Payment-type domain issues are warning-only and belong to later policy
    decisions if the capstone chooses to act on them.
    """
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("df_raw must be a pandas DataFrame.")

    df = df_raw.copy()
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, float | int] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "labels_required": int(bool(labels_required)),
    }

    required = required_columns(labels_required=labels_required)
    missing_required = [column for column in required if column not in df.columns]
    metrics["n_missing_required_cols"] = int(len(missing_required))
    if missing_required:
        errors.append(f"missing required columns: {missing_required}")

    schema_table = _build_schema_table(df, required)

    parsed_pickup, pickup_invalid_frac = _datetime_invalid_frac(
        df,
        PICKUP_DATETIME_COL,
    )
    parsed_dropoff, dropoff_invalid_frac = _datetime_invalid_frac(
        df,
        DROPOFF_DATETIME_COL,
    )
    metrics["pickup_datetime_invalid_frac"] = pickup_invalid_frac
    metrics["dropoff_datetime_invalid_frac"] = dropoff_invalid_frac

    if PICKUP_DATETIME_COL in df.columns and _has_no_parseable_values(parsed_pickup):
        errors.append(f"{PICKUP_DATETIME_COL} has no parseable datetime values")
    if DROPOFF_DATETIME_COL in df.columns and _has_no_parseable_values(parsed_dropoff):
        errors.append(f"{DROPOFF_DATETIME_COL} has no parseable datetime values")

    trip_distance = _coerce_numeric(df, "trip_distance")
    trip_distance_missing_frac = _missing_frac(trip_distance)
    trip_distance_negative_frac = _bad_frac(trip_distance, trip_distance < 0)
    metrics["trip_distance_missing_frac"] = trip_distance_missing_frac
    metrics["trip_distance_negative_frac"] = trip_distance_negative_frac
    if trip_distance_negative_frac > max_negative_distance_frac:
        errors.append(
            "trip_distance negative fraction "
            f"{trip_distance_negative_frac:.6f} exceeds allowed "
            f"{max_negative_distance_frac:.6f}"
        )
    elif trip_distance_negative_frac > 0:
        warnings.append(
            "trip_distance negative fraction "
            f"{trip_distance_negative_frac:.6f} within tolerance "
            f"{max_negative_distance_frac:.6f}"
        )

    duration_min, duration_missing_frac, duration_negative_frac = _duration_metrics(
        parsed_pickup,
        parsed_dropoff,
    )
    metrics["duration_missing_frac"] = duration_missing_frac
    metrics["duration_negative_frac"] = duration_negative_frac
    if duration_negative_frac > max_negative_duration_frac:
        errors.append(
            "duration negative fraction "
            f"{duration_negative_frac:.6f} exceeds allowed "
            f"{max_negative_duration_frac:.6f}"
        )
    elif duration_negative_frac > 0:
        warnings.append(
            "duration negative fraction "
            f"{duration_negative_frac:.6f} within tolerance "
            f"{max_negative_duration_frac:.6f}"
        )

    target = _coerce_numeric(df, TARGET_COL)
    if target is not None:
        target_missing_frac = _missing_frac(target)
        metrics["target_missing_frac"] = target_missing_frac
        if labels_required:
            if _has_no_valid_numeric_values(target):
                errors.append(f"{TARGET_COL} has no valid numeric label values")
            elif target_missing_frac > 0:
                warnings.append(
                    f"{TARGET_COL} has missing or invalid labels: "
                    f"{target_missing_frac:.6f}"
                )

    payment_type = _coerce_numeric(df, "payment_type")
    payment_type_invalid_frac = _domain_invalid_frac(
        payment_type,
        allowed_values=ALLOWED_PAYMENT_TYPES,
    )
    metrics["payment_type_invalid_frac"] = payment_type_invalid_frac
    if payment_type_invalid_frac > 0:
        warnings.append(
            "payment_type contains values outside "
            f"{list(ALLOWED_PAYMENT_TYPES)}: {payment_type_invalid_frac:.6f}"
        )

    hard_passed = len(errors) == 0
    tables = {
        "schema": schema_table,
        "datetime_checks": _build_datetime_table(
            pickup_invalid_frac,
            dropoff_invalid_frac,
        ),
        "numeric_checks": _build_numeric_table(
            trip_distance_missing_frac=trip_distance_missing_frac,
            trip_distance_negative_frac=trip_distance_negative_frac,
            duration_missing_frac=duration_missing_frac,
            duration_negative_frac=duration_negative_frac,
            target_missing_frac=metrics.get("target_missing_frac"),
        ),
        "domain_checks": _build_domain_table(payment_type_invalid_frac),
        "summary": _build_summary_table(
            hard_passed=hard_passed,
            errors=errors,
            warnings=warnings,
            n_rows=metrics["n_rows"],
            n_cols=metrics["n_cols"],
        ),
    }

    return IntegrityResult(
        hard_passed=hard_passed,
        errors=errors,
        warnings=warnings,
        metrics=_plain_metrics(metrics),
        tables=tables,
    )


def required_columns(*, labels_required: bool = True) -> tuple[str, ...]:
    """Return raw columns required by the capstone hard gate."""
    if labels_required:
        return BASE_REQUIRED_COLUMNS + (TARGET_COL,)
    return BASE_REQUIRED_COLUMNS


def summarize_hard_checks(check_result: IntegrityResult) -> dict[str, Any]:
    """Return a compact summary for decision records or tags."""
    return {
        "hard_passed": bool(check_result.hard_passed),
        "errors": list(check_result.errors),
        "warnings": list(check_result.warnings),
        "n_errors": int(len(check_result.errors)),
        "n_warnings": int(len(check_result.warnings)),
        "metrics": dict(check_result.metrics),
    }


def _build_schema_table(df: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    """Build an MLflow-friendly schema presence table."""
    columns = sorted(set(required) | set(df.columns))
    return pd.DataFrame(
        [
            {
                "column": column,
                "required": bool(column in required),
                "present": bool(column in df.columns),
            }
            for column in columns
        ],
        columns=["column", "required", "present"],
    )


def _datetime_invalid_frac(
    df: pd.DataFrame,
    column: str,
) -> tuple[pd.Series | None, float]:
    """Parse a datetime column and return invalid fraction."""
    if column not in df.columns:
        return None, 1.0

    parsed = pd.to_datetime(df[column], errors="coerce")
    return parsed, _missing_frac(parsed)


def _duration_metrics(
    pickup: pd.Series | None,
    dropoff: pd.Series | None,
) -> tuple[pd.Series | None, float, float]:
    """Compute duration missing and negative fractions from parsed datetimes."""
    if pickup is None or dropoff is None:
        return None, 1.0, 0.0

    duration_min = (dropoff - pickup).dt.total_seconds() / 60.0
    return (
        duration_min,
        _missing_frac(duration_min),
        _bad_frac(duration_min, duration_min < 0),
    )


def _coerce_numeric(df: pd.DataFrame, column: str) -> pd.Series | None:
    """Coerce a numeric column if present."""
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def _missing_frac(series: pd.Series | None) -> float:
    """Return missing fraction as a plain float."""
    if series is None:
        return 1.0
    if len(series) == 0:
        return 1.0
    return float(series.isna().mean())


def _bad_frac(series: pd.Series | None, bad_mask: pd.Series | None) -> float:
    """Return fraction of bad values over all rows."""
    if series is None or bad_mask is None:
        return 0.0
    if len(series) == 0:
        return 0.0
    return float(bad_mask.fillna(False).mean())


def _domain_invalid_frac(
    series: pd.Series | None,
    *,
    allowed_values: tuple[int, ...],
) -> float:
    """Return fraction of non-missing values outside an allowed set."""
    if series is None:
        return 0.0
    if len(series) == 0:
        return 0.0

    invalid = series.notna() & ~series.isin(allowed_values)
    return float(invalid.mean())


def _has_no_parseable_values(parsed: pd.Series | None) -> bool:
    """Return whether a parsed datetime column has no valid values."""
    return parsed is None or int(parsed.notna().sum()) == 0


def _has_no_valid_numeric_values(series: pd.Series | None) -> bool:
    """Return whether a numeric series has no valid values."""
    return series is None or int(series.notna().sum()) == 0


def _build_datetime_table(
    pickup_invalid_frac: float,
    dropoff_invalid_frac: float,
) -> pd.DataFrame:
    """Build datetime-check table."""
    return pd.DataFrame(
        [
            {
                "column": PICKUP_DATETIME_COL,
                "invalid_frac": float(pickup_invalid_frac),
            },
            {
                "column": DROPOFF_DATETIME_COL,
                "invalid_frac": float(dropoff_invalid_frac),
            },
        ],
        columns=["column", "invalid_frac"],
    )


def _build_numeric_table(
    *,
    trip_distance_missing_frac: float,
    trip_distance_negative_frac: float,
    duration_missing_frac: float,
    duration_negative_frac: float,
    target_missing_frac: float | int | None,
) -> pd.DataFrame:
    """Build numeric-check table."""
    rows: list[dict[str, Any]] = [
        {
            "feature": "trip_distance",
            "missing_frac": float(trip_distance_missing_frac),
            "bad_frac": float(trip_distance_negative_frac),
            "check_type": "negative",
        },
        {
            "feature": "duration_min",
            "missing_frac": float(duration_missing_frac),
            "bad_frac": float(duration_negative_frac),
            "check_type": "negative",
        },
    ]
    if target_missing_frac is not None:
        rows.append(
            {
                "feature": TARGET_COL,
                "missing_frac": float(target_missing_frac),
                "bad_frac": float(target_missing_frac),
                "check_type": "missing_or_invalid_target",
            }
        )

    return pd.DataFrame(
        rows,
        columns=["feature", "missing_frac", "bad_frac", "check_type"],
    )


def _build_domain_table(payment_type_invalid_frac: float) -> pd.DataFrame:
    """Build domain-check table."""
    return pd.DataFrame(
        [
            {
                "feature": "payment_type",
                "invalid_frac": float(payment_type_invalid_frac),
                "allowed_values": ",".join(str(value) for value in ALLOWED_PAYMENT_TYPES),
            }
        ],
        columns=["feature", "invalid_frac", "allowed_values"],
    )


def _build_summary_table(
    *,
    hard_passed: bool,
    errors: list[str],
    warnings: list[str],
    n_rows: float | int,
    n_cols: float | int,
) -> pd.DataFrame:
    """Build one-row summary table."""
    return pd.DataFrame(
        [
            {
                "hard_passed": bool(hard_passed),
                "n_errors": int(len(errors)),
                "n_warnings": int(len(warnings)),
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
            }
        ],
        columns=["hard_passed", "n_errors", "n_warnings", "n_rows", "n_cols"],
    )


def _plain_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    """Convert pandas/numpy scalar-like values to plain Python numbers."""
    out: dict[str, float | int] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, int):
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out
