"""Warning-only soft monitoring checks for the Unit 8 capstone.

The hard integrity gate lives in ``quality.py``. This module compares already
engineered reference and batch feature frames, returns MLflow-friendly metrics
and tables, and never stops the workflow automatically.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SoftCheckResult:
    """Structured output for warning-only monitoring checks."""

    warning: bool
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | int] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    method: str = "manual"

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like access for flow code that consumes result objects."""
        return getattr(self, key, default)


def run_soft_checks(
    reference_features: pd.DataFrame,
    batch_features: pd.DataFrame,
    *,
    feature_spec: object | None = None,
    missingness_warn_threshold: float = 0.10,
    unseen_category_warn_threshold: float = 0.02,
    enable_nannyml: bool = True,
) -> SoftCheckResult:
    """Run warning-only missingness, categorical, and optional NannyML checks.

    Manual checks always run. NannyML is used opportunistically when installed
    and compatible with the local API; any NannyML failure is reported as a
    warning and the manual soft-gate result is still returned.
    """
    if not isinstance(reference_features, pd.DataFrame):
        raise TypeError("reference_features must be a pandas DataFrame.")
    if not isinstance(batch_features, pd.DataFrame):
        raise TypeError("batch_features must be a pandas DataFrame.")

    reference = reference_features.copy()
    batch = batch_features.copy()
    warnings: list[str] = []
    metrics: dict[str, float | int] = {
        "n_reference_rows": int(len(reference)),
        "n_batch_rows": int(len(batch)),
        "missingness_warn_threshold": float(missingness_warn_threshold),
        "unseen_category_warn_threshold": float(unseen_category_warn_threshold),
        "nannyml_enabled": int(bool(enable_nannyml)),
        "nannyml_succeeded": 0,
    }

    numeric_features, categorical_features = _feature_groups(
        reference,
        batch,
        feature_spec=feature_spec,
    )
    expected_features = _unique(numeric_features + categorical_features)
    common_features = [
        feature
        for feature in expected_features
        if feature in reference.columns and feature in batch.columns
    ]
    common_categorical = [
        feature for feature in categorical_features if feature in common_features
    ]

    missing_columns_table = _missing_columns_table(
        expected_features,
        reference,
        batch,
    )
    missing_feature_warnings = _missing_feature_warnings(missing_columns_table)
    warnings.extend(missing_feature_warnings)

    metrics["n_common_features"] = int(len(common_features))
    metrics["n_missing_feature_columns"] = int(len(missing_feature_warnings))

    if reference.empty or batch.empty:
        warnings.append(
            "soft checks received empty reference or batch features; drift comparisons skipped"
        )
        metrics.update(
            {
                "n_missingness_warnings": 0,
                "n_unseen_category_warnings": 0,
                "n_total_warnings": int(len(warnings)),
            }
        )
        tables = {
            "missingness": _empty_missingness_table(),
            "unseen_categories": _empty_unseen_categories_table(),
            "feature_columns": missing_columns_table,
            "summary": _summary_table(
                warning=True,
                method="manual_empty",
                metrics=metrics,
                n_total_warnings=len(warnings),
            ),
        }
        return SoftCheckResult(
            warning=True,
            warnings=warnings,
            metrics=_plain_metrics(metrics),
            tables=tables,
            method="manual_empty",
        )

    missingness_table, missingness_warnings = _missingness_table(
        reference,
        batch,
        common_features,
        threshold=missingness_warn_threshold,
    )
    warnings.extend(missingness_warnings)

    unseen_table, unseen_warnings = _unseen_categories_table(
        reference,
        batch,
        common_categorical,
        threshold=unseen_category_warn_threshold,
    )
    warnings.extend(unseen_warnings)

    metrics["n_missingness_warnings"] = int(len(missingness_warnings))
    metrics["n_unseen_category_warnings"] = int(len(unseen_warnings))

    method = "manual"
    nannyml_table = None
    if enable_nannyml and common_features:
        nannyml_table, nannyml_metrics, nannyml_warning = _try_nannyml_drift(
            reference,
            batch,
            common_features,
            common_categorical,
        )
        if nannyml_table is not None:
            method = "manual+nannyml"
            metrics["nannyml_succeeded"] = 1
            metrics.update(nannyml_metrics)
            nannyml_alert_count = int(nannyml_metrics.get("nannyml_alert_count", 0))
            if nannyml_alert_count > 0:
                warnings.append(f"NannyML drift alerts detected: {nannyml_alert_count}")
        elif nannyml_warning:
            method = "manual_fallback"
            warnings.append(nannyml_warning)

    metrics["n_total_warnings"] = int(len(warnings))
    warning = bool(warnings)
    tables = {
        "missingness": missingness_table,
        "unseen_categories": unseen_table,
        "feature_columns": missing_columns_table,
        "summary": _summary_table(
            warning=warning,
            method=method,
            metrics=metrics,
            n_total_warnings=len(warnings),
        ),
    }
    if nannyml_table is not None:
        tables["nannyml_drift"] = nannyml_table

    return SoftCheckResult(
        warning=warning,
        warnings=warnings,
        metrics=_plain_metrics(metrics),
        tables=tables,
        method=method,
    )


def run_nannyml_checks(
    reference_features: pd.DataFrame,
    batch_features: pd.DataFrame,
    **kwargs: Any,
) -> SoftCheckResult:
    """Compatibility alias for ``run_soft_checks``."""
    return run_soft_checks(reference_features, batch_features, **kwargs)


def run_soft_monitoring_checks(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    **kwargs: Any,
) -> SoftCheckResult:
    """Compatibility alias for ``run_soft_checks``."""
    return run_soft_checks(reference_data, current_data, **kwargs)


def run_drift_summary(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    **kwargs: Any,
) -> SoftCheckResult:
    """Compatibility wrapper returning the soft-check summary result."""
    return run_soft_checks(reference_features, current_features, **kwargs)


def format_nannyml_artifacts(results: SoftCheckResult | Mapping[str, object]) -> dict[str, object]:
    """Return MLflow-ready tables and metrics from a soft-check result."""
    if isinstance(results, SoftCheckResult):
        return {
            "method": results.method,
            "warning": results.warning,
            "warnings": list(results.warnings),
            "metrics": dict(results.metrics),
            "tables": dict(results.tables),
        }
    return dict(results)


def _feature_groups(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    *,
    feature_spec: object | None,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names from spec or dataframes."""
    if feature_spec is not None:
        numeric = _list_field(feature_spec, "numeric_features")
        categorical = _list_field(feature_spec, "categorical_features")
        return numeric, categorical

    all_columns = _unique(list(reference.columns) + list(batch.columns))
    numeric = [
        column
        for column in all_columns
        if column in reference.columns
        and column in batch.columns
        and pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(batch[column])
    ]
    categorical = [column for column in all_columns if column not in numeric]
    return numeric, categorical


def _missingness_table(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    common_features: list[str],
    *,
    threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Build missingness comparison table and threshold warnings."""
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for feature in common_features:
        reference_missing = _missing_frac(reference[feature])
        batch_missing = _missing_frac(batch[feature])
        delta = batch_missing - reference_missing
        warned = bool(delta > threshold)
        if warned:
            warnings.append(
                "missingness spike for "
                f"{feature}: delta {delta:.6f} exceeds threshold {threshold:.6f}"
            )
        rows.append(
            {
                "feature": feature,
                "reference_missing_frac": reference_missing,
                "batch_missing_frac": batch_missing,
                "delta_missing_frac": float(delta),
                "warned": warned,
            }
        )

    return (
        pd.DataFrame(
            rows,
            columns=[
                "feature",
                "reference_missing_frac",
                "batch_missing_frac",
                "delta_missing_frac",
                "warned",
            ],
        ),
        warnings,
    )


def _unseen_categories_table(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    categorical_features: list[str],
    *,
    threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Build unseen-category table and threshold warnings."""
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for feature in categorical_features:
        reference_values = set(_non_missing_strings(reference[feature]))
        batch_values = _non_missing_strings(batch[feature])
        unseen_mask = batch_values.map(lambda value: value not in reference_values)
        unseen_count = int(unseen_mask.sum()) if len(unseen_mask) else 0
        unseen_frac = float(unseen_count / len(batch)) if len(batch) else 0.0
        unseen_values = sorted(set(batch_values[unseen_mask]))
        warned = bool(unseen_frac > threshold)
        if warned:
            warnings.append(
                "unseen categorical values for "
                f"{feature}: fraction {unseen_frac:.6f} exceeds threshold {threshold:.6f}"
            )
        rows.append(
            {
                "feature": feature,
                "unseen_frac": unseen_frac,
                "n_unseen_values": int(len(unseen_values)),
                "sample_unseen_values": ",".join(unseen_values[:10]),
                "warned": warned,
            }
        )

    return (
        pd.DataFrame(
            rows,
            columns=[
                "feature",
                "unseen_frac",
                "n_unseen_values",
                "sample_unseen_values",
                "warned",
            ],
        ),
        warnings,
    )


def _missing_columns_table(
    expected_features: list[str],
    reference: pd.DataFrame,
    batch: pd.DataFrame,
) -> pd.DataFrame:
    """Build table showing feature presence in reference and batch frames."""
    rows = [
        {
            "feature": feature,
            "present_in_reference": bool(feature in reference.columns),
            "present_in_batch": bool(feature in batch.columns),
            "common": bool(feature in reference.columns and feature in batch.columns),
        }
        for feature in expected_features
    ]
    return pd.DataFrame(
        rows,
        columns=["feature", "present_in_reference", "present_in_batch", "common"],
    )


def _missing_feature_warnings(feature_columns: pd.DataFrame) -> list[str]:
    """Return warnings for expected features missing from either side."""
    warnings: list[str] = []
    if feature_columns.empty:
        return warnings
    missing = feature_columns.loc[~feature_columns["common"], "feature"].tolist()
    if missing:
        warnings.append(f"features missing from reference or batch comparison: {missing}")
    return warnings


def _try_nannyml_drift(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    common_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame | None, dict[str, float | int], str | None]:
    """Try a compatible NannyML univariate drift calculator."""
    try:
        import nannyml as nml  # type: ignore[import-not-found]

        calculator_cls = getattr(nml, "UnivariateDriftCalculator", None)
        if calculator_cls is None:
            calculator_cls = getattr(nml, "UnivariateStatisticalDriftCalculator", None)
        if calculator_cls is None:
            return None, {}, "NannyML drift check failed; using manual soft checks only: no compatible univariate drift calculator found"

        kwargs = _nannyml_kwargs(
            calculator_cls,
            common_features,
            categorical_features,
            n_rows=len(reference) + len(batch),
        )
        calculator = calculator_cls(**kwargs)
        calculator.fit(reference.loc[:, common_features])
        results = calculator.calculate(batch.loc[:, common_features])
        drift_table = _nannyml_results_to_frame(results)
        metrics = _nannyml_metrics(drift_table)
        return drift_table, metrics, None
    except Exception as exc:  # pragma: no cover - depends on optional NannyML/API.
        return (
            None,
            {},
            "NannyML drift check failed; using manual soft checks only: "
            f"{type(exc).__name__}: {exc}",
        )


def _nannyml_kwargs(
    calculator_cls: object,
    common_features: list[str],
    categorical_features: list[str],
    *,
    n_rows: int,
) -> dict[str, object]:
    """Build constructor kwargs across common NannyML API versions."""
    candidate_kwargs: dict[str, object] = {
        "column_names": common_features,
        "chunk_size": max(1, min(5000, n_rows)),
    }
    try:
        signature = inspect.signature(calculator_cls)
    except (TypeError, ValueError):
        return candidate_kwargs

    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if "categorical_column_names" in parameters:
        candidate_kwargs["categorical_column_names"] = categorical_features
    elif "treat_as_categorical" in parameters:
        candidate_kwargs["treat_as_categorical"] = categorical_features

    if accepts_kwargs:
        return candidate_kwargs
    return {
        key: value
        for key, value in candidate_kwargs.items()
        if key in parameters
    }


def _nannyml_results_to_frame(results: object) -> pd.DataFrame:
    """Extract a small DataFrame from NannyML results when possible."""
    for method_name in ("to_df", "to_dataframe"):
        method = getattr(results, method_name, None)
        if callable(method):
            table = method()
            if isinstance(table, pd.DataFrame):
                return table.reset_index(drop=False)

    data = getattr(results, "data", None)
    if isinstance(data, pd.DataFrame):
        return data.reset_index(drop=False)

    return pd.DataFrame([{"nannyml_result": str(results)}])


def _nannyml_metrics(drift_table: pd.DataFrame) -> dict[str, float | int]:
    """Extract compact drift metrics from a NannyML result table."""
    metrics: dict[str, float | int] = {"nannyml_rows": int(len(drift_table))}
    if drift_table.empty:
        return metrics

    lower_columns = {str(column).lower(): column for column in drift_table.columns}
    alert_columns = [
        column
        for lower, column in lower_columns.items()
        if "alert" in lower or "drift" in lower and "detected" in lower
    ]
    if alert_columns:
        alert_counts = 0
        for column in alert_columns:
            alert_counts += int(drift_table[column].fillna(False).astype(bool).sum())
        metrics["nannyml_alert_count"] = int(alert_counts)

    value_columns = [
        column
        for lower, column in lower_columns.items()
        if lower.endswith("value") or "statistic" in lower or "drift_score" in lower
    ]
    for column in value_columns[:5]:
        numeric = pd.to_numeric(drift_table[column], errors="coerce")
        if numeric.notna().any():
            safe_name = str(column).replace(" ", "_")
            metrics[f"nannyml_{safe_name}_max"] = float(numeric.max())

    return metrics


def _summary_table(
    *,
    warning: bool,
    method: str,
    metrics: dict[str, float | int],
    n_total_warnings: int,
) -> pd.DataFrame:
    """Build one-row MLflow-friendly soft-check summary table."""
    return pd.DataFrame(
        [
            {
                "warning": bool(warning),
                "method": method,
                "n_reference_rows": int(metrics.get("n_reference_rows", 0)),
                "n_batch_rows": int(metrics.get("n_batch_rows", 0)),
                "n_common_features": int(metrics.get("n_common_features", 0)),
                "n_total_warnings": int(n_total_warnings),
            }
        ],
        columns=[
            "warning",
            "method",
            "n_reference_rows",
            "n_batch_rows",
            "n_common_features",
            "n_total_warnings",
        ],
    )


def _empty_missingness_table() -> pd.DataFrame:
    """Return an empty missingness table with stable columns."""
    return pd.DataFrame(
        columns=[
            "feature",
            "reference_missing_frac",
            "batch_missing_frac",
            "delta_missing_frac",
            "warned",
        ]
    )


def _empty_unseen_categories_table() -> pd.DataFrame:
    """Return an empty unseen-category table with stable columns."""
    return pd.DataFrame(
        columns=[
            "feature",
            "unseen_frac",
            "n_unseen_values",
            "sample_unseen_values",
            "warned",
        ]
    )


def _missing_frac(series: pd.Series) -> float:
    """Return missing fraction as a plain float."""
    if len(series) == 0:
        return 1.0
    return float(series.isna().mean())


def _non_missing_strings(series: pd.Series) -> pd.Series:
    """Return non-missing category values as strings."""
    return series.dropna().astype("string").astype(str).reset_index(drop=True)


def _list_field(source: object, field_name: str) -> list[str]:
    """Extract a list field from a FeatureSpec-like object or mapping."""
    if isinstance(source, Mapping):
        value = source.get(field_name, [])
    else:
        value = getattr(source, field_name, [])
    if value is None:
        return []
    return [str(item) for item in list(value)]


def _unique(values: list[str]) -> list[str]:
    """Return values with duplicates removed while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _plain_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    """Convert pandas/numpy scalar-like values to plain Python numbers."""
    out: dict[str, float | int] = {}
    for key, value in metrics.items():
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, int):
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out
