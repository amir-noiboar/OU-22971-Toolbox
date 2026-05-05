"""Small MLflow helpers for the Unit 8 capstone workflow."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

try:
    from mlflow.data.sources import LocalArtifactDatasetSource
except ImportError:  # pragma: no cover - depends on MLflow version.
    LocalArtifactDatasetSource = None  # type: ignore[assignment]

try:
    from .decisions import assert_json_serializable, build_decision_tags
except ImportError:  # pragma: no cover - allows direct module imports in simple scripts.
    from decisions import assert_json_serializable, build_decision_tags


def init_mlflow(
    *,
    tracking_uri: str,
    experiment_name: str,
) -> None:
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_params_safe(params: Mapping[str, object]) -> None:
    """Log MLflow params after converting values to safe scalar strings."""
    safe_params = {
        str(key): safe_value
        for key, value in params.items()
        if value is not None and (safe_value := _param_value(value)) is not None
    }
    if safe_params:
        mlflow.log_params(safe_params)


def log_metrics_safe(metrics: Mapping[str, object], *, prefix: str | None = None) -> None:
    """Log numeric MLflow metrics, skipping non-numeric and non-finite values."""
    safe_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        metric_value = _metric_value(value)
        if metric_value is None:
            continue
        metric_key = _prefixed_key(str(key), prefix)
        safe_metrics[metric_key] = metric_value

    if safe_metrics:
        mlflow.log_metrics(safe_metrics)


def log_tags_safe(tags: Mapping[str, object]) -> None:
    """Set MLflow tags after converting values to strings."""
    safe_tags = {
        str(key): tag_value
        for key, value in tags.items()
        if value is not None and (tag_value := _tag_value(value)) is not None
    }
    if safe_tags:
        mlflow.set_tags(safe_tags)


def log_tables(
    tables: Mapping[str, pd.DataFrame] | str,
    tables_legacy: Mapping[str, object] | None = None,
    *,
    artifact_dir: str | None = None,
) -> None:
    """Log small pandas tables as JSON artifacts.

    The preferred call is ``log_tables(tables, artifact_dir="integrity")``.
    The ``log_tables("integrity", tables)`` form is kept for the current flow
    skeleton.
    """
    if isinstance(tables, str):
        artifact_dir = artifact_dir or tables
        table_mapping = tables_legacy or {}
    else:
        artifact_dir = artifact_dir or "tables"
        table_mapping = tables

    safe_dir = _safe_artifact_part(artifact_dir)
    for name, table in table_mapping.items():
        dataframe = _as_dataframe(table)
        if dataframe is None:
            continue
        artifact_name = _safe_artifact_part(str(name))
        mlflow.log_table(dataframe, artifact_file=f"{safe_dir}/{artifact_name}.json")


def log_integrity_result(
    result: object,
    *,
    artifact_dir: str = "integrity",
    metric_prefix: str = "integrity",
) -> None:
    """Log an IntegrityResult-like object to MLflow."""
    metrics = _extract_mapping(result, "metrics")
    tables = _extract_mapping(result, "tables")
    warnings = _extract_sequence(result, "warnings")
    hard_passed = bool(_extract_field(result, "hard_passed", False))

    log_metrics_safe(metrics, prefix=metric_prefix)
    log_tables(tables, artifact_dir=artifact_dir)
    log_tags_safe(
        {
            "hard_gate_passed": hard_passed,
            "integrity_warn": bool(warnings),
        }
    )


def log_feature_spec(
    feature_spec: object,
    *,
    artifact_file: str = "feature_spec.json",
) -> None:
    """Log a FeatureSpec-like object as JSON."""
    payload = {
        "feature_names": _list_field(feature_spec, "feature_names"),
        "numeric_features": _list_field(feature_spec, "numeric_features"),
        "categorical_features": _list_field(feature_spec, "categorical_features"),
        "target_name": _extract_field(feature_spec, "target_name", "tip_amount"),
        "labels_available": bool(_extract_field(feature_spec, "labels_available", True)),
    }
    mlflow.log_dict(_json_safe(payload), artifact_file=artifact_file)


def log_decision(
    decision: Mapping[str, object],
    *,
    artifact_file: str = "decision.json",
    set_tags: bool = True,
) -> None:
    """Log a decision dictionary and optionally mirror key fields to MLflow tags."""
    payload = _json_safe(dict(decision))
    assert isinstance(payload, dict)
    assert_json_serializable(payload)
    mlflow.log_dict(payload, artifact_file=artifact_file)

    if set_tags:
        tags = build_decision_tags(payload)
        for key in ("retrain_recommended", "promotion_recommended"):
            if key in payload:
                tags[key] = _tag_value(payload[key]) or ""
        log_tags_safe(tags)


def log_dataset_input_from_pandas(
    df: pd.DataFrame,
    *,
    source_path: str,
    name: str,
    context: str,
) -> None:
    """Log pandas dataset lineage when the active MLflow version supports it."""
    try:
        if LocalArtifactDatasetSource is not None:
            source = LocalArtifactDatasetSource(source_path)
            dataset = mlflow.data.from_pandas(df, source=source, name=name)
        else:
            dataset = mlflow.data.from_pandas(df, source=source_path, name=name)
        mlflow.log_input(dataset, context=context)
    except Exception as exc:  # pragma: no cover - API differences are environment-specific.
        log_tags_safe({"dataset_lineage_warning": "true"})
        mlflow.log_text(
            f"Dataset lineage logging failed for {name}: {type(exc).__name__}: {exc}",
            artifact_file=f"lineage/{_safe_artifact_part(name)}_warning.txt",
        )


def log_artifact_if_exists(
    path: str | Path,
    *,
    artifact_path: str | None = None,
) -> None:
    """Log an artifact file if it exists."""
    artifact = Path(path)
    if artifact.exists():
        mlflow.log_artifact(str(artifact), artifact_path=artifact_path)


# Compatibility wrappers for the current flow skeleton.


def initialize_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Compatibility wrapper for ``init_mlflow``."""
    init_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)


def log_params(params: Mapping[str, object]) -> None:
    """Compatibility wrapper for ``log_params_safe``."""
    log_params_safe(params)


def set_decision_tags(tags: Mapping[str, object]) -> None:
    """Compatibility wrapper for ``log_tags_safe``."""
    log_tags_safe(tags)


def log_metrics(prefix: str, metrics: Mapping[str, object]) -> None:
    """Compatibility wrapper for ``log_metrics_safe``."""
    log_metrics_safe(metrics, prefix=prefix)


def log_artifact_dict(payload: Mapping[str, object], artifact_file: str) -> None:
    """Log a JSON dictionary artifact."""
    mlflow.log_dict(_json_safe(dict(payload)), artifact_file=artifact_file)


def log_artifact_file(path: str | Path) -> None:
    """Compatibility wrapper for ``log_artifact_if_exists``."""
    log_artifact_if_exists(path)


def log_decision_json(decision: Mapping[str, object]) -> None:
    """Compatibility wrapper for ``log_decision``."""
    log_decision(decision)


def log_dataset_lineage(datasets: Mapping[str, object]) -> None:
    """Log simple dataset-lineage metadata as a fallback artifact.

    Raw dataframe lineage should use ``log_dataset_input_from_pandas`` when a
    dataframe is available.
    """
    mlflow.log_dict(_json_safe(dict(datasets)), artifact_file="lineage/datasets.json")


def _param_value(value: object) -> str | int | float | bool | None:
    """Convert a Python value to something accepted by MLflow params."""
    value = _scalar_item(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping | list | tuple):
        try:
            return json.dumps(_json_safe(value), sort_keys=True)
        except TypeError:
            return str(value)
    return str(value)


def _metric_value(value: object) -> float | None:
    """Convert a value to a finite metric float when possible."""
    value = _scalar_item(value)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        metric = float(value)
        if math.isfinite(metric):
            return metric
    return None


def _tag_value(value: object) -> str | None:
    """Convert a value to an MLflow tag string."""
    value = _scalar_item(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Mapping | list | tuple):
        try:
            return json.dumps(_json_safe(value), sort_keys=True)
        except TypeError:
            return str(value)
    return str(value)


def _prefixed_key(key: str, prefix: str | None) -> str:
    """Return a metric key with an optional prefix."""
    safe_key = key.replace(" ", "_")
    if not prefix:
        return safe_key
    return f"{prefix}_{safe_key}"


def _safe_artifact_part(value: str) -> str:
    """Make a conservative artifact path component."""
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe.strip("_") or "artifact"


def _as_dataframe(value: object) -> pd.DataFrame | None:
    """Return a dataframe for table-like values when safe."""
    if isinstance(value, pd.DataFrame):
        return value
    if value is None:
        return None
    try:
        return pd.DataFrame(value)
    except Exception:
        return None


def _extract_field(source: object, field_name: str, default: object = None) -> object:
    """Extract a field from a mapping or loose object."""
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(field_name, default)
    return getattr(source, field_name, default)


def _extract_mapping(source: object, field_name: str) -> Mapping[str, object]:
    """Extract a mapping field from a loose object."""
    value = _extract_field(source, field_name, {})
    return value if isinstance(value, Mapping) else {}


def _extract_sequence(source: object, field_name: str) -> list[object]:
    """Extract a list-like field from a loose object."""
    value = _extract_field(source, field_name, [])
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _list_field(source: object, field_name: str) -> list[object]:
    """Extract a field as a JSON-safe list."""
    value = _extract_field(source, field_name, [])
    if isinstance(value, list):
        return _json_safe(value)
    if isinstance(value, tuple):
        return _json_safe(list(value))
    return []


def _json_safe(value: object) -> object:
    """Recursively convert common values to JSON-safe objects."""
    value = _scalar_item(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_json_safe(item) for item in value]
    return str(value)


def _scalar_item(value: object) -> object:
    """Convert NumPy/pandas scalar-like objects without importing NumPy."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value
