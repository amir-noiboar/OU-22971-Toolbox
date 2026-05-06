"""MLflow Model Registry helpers for the Unit 8 capstone."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


CHAMPION_ALIAS = "champion"
CANDIDATE_ALIAS = "candidate"
PREVIOUS_CHAMPION_ROLE = "previous_champion"


@dataclass(frozen=True)
class ModelVersionInfo:
    """Small JSON-friendly view of an MLflow model version."""

    name: str
    version: str
    source: str | None = None
    run_id: str | None = None
    status: str | None = None
    aliases: list[str] | None = None
    model_uri: str | None = None

    def __str__(self) -> str:
        """Return version for simple string interpolation compatibility."""
        return self.version


def get_client() -> MlflowClient:
    """Return an MLflow tracking client for the active tracking URI."""
    return MlflowClient()


def ensure_registered_model(model_name: str) -> None:
    """Create a registered model if it does not already exist."""
    client = get_client()
    try:
        client.get_registered_model(model_name)
    except Exception:
        try:
            client.create_registered_model(model_name)
        except Exception:
            client.get_registered_model(model_name)


def get_model_version_by_alias(
    model_name: str,
    alias: str,
) -> ModelVersionInfo | None:
    """Return model version information for an alias, or None when absent."""
    client = get_client()
    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None
    return _version_info(model_version)


def get_champion(model_name: str) -> ModelVersionInfo | None:
    """Return the current champion version."""
    return get_model_version_by_alias(model_name, CHAMPION_ALIAS)


def log_and_register_model(
    model: object,
    *,
    model_name: str,
    artifact_path: str = "model",
    input_example: object | None = None,
    registered_model_name: str | None = None,
    tags: dict[str, object] | None = None,
) -> ModelVersionInfo:
    """Log an sklearn-compatible model and register it as a model version."""
    registered_name = registered_model_name or model_name
    ensure_registered_model(registered_name)
    sk_model = _unwrap_model(model)

    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run is not None else None

    try:
        model_info = mlflow.sklearn.log_model(
            sk_model=sk_model,
            name=artifact_path,
            registered_model_name=registered_name,
            input_example=input_example,
            await_registration_for=300,
        )
    except TypeError:
        model_info = mlflow.sklearn.log_model(
            sk_model=sk_model,
            artifact_path=artifact_path,
            registered_model_name=registered_name,
            input_example=input_example,
            await_registration_for=300,
        )

    model_uri = getattr(model_info, "model_uri", None)
    if model_uri:
        try:
            mlflow.log_text(str(model_uri), "model_uri.txt")
        except Exception:
            pass

    version = _registered_version_from_model_info(model_info)
    if version is None:
        version = _find_registered_version(registered_name, run_id=run_id, source=model_uri)
    if version is None:
        raise RuntimeError(f"Could not find registered model version for {registered_name!r}.")

    if tags:
        set_model_version_tags(model_name=registered_name, version=version, tags=tags)

    client = get_client()
    return _version_info(client.get_model_version(registered_name, version), model_uri=model_uri)


def bootstrap_champion(
    model: object,
    *,
    model_name: str,
    input_example: object | None = None,
    tags: dict[str, object] | None = None,
    promotion_reason: str = "bootstrap",
    metadata: dict[str, object] | None = None,
) -> ModelVersionInfo:
    """Register the first model and set the champion alias."""
    merged_tags = {
        **(metadata or {}),
        **(tags or {}),
        "role": CHAMPION_ALIAS,
        "validation_status": "approved",
        "promotion_reason": promotion_reason,
    }
    info = log_and_register_model(
        model,
        model_name=model_name,
        input_example=input_example,
        tags=merged_tags,
    )
    client = get_client()
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, info.version)
    return _version_info(client.get_model_version(model_name, info.version), model_uri=info.model_uri)


def register_candidate(
    model: object,
    *,
    model_name: str,
    input_example: object | None = None,
    trained_on_batches: str | None = None,
    eval_batch_id: str | None = None,
    validation_status: str = "pending",
    decision_reason: str | None = None,
    tags: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> ModelVersionInfo:
    """Register a candidate model version and tag it for validation."""
    merged_tags = {
        **(metadata or {}),
        **(tags or {}),
        "role": CANDIDATE_ALIAS,
        "validation_status": validation_status,
        "trained_on_batches": trained_on_batches,
        "eval_batch_id": eval_batch_id,
        "decision_reason": decision_reason,
    }
    info = log_and_register_model(
        model,
        model_name=model_name,
        input_example=input_example,
        tags=merged_tags,
    )
    client = get_client()
    client.set_registered_model_alias(model_name, CANDIDATE_ALIAS, info.version)
    return _version_info(client.get_model_version(model_name, info.version), model_uri=info.model_uri)


def promote_candidate(
    *,
    model_name: str,
    candidate_version: str,
    old_champion_version: str | None = None,
    promotion_reason: str | None = None,
    tags: dict[str, object] | None = None,
    reason: str | None = None,
) -> ModelVersionInfo:
    """Promote a candidate by moving the champion alias to its version."""
    if not str(candidate_version).strip():
        raise ValueError("candidate_version is required for promotion.")
    promotion_reason = promotion_reason or reason or ""
    if not promotion_reason.strip():
        raise ValueError("promotion_reason is required for promotion.")

    client = get_client()
    now = _utc_now()

    if old_champion_version is None:
        old_champion = get_champion(model_name)
        old_champion_version = old_champion.version if old_champion is not None else None

    if old_champion_version:
        set_model_version_tags(
            model_name=model_name,
            version=old_champion_version,
            tags={
                "role": PREVIOUS_CHAMPION_ROLE,
                "demoted_at": now,
            },
        )

    set_model_version_tags(
        model_name=model_name,
        version=str(candidate_version),
        tags={
            **(tags or {}),
            "role": CHAMPION_ALIAS,
            "validation_status": "approved",
            "promoted_at": now,
            "promotion_reason": promotion_reason,
        },
    )
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, str(candidate_version))
    return _version_info(client.get_model_version(model_name, str(candidate_version)))


def set_model_version_tags(
    *,
    model_name: str,
    version: str,
    tags: dict[str, object],
) -> None:
    """Set stringified tags on a model version."""
    client = get_client()
    for key, value in _safe_tags(tags).items():
        client.set_model_version_tag(model_name, str(version), key, value)


def model_uri_for_alias(model_name: str, alias: str = CHAMPION_ALIAS) -> str:
    """Return a models:/ URI for a registered model alias."""
    return f"models:/{model_name}@{alias}"


def model_uri_for_version(model_name: str, version: str) -> str:
    """Return a models:/ URI for a registered model version."""
    return f"models:/{model_name}/{version}"


# Compatibility helpers for the current flow skeleton.


def get_champion_by_alias(model_name: str) -> ModelVersionInfo | None:
    """Compatibility wrapper for get_champion."""
    return get_champion(model_name)


def set_candidate_alias(model_name: str, candidate_version: str) -> None:
    """Set the candidate alias for a registered model version."""
    get_client().set_registered_model_alias(model_name, CANDIDATE_ALIAS, str(candidate_version))


def _version_info(model_version: Any, *, model_uri: str | None = None) -> ModelVersionInfo:
    """Convert an MLflow model version object to ModelVersionInfo."""
    name = str(getattr(model_version, "name", ""))
    version = str(getattr(model_version, "version", ""))
    aliases = getattr(model_version, "aliases", None)
    if aliases is not None:
        aliases = [str(alias) for alias in aliases]
    return ModelVersionInfo(
        name=name,
        version=version,
        source=getattr(model_version, "source", None),
        run_id=getattr(model_version, "run_id", None),
        status=getattr(model_version, "status", None),
        aliases=aliases,
        model_uri=model_uri or model_uri_for_version(name, version),
    )


def _registered_version_from_model_info(model_info: object) -> str | None:
    """Extract registered version from a log_model result when available."""
    version = getattr(model_info, "registered_model_version", None)
    if version is None:
        return None
    return str(version)


def _find_registered_version(
    model_name: str,
    *,
    run_id: str | None,
    source: str | None,
) -> str | None:
    """Find a newly registered version by run ID, source, then highest version."""
    client = get_client()
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    if not versions:
        return None

    if run_id is not None:
        run_matches = [version for version in versions if getattr(version, "run_id", None) == run_id]
        if run_matches:
            return _highest_version(run_matches)

    if source is not None:
        source_matches = [version for version in versions if getattr(version, "source", None) == source]
        if source_matches:
            return _highest_version(source_matches)

    return _highest_version(versions)


def _highest_version(versions: list[Any]) -> str:
    """Return the highest model version number from MLflow version objects."""
    return str(max(int(getattr(version, "version")) for version in versions))


def _unwrap_model(model: object) -> object:
    """Accept raw sklearn models or TrainResult-like objects."""
    return getattr(model, "model", model)


def _safe_tags(tags: dict[str, object]) -> dict[str, str]:
    """Convert tag values to MLflow-safe strings, skipping None."""
    safe: dict[str, str] = {}
    for key, value in tags.items():
        if value is None:
            continue
        if isinstance(value, bool):
            safe[str(key)] = str(value).lower()
        else:
            safe[str(key)] = str(value)
    return safe


def _utc_now() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()
