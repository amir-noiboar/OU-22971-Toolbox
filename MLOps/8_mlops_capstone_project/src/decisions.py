"""Pure decision record builders for the Unit 8 capstone workflow."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


Decision = dict[str, object]


def build_reject_decision(
    integrity_result: object | None = None,
    *,
    hard_passed: bool | None = None,
    errors: Sequence[object] | None = None,
    warnings: Sequence[object] | None = None,
    metrics: Mapping[str, object] | None = None,
    reason: str = "hard_integrity_failure",
    evidence: object | None = None,
) -> Decision:
    """Build a reject-batch decision after the hard integrity gate fails."""
    source = integrity_result if integrity_result is not None else evidence
    source_errors = _extract_field(source, "errors", [])
    source_warnings = _extract_field(source, "warnings", [])
    source_metrics = _extract_field(source, "metrics", {})
    source_hard_passed = _extract_field(source, "hard_passed", hard_passed)

    decision = _base_decision(
        action="reject_batch",
        final_decision="rejected",
        reason=reason,
        criteria=[
            "hard integrity gate must pass before feature engineering",
            "required schema, parseable datetimes, feasible distance/duration, and labels are checked",
        ],
        metrics=metrics if metrics is not None else source_metrics,
        warnings=warnings if warnings is not None else source_warnings,
        errors=errors if errors is not None else source_errors,
        hard_gate_passed=bool(source_hard_passed) if source_hard_passed is not None else False,
        retrain_recommended=False,
        promotion_recommended=False,
        promotion_executed=False,
    )
    decision["hard_gate_passed"] = False
    return _finalize(decision)


def build_no_retrain_decision(
    champion_metrics: Mapping[str, object] | None = None,
    *,
    rmse_champion: float | None = None,
    rmse_baseline: float | None = None,
    rmse_increase_pct: float | None = None,
    rmse_increase_threshold: float | None = None,
    reason: str | None = None,
    evidence: object | None = None,
    warnings: Sequence[object] | None = None,
    errors: Sequence[object] | None = None,
) -> Decision:
    """Build a decision record for keeping the current champion."""
    merged_metrics = _merge_metrics(champion_metrics, _extract_field(evidence, "metrics", {}))
    _add_optional_metrics(
        merged_metrics,
        rmse_champion=rmse_champion,
        rmse_baseline=rmse_baseline,
        rmse_increase_pct=rmse_increase_pct,
    )
    if rmse_increase_threshold is not None:
        merged_metrics["rmse_increase_threshold"] = rmse_increase_threshold

    if reason is None:
        reason = "champion degradation did not exceed retrain threshold"

    decision = _base_decision(
        action="keep_champion",
        final_decision="no_retrain",
        reason=reason,
        criteria=[
            "hard integrity gate passed",
            "champion evaluation is acceptable",
            "rmse_increase_pct must exceed rmse_increase_threshold to retrain",
        ],
        metrics=merged_metrics,
        warnings=warnings if warnings is not None else _extract_field(evidence, "warnings", []),
        errors=errors if errors is not None else _extract_field(evidence, "errors", []),
        hard_gate_passed=True,
        retrain_recommended=False,
        promotion_recommended=False,
        promotion_executed=False,
    )
    decision["retrain_needed"] = False
    return _finalize(decision)


def build_retrain_decision(
    champion_metrics: Mapping[str, object] | None = None,
    soft_integrity: Mapping[str, object] | None = None,
    rmse_increase_threshold: float | None = None,
    *,
    rmse_champion: float | None = None,
    rmse_baseline: float | None = None,
    rmse_increase_pct: float | None = None,
    reason: str | None = None,
    warnings: Sequence[object] | None = None,
    errors: Sequence[object] | None = None,
) -> Decision:
    """Build a decision record recommending candidate retraining."""
    merged_metrics = _merge_metrics(champion_metrics)
    _add_optional_metrics(
        merged_metrics,
        rmse_champion=rmse_champion,
        rmse_baseline=rmse_baseline,
        rmse_increase_pct=rmse_increase_pct,
    )
    if rmse_increase_threshold is not None:
        merged_metrics["rmse_increase_threshold"] = rmse_increase_threshold

    soft_warnings = _extract_field(soft_integrity, "warnings", [])
    if warnings is not None:
        soft_warnings = list(warnings)

    if reason is None:
        reason = _retrain_reason(merged_metrics, rmse_increase_threshold, soft_integrity)

    decision = _base_decision(
        action="retrain_candidate",
        final_decision="retrain_recommended",
        reason=reason,
        criteria=[
            "hard integrity gate passed",
            "champion degradation or monitoring warnings indicate retraining",
            "rmse_increase_pct is compared with rmse_increase_threshold when available",
        ],
        metrics=merged_metrics,
        warnings=soft_warnings,
        errors=errors,
        hard_gate_passed=True,
        retrain_recommended=True,
        promotion_recommended=False,
        promotion_executed=False,
    )
    decision["retrain_needed"] = True
    return _finalize(decision)


def build_candidate_rejected_decision(
    champion_metrics: Mapping[str, object] | None = None,
    candidate_metrics: Mapping[str, object] | None = None,
    *,
    min_improvement: float,
    rmse_champion: float | None = None,
    rmse_candidate: float | None = None,
    stability_check_passed: bool | None = None,
    reason: str | None = None,
    evidence: object | None = None,
    warnings: Sequence[object] | None = None,
    errors: Sequence[object] | None = None,
) -> Decision:
    """Build a decision record for rejecting a retrained candidate."""
    champion_metrics = _with_optional_metrics(champion_metrics, rmse_champion=rmse_champion)
    candidate_metrics = _with_optional_metrics(candidate_metrics, rmse_candidate=rmse_candidate)
    metrics = _promotion_metrics(
        champion_metrics=champion_metrics,
        candidate_metrics=candidate_metrics,
        min_improvement=min_improvement,
    )
    if stability_check_passed is not None:
        metrics["stability_check_passed"] = bool(stability_check_passed)

    if reason is None:
        reason = _candidate_rejection_reason(metrics, stability_check_passed)

    decision = _base_decision(
        action="keep_champion",
        final_decision="candidate_rejected",
        reason=reason,
        criteria=[
            "candidate evaluation metrics must exist",
            "rmse_candidate must be below rmse_champion * (1 - min_improvement)",
            "stability check must pass when provided",
        ],
        metrics=metrics,
        warnings=warnings if warnings is not None else _extract_field(evidence, "warnings", []),
        errors=errors if errors is not None else _extract_field(evidence, "errors", []),
        hard_gate_passed=True,
        retrain_recommended=True,
        promotion_recommended=False,
        promotion_executed=False,
    )
    return _finalize(decision)


def build_promotion_decision(
    champion_metrics: Mapping[str, object] | None = None,
    candidate_metrics: Mapping[str, object] | None = None,
    min_improvement: float = 0.01,
    evidence: Mapping[str, object] | None = None,
    *,
    rmse_champion: float | None = None,
    rmse_candidate: float | None = None,
    old_champion_version: object | None = None,
    new_champion_version: object | None = None,
    candidate_version: object | None = None,
    stability_check_passed: bool | None = None,
    reason: str | None = None,
    warnings: Sequence[object] | None = None,
    errors: Sequence[object] | None = None,
) -> Decision:
    """Build a promotion decision, or a rejection if promotion criteria fail.

    The capstone flow can call this from the promotion gate: when candidate
    metrics fail the criteria, the returned decision is ``candidate_rejected``.
    """
    evidence = evidence or {}
    if candidate_version is None:
        candidate_version = evidence.get("candidate_version")
    if stability_check_passed is None:
        maybe_stability = evidence.get("stability_check_passed")
        if maybe_stability is not None:
            stability_check_passed = bool(maybe_stability)

    champion_metrics = _with_optional_metrics(champion_metrics, rmse_champion=rmse_champion)
    candidate_metrics = _with_optional_metrics(candidate_metrics, rmse_candidate=rmse_candidate)
    metrics = _promotion_metrics(
        champion_metrics=champion_metrics,
        candidate_metrics=candidate_metrics,
        min_improvement=min_improvement,
    )
    if stability_check_passed is not None:
        metrics["stability_check_passed"] = bool(stability_check_passed)

    if not _promotion_criteria_pass(metrics, stability_check_passed):
        return build_candidate_rejected_decision(
            champion_metrics=champion_metrics,
            candidate_metrics=candidate_metrics,
            min_improvement=min_improvement,
            rmse_champion=rmse_champion,
            rmse_candidate=rmse_candidate,
            stability_check_passed=stability_check_passed,
            reason=reason,
            evidence=evidence,
            warnings=warnings,
            errors=errors,
        )

    if reason is None:
        reason = "candidate rmse beats champion by at least min_improvement"

    if old_champion_version is not None:
        metrics["old_champion_version"] = old_champion_version
    if candidate_version is not None:
        metrics["candidate_version"] = candidate_version
    if new_champion_version is not None:
        metrics["new_champion_version"] = new_champion_version
    elif candidate_version is not None:
        metrics["new_champion_version"] = candidate_version

    decision = _base_decision(
        action="promote_candidate",
        final_decision="candidate_promoted",
        reason=reason,
        criteria=[
            "hard integrity gate passed",
            "candidate evaluation metrics exist",
            "rmse_candidate < rmse_champion * (1 - min_improvement)",
            "stability check passed when provided",
        ],
        metrics=metrics,
        warnings=warnings if warnings is not None else _extract_field(evidence, "warnings", []),
        errors=errors if errors is not None else _extract_field(evidence, "errors", []),
        hard_gate_passed=True,
        retrain_recommended=True,
        promotion_recommended=True,
        promotion_executed=True,
    )
    return _finalize(decision)


def build_rejection_decision(
    reason: str,
    evidence: Mapping[str, object] | None = None,
) -> Decision:
    """Backward-compatible alias for candidate rejection records."""
    evidence = evidence or {}
    return build_candidate_rejected_decision(
        champion_metrics=_as_mapping(evidence.get("champion_metrics")),
        candidate_metrics=_as_mapping(evidence.get("candidate_metrics")),
        min_improvement=float(evidence.get("min_improvement", 0.01)),
        stability_check_passed=_optional_bool(evidence.get("stability_check_passed")),
        reason=reason,
        evidence=evidence,
    )


def build_decision_tags(decision: Mapping[str, object]) -> dict[str, str]:
    """Return MLflow-safe tags from a decision record."""
    tag_keys = (
        "action",
        "final_decision",
        "hard_gate_passed",
        "retrain_recommended",
        "promotion_recommended",
        "promotion_executed",
    )
    tags: dict[str, str] = {}
    for key in tag_keys:
        value = decision.get(key)
        if isinstance(value, bool):
            tags[key] = str(value).lower()
        elif value is None:
            tags[key] = ""
        else:
            tags[key] = str(value)
    return tags


def assert_json_serializable(decision: Mapping[str, object]) -> None:
    """Raise TypeError when a decision record cannot be written as JSON."""
    try:
        json.dumps(decision, sort_keys=True)
    except TypeError as exc:
        raise TypeError(f"decision is not JSON serializable: {exc}") from exc


def _base_decision(
    *,
    action: str,
    final_decision: str,
    reason: str,
    criteria: Sequence[object],
    metrics: Mapping[str, object] | None,
    warnings: Sequence[object] | None,
    errors: Sequence[object] | None,
    hard_gate_passed: bool,
    retrain_recommended: bool,
    promotion_recommended: bool,
    promotion_executed: bool,
) -> Decision:
    """Create the shared decision.json shape."""
    return {
        "action": action,
        "final_decision": final_decision,
        "reason": reason,
        "criteria": _to_json_safe(list(criteria)),
        "metrics": _to_json_safe(dict(metrics or {})),
        "warnings": _to_json_safe(list(warnings or [])),
        "errors": _to_json_safe(list(errors or [])),
        "hard_gate_passed": bool(hard_gate_passed),
        "retrain_recommended": bool(retrain_recommended),
        "promotion_recommended": bool(promotion_recommended),
        "promotion_executed": bool(promotion_executed),
    }


def _finalize(decision: Decision) -> Decision:
    """Normalize and validate a decision record."""
    normalized = _to_json_safe(decision)
    assert isinstance(normalized, dict)
    assert_json_serializable(normalized)
    return normalized


def _extract_field(source: object, field_name: str, default: object) -> object:
    """Extract a field from a mapping or loose object."""
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(field_name, default)
    return getattr(source, field_name, default)


def _merge_metrics(*sources: Mapping[str, object] | object | None) -> dict[str, object]:
    """Merge metric dictionaries from mappings or result-like objects."""
    merged: dict[str, object] = {}
    for source in sources:
        if source is None:
            continue
        metrics = _extract_field(source, "metrics", source)
        if isinstance(metrics, Mapping):
            merged.update(metrics)
    return merged


def _with_optional_metrics(
    metrics: Mapping[str, object] | None,
    **optional_metrics: object,
) -> dict[str, object]:
    """Return a metric dict updated with non-None explicit keyword metrics."""
    merged = _merge_metrics(metrics)
    _add_optional_metrics(merged, **optional_metrics)
    return merged


def _add_optional_metrics(metrics: dict[str, object], **optional_metrics: object) -> None:
    """Add non-None explicit metric keyword values in place."""
    for key, value in optional_metrics.items():
        if value is not None:
            metrics[key] = value


def _promotion_metrics(
    *,
    champion_metrics: Mapping[str, object] | None,
    candidate_metrics: Mapping[str, object] | None,
    min_improvement: float,
) -> dict[str, object]:
    """Build metric fields needed by promotion/rejection decisions."""
    metrics = _merge_metrics(champion_metrics, candidate_metrics)
    rmse_champion = _first_number(
        champion_metrics,
        "rmse_champion",
        "champion_rmse",
        "rmse",
        "root_mean_squared_error",
    )
    rmse_candidate = _first_number(
        candidate_metrics,
        "rmse_candidate",
        "candidate_rmse",
        "rmse",
        "root_mean_squared_error",
    )

    metrics["rmse_champion"] = rmse_champion
    metrics["rmse_candidate"] = rmse_candidate
    metrics["min_improvement"] = float(min_improvement)
    metrics["required_rmse_candidate_threshold"] = (
        rmse_champion * (1.0 - float(min_improvement))
        if rmse_champion is not None
        else None
    )
    return metrics


def _promotion_criteria_pass(
    metrics: Mapping[str, object],
    stability_check_passed: bool | None,
) -> bool:
    """Return whether candidate promotion criteria pass."""
    rmse_candidate = _as_float(metrics.get("rmse_candidate"))
    required_threshold = _as_float(metrics.get("required_rmse_candidate_threshold"))
    if rmse_candidate is None or required_threshold is None:
        return False
    if not rmse_candidate < required_threshold:
        return False
    if stability_check_passed is False:
        return False
    return True


def _candidate_rejection_reason(
    metrics: Mapping[str, object],
    stability_check_passed: bool | None,
) -> str:
    """Explain why a candidate was not promoted."""
    if _as_float(metrics.get("rmse_champion")) is None:
        return "candidate was not promoted because champion rmse is missing"
    if _as_float(metrics.get("rmse_candidate")) is None:
        return "candidate was not promoted because candidate rmse is missing"
    if stability_check_passed is False:
        return "candidate was not promoted because stability check failed"
    return "candidate rmse did not beat the required improvement threshold"


def _retrain_reason(
    metrics: Mapping[str, object],
    rmse_increase_threshold: float | None,
    soft_integrity: Mapping[str, object] | None,
) -> str:
    """Explain why retraining is recommended."""
    rmse_increase_pct = _first_number(metrics, "rmse_increase_pct", "rmse_increase_pct_vs_baseline")
    if rmse_increase_pct is not None and rmse_increase_threshold is not None:
        return (
            "rmse increase "
            f"{rmse_increase_pct:.6f} exceeded threshold {rmse_increase_threshold:.6f}"
        )
    if bool(_extract_field(soft_integrity, "integrity_warn", False)):
        return "soft monitoring checks raised integrity warnings"
    return "champion evaluation indicates retraining is recommended"


def _first_number(source: Mapping[str, object] | None, *keys: str) -> float | None:
    """Return the first numeric value found for the given keys."""
    if source is None:
        return None
    for key in keys:
        value = _as_float(source.get(key))
        if value is not None:
            return value
    return None


def _as_float(value: object) -> float | None:
    """Convert numeric-like values to float, excluding booleans and NaN."""
    value = _numpy_item(value)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return None


def _optional_bool(value: object) -> bool | None:
    """Return bool(value) unless value is None."""
    if value is None:
        return None
    return bool(value)


def _as_mapping(value: object) -> Mapping[str, object] | None:
    """Return a mapping only when value is mapping-like."""
    return value if isinstance(value, Mapping) else None


def _to_json_safe(value: object) -> object:
    """Recursively convert common scalar containers to JSON-safe values."""
    value = _numpy_item(value)

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_to_json_safe(item) for item in value]

    # Keep custom objects out of decision.json but preserve a readable breadcrumb.
    return str(value)


def _numpy_item(value: object) -> object:
    """Convert NumPy scalar-like objects without importing NumPy."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value
