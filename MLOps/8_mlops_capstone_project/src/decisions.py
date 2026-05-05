"""Decision record builders for the Unit 8 capstone."""

from __future__ import annotations


def build_reject_decision(reason: str, evidence: dict[str, object]) -> dict[str, object]:
    """Build a reject-batch decision after hard integrity failure.

    TODO: decision.json must include action, criteria used, metric values, final decision, and reason.
    """
    raise NotImplementedError("Reject decision construction is not implemented yet.")


def build_no_retrain_decision(reason: str, evidence: dict[str, object]) -> dict[str, object]:
    """Build a decision record when retraining is skipped.

    TODO: decision.json must include action, criteria used, metric values, final decision, and reason.
    """
    raise NotImplementedError("No-retrain decision construction is not implemented yet.")


def build_retrain_decision(
    champion_metrics: dict[str, float],
    soft_integrity: dict[str, object],
    rmse_increase_threshold: float,
) -> dict[str, object]:
    """Build a decision record indicating whether retraining is recommended.

    TODO: Compare RMSE degradation, NannyML warnings, and policy thresholds.
    TODO: decision.json must include action, criteria used, metric values, final decision, and reason.
    """
    raise NotImplementedError("Retrain decision construction is not implemented yet.")


def build_promotion_decision(
    champion_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    min_improvement: float,
    evidence: dict[str, object],
) -> dict[str, object]:
    """Build a candidate promotion decision.

    TODO: Enforce min_improvement, stability, integrity sanity, and metric availability.
    TODO: decision.json must include action, criteria used, metric values, final decision, and reason.
    """
    raise NotImplementedError("Promotion decision construction is not implemented yet.")


def build_rejection_decision(reason: str, evidence: dict[str, object]) -> dict[str, object]:
    """Build a candidate rejection decision.

    TODO: Record why a candidate failed promotion criteria.
    TODO: decision.json must include action, criteria used, metric values, final decision, and reason.
    """
    raise NotImplementedError("Candidate rejection decision construction is not implemented yet.")
