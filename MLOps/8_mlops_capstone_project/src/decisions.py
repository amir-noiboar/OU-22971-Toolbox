"""Decision policy stubs for retraining and promotion.

TODO: Encode auditable policies for reject, retrain, promote, and hold decisions.
"""

from __future__ import annotations


def decide_retrain(
    champion_metrics: dict[str, float],
    drift_summary: dict[str, object],
    thresholds: dict[str, float],
) -> dict[str, object]:
    """Decide whether retraining should run for the current batch.

    TODO: Combine model degradation, drift signals, and integrity warnings.
    """
    raise NotImplementedError("Retrain decision policy is not implemented yet.")


def decide_promotion(
    champion_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, object]:
    """Decide whether a candidate model should become champion.

    TODO: Enforce meaningful improvement, stability, and integrity sanity criteria.
    """
    raise NotImplementedError("Promotion decision policy is not implemented yet.")


def build_decision_record(action: str, reason: str, evidence: dict[str, object]) -> dict[str, object]:
    """Build the serializable decision record logged for every run.

    TODO: Include criteria, metric values, artifact references, and final action.
    """
    raise NotImplementedError("Decision record construction is not implemented yet.")
