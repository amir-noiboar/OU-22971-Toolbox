"""MLflow Model Registry operation stubs for the Unit 8 capstone."""

from __future__ import annotations


def get_champion_by_alias(model_name: str) -> object | None:
    """Get the champion model by alias models:/<model_name>@champion.

    TODO: Return None only when the registered model or champion alias is absent.
    """
    raise NotImplementedError("Champion alias lookup is not implemented yet.")


def bootstrap_champion(model: object, model_name: str, metadata: dict[str, object]) -> object:
    """Register the first model version and set the champion alias.

    TODO: Register under model_name, set @champion, and tag role=champion.
    """
    raise NotImplementedError("Champion bootstrap is not implemented yet.")


def register_candidate(model: object, model_name: str, metadata: dict[str, object]) -> str:
    """Register a candidate model version.

    TODO: Log/register the model, set @candidate, and tag role=candidate.
    TODO: Attach trained_on_batches, eval_batch_id, validation_status, and decision_reason.
    """
    raise NotImplementedError("Candidate registration is not implemented yet.")


def promote_candidate(model_name: str, candidate_version: str, reason: str) -> None:
    """Promote a validated candidate to champion.

    TODO: Prevent promotion without evaluation metrics.
    TODO: Tag old champion as role=previous_champion with demoted_at.
    TODO: Tag new champion as role=champion with promoted_at and promotion_reason.
    TODO: Update the @champion alias to candidate_version.
    """
    raise NotImplementedError("Candidate promotion is not implemented yet.")


def set_candidate_alias(model_name: str, candidate_version: str) -> None:
    """Set the candidate alias for a registered model version.

    TODO: Point @candidate to the newly registered candidate version.
    """
    raise NotImplementedError("Candidate alias update is not implemented yet.")
