"""MLflow Model Registry helpers for the capstone workflow.

TODO: Implement champion lookup, candidate registration, and alias promotion.
"""

from __future__ import annotations


def load_champion(model_name: str) -> object:
    """Load the current champion model by registry alias.

    TODO: Resolve models:/<model_name>@champion and handle bootstrap when missing.
    """
    raise NotImplementedError("Champion loading is not implemented yet.")


def register_candidate(model: object, model_name: str, metadata: dict[str, object]) -> str:
    """Register a trained candidate model version.

    TODO: Log the model, attach tags, and return the created version identifier.
    """
    raise NotImplementedError("Candidate registration is not implemented yet.")


def promote_candidate(model_name: str, candidate_version: str, reason: str) -> None:
    """Promote an approved candidate to champion.

    TODO: Flip the champion alias and tag old and new model versions.
    """
    raise NotImplementedError("Candidate promotion is not implemented yet.")
