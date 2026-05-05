"""Data loading and persistence helpers for the capstone workflow.

TODO: Implement robust loaders for TLC data and decision artifacts.
"""

from __future__ import annotations

from pathlib import Path


def resolve_input_path(path: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve user-provided input paths for capstone runs.

    TODO: Mirror the path ergonomics from earlier units without duplicating logic blindly.
    """
    raise NotImplementedError("Input path resolution is not implemented yet.")


def load_batch(path: str | Path) -> object:
    """Load a raw batch dataset.

    TODO: Support the selected capstone data format and datetime normalization.
    """
    raise NotImplementedError("Batch loading is not implemented yet.")


def write_decision(decision: dict[str, object], path: str | Path) -> None:
    """Persist a workflow decision artifact.

    TODO: Write reproducible JSON with all gate inputs and final actions.
    """
    raise NotImplementedError("Decision persistence is not implemented yet.")
