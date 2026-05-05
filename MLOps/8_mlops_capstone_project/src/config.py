"""Configuration helpers for the capstone workflow.

TODO: Define project defaults and runtime configuration loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CapstoneConfig:
    """Runtime configuration for the capstone flow.

    TODO: Add all required paths, MLflow settings, thresholds, and model names.
    """

    project_dir: Path
    tracking_uri: str
    experiment_name: str
    model_name: str


def default_config(project_dir: Path | None = None) -> CapstoneConfig:
    """Build a default capstone configuration.

    TODO: Decide final defaults and allow command-line overrides from the flow.
    """
    raise NotImplementedError("Configuration defaults are not implemented yet.")


def validate_config(config: CapstoneConfig) -> None:
    """Validate a capstone configuration before running the workflow.

    TODO: Check paths, thresholds, registry names, and required settings.
    """
    raise NotImplementedError("Configuration validation is not implemented yet.")
