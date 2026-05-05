"""Warning-only NannyML monitoring checks for the Unit 8 capstone."""

from __future__ import annotations


def run_soft_monitoring_checks(reference_data: object, current_data: object) -> dict[str, object]:
    """Run NannyML soft checks without failing the workflow automatically.

    TODO: Detect missingness spikes versus reference.
    TODO: Detect unseen categorical values versus reference.
    TODO: Produce a drift summary for raw and/or engineered features.
    TODO: Set warning tag integrity_warn=true when soft checks cross thresholds.
    TODO: Never fail the workflow automatically from NannyML checks.
    """
    raise NotImplementedError("NannyML soft monitoring checks are not implemented yet.")


def run_drift_summary(reference_features: object, current_features: object) -> dict[str, object]:
    """Build a compact NannyML drift summary for decision policies.

    TODO: Select NannyML calculators, chunking, thresholds, and artifact tables.
    """
    raise NotImplementedError("NannyML drift summary is not implemented yet.")


def format_nannyml_artifacts(results: dict[str, object]) -> dict[str, object]:
    """Prepare NannyML outputs for MLflow tables and artifacts.

    TODO: Return small JSON-table-friendly summaries and links to richer reports if generated.
    """
    raise NotImplementedError("NannyML artifact formatting is not implemented yet.")
