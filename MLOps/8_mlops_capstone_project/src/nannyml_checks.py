"""NannyML-based monitoring checks for the capstone workflow.

TODO: Add soft-gate data quality, drift, and performance-estimation checks.
"""

from __future__ import annotations


def run_soft_integrity_checks(reference_data: object, current_data: object) -> dict[str, object]:
    """Run warning-oriented NannyML checks comparing reference and current data.

    TODO: Check missingness spikes, unseen categories, and other soft integrity signals.
    """
    raise NotImplementedError("NannyML soft integrity checks are not implemented yet.")


def run_drift_checks(reference_features: object, current_features: object) -> dict[str, object]:
    """Run NannyML drift checks on engineered features.

    TODO: Select methods, thresholds, and artifact formats for capstone reporting.
    """
    raise NotImplementedError("NannyML drift checks are not implemented yet.")


def summarize_nannyml_results(results: dict[str, object]) -> dict[str, object]:
    """Summarize NannyML outputs for MLflow logging and decision policies.

    TODO: Produce compact metrics, tables, and warning flags.
    """
    raise NotImplementedError("NannyML result summarization is not implemented yet.")
