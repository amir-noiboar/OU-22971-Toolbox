"""Hard data quality checks for the Unit 8 capstone."""

from __future__ import annotations


def run_hard_integrity_checks(raw_data: object, labels_required: bool = True) -> dict[str, object]:
    """Run fail-fast integrity checks on raw batch data.

    The returned structure should include:
    hard_passed, errors, warnings, metrics, and tables.

    TODO: Check required columns.
    TODO: Check invalid pickup/dropoff datetimes.
    TODO: Check missing target when labels are required.
    TODO: Check negative trip_distance values.
    TODO: Check dropoff before pickup.
    TODO: Return structured errors, warnings, scalar metrics, and small tables.
    """
    raise NotImplementedError("Hard integrity checks are not implemented yet.")


def required_columns() -> tuple[str, ...]:
    """Return the raw columns required by the capstone policy.

    TODO: Finalize the minimum schema for training, monitoring, and inference.
    """
    raise NotImplementedError("Required-column policy is not implemented yet.")


def summarize_hard_checks(check_result: dict[str, object]) -> dict[str, object]:
    """Summarize hard checks for MLflow logging and decision records.

    TODO: Normalize pass/fail fields and user-readable reasons.
    """
    raise NotImplementedError("Hard-check summarization is not implemented yet.")
