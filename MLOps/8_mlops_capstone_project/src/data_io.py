"""Data loading and path helpers for the Unit 8 capstone."""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a dataset path from the current working directory or a base directory.

    TODO: Reuse the Unit 6 ergonomics where relative paths work from repo root or unit dir.
    """
    raise NotImplementedError("Path resolution is not implemented yet.")


def load_parquet(path: str | Path) -> object:
    """Load a parquet file for reference, batch, or training data.

    TODO: Read parquet with pandas and normalize timestamp columns.
    TODO: Preserve dataset lineage in MLflow via mlflow.log_input.
    """
    raise NotImplementedError("Parquet loading is not implemented yet.")


def load_optional_parquets(paths: list[str] | tuple[str, ...]) -> list[object]:
    """Load optional extra training parquet files.

    TODO: Resolve and load each path while preserving per-dataset lineage.
    """
    raise NotImplementedError("Optional training data loading is not implemented yet.")
