"""Data loading and path helpers for the Unit 8 capstone.

This module is intentionally pure: it resolves input paths, reads parquet
files, and returns lightweight dataset metadata. Feature engineering and
MLflow lineage logging live in dedicated modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


UNIT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LoadedDataset:
    """Loaded parquet data plus stable metadata for tags and decisions."""

    path: Path
    batch_id: str
    df: pd.DataFrame
    n_rows: int
    n_cols: int


def resolve_input_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Resolve an input path from cwd, an optional base, Unit 8, or repo root.

    Relative paths are tried in this order: current working directory,
    ``base_dir`` when provided, the Unit 8 folder, and the detected repository
    root. A ``FileNotFoundError`` lists every attempted candidate.
    """
    raw_path = Path(path).expanduser()
    attempted: list[Path] = []

    if raw_path.is_absolute():
        candidate = raw_path.resolve()
        attempted.append(candidate)
        if candidate.exists():
            return candidate
        raise _missing_path_error(path, attempted)

    candidates = [Path.cwd() / raw_path]
    if base_dir is not None:
        candidates.append(Path(base_dir).expanduser() / raw_path)
    candidates.append(UNIT_DIR / raw_path)

    repo_root = _find_repo_root()
    if repo_root is not None:
        candidates.append(repo_root / raw_path)

    for candidate in _unique_paths(candidates):
        resolved = candidate.resolve()
        attempted.append(resolved)
        if resolved.exists():
            return resolved

    raise _missing_path_error(path, attempted)


def load_parquet_dataset(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> LoadedDataset:
    """Load a parquet dataset and return dataframe plus dataset metadata."""
    resolved_path = resolve_input_path(path, base_dir=base_dir)
    try:
        df = pd.read_parquet(resolved_path)
    except ImportError as exc:
        raise ImportError(
            "Parquet support is missing. Install 'pyarrow' or 'fastparquet' "
            "to load TLC parquet files."
        ) from exc

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame from parquet: {resolved_path}")

    return LoadedDataset(
        path=resolved_path,
        batch_id=batch_id_from_path(resolved_path),
        df=df.copy(),
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
    )


def load_reference(path: str | Path, *, base_dir: str | Path | None = None) -> LoadedDataset:
    """Load the reference parquet slice for monitoring and bootstrap training."""
    return load_parquet_dataset(path, base_dir=base_dir)


def load_batch(path: str | Path, *, base_dir: str | Path | None = None) -> LoadedDataset:
    """Load the current batch parquet slice for gates, evaluation, and inference."""
    return load_parquet_dataset(path, base_dir=base_dir)


def load_optional_training_datasets(
    paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    *,
    base_dir: str | Path | None = None,
) -> list[LoadedDataset]:
    """Load optional extra training parquet slices, preserving input order."""
    return [
        load_parquet_dataset(path, base_dir=base_dir)
        for path in parse_extra_train_paths(paths)
    ]


def batch_id_from_path(path: str | Path) -> str:
    """Return a stable, tag-safe batch id from a file path stem."""
    stem = Path(path).stem
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in stem.replace(" ", "_")
    )
    return safe.strip("_") or "dataset"


def dataset_summary(dataset: LoadedDataset) -> dict[str, object]:
    """Return JSON-safe summary metadata for a loaded dataset."""
    return {
        "path": str(dataset.path),
        "batch_id": dataset.batch_id,
        "n_rows": int(dataset.n_rows),
        "n_cols": int(dataset.n_cols),
    }


def datasets_summary(datasets: list[LoadedDataset]) -> list[dict[str, object]]:
    """Return JSON-safe summaries for loaded datasets."""
    return [dataset_summary(dataset) for dataset in datasets]


def parse_extra_train_paths(
    value: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> list[str | Path]:
    """Parse optional extra training path input into an ordered path list.

    ``None`` and empty strings become ``[]``. Strings may be comma-separated.
    Lists and tuples are flattened only for comma-separated string elements and
    empty parts are ignored.
    """
    if value is None:
        return []
    if isinstance(value, Path):
        return [value]
    if isinstance(value, str):
        return _split_path_string(value)

    paths: list[str | Path] = []
    for item in value:
        if isinstance(item, Path):
            paths.append(item)
        elif isinstance(item, str):
            paths.extend(_split_path_string(item))
        elif item is not None:
            paths.append(Path(item))
    return paths


def concatenate_loaded_datasets(datasets: list[LoadedDataset]) -> pd.DataFrame:
    """Concatenate loaded dataset dataframes with a fresh integer index.

    An empty input list returns an empty ``DataFrame`` so callers can decide
    whether optional extra training data is required.
    """
    if not datasets:
        return pd.DataFrame()
    return pd.concat([dataset.df for dataset in datasets], ignore_index=True)


def load_taxi_table(path: str | Path, *, base_dir: str | Path | None = None) -> pd.DataFrame:
    """Compatibility helper returning only the loaded dataframe."""
    return load_parquet_dataset(path, base_dir=base_dir).df


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Compatibility wrapper for ``resolve_input_path``."""
    return resolve_input_path(path, base_dir=base_dir)


def load_parquet(path: str | Path, *, base_dir: str | Path | None = None) -> LoadedDataset:
    """Compatibility wrapper for ``load_parquet_dataset``."""
    return load_parquet_dataset(path, base_dir=base_dir)


def load_optional_parquets(
    paths: list[str] | tuple[str, ...],
    *,
    base_dir: str | Path | None = None,
) -> list[LoadedDataset]:
    """Compatibility wrapper for ``load_optional_training_datasets``."""
    return load_optional_training_datasets(paths, base_dir=base_dir)


def _split_path_string(value: str) -> list[str]:
    """Split a comma-separated path string while ignoring empty parts."""
    return [part.strip() for part in value.split(",") if part.strip()]


def _find_repo_root() -> Optional[Path]:
    """Find the repository root by walking upward from cwd and Unit 8."""
    for start in (Path.cwd(), UNIT_DIR):
        for candidate in (start.resolve(), *start.resolve().parents):
            if (candidate / ".git").exists() or (candidate / "README.md").exists():
                return candidate
    return None


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Return paths with duplicates removed while preserving order."""
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _missing_path_error(path: str | Path, attempted: list[Path]) -> FileNotFoundError:
    """Build a helpful missing-path error with every attempted candidate."""
    attempts = "\n".join(f"- {candidate}" for candidate in attempted)
    return FileNotFoundError(
        f"Could not resolve input path {path!s}. Tried:\n{attempts}"
    )
