"""Irreversibly hard-clean a local MLflow tracking server for demos.

This utility is destructive. With --yes, it permanently deletes registered
models and garbage-collects deleted experiments/runs/artifacts so experiment
names can be reused. MLflow garbage collection usually needs direct access to
the backend store via --backend-store-uri; using only an HTTP tracking URI may
fail depending on the server backend. Use --dry-run first to inspect what would
be removed.

Examples:
    python MLOps/8_mlops_capstone_project/scripts/hard_clean_mlflow.py \
      --tracking-uri http://127.0.0.1:5000 \
      --dry-run

    python MLOps/8_mlops_capstone_project/scripts/hard_clean_mlflow.py \
      --tracking-uri http://127.0.0.1:5000 \
      --backend-store-uri sqlite:////absolute/path/to/mlflow.db \
      --yes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

import mlflow
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT_ID = "0"
DEFAULT_EXPERIMENT_NAME = "Default"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the irreversible cleanup utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Irreversibly hard-clean an MLflow tracking server by deleting "
            "registered models and garbage-collecting deleted experiments. "
            "Use --dry-run first. Actual deletion requires --yes."
        )
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow tracking URI to clean. Default: {DEFAULT_TRACKING_URI}",
    )
    parser.add_argument(
        "--backend-store-uri",
        default=None,
        help=(
            "Direct MLflow backend store URI for mlflow gc, for example "
            "sqlite:////absolute/path/to/mlflow.db. If omitted, gc falls back "
            "to --tracking-uri and may fail for HTTP tracking servers."
        ),
    )
    parser.add_argument(
        "--artifacts-destination",
        default=None,
        help=(
            "Optional artifacts destination to pass to mlflow gc when using "
            "--backend-store-uri."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform irreversible deletion. Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--include-default",
        action="store_true",
        help=(
            "Also delete and garbage-collect the Default experiment. "
            "By default, experiment id 0 / name Default is skipped."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be deleted without deleting anything or running mlflow gc.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Do not delete registered models.",
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Do not delete or garbage-collect experiments.",
    )
    return parser.parse_args()


def list_registered_models(client: MlflowClient) -> list[str]:
    """Return all registered model names."""
    return [model.name for model in client.search_registered_models()]


def delete_registered_models(
    client: MlflowClient,
    names: list[str],
    dry_run: bool,
) -> int:
    """Delete registered models, continuing after per-model failures."""
    if dry_run:
        return 0

    deleted_count = 0
    for name in names:
        try:
            client.delete_registered_model(name)
            deleted_count += 1
            print(f"Deleted registered model: {name}")
        except Exception as exc:
            print(
                f"ERROR: could not delete registered model {name!r}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    return deleted_count


def list_experiments(
    client: MlflowClient,
    include_default: bool,
) -> list[Any]:
    """Return active and deleted experiments, optionally excluding Default."""
    experiments = client.search_experiments(view_type=ViewType.ALL)
    if include_default:
        return list(experiments)
    return [
        experiment
        for experiment in experiments
        if not (
            str(experiment.experiment_id) == DEFAULT_EXPERIMENT_ID
            or experiment.name == DEFAULT_EXPERIMENT_NAME
        )
    ]


def delete_experiments(
    client: MlflowClient,
    experiments: list[Any],
    dry_run: bool,
) -> tuple[int, list[str]]:
    """Soft-delete active experiments and collect deleted IDs for mlflow gc."""
    if dry_run:
        return 0, []

    soft_deleted_count = 0
    deleted_experiment_ids: list[str] = []
    for experiment in experiments:
        experiment_id = str(experiment.experiment_id)
        lifecycle_stage = str(getattr(experiment, "lifecycle_stage", ""))
        if lifecycle_stage == "deleted":
            deleted_experiment_ids.append(experiment_id)
            continue

        client.delete_experiment(experiment_id)
        soft_deleted_count += 1
        deleted_experiment_ids.append(experiment_id)
        print(f"Soft-deleted experiment: {experiment_id} ({experiment.name})")

    return soft_deleted_count, deleted_experiment_ids


def run_mlflow_gc(
    tracking_uri: str,
    backend_store_uri: str | None,
    artifacts_destination: str | None,
    experiment_ids: list[str],
    dry_run: bool,
) -> bool:
    """Run MLflow garbage collection for deleted experiment IDs."""
    if dry_run:
        return False
    if not experiment_ids:
        print("No deleted experiment IDs to pass to mlflow gc.")
        return True

    command = ["mlflow", "gc"]
    if backend_store_uri:
        command.extend(["--backend-store-uri", backend_store_uri])
        if artifacts_destination:
            command.extend(["--artifacts-destination", artifacts_destination])
    else:
        print(
            "WARNING: No --backend-store-uri supplied; GC via HTTP tracking URI "
            "may fail depending on backend.",
            file=sys.stderr,
        )
        command.extend(["--tracking-uri", tracking_uri])
    command.extend(["--experiment-ids", ",".join(experiment_ids)])

    print(f"Running MLflow garbage collection: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(
            "mlflow gc could not be started. Models may already have been "
            "hard-deleted. Experiments may remain soft-deleted. To hard-delete "
            "experiments, rerun with --backend-store-uri pointing to the "
            "MLflow backend store used by the server."
        ) from exc
    if result.stdout:
        print("mlflow gc stdout:")
        print(result.stdout.rstrip())
    if result.stderr:
        print("mlflow gc stderr:", file=sys.stderr)
        print(result.stderr.rstrip(), file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            "mlflow gc failed. Models may already have been hard-deleted. "
            "Experiments may remain soft-deleted. To hard-delete experiments, "
            "rerun with --backend-store-uri pointing to the MLflow backend "
            "store used by the server."
        )
    return True


def _print_plan(
    model_names: list[str],
    experiments: list[Any],
    skip_models: bool,
    skip_experiments: bool,
    dry_run: bool,
) -> None:
    mode = "DRY RUN" if dry_run else "DELETE"
    print(f"MLflow hard-clean plan ({mode})")
    if skip_models:
        print("Registered models: skipped")
    elif model_names:
        print(f"Registered models to delete ({len(model_names)}):")
        for name in model_names:
            print(f"  - {name}")
    else:
        print("Registered models to delete: none")

    if skip_experiments:
        print("Experiments: skipped")
    elif experiments:
        print(f"Experiments to delete/gc ({len(experiments)}):")
        for experiment in experiments:
            print(
                "  - "
                f"{experiment.experiment_id} "
                f"({experiment.name}, {experiment.lifecycle_stage})"
            )
    else:
        print("Experiments to delete/gc: none")


def main() -> None:
    """Run the irreversible MLflow cleanup workflow."""
    args = parse_args()
    if not args.yes and not args.dry_run:
        print(
            "WARNING: This script permanently deletes MLflow models and "
            "garbage-collects deleted experiments/runs/artifacts. Re-run with "
            "--dry-run to inspect or --yes to delete.",
            file=sys.stderr,
        )
        raise SystemExit(0)

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    try:
        model_names = (
            []
            if args.skip_models
            else list_registered_models(client)
        )
        experiments = (
            []
            if args.skip_experiments
            else list_experiments(client, include_default=args.include_default)
        )
    except (MlflowException, ConnectionError, OSError) as exc:
        print(
            f"ERROR: MLflow tracking server is unreachable at {args.tracking_uri}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception as exc:
        print(
            f"ERROR: could not inspect MLflow tracking server at {args.tracking_uri}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    _print_plan(
        model_names=model_names,
        experiments=experiments,
        skip_models=args.skip_models,
        skip_experiments=args.skip_experiments,
        dry_run=args.dry_run,
    )

    deleted_model_count = delete_registered_models(
        client=client,
        names=model_names,
        dry_run=args.dry_run,
    )
    soft_deleted_experiment_count, deleted_experiment_ids = delete_experiments(
        client=client,
        experiments=experiments,
        dry_run=args.dry_run,
    )
    try:
        gc_succeeded = run_mlflow_gc(
            tracking_uri=args.tracking_uri,
            backend_store_uri=args.backend_store_uri,
            artifacts_destination=args.artifacts_destination,
            experiment_ids=deleted_experiment_ids,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print("Summary:")
    print(f"  registered models deleted: {deleted_model_count}")
    print(f"  experiments soft-deleted: {soft_deleted_experiment_count}")
    print(f"  deleted experiment IDs passed to gc: {len(deleted_experiment_ids)}")
    print(f"  gc succeeded: {gc_succeeded}")


if __name__ == "__main__":
    main()
