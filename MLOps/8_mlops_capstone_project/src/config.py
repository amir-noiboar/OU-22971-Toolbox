"""Central defaults and constants for the Unit 8 MLOps capstone.

This module has no runtime side effects. Paths are declared as ``Path`` objects
for flow defaults and artifact planning, but they are not required to exist at
import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


UNIT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = UNIT_DIR / "data"
OUTPUT_DIR = UNIT_DIR / "outputs"

DEFAULT_REFERENCE_PATH = DATA_DIR / "TLC_data" / "green_tripdata_2020-01.parquet"
DEFAULT_BATCH_PATH = DATA_DIR / "TLC_data" / "green_tripdata_2020-04.parquet"
DEFAULT_INFERENCE_OUTPUT_PATH = OUTPUT_DIR / "predictions.parquet"

DEFAULT_MODEL_NAME = "green_taxi_tip_model"
DEFAULT_EXPERIMENT_NAME = "8_green_taxi_capstone"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

CHAMPION_ALIAS = "champion"
CANDIDATE_ALIAS = "candidate"
PREVIOUS_CHAMPION_ROLE = "previous_champion"

DECISION_ARTIFACT = "decision.json"
FEATURE_SPEC_ARTIFACT = "feature_spec.json"
PREDICTIONS_ARTIFACT_DIR = "inference"
INTEGRITY_ARTIFACT_DIR = "integrity"
SOFT_CHECK_ARTIFACT_DIR = "soft_checks"
MODEL_ARTIFACT_PATH = "model"
OPTUNA_TRIALS_ARTIFACT = "optuna_trials.json"

ACTION_REJECT_BATCH = "reject_batch"
ACTION_KEEP_CHAMPION = "keep_champion"
ACTION_RETRAIN_CANDIDATE = "retrain_candidate"
ACTION_PROMOTE_CANDIDATE = "promote_candidate"

TARGET_COLUMN = "tip_amount"
LEAKAGE_COLUMNS = ("tip_amount", "total_amount")

# Compatibility aliases kept for earlier skeleton names.
DEFAULT_MIN_IMPROVEMENT = 0.01
DEFAULT_RMSE_INCREASE_THRESHOLD = 0.10
DEFAULT_N_TRIALS = 20
PREDICTIONS_ARTIFACT = "predictions.parquet"
RAW_INTEGRITY_ARTIFACT_DIR = INTEGRITY_ARTIFACT_DIR
NANNYML_ARTIFACT_DIR = SOFT_CHECK_ARTIFACT_DIR


@dataclass(frozen=True)
class CapstoneDefaults:
    """Default command and workflow values for the capstone flow."""

    tracking_uri: str = DEFAULT_TRACKING_URI
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    model_name: str = DEFAULT_MODEL_NAME
    min_improvement: float = 0.01
    rmse_increase_threshold: float = 0.10
    missingness_warn_threshold: float = 0.10
    unseen_category_warn_threshold: float = 0.02
    model_type: str = "random_forest"
    random_state: int = 0
    use_optuna: bool = False
    n_trials: int = DEFAULT_N_TRIALS
    test_size: float = 0.2
    n_jobs: int = -1
    prediction_column: str = "predicted_tip_amount"


def defaults() -> CapstoneDefaults:
    """Return a fresh immutable defaults object."""
    return CapstoneDefaults()


def default_params_dict() -> dict[str, object]:
    """Return flow-ready default parameters and common paths as a dictionary."""
    values = defaults()
    return {
        "tracking_uri": values.tracking_uri,
        "experiment_name": values.experiment_name,
        "model_name": values.model_name,
        "reference_path": DEFAULT_REFERENCE_PATH,
        "batch_path": DEFAULT_BATCH_PATH,
        "extra_train_paths": "",
        "inference_output_path": DEFAULT_INFERENCE_OUTPUT_PATH,
        "min_improvement": values.min_improvement,
        "rmse_increase_threshold": values.rmse_increase_threshold,
        "missingness_warn_threshold": values.missingness_warn_threshold,
        "unseen_category_warn_threshold": values.unseen_category_warn_threshold,
        "model_type": values.model_type,
        "random_state": values.random_state,
        "use_optuna": values.use_optuna,
        "n_trials": values.n_trials,
        "test_size": values.test_size,
        "n_jobs": values.n_jobs,
        "prediction_column": values.prediction_column,
        "decision_artifact": DECISION_ARTIFACT,
        "feature_spec_artifact": FEATURE_SPEC_ARTIFACT,
        "predictions_artifact_dir": PREDICTIONS_ARTIFACT_DIR,
        "integrity_artifact_dir": INTEGRITY_ARTIFACT_DIR,
        "soft_check_artifact_dir": SOFT_CHECK_ARTIFACT_DIR,
        "model_artifact_path": MODEL_ARTIFACT_PATH,
        "optuna_trials_artifact": OPTUNA_TRIALS_ARTIFACT,
        "champion_alias": CHAMPION_ALIAS,
        "candidate_alias": CANDIDATE_ALIAS,
    }
