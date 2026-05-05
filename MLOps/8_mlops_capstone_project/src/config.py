"""Constants and defaults for the Unit 8 MLOps capstone."""

from __future__ import annotations

DEFAULT_MODEL_NAME = "green_taxi_tip_model"
DEFAULT_EXPERIMENT_NAME = "8_green_taxi_capstone"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MIN_IMPROVEMENT = 0.01
DEFAULT_RMSE_INCREASE_THRESHOLD = 0.10
DEFAULT_N_TRIALS = 20

CHAMPION_ALIAS = "champion"
CANDIDATE_ALIAS = "candidate"
PREVIOUS_CHAMPION_ROLE = "previous_champion"

DECISION_ARTIFACT = "decision.json"
FEATURE_SPEC_ARTIFACT = "feature_spec.json"
PREDICTIONS_ARTIFACT = "predictions.parquet"
RAW_INTEGRITY_ARTIFACT_DIR = "raw_integrity"
NANNYML_ARTIFACT_DIR = "nannyml"

TARGET_COLUMN = "tip_amount"
LEAKAGE_COLUMNS = ("tip_amount", "total_amount")
