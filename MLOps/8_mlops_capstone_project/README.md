# Green Taxi Tip Prediction MLOps Capstone

This README contains the setup, validation commands, demo scenarios for the Unit 8 MLOps capstone.

## Overview

This project implements a Metaflow workflow for NYC Green Taxi tip prediction. It includes raw-data integrity checks, feature engineering, soft monitoring with manual checks and optional NannyML, MLflow tracking, model registry with champion/candidate aliases, retraining/promotion logic, batch inference, and a failure/resume demonstration.

Main workflow:

raw batch -> hard integrity gate -> feature engineering -> soft monitoring -> champion bootstrap/load -> champion evaluation -> retrain decision -> optional candidate retraining -> promotion gate -> batch inference -> end

## Implemented Modules

- `src/config.py`: central defaults, paths, artifact names, aliases, thresholds, and flow parameter defaults.
- `src/data_io.py`: path resolution, parquet loading, dataset IDs, and dataset summaries.
- `src/features.py`: reusable Green Taxi feature engineering with stable numeric/categorical schema.
- `src/quality.py`: hard raw-data integrity gate for schema, datetimes, distances, durations, labels, and domains.
- `src/nannyml_checks.py`: soft monitoring checks for missingness spikes, unseen categories, and optional NannyML drift alerts.
- `src/modeling.py`: sklearn pipelines, preprocessing, RandomForest, CPU/GPU XGBoost, Optuna tuning, and regression evaluation.
- `src/registry.py`: MLflow Model Registry helpers for champion/candidate aliases and promotion tags.
- `src/decisions.py`: JSON-serializable decision records for reject, no-retrain, retrain, reject candidate, and promote candidate.
- `src/mlflow_utils.py`: safe MLflow logging helpers for params, metrics, tags, tables, artifacts, decisions, and dataset lineage.
- `src/inference.py`: offline batch prediction and parquet output utilities.
- `capstone_flow.py`: Metaflow workflow wiring the full monitoring, retraining, promotion, inference, and resume demo.

## Environment Assumptions

- WSL2 Ubuntu.
- Conda environment: `22971-capstone`.
- MLflow server running at `http://127.0.0.1:5000`.
- TLC parquet files under `MLOps/8_mlops_capstone_project/data/TLC_data/`.
- XGBoost GPU is optional. `random_forest` and CPU `xgboost` are supported too.

## Common Setup

```bash
cd ~/dev/OU-22971-Toolbox/MLOps/8_mlops_capstone_project
conda activate 22971-capstone
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

## Optional: Clean MLflow Before Recording

`scripts/hard_clean_mlflow.py` is destructive and should only be used in a local demo environment. It permanently deletes non-default experiments and registered models, then garbage-collects deleted experiments so old experiment names can be reused. Run this before recording if you want Demo Run 1 to clearly show bootstrap with no existing `@champion`.

Dry run first:

```bash
cd ~/dev/OU-22971-Toolbox
conda activate 22971-capstone
python MLOps/8_mlops_capstone_project/scripts/hard_clean_mlflow.py \
  --tracking-uri http://127.0.0.1:5000 \
  --backend-store-uri sqlite:////home/amir_noiboar/dev/OU-22971-Toolbox/mlflow_tracking/mlflow.db \
  --artifacts-destination /home/amir_noiboar/dev/OU-22971-Toolbox/mlflow_tracking/mlruns \
  --dry-run
```

Actual clean:

```bash
cd ~/dev/OU-22971-Toolbox
conda activate 22971-capstone
python MLOps/8_mlops_capstone_project/scripts/hard_clean_mlflow.py \
  --tracking-uri http://127.0.0.1:5000 \
  --backend-store-uri sqlite:////home/amir_noiboar/dev/OU-22971-Toolbox/mlflow_tracking/mlflow.db \
  --artifacts-destination /home/amir_noiboar/dev/OU-22971-Toolbox/mlflow_tracking/mlruns \
  --yes
```

## Validation Commands

```bash
cd ~/dev/OU-22971-Toolbox
conda activate 22971-capstone
git status
python -m py_compile MLOps/8_mlops_capstone_project/capstone_flow.py MLOps/8_mlops_capstone_project/src/*.py
cd MLOps/8_mlops_capstone_project
python capstone_flow.py check
```

## MLflow Server Startup
```bash
cd ~/dev/OU-22971-Toolbox
conda activate 22971-capstone
mlflow server   --workers 1   --host 127.0.0.1   --port 5000   --backend-store-uri sqlite:///mlflow_tracking/mlflow.db   --default-artifact-root "$(pwd)/mlflow_tracking/mlruns"
```

## Demo Run 1: Baseline / Bootstrap / No-Action Path

```bash
cd ~/dev/OU-22971-Toolbox/MLOps/8_mlops_capstone_project
conda activate 22971-capstone
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python capstone_flow.py run \
  --reference_path data/TLC_data/green_tripdata_2020-01.parquet \
  --batch_path data/TLC_data/green_tripdata_2020-01.parquet \
  --model_type xgboost_gpu \
  --retrain_on_soft_warning false \
  --experiment_name 8_green_taxi_capstone_demo_final \
  --model_name green_taxi_tip_model_capstone_demo_final \
  --inference_output_path outputs/predictions_demo_baseline.parquet
```

Expected final summary: `no_retrain`, `retrain_recommended=False`, `promotion_executed=False`.

This is the required baseline/no-action run. It uses the same January 2020 file as both reference and batch.

Because the MLflow registry is clean and no `@champion` exists, the flow bootstraps the initial champion. Bootstrap registers model version 1 and sets `@champion`. 

## Demo Run 2: Retrain + Promotion Using Existing Champion

```bash
cd ~/dev/OU-22971-Toolbox/MLOps/8_mlops_capstone_project
conda activate 22971-capstone
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python capstone_flow.py run \
  --reference_path data/TLC_data/green_tripdata_2020-01.parquet \
  --batch_path data/TLC_data/green_tripdata_2020-04.parquet \
  --model_type xgboost_gpu \
  --retrain_on_soft_warning false \
  --experiment_name 8_green_taxi_capstone_demo_final \
  --model_name green_taxi_tip_model_capstone_demo_final \
  --inference_output_path outputs/predictions_demo_promote.parquet
```

Expected final summary: `candidate_promoted`, `retrain_recommended=True`, `promotion_executed=True`.

This run uses January 2020 as the reference slice and April 2020 as the batch. 
The champion already exists from Run 1, so this run loads the existing `@champion` instead of bootstrapping. Retraining is triggered by performance drift: champion batch RMSE versus champion reference RMSE. The candidate is promoted only if it beats the champion by `min_improvement` and passes stability checks.

## Demo Run 3: Failure / Resume

```bash
cd ~/dev/OU-22971-Toolbox/MLOps/8_mlops_capstone_project
conda activate 22971-capstone
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

rm -f /tmp/capstone_flow_fail_once_batch_inference.marker

python capstone_flow.py run \
  --reference_path data/TLC_data/green_tripdata_2020-04.parquet \
  --batch_path data/TLC_data/green_tripdata_2020-08.parquet \
  --model_type xgboost_gpu \
  --retrain_on_soft_warning false \
  --experiment_name 8_green_taxi_capstone_demo_final \
  --model_name green_taxi_tip_model_capstone_demo_final \
  --inference_output_path outputs/predictions_demo_resume.parquet \
  --fail_at_step batch_inference

python capstone_flow.py resume
```

Expected behavior: the first run intentionally fails at `batch_inference`; `resume` continues from `batch_inference`; predictions are written.
The failure is intentionally triggered at `batch_inference` because earlier expensive steps are already completed. Resume should not rerun previous successful tasks and rerun only the failed step.


## Note

ChatGPT/Codex where used in this project