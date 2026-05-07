# Green Taxi Tip Prediction MLOps Capstone

This README contains the setup, validation commands, demo scenarios, MLflow inspection guide, and submission checklist for the Unit 8 MLOps capstone.

## Overview

This project implements a Metaflow workflow for NYC Green Taxi tip prediction. It includes raw-data integrity checks, feature engineering, soft monitoring with manual checks and optional NannyML, MLflow tracking, model registry with champion/candidate aliases, retraining/promotion logic, batch inference, and a failure/resume demonstration.

For course submission, this file is the main capstone README/demo guide. The extended duplicate guide is also available in `README_CAPSTONE_DEMO.md`.

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

## Recommended Recording Order

0. Optional hard-clean MLflow so no champion exists.
1. Baseline/no-action run using January as both reference and batch. Because no champion exists, this also demonstrates bootstrap: initial model registration and `@champion` alias creation.
2. Retrain + promotion run using January reference and April batch, with the same `experiment_name` and `model_name` as run 1, so it loads the champion created in run 1.
3. Failure + resume run.

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

Because the MLflow registry is clean and no `@champion` exists, the flow bootstraps the initial champion. Bootstrap registers model version 1 and sets `@champion`. This is not retraining/promotion; it is initial champion creation.

What to show in MLflow:

- `capstone_load_or_bootstrap_champion` run.
- Model Registry version 1.
- `@champion` alias.
- `role=champion`.
- `promotion_reason=bootstrap`.
- `decide_retrain` `decision.json` with `retrain_recommended=false` and `promotion_recommended=false`.

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

This run uses January 2020 as the reference slice and April 2020 as the batch. `--retrain_on_soft_warning false` means soft monitoring warnings are logged but do not trigger retraining.

The champion already exists from Run 1, so this run loads the existing `@champion` instead of bootstrapping. Retraining is triggered by performance drift: champion batch RMSE versus champion reference RMSE. The candidate is promoted only if it beats the champion by `min_improvement` and passes stability checks.

What to show in MLflow:

- `evaluate_champion` metrics: `champion_rmse`, `champion_reference_rmse`, `rmse_increase_pct`, `naive_baseline_rmse`.
- `decide_retrain` `decision.json` with `retrain_recommended=true`.
- `evaluate_candidate` `candidate_rmse`.
- `promotion_gate` `decision.json` with `promotion_executed=true`.
- Model Registry: old version tagged `role=previous_champion`, new version has `@champion`.

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

Expected behavior: the first run intentionally fails at `batch_inference`; `resume` clones previous successful steps and continues from `batch_inference`; predictions are written.

The failure is intentionally triggered at `batch_inference` because earlier expensive steps are already completed. Resume should clone previous successful tasks and rerun only the failed step.

## What To Show In MLflow

- The shared demo experiment: `8_green_taxi_capstone_demo_final`, containing the baseline/bootstrap, retrain/promotion, and failure/resume runs.
- Per-step runs named `capstone_<step>_<batch_id>`.
- Integrity metrics and tables.
- `soft_checks` tables plus NannyML/manual warnings.
- `feature_spec.json`.
- In `evaluate_champion`: `champion_rmse`, `champion_reference_rmse`, `rmse_increase_pct`, `naive_baseline_rmse`.
- In `evaluate_candidate`: `candidate_rmse` and stability metrics.
- In `promotion_gate`: `decision.json` with `final_decision` and `promotion_executed`.
- Model Registry aliases: `champion` and `candidate`.
- Previous champions are tracked by the model-version tag `role=previous_champion`, not necessarily by an alias.
- Version tags: `role`, `validation_status`, `promotion_reason`.
- Inference artifact with predictions parquet.

## What To Show In Metaflow

```bash
python capstone_flow.py show
```

Also show terminal output with successful steps, and resume output showing cloned tasks plus rerun `batch_inference`.

## Known Limitations

- Per-step MLflow runs are used for robustness because Metaflow steps run as separate processes.
- Soft monitoring warnings can be configured to trigger or not trigger retraining using `retrain_on_soft_warning`.
- GPU XGBoost requires a CUDA-visible GPU in WSL. Otherwise use `model_type xgboost` or `model_type random_forest`.
- This project is a course capstone, not a production-grade deployment.

## Submission Checklist

- [ ] `git status` clean
- [ ] `capstone_flow.py check` passes
- [ ] Baseline/bootstrap/no-action run
- [ ] Retrain/promotion run
- [ ] Failure/resume scenario run
- [ ] MLflow `decision.json` visible
- [ ] Predictions parquet visible
- [ ] Model Registry `champion` alias visible
