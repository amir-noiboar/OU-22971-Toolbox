# Green Taxi Tip Prediction MLOps Capstone

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

## Validation Commands

```bash
cd ~/dev/OU-22971-Toolbox
conda activate 22971-capstone
git status
python -m py_compile MLOps/8_mlops_capstone_project/capstone_flow.py MLOps/8_mlops_capstone_project/src/*.py
cd MLOps/8_mlops_capstone_project
python capstone_flow.py check
```

## Demo Scenario A: Promotion / Retraining Path

```bash
cd ~/dev/OU-22971-Toolbox/MLOps/8_mlops_capstone_project
conda activate 22971-capstone
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python capstone_flow.py run \
  --reference_path data/TLC_data/green_tripdata_2020-01.parquet \
  --batch_path data/TLC_data/green_tripdata_2020-04.parquet \
  --model_type xgboost_gpu \
  --retrain_on_soft_warning false \
  --experiment_name 8_green_taxi_capstone_demo \
  --model_name green_taxi_tip_model_capstone_demo \
  --inference_output_path outputs/predictions_demo_promote.parquet
```

Expected final summary: `candidate_promoted`, `retrain_recommended=True`, `promotion_executed=True`.

## Demo Scenario B: No-Retrain Path

```bash
cd ~/dev/OU-22971-Toolbox/MLOps/8_mlops_capstone_project
conda activate 22971-capstone
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python capstone_flow.py run \
  --reference_path data/TLC_data/green_tripdata_2020-01.parquet \
  --batch_path data/TLC_data/green_tripdata_2020-01.parquet \
  --model_type xgboost_gpu \
  --retrain_on_soft_warning false \
  --experiment_name 8_green_taxi_capstone_demo_no_retrain \
  --model_name green_taxi_tip_model_capstone_demo_no_retrain \
  --inference_output_path outputs/predictions_demo_no_retrain.parquet
```

Expected final summary: `keep_champion`, `no_retrain`, `retrain_recommended=False`, `promotion_executed=False`.

## Demo Scenario C: Failure / Resume

```bash
cd ~/dev/OU-22971-Toolbox/MLOps/8_mlops_capstone_project
conda activate 22971-capstone
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

rm -f /tmp/capstone_flow_fail_once_batch_inference.marker

python capstone_flow.py run \
  --reference_path data/TLC_data/green_tripdata_2020-01.parquet \
  --batch_path data/TLC_data/green_tripdata_2020-04.parquet \
  --model_type xgboost_gpu \
  --experiment_name 8_green_taxi_capstone_demo_resume \
  --model_name green_taxi_tip_model_capstone_demo_resume \
  --inference_output_path outputs/predictions_demo_resume.parquet \
  --fail_at_step batch_inference

python capstone_flow.py resume
```

Expected behavior: the first run intentionally fails at `batch_inference`; `resume` clones previous successful steps and continues from `batch_inference`; predictions are written.

## What To Show In MLflow

- Experiments for each demo scenario.
- Per-step runs named `capstone_<step>_<batch_id>`.
- Integrity metrics and tables.
- `soft_checks` tables plus NannyML/manual warnings.
- `feature_spec.json`.
- `decision.json`.
- Model Registry aliases: `champion` and `candidate`.
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
- [ ] Promotion scenario run
- [ ] No-retrain scenario run
- [ ] Failure/resume scenario run
- [ ] MLflow `decision.json` visible
- [ ] Predictions parquet visible
- [ ] Model Registry `champion` alias visible
