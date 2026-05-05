# Capstone Working Routine

This project is developed on Windows 11 using VS Code with WSL2 Ubuntu.

## Fixed paths and environment

Repo path inside WSL:

```bash
~/dev/OU-22971-Toolbox
Conda environment:

22971-capstone

MLflow tracking server:

http://127.0.0.1:5000
Start a work session

Open Ubuntu / WSL and run:

cd ~/dev/OU-22971-Toolbox
code .

In every new VS Code WSL terminal:

conda activate 22971-capstone
cd ~/dev/OU-22971-Toolbox
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
Start MLflow server

Use a dedicated terminal and keep it open:

conda activate 22971-capstone
cd ~/dev/OU-22971-Toolbox

mlflow server \
  --workers 1 \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow_tracking/mlflow.db \
  --default-artifact-root "$(pwd)/mlflow_tracking/mlruns"

Open MLflow UI:

http://127.0.0.1:5000
Useful Unit 6 commands

Go to the Unit 6 folder:

conda activate 22971-capstone
cd ~/dev/OU-22971-Toolbox/MLOps/6_monitoring_data_drift
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

Initial training:

python train_initial.py \
  --tracking-uri http://127.0.0.1:5000 \
  --experiment 6_green_taxi_drift_wsl_test \
  --data-parquet TLC_data/green_tripdata_2020-01.parquet \
  --run-name initial_wsl_test

Monitoring:

python check_drift.py \
  --tracking-uri http://127.0.0.1:5000 \
  --experiment 6_green_taxi_drift_wsl_test \
  --ref-parquet TLC_data/green_tripdata_2020-01.parquet \
  --cur-parquet TLC_data/green_tripdata_2020-04.parquet \
  --run-name monitor_wsl_test

Retraining:

python retrain.py \
  --tracking-uri http://127.0.0.1:5000 \
  --experiment 6_green_taxi_drift_wsl_test \
  --train-parquets TLC_data/green_tripdata_2020-01.parquet TLC_data/green_tripdata_2020-04.parquet \
  --eval-parquet TLC_data/green_tripdata_2020-08.parquet \
  --run-name retrain_wsl_test
Git workflow

Current branch:

capstone-mlops-flow

Before work:

git status

After a logical change:

git add <files>
git commit -m "Clear commit message"

Local artifacts that should stay uncommitted are ignored in .gitignore:

.metaflow/
.vscode/
mlflow_tracking/
mlruns/
outputs/
environment-capstone-lock.yml
Environment file

The reproducible environment file is:

environment-capstone.yml

The local lock file is not committed:

environment-capstone-lock.yml
