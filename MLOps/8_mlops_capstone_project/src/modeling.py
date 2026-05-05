"""Model training, tuning, and evaluation stubs for the Unit 8 capstone."""

from __future__ import annotations


def train_initial_model(features: object, use_optuna: bool = False, n_trials: int = 20) -> object:
    """Train the first champion model from the reference dataset.

    TODO: Choose a baseline model, such as RandomForestRegressor or XGBoost.
    TODO: Optionally call Optuna tuning when use_optuna is true.
    """
    raise NotImplementedError("Initial model training is not implemented yet.")


def train_candidate_model(
    reference_features: object,
    batch_features: object,
    extra_train_features: list[object] | None = None,
    use_optuna: bool = False,
    n_trials: int = 20,
) -> object:
    """Train a candidate model for a retraining run.

    TODO: Build a rolling or expanding training window.
    TODO: Consider RandomForestRegressor for a simple baseline.
    TODO: Consider XGBoost when the environment supports it.
    TODO: Optionally tune hyperparameters with Optuna and log child trials to MLflow.
    """
    raise NotImplementedError("Candidate model training is not implemented yet.")


def tune_with_optuna(training_data: object, n_trials: int = 20) -> dict[str, object]:
    """Run optional Optuna hyperparameter tuning.

    TODO: Define an objective, sampler, pruner, search space, and MLflow trial logging.
    """
    raise NotImplementedError("Optuna tuning is not implemented yet.")


def evaluate_regression(model: object, features: object) -> dict[str, float]:
    """Evaluate regression metrics for champion or candidate models.

    TODO: Compute RMSE, MAE, and any stability/slice diagnostics used by gates.
    """
    raise NotImplementedError("Regression evaluation is not implemented yet.")
