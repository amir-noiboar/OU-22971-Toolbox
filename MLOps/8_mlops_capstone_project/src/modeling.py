"""Model training, tuning, and evaluation utilities for the Unit 8 capstone."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for capstone model training."""

    model_type: str = "random_forest"
    random_state: int = 0
    use_optuna: bool = False
    n_trials: int = 20
    test_size: float = 0.2
    n_jobs: int = -1


@dataclass
class TrainResult:
    """Result object returned by model training utilities."""

    model: object
    feature_spec: object
    model_config: ModelConfig
    train_metrics: dict[str, float | int]
    validation_metrics: dict[str, float | int]
    best_params: dict[str, object]
    optuna_trials: Optional[pd.DataFrame] = None


def build_preprocessor(feature_spec: object) -> ColumnTransformer:
    """Build preprocessing from a FeatureSpec-like object."""
    numeric_features = _list_field(feature_spec, "numeric_features")
    categorical_features = _list_field(feature_spec, "categorical_features")
    if not numeric_features and not categorical_features:
        raise ValueError("FeatureSpec must contain numeric or categorical features.")

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_estimator(
    *,
    model_type: str,
    random_state: int = 0,
    n_jobs: int = -1,
    params: Mapping[str, object] | None = None,
) -> object:
    """Build a random_forest, xgboost, or xgboost_gpu regression estimator."""
    normalized = _normalize_model_type(model_type)
    final_params = _default_estimator_params(
        model_type=normalized,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    if params:
        final_params.update(dict(params))

    if normalized == "random_forest":
        return RandomForestRegressor(**final_params)
    if normalized in {"xgboost", "xgboost_gpu"}:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install xgboost or use "
                "model_type='random_forest'."
            ) from exc
        return XGBRegressor(**final_params)

    raise ValueError(
        f"Unsupported model_type: {model_type!r}. Expected "
        "'random_forest', 'xgboost', or 'xgboost_gpu'."
    )


def build_model_pipeline(
    feature_spec: object,
    *,
    model_type: str = "random_forest",
    random_state: int = 0,
    n_jobs: int = -1,
    params: Mapping[str, object] | None = None,
) -> Pipeline:
    """Build a self-contained sklearn model pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_spec)),
            (
                "regressor",
                build_estimator(
                    model_type=model_type,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    params=params,
                ),
            ),
        ]
    )


def evaluate_regression_model(
    model: object,
    feature_frame: object,
    *,
    metric_prefix: str | None = None,
) -> dict[str, float | int]:
    """Evaluate a fitted regression model on a labeled FeatureFrame."""
    X, y, _ = _validate_feature_frame(feature_frame, require_y=True)
    predictions = np.asarray(model.predict(X), dtype=float)
    return _regression_metrics(y_true=y, y_pred=predictions, metric_prefix=metric_prefix)


def train_model(
    feature_frame: object,
    *,
    config: ModelConfig | None = None,
    params: Mapping[str, object] | None = None,
) -> TrainResult:
    """Train a model pipeline from a labeled FeatureFrame."""
    config = config or ModelConfig()
    X, y, feature_spec = _validate_feature_frame(feature_frame, require_y=True)
    if len(X) < 2:
        raise ValueError("At least two labeled rows are required for train/validation split.")

    if config.use_optuna:
        best_params, optuna_trials, train_metrics, validation_metrics = _train_with_optuna(
            X=X,
            y=y,
            feature_spec=feature_spec,
            config=config,
        )
    else:
        best_params = _best_params_for_return(
            model_type=config.model_type,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            params=params,
        )
        train_metrics, validation_metrics = _fit_and_score_split(
            X=X,
            y=y,
            feature_spec=feature_spec,
            config=config,
            params=params,
        )
        optuna_trials = None

    final_model = build_model_pipeline(
        feature_spec,
        model_type=config.model_type,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        params=best_params,
    )
    final_model.fit(X.copy(), y.copy())

    return TrainResult(
        model=final_model,
        feature_spec=feature_spec,
        model_config=config,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_params=best_params,
        optuna_trials=optuna_trials,
    )


def train_candidate_model(
    feature_frame: object | None = None,
    *,
    config: ModelConfig | None = None,
    reference_features: object | None = None,
    batch_features: object | None = None,
    extra_train_features: list[object] | None = None,
    use_optuna: bool | None = None,
    n_trials: int | None = None,
) -> TrainResult:
    """Train a candidate model.

    ``feature_frame`` is the preferred API. The keyword-only compatibility
    inputs allow the current flow skeleton to pass reference/batch/extra frames
    until it is wired to assemble a training window explicitly.
    """
    if feature_frame is None:
        feature_frame = _combine_feature_frames(
            reference_features=reference_features,
            batch_features=batch_features,
            extra_train_features=extra_train_features or [],
        )
    if config is None and (use_optuna is not None or n_trials is not None):
        config = ModelConfig(
            use_optuna=bool(use_optuna),
            n_trials=20 if n_trials is None else int(n_trials),
        )
    return train_model(feature_frame, config=config)


def train_initial_model(
    features: object,
    use_optuna: bool = False,
    n_trials: int = 20,
) -> TrainResult:
    """Train the first champion model from reference features."""
    return train_model(
        features,
        config=ModelConfig(use_optuna=use_optuna, n_trials=n_trials),
    )


def tune_with_optuna(training_data: object, n_trials: int = 20) -> dict[str, object]:
    """Return best Optuna params for a labeled FeatureFrame."""
    config = ModelConfig(use_optuna=True, n_trials=n_trials)
    X, y, feature_spec = _validate_feature_frame(training_data, require_y=True)
    best_params, _, _, _ = _train_with_optuna(
        X=X,
        y=y,
        feature_spec=feature_spec,
        config=config,
    )
    return best_params


def evaluate_regression(model: object, features: object) -> dict[str, float]:
    """Compatibility wrapper for champion/candidate evaluation."""
    return {
        key: float(value)
        for key, value in evaluate_regression_model(model, features).items()
    }


def _fit_and_score_split(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_spec: object,
    config: ModelConfig,
    params: Mapping[str, object] | None,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """Fit on a split and return train/validation metrics."""
    X_train, X_val, y_train, y_val = train_test_split(
        X.copy(),
        y.copy(),
        test_size=config.test_size,
        random_state=config.random_state,
    )
    model = build_model_pipeline(
        feature_spec,
        model_type=config.model_type,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        params=params,
    )
    model.fit(X_train, y_train)
    train_metrics = _regression_metrics(
        y_true=y_train,
        y_pred=np.asarray(model.predict(X_train), dtype=float),
        metric_prefix="train",
    )
    validation_metrics = _regression_metrics(
        y_true=y_val,
        y_pred=np.asarray(model.predict(X_val), dtype=float),
        metric_prefix="validation",
    )
    return train_metrics, validation_metrics


def _train_with_optuna(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_spec: object,
    config: ModelConfig,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, float | int], dict[str, float | int]]:
    """Tune hyperparameters with Optuna and score the best split model."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is not installed. Install optuna or set use_optuna=False."
        ) from exc

    X_train, X_val, y_train, y_val = train_test_split(
        X.copy(),
        y.copy(),
        test_size=config.test_size,
        random_state=config.random_state,
    )

    def objective(trial: object) -> float:
        params = _sample_params(trial, config.model_type)
        model = build_model_pipeline(
            feature_spec,
            model_type=config.model_type,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            params=params,
        )
        model.fit(X_train, y_train)
        predictions = np.asarray(model.predict(X_val), dtype=float)
        return _rmse(y_val, predictions)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=max(1, int(config.n_trials)))

    best_params = dict(study.best_params)
    best_model = build_model_pipeline(
        feature_spec,
        model_type=config.model_type,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        params=best_params,
    )
    best_model.fit(X_train, y_train)
    train_metrics = _regression_metrics(
        y_true=y_train,
        y_pred=np.asarray(best_model.predict(X_train), dtype=float),
        metric_prefix="train",
    )
    validation_metrics = _regression_metrics(
        y_true=y_val,
        y_pred=np.asarray(best_model.predict(X_val), dtype=float),
        metric_prefix="validation",
    )
    return best_params, study.trials_dataframe(), train_metrics, validation_metrics


def _sample_params(trial: object, model_type: str) -> dict[str, object]:
    """Sample bounded Optuna params for supported model types."""
    normalized = _normalize_model_type(model_type)
    if normalized == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220),
            "max_depth": trial.suggest_int("max_depth", 6, 18),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 120),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.6, 0.8, 1.0]),
        }
    if normalized in {"xgboost", "xgboost_gpu"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    raise ValueError(
        f"Unsupported model_type for Optuna: {model_type!r}. Expected "
        "'random_forest', 'xgboost', or 'xgboost_gpu'."
    )


def _validate_feature_frame(
    feature_frame: object,
    *,
    require_y: bool,
) -> tuple[pd.DataFrame, np.ndarray, object]:
    """Validate a FeatureFrame-like object and return X, y, spec."""
    X = getattr(feature_frame, "X", None)
    y = getattr(feature_frame, "y", None)
    feature_spec = getattr(feature_frame, "spec", None)

    if not isinstance(X, pd.DataFrame):
        raise TypeError("feature_frame.X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("feature_frame.X is empty.")
    if feature_spec is None:
        raise ValueError("feature_frame.spec is required.")
    if not _list_field(feature_spec, "numeric_features") and not _list_field(
        feature_spec,
        "categorical_features",
    ):
        raise ValueError("FeatureSpec must include numeric or categorical features.")
    if require_y and y is None:
        raise ValueError("feature_frame.y is required for training/evaluation.")

    y_array = np.asarray(y, dtype=float) if y is not None else np.asarray([], dtype=float)
    if require_y and len(y_array) != len(X):
        raise ValueError("feature_frame.y length must match feature_frame.X rows.")
    return X.copy(), y_array, feature_spec


def _combine_feature_frames(
    *,
    reference_features: object | None,
    batch_features: object | None,
    extra_train_features: Sequence[object],
) -> object:
    """Combine FeatureFrame-like objects for compatibility training windows."""
    frames = [frame for frame in [reference_features, *extra_train_features, batch_features] if frame is not None]
    if not frames:
        raise ValueError("No feature frames were provided for candidate training.")

    first = frames[0]
    feature_spec = getattr(first, "spec", None)
    X_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []
    for frame in frames:
        X, y, _ = _validate_feature_frame(frame, require_y=True)
        X_parts.append(X)
        y_parts.append(y)

    class _CombinedFeatureFrame:
        pass

    combined = _CombinedFeatureFrame()
    combined.X = pd.concat(X_parts, ignore_index=True)
    combined.y = np.concatenate(y_parts)
    combined.spec = feature_spec
    combined.row_ids = None
    return combined


def _regression_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_prefix: str | None = None,
) -> dict[str, float | int]:
    """Compute standard regression metrics."""
    metrics = {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_obs": int(len(y_true)),
    }
    if metric_prefix:
        return {f"{metric_prefix}_{key}": value for key, value in metrics.items()}
    return metrics


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE with sklearn-version compatibility."""
    try:
        from sklearn.metrics import root_mean_squared_error

        return float(root_mean_squared_error(y_true, y_pred))
    except ImportError:
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _default_estimator_params(
    *,
    model_type: str,
    random_state: int,
    n_jobs: int,
) -> dict[str, object]:
    """Return default estimator params for supported model types."""
    normalized = _normalize_model_type(model_type)
    if normalized == "random_forest":
        return {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_leaf": 50,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
    if normalized == "xgboost":
        return {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
    if normalized == "xgboost_gpu":
        return {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda",
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
    raise ValueError(
        f"Unsupported model_type: {model_type!r}. Expected "
        "'random_forest', 'xgboost', or 'xgboost_gpu'."
    )


def _best_params_for_return(
    *,
    model_type: str,
    random_state: int,
    n_jobs: int,
    params: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return default params plus user overrides for TrainResult.best_params."""
    best_params = _default_estimator_params(
        model_type=model_type,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    if params:
        best_params.update(dict(params))
    return best_params


def _normalize_model_type(model_type: str) -> str:
    """Normalize model type aliases."""
    normalized = model_type.strip().lower()
    if normalized in {"random_forest", "rf"}:
        return "random_forest"
    if normalized in {"xgboost", "xgb"}:
        return "xgboost"
    if normalized in {"xgboost_gpu", "xgb_gpu", "gpu_xgboost"}:
        return "xgboost_gpu"
    return normalized


def _list_field(source: object, field_name: str) -> list[str]:
    """Extract a string list field from a FeatureSpec-like object."""
    if isinstance(source, Mapping):
        value = source.get(field_name, [])
    else:
        value = getattr(source, field_name, [])
    if value is None:
        return []
    return [str(item) for item in list(value)]
