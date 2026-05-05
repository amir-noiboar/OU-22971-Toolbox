"""Model training helpers for the capstone workflow.

TODO: Implement baseline and candidate model training.
"""

from __future__ import annotations


def train_model(features: object, target: object, params: dict[str, object] | None = None) -> object:
    """Train a candidate regression model.

    TODO: Choose the model family, fit it, and return a serializable model object.
    """
    raise NotImplementedError("Model training is not implemented yet.")


def predict(model: object, features: object) -> object:
    """Generate predictions from a trained model.

    TODO: Validate feature alignment before calling the model.
    """
    raise NotImplementedError("Prediction is not implemented yet.")


def build_training_window(reference_features: object, current_features: object) -> object:
    """Create the training window used for initial training or retraining.

    TODO: Decide rolling versus expanding window behavior.
    """
    raise NotImplementedError("Training window construction is not implemented yet.")
