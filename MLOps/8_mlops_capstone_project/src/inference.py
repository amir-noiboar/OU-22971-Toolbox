"""Inference helpers for the capstone model.

TODO: Keep serving-time feature preparation consistent with training.
"""

from __future__ import annotations


def prepare_inference_frame(raw_records: object, feature_spec: dict[str, object]) -> object:
    """Prepare raw records for model inference.

    TODO: Apply the production feature spec and reject incompatible inputs.
    """
    raise NotImplementedError("Inference frame preparation is not implemented yet.")


def run_inference(model: object, features: object) -> object:
    """Run model inference on prepared features.

    TODO: Return predictions in the response shape selected for deployment.
    """
    raise NotImplementedError("Inference execution is not implemented yet.")
