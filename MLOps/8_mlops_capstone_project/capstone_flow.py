"""Metaflow skeleton for the MLOps capstone workflow.

TODO: Fill each step with the concrete capstone implementation.
"""

from __future__ import annotations

from metaflow import FlowSpec, step


class CapstoneFlow(FlowSpec):
    """Manual monitoring, retraining, and promotion workflow skeleton."""

    @step
    def start(self) -> None:
        """Initialize flow parameters and run context.

        TODO: Add parameters for input paths, tracking URI, model name, and gates.
        """
        self.next(self.load_data)

    @step
    def load_data(self) -> None:
        """Load reference and current batch data.

        TODO: Call data_io helpers and record batch metadata.
        """
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self) -> None:
        """Run hard integrity checks and soft NannyML checks.

        TODO: Reject unusable batches and log warning-only findings.
        """
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self) -> None:
        """Create stable model features for reference and current data.

        TODO: Apply shared transformations and log the feature specification.
        """
        self.next(self.load_champion)

    @step
    def load_champion(self) -> None:
        """Load or bootstrap the champion model.

        TODO: Resolve the MLflow champion alias or create the first champion.
        """
        self.next(self.model_gate)

    @step
    def model_gate(self) -> None:
        """Evaluate champion performance and decide whether retraining is needed.

        TODO: Compute metrics, compare thresholds, and persist decision evidence.
        """
        self.next(self.retrain)

    @step
    def retrain(self) -> None:
        """Train a candidate model when the decision policy requires it.

        TODO: Add conditional retraining and candidate evaluation.
        """
        self.next(self.candidate_acceptance)

    @step
    def candidate_acceptance(self) -> None:
        """Apply promotion gates for the candidate model.

        TODO: Register, approve, promote, or reject the candidate.
        """
        self.next(self.end)

    @step
    def end(self) -> None:
        """Finish the flow.

        TODO: Emit a final run summary.
        """


if __name__ == "__main__":
    CapstoneFlow()
