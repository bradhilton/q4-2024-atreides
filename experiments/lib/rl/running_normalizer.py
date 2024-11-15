import torch
from typing import Literal

DeviationType = Literal["std", "mad"]


class RunningNormalizer:
    def __init__(self, alpha: float, deviation: DeviationType) -> None:
        """
        Normalizes values using a running mean and deviation.

        Args:
            alpha (float): Alpha parameter [0, 1] for the running mean and deviation:
                - alpha == 0.0: Expanding window/equal-weight normalization.
                - 0.0 < alpha < 1.0: Exponentially weighted normalization.
                - alpha == 1.0: Batch normalization.
            deviation (DeviationType): Type of deviation to use for normalization.
                - "std": Standard deviation.
                - "mad": Mean absolute deviation.
        """
        self.alpha = alpha
        self.deviation = deviation
        self.batch_count = 0
        self.running_mean_sum = 0
        self.running_deviation_sum = 0
        self.running_weight = 0

    @property
    def running_mean(self) -> float:
        return (
            self.running_mean_sum / self.running_weight
            if self.running_weight > 0
            else 0.0
        )

    @property
    def running_deviation(self) -> float:
        return (
            self.running_deviation_sum / self.running_weight
            if self.running_weight > 0
            else 1.0
        )

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the values tensor using running mean and mean absolute deviation.

        Args:
            values (Tensor): Shape (batch_size,)

        Returns:
            Tensor: Normalized values, Shape (batch_size,)
            float: Batch mean
            float: Batch deviation
        """
        batch_mean = float(values.mean())
        batch_deviation = float(
            values.std() if self.deviation == "std" else abs(values - batch_mean).mean()
        )

        self.running_mean_sum = batch_mean + (1 - self.alpha) * self.running_mean_sum
        self.running_deviation_sum = (
            batch_deviation + (1 - self.alpha) * self.running_deviation_sum
        )
        self.running_weight += (1 - self.alpha) ** self.batch_count
        self.batch_count += 1

        return (values - self.running_mean) / (self.running_deviation + 1e-8)
