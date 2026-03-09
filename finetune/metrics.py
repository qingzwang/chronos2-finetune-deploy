"""
Evaluation metrics for time series forecasting.
"""

import numpy as np
import torch
from typing import Dict, Union, Optional


def mae(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Mean Absolute Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional mask for valid values

    Returns:
        MAE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    errors = np.abs(predictions - targets)

    if mask is not None:
        errors = errors[mask > 0]

    return float(np.mean(errors))


def mse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Mean Squared Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional mask for valid values

    Returns:
        MSE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    errors = (predictions - targets) ** 2

    if mask is not None:
        errors = errors[mask > 0]

    return float(np.mean(errors))


def rmse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Root Mean Squared Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional mask for valid values

    Returns:
        RMSE value
    """
    return float(np.sqrt(mse(predictions, targets, mask)))


def smape(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    sMAPE = (200/n) * Σ |pred - target| / (|pred| + |target|)

    Handles zero values gracefully unlike MAPE.
    Range: [0, 200], where 0 is perfect prediction.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional mask for valid values

    Returns:
        sMAPE value (percentage)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    numerator = np.abs(predictions - targets)
    denominator = np.abs(predictions) + np.abs(targets)

    # Avoid division by zero: when both pred and target are 0, error is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(denominator > 0, numerator / denominator, 0.0)

    if mask is not None:
        ratios = ratios[mask > 0]

    return float(200.0 * np.mean(ratios))


def compute_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Compute all forecasting metrics.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional mask for valid values

    Returns:
        Dictionary with MAE, MSE, RMSE, sMAPE
    """
    return {
        "mae": mae(predictions, targets, mask),
        "mse": mse(predictions, targets, mask),
        "rmse": rmse(predictions, targets, mask),
        "smape": smape(predictions, targets, mask),
    }


class MetricsAccumulator:
    """Accumulator for computing metrics over multiple batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated values."""
        self._predictions = []
        self._targets = []
        self._loss_sum = 0.0
        self._loss_count = 0

    def update(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        loss: Optional[float] = None,
    ):
        """
        Add batch predictions and targets.

        Args:
            predictions: Batch predictions
            targets: Batch targets
            loss: Optional loss value for this batch
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self._predictions.append(predictions.flatten())
        self._targets.append(targets.flatten())

        if loss is not None:
            self._loss_sum += loss
            self._loss_count += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute metrics from accumulated values.

        Returns:
            Dictionary with all metrics
        """
        if not self._predictions:
            return {}

        predictions = np.concatenate(self._predictions)
        targets = np.concatenate(self._targets)

        metrics = compute_metrics(predictions, targets)

        if self._loss_count > 0:
            metrics["loss"] = self._loss_sum / self._loss_count

        return metrics
