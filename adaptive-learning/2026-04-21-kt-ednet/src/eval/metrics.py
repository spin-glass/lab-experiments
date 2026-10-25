"""Evaluation metrics for Knowledge Tracing models.

Provides AUC-ROC, accuracy, log-loss, and calibration helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

__all__ = ["EvalResult", "evaluate_predictions", "calibration_data"]


@dataclass(frozen=True)
class EvalResult:
    """Bundle of evaluation metrics.

    Attributes
    ----------
    auc_roc : float
    accuracy : float
    log_loss : float
    n_samples : int
    """

    auc_roc: float
    accuracy: float
    log_loss: float
    n_samples: int


def evaluate_predictions(
    y_true: npt.NDArray[np.int64],
    y_prob: npt.NDArray[np.float64],
) -> EvalResult:
    """Compute AUC-ROC, accuracy (threshold=0.5), and log-loss.

    Parameters
    ----------
    y_true
        Binary ground-truth labels (0 or 1).
    y_prob
        Predicted probabilities in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return EvalResult(
        auc_roc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        log_loss=float(log_loss(y_true, y_prob)),
        n_samples=len(y_true),
    )


def calibration_data(
    y_true: npt.NDArray[np.int64],
    y_prob: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(fraction_of_positives, mean_predicted_value)`` for a calibration plot.

    Parameters
    ----------
    y_true
        Binary ground-truth labels.
    y_prob
        Predicted probabilities.
    n_bins
        Number of bins for the calibration curve.
    """
    frac_pos, mean_pred = calibration_curve(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_prob, dtype=np.float64),
        n_bins=n_bins,
        strategy="uniform",
    )
    return frac_pos, mean_pred
