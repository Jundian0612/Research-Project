"""
Evaluation metrics for geospatial neural adapter.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import r2_score


def compute_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> Tuple[float, float, float]:
    """Compute RMSE, MAE, and R² metrics."""
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    rmse = torch.sqrt(((y_true - y_pred) ** 2).mean()).item()
    mae = torch.abs(y_true - y_pred).mean().item()
    r2 = r2_score(y_true_np, y_pred_np)

    return rmse, mae, r2


def fusion_score(rmse: float, proj_gap: Optional[float], p: Optional[int]) -> float:
    """
    Combine RMSE with an average projection gap.
    Returns a single scalar (lower = better).
    """
    if proj_gap is None or p is None or p == 0:
        return rmse
    avg_gap = proj_gap / p
    return rmse + avg_gap


def frobenius_norm(A, B):
    """
    Frobenius norm between two matrices.
    """
    return float(np.linalg.norm(A - B, ord="fro"))
