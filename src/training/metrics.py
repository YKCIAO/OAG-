from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def _mae_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Prefer sklearn if available; fallback to numpy.
    """
    try:
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"mae": mae, "r2": r2}
    except Exception:
        return {"mae": _mae_np(y_true, y_pred), "r2": _r2_np(y_true, y_pred)}
