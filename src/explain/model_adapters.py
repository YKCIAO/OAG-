from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch


@dataclass
class PCA2AgeWrapper:
    """
    Adapter for SHAP:
    maps PCA-reduced FC features → predicted age
    via inverse PCA + encoder + regressor.

    Callable wrapper for SHAP KernelExplainer.

    Input:  x_pca (N, K) numpy
    Output: y_pred (N,) numpy
    """
    encoder: torch.nn.Module
    regressor: torch.nn.Module
    pca: object  # sklearn PCA
    fc_shape: Tuple[int, int] = (278, 278)
    device: str = "cpu"

    def __post_init__(self):
        self.encoder.to(self.device).eval()
        self.regressor.to(self.device).eval()

    def __call__(self, x_pca: np.ndarray) -> np.ndarray:
        # x_pca: (N, K)
        x_flat = self.pca.inverse_transform(x_pca)  # (N, H*W)
        x_fc = x_flat.reshape((-1, 1, *self.fc_shape)).astype(np.float32)  # (N,1,H,W)

        x_tensor = torch.from_numpy(x_fc).to(self.device)

        with torch.no_grad():
            z_age, z_noise = self.encoder.encode(x_tensor)
            y_pred = self.regressor(z_age)  # expected (N,) or (N,1)

        y = y_pred.detach().cpu().numpy().reshape(-1)
        return y
