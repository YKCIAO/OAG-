from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import shap


@dataclass
class KernelShapConfig:
    background_size: int = 20
    nsamples: int = 200
    random_seed: int = 42


def run_kernelshap_on_pca(
    model_wrapper,
    x_pca: np.ndarray,
    cfg: KernelShapConfig
) -> Tuple[np.ndarray, float]:
    """
    Returns:
      shap_values: (N, K) for single-output regression
      expected_value: float
    """
    if x_pca.shape[0] < cfg.background_size:
        bg = x_pca
    else:
        bg = x_pca[:cfg.background_size]

    explainer = shap.KernelExplainer(model_wrapper, bg)

    # guard nsamples if data tiny
    ns = min(cfg.nsamples, max(10, x_pca.shape[0] * 10)) if x_pca.shape[0] < 20 else cfg.nsamples
    shap_values = explainer.shap_values(x_pca, nsamples=ns)

    # KernelExplainer for regression usually returns (N,K) array; sometimes list for multi-output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.array(expected_value).reshape(-1)[0])

    return np.asarray(shap_values), expected_value


def backproject_shap_to_fc(
    shap_values: np.ndarray,
    expected_value: float,
    pca,
    fc_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """
    Convert PCA-space SHAP values back to FC-space contributions.

    shap_values: (N,K)
    PCA components_: (K, H*W)

    returns dict:
      shap_fc_per_sample: (N, H*W)
      shap_fc_mean_map: (H, W)
    """
    # Optionally center by expected value? Usually SHAP values already sum to (pred - expected)
    # So you should NOT subtract expected_value from shap_values.
    # Your previous "delta" was not necessary.
    comps = pca.components_  # (K, H*W)
    shap_fc_per_sample = shap_values @ comps  # (N, H*W)
    shap_fc_mean_map = shap_fc_per_sample.mean(axis=0).reshape(fc_shape)
    return {
        "shap_fc_per_sample": shap_fc_per_sample,
        "shap_fc_mean_map": shap_fc_mean_map
    }
