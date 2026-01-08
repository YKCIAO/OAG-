from __future__ import annotations
import os
from typing import List, Optional

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npy(path: str, arr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def save_expected_value(path: str, v: float) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, np.array([v], dtype=np.float32))


def save_shap_pca_table_xlsx(path: str, shap_values: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    try:
        import pandas as pd
        cols = [f"PC{i+1}" for i in range(shap_values.shape[1])]
        pd.DataFrame(shap_values, columns=cols).to_excel(path, index=False)
    except Exception:
        # fallback to npy
        np.save(path.replace(".xlsx", ".npy"), shap_values)


def save_beeswarm(path_png: str, shap_values: np.ndarray, x_pca: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path_png))
    try:
        import shap
        from matplotlib import pyplot as plt

        exp = shap.Explanation(
            values=shap_values,
            data=x_pca,
            feature_names=[f"PC{i+1}" for i in range(shap_values.shape[1])]
        )
        shap.plots.beeswarm(exp, show=False)
        plt.title("PCA → Age | KernelSHAP")
        plt.savefig(path_png, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("[WARN] beeswarm failed:", e)
