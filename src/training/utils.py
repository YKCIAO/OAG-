from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch


def reset_seeds(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def z_score_normalize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit z-score parameters on train set only."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def z_score_normalize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def filter_age(data, labels, max_age: Optional[float] = None):
    """
    Robust age filter that works for numpy arrays / torch tensors / lists.
    """
    if max_age is None:
        return data, labels

    # numpy path
    if isinstance(labels, np.ndarray):
        mask = labels <= max_age
        return data[mask], labels[mask]

    # torch path
    if torch.is_tensor(labels):
        mask = labels <= max_age
        return data[mask], labels[mask]

    # list -> numpy
    labels_np = np.asarray(labels)
    mask = labels_np <= max_age
    data_np = np.asarray(data)
    return data_np[mask], labels_np[mask]


def age_to_group(age_value: Union[float, np.ndarray, torch.Tensor],
                 start_age: float,
                 interval: float,
                 num_classes: int,
                 age_unit: str = "YEARS"):
    """
    Map age to discrete groups. If age is in MONTHS, caller should convert before passing.
    """
    if torch.is_tensor(age_value):
        g = torch.floor((age_value - start_age) / interval).long()
        return torch.clamp(g, 0, num_classes - 1)

    age = np.asarray(age_value)
    g = np.floor((age - start_age) / interval).astype(int)
    g = np.clip(g, 0, num_classes - 1)
    return g
