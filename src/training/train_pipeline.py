from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.utils import reset_seeds, z_score_normalize_fit, z_score_normalize_apply
from src.training.stage1_train import train_stage1
from src.training.stage2_train import train_stage2
from src.training.metrics import compute_metrics
from src.training.io_training import save_json, try_export_csv

# 你的模型 / loss（按你现有文件名改一下 import 即可）
from src.models.OAG_CAE import OrthogonalAutoEncoder
from src.models.regressors import ConvAgeRegressor, ConvAgeRegressorConfig
from src.training.losses import orthogonal_guided_loss  # <- 如果你文件名不是 losses.py，改这里即可


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"  # or "cpu"
    num_workers: int = 0
    batch_size: int = 3

    # stage1
    epochs_stage1: int = 20
    lr_stage1: float = 1e-3
    wd_stage1: float = 1e-4

    # stage2
    epochs_stage2: int = 50
    lr_stage2: float = 1e-3
    wd_stage2: float = 1e-4
    early_stop_patience: int = 10

    # loss weights
    w_recon: float = 1.0
    w_age: float = 0.3
    w_ortho: float = 0.3
    w_class: float = 0.01

    grad_clip: float = 5.0
    verbose: bool = True

    # model dims
    input_dim: int = 278
    z_age_dim: int = 32
    z_noise_dim: int = 32

    # outputs
    out_dir: str = "outputs"
    # inputs
    input_dir: str = "inputs"

def _resolve_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_loaders(train_ds, val_ds, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, val_loader


def _build_models(cfg: TrainConfig, device: torch.device):
    encoder = OrthogonalAutoEncoder(cfg.input_dim, cfg.z_age_dim, cfg.z_noise_dim).to(device)

    # Stage2 regressor（你可改成 AttentionRegressor 或你的配置）
    reg_cfg = ConvAgeRegressorConfig(in_dim=cfg.z_age_dim, hidden_channels=4, length=cfg.z_age_dim // 4)
    regressor = ConvAgeRegressor(reg_cfg).to(device)
    return encoder, regressor


def train_and_eval(
    folds: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    train_dataset_ctor,
    cfg: TrainConfig
) -> float:
    """
    folds: list of (train_x, train_y, val_x, val_y)
    train_dataset_ctor: callable -> (train_ds, val_ds) from arrays
        你把你原本的 Dataset 类（FCDataset / fMRIDataset 等）用一个 lambda 包一层传进来即可
    """
    reset_seeds(cfg.seed)
    device = _resolve_device(cfg)

    fold_mae = []

    for i, (train_x, train_y, val_x, val_y) in enumerate(folds):
        print(f"\n===== Fold {i+1}/{len(folds)} =====")

        # 只用 train 拟合 normalize，再 apply 到 val
        mean, std = z_score_normalize_fit(train_x)
        train_x_n = z_score_normalize_apply(train_x, mean, std)
        val_x_n = z_score_normalize_apply(val_x, mean, std)

        train_ds, val_ds = train_dataset_ctor(train_x_n, train_y, val_x_n, val_y)

        train_loader, val_loader = _build_loaders(train_ds, val_ds, cfg)

        encoder, regressor = _build_models(cfg, device)

        optimizer1 = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr_stage1, weight_decay=cfg.wd_stage1)
        s1 = train_stage1(
            model=encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer1,
            loss_fn=orthogonal_guided_loss,
            device=device,
            cfg=cfg
        )
        encoder.load_state_dict(s1.best_state_dict)

        optimizer2 = torch.optim.AdamW(regressor.parameters(), lr=cfg.lr_stage2, weight_decay=cfg.wd_stage2)
        s2 = train_stage2(
            encoder=encoder,
            regressor=regressor,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer2,
            device=device,
            cfg=cfg
        )
        fold_mae.append(s2.best_val_mae)

        # 输出 fold 结果
        save_json(f"{cfg.out_dir}/fold{i+1}_summary.json", {
            "fold": i + 1,
            "stage1_best_val_loss": s1.best_val_loss,
            "stage2_best_val_mae": s2.best_val_mae
        })

    mean_mae = float(np.mean(fold_mae))
    save_json(f"{cfg.out_dir}/cv_summary.json", {"mean_mae": mean_mae, "fold_mae": fold_mae})
    print(f"\n===== CV DONE | mean MAE = {mean_mae:.4f} =====")
    return mean_mae
