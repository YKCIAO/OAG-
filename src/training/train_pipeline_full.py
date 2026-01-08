# src/training/train_pipeline_full.py
from src.models.regressors import ConvAgeRegressor
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# keep your original imports / names
from src.data.datasetFC import fMRIDataset

from src.models.OAG_CAE import OrthogonalAutoEncoder
from src.training.losses import orthogonal_guided_loss


# -------------------------
# Utils
# -------------------------
def reset_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def z_scale_normalize(data, mean=None, std=None, epsilon=1e-8):
    """Z-score normalize along axis=0. Accepts numpy arrays or torch tensors."""
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
        return_numpy = False
    else:
        data_np = data.copy()
        return_numpy = True

    if mean is None:
        mean = np.nanmean(data_np, axis=0)
    if std is None:
        std = np.nanstd(data_np, axis=0)

    std = np.where(std < epsilon, epsilon, std)
    normalized = (data_np - mean) / std
    return normalized if return_numpy else torch.from_numpy(normalized)


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
    else:
        pearson_corr, spearman_corr = np.nan, np.nan

    return {"MAE": mae, "R2": r2, "Pearson": pearson_corr, "Spearman": spearman_corr}


def save_predictions(loader, model, device, filename_prefix: str):
    predictions, labels, cores = [], [], []
    model.eval()
    with torch.no_grad():
        for matrix_FC, _, label, mask_FC, _ in loader:
            x = matrix_FC.to(device)
            recon_matrix, z_age, *_ = model(x)
            predictions.append(x.detach().cpu())
            cores.append(z_age.detach().cpu())
            labels.append(label.detach().cpu())

    predictions = torch.cat(predictions, dim=0).numpy()
    cores = torch.cat(cores, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    np.save(f"{filename_prefix}_prediction_matrix.npy", predictions)
    np.save(f"{filename_prefix}_prediction_core.npy", cores)
    np.save(f"{filename_prefix}_label.npy", labels)


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    device: str = "cuda"
    batch_size: int = 2
    lr_stage1: float = 2e-4
    wd_stage1: float = 2e-3
    lr_regressor: float = 1e-2
    lr_encoder_stage2: float = 1e-5
    wd_stage2: float = 1e-3
    stage2_epochs: int = 2000
    val_interval_stage1: int = 10
    val_interval_stage2: int = 10
    early_stop_patience: int = 7
    noise_z: float = 0.03
    noise_age: float = 0.5
    max_age: float = 1150
    out_root: str = "D:/CodeHome/python/OAG-CAE/BN278_FC/FC_/"


# -------------------------
# Train + Eval (5-fold)
# -------------------------
def train_and_eval(
    group_paths: List[Tuple[str, str, str, str, str, str]],
    cfg: TrainConfig,
    w_recon=0.27,
    w_age=0.7,
    w_ortho=0.4,
    w_class=0.08,
    use_wandb: bool = False,
):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    val_mae_list = []

    def filter_age(data, labels, max_age=cfg.max_age):
        mask = labels <= max_age
        return data[mask], labels[mask]

    def age_to_group(age_value: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT: ensure unit consistency.
        If age_value is MONTHS, use start_age=35*12, interval=10*12.
        If age_value is YEARS, keep below.
        """
        start_age = 35
        interval = 10
        group = ((age_value - start_age) // interval).clamp(0, 6)
        return group.long()

    # preload folds
    fold_data = [np.load(p[0]) for p in group_paths]
    fold_labels = [np.load(p[1]) for p in group_paths]


    for fold in range(5):
        result_path = os.path.join(cfg.out_root, f"fold{fold+1}")
        os.makedirs(result_path, exist_ok=True)
        print(f"\n===== Fold {fold + 1}/5 =====")

        # split
        test_x, test_y = fold_data[fold], fold_labels[fold]

        test_x, test_y,  = filter_age(test_x, test_y)

        train_x = np.concatenate([fold_data[i] for i in range(5) if i != fold], axis=0)
        train_y = np.concatenate([fold_labels[i] for i in range(5) if i != fold], axis=0)

        train_x, train_y = filter_age(train_x, train_y)

        # normalize (training stats -> test)
        x_mean, x_std = np.nanmean(train_x, axis=0), np.nanstd(train_x, axis=0)


        train_x_norm = z_scale_normalize(train_x, x_mean, x_std)
        test_x_norm  = z_scale_normalize(test_x,  x_mean, x_std)



        train_set = fMRIDataset(train_x_norm, train_y, argument=False)
        test_set  = fMRIDataset(test_x_norm,  test_y,  argument=False)

        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        test_loader  = DataLoader(test_set,  batch_size=len(test_set), shuffle=False, drop_last=False)

        # ---------------- Stage 1 ----------------
        encoderM = OrthogonalAutoEncoder(278, 32, 32).to(device)
        optimizer1 = torch.optim.AdamW(encoderM.parameters(), lr=cfg.lr_stage1, weight_decay=cfg.wd_stage1)

        N_max_epochs = [20, 20, 20, 20, 20][fold]
        print("\n--- Stage 1: AE + Classification ---")

        for epoch in range(N_max_epochs):
            encoderM.train()
            for matrix_FC, VBM, label, mask_FC, _ in train_loader:
                x = matrix_FC.to(device)
                age = label.to(device)
                mask = mask_FC.to(device)

                age_group = age_to_group(age).to(device)
                recon, z_age, z_noise, mu, logits = encoderM(x)

                loss, log = orthogonal_guided_loss(
                    recon, x, mu, age, logits, age_group, z_age, z_noise, mask,
                    w_recon, w_age, w_ortho, w_class, epoch=epoch, max_epoch=N_max_epochs
                )

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

            if (epoch + 1) % cfg.val_interval_stage1 == 0:
                encoderM.eval()
                with torch.no_grad():
                    for matrix_FC, _, label, mask_FC, _ in test_loader:
                        x = matrix_FC.to(device)
                        age = label.to(device)
                        mask = mask_FC.to(device)
                        age_group = age_to_group(age).to(device)

                        recon, z_age, z_noise, mu, logits = encoderM(x)
                        loss_v, log_v = orthogonal_guided_loss(
                            recon, x, mu, age, logits, age_group, z_age, z_noise, mask,
                            w_recon, w_age, w_ortho, w_class, epoch=epoch, max_epoch=N_max_epochs
                        )
                        pred_group = torch.argmax(logits, dim=1)
                        acc = (pred_group == age_group).float().mean().item()

                print(f"[Stage1][Epoch {epoch+1}] Valid: {log_v} | Acc: {acc:.3f}")
                print(f"[Stage1][Epoch {epoch+1}] Train: {log}")

        torch.save(encoderM.state_dict(), os.path.join(result_path, f"fold{fold}_best_refine_encoder.pth"))

        # ---------------- Stage 2 ----------------
        print("\n--- Stage 2: Add Regressor ---")
        regressor = ConvAgeRegressor(hidden_dim=4).to(device)
        reg_loss = torch.nn.L1Loss()

        optimizer2 = torch.optim.AdamW(
            [
                {"params": regressor.parameters(), "lr": cfg.lr_regressor},
                {"params": encoderM.parameters(), "lr": cfg.lr_encoder_stage2},
            ],
            weight_decay=cfg.wd_stage2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer2, T_0=100, T_mult=2, eta_min=1e-6
        )

        # freeze encoder (as your original design)
        for p in encoderM.parameters():
            p.requires_grad = False
        encoderM.eval()

        best_mae = float("inf")
        patience = 0

        for epoch in range(cfg.stage2_epochs):
            regressor.train()
            train_losses = []

            for matrix_FC, VBM, label, mask_FC, _ in train_loader:
                x = matrix_FC.to(device)
                age = label.to(device)

                with torch.no_grad():
                    recon, z_age, z_noise, mu, logits, *_ = encoderM(x)

                z_age_aug = z_age + torch.randn_like(z_age) * cfg.noise_z
                age_aug = age + torch.randn_like(age) * cfg.noise_age

                pred = regressor(z_age_aug)
                loss = reg_loss(pred.squeeze(), age_aug.squeeze().float())

                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
                scheduler.step()

                train_losses.append(loss.item())

            if (epoch + 1) % cfg.val_interval_stage2 == 0:
                regressor.eval()
                with torch.no_grad():
                    for matrix_FC, _, label, mask_FC, _ in test_loader:
                        x = matrix_FC.to(device)
                        age = label.to(device)
                        recon, z_age, z_noise, mu, logits, *_ = encoderM(x)
                        pred = regressor(z_age).view(-1)
                        age_vec = age.view(-1).float()
                        val_mae = torch.mean(torch.abs(pred - age_vec)).item()

                print(f"[Stage2][Epoch {epoch+1}] Val MAE: {val_mae:.3f} | Train MAE: {np.mean(train_losses):.3f}")

                if use_wandb:
                    import wandb
                    wandb.log({"fold": fold, "epoch": epoch + 1, "val_mae": val_mae})

                if val_mae < best_mae:
                    best_mae = val_mae
                    patience = 0
                    torch.save(regressor.state_dict(), os.path.join(result_path, f"fold{fold}_best_refine_regressor.pth"))
                else:
                    patience += 1
                    if patience >= cfg.early_stop_patience:
                        print("Early stopping.")
                        break

        val_mae_list.append(best_mae)

        # ---------------- Save predictions ----------------
        save_predictions(test_loader, encoderM, device, os.path.join(result_path, "test"))
        save_predictions(train_loader, encoderM, device, os.path.join(result_path, "training"))

        # ---------------- Export CSV + metrics ----------------
        regressor.load_state_dict(torch.load(os.path.join(result_path, f"fold{fold}_best_refine_regressor.pth"), map_location=device))
        regressor.eval()

        def export_split(loader, split_name: str):
            real_age, predicted, middle_age = [], [], []
            with torch.no_grad():
                for matrix_FC, VBM, label, mask_FC, _ in loader:
                    x = matrix_FC.to(device)
                    age = label.to(device)
                    recon, z_age, z_noise, mu, logits, *_ = encoderM(x)
                    pred = regressor(z_age)

                    real_age.extend(age.detach().cpu().numpy().flatten())
                    predicted.extend(pred.detach().cpu().numpy().flatten())
                    middle_age.extend(mu.detach().cpu().numpy().flatten())

            df = pd.DataFrame(
                np.column_stack((real_age, predicted, middle_age)),
                columns=["Real Age", "Predicted Age", "Middle Age"],
            )
            df.to_csv(os.path.join(result_path, f"{fold}_{split_name}.csv"), index=False)

            metrics = compute_metrics(real_age, predicted)
            pd.DataFrame([metrics]).to_csv(os.path.join(result_path, f"{fold}_{split_name}_metrics.csv"), index=False)
            print(f"Fold {fold+1} {split_name} Metrics:", metrics)

        export_split(test_loader, "test")
        export_split(train_loader, "training")

    return float(np.mean(val_mae_list))
