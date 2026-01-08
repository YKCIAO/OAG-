from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class Stage2Result:
    best_state_dict: Dict[str, torch.Tensor]
    best_val_mae: float


@torch.no_grad()
def _eval_stage2(encoder, regressor, loader, device) -> float:
    encoder.eval()
    regressor.eval()

    all_true = []
    all_pred = []

    for batch in loader:
        if len(batch) == 3:
            x, age_true, _ = batch
        elif len(batch) == 5:
            x, _, age_true, _, _ = batch
        else:
            x = batch[0]
            age_true = batch[1]

        x = x.to(device)
        age_true = age_true.to(device)

        z_age, z_noise = encoder.encode(x)
        pred = regressor(z_age)

        all_true.append(age_true.detach().cpu())
        all_pred.append(pred.detach().cpu())

    y_true = torch.cat(all_true).numpy().reshape(-1)
    y_pred = torch.cat(all_pred).numpy().reshape(-1)
    mae = float(abs(y_true - y_pred).mean())
    return mae


def train_stage2(encoder, regressor, train_loader, val_loader, optimizer, device, cfg) -> Stage2Result:
    best_mae = float("inf")
    best_sd = None
    patience = 0

    for epoch in range(cfg.epochs_stage2):
        encoder.eval()  # 通常 stage2 冻结 encoder；如果你要 finetune，可以改成 train()
        regressor.train()

        for batch in train_loader:
            if len(batch) == 3:
                x, age_true, _ = batch
            elif len(batch) == 5:
                x, _, age_true, _, _ = batch
            else:
                x = batch[0]
                age_true = batch[1]

            x = x.to(device)
            age_true = age_true.to(device)

            with torch.no_grad():
                z_age, z_noise = encoder.encode(x)

            pred = regressor(z_age)
            loss = F.huber_loss(pred, age_true.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(regressor.parameters(), cfg.grad_clip)
            optimizer.step()

        val_mae = _eval_stage2(encoder, regressor, val_loader, device)

        if val_mae < best_mae:
            best_mae = val_mae
            patience = 0
            best_sd = {k: v.detach().cpu().clone() for k, v in regressor.state_dict().items()}
        else:
            patience += 1

        if cfg.verbose:
            print(f"[Stage2] epoch {epoch+1}/{cfg.epochs_stage2} val_mae={val_mae:.4f} best={best_mae:.4f}")

        if patience >= cfg.early_stop_patience:
            if cfg.verbose:
                print(f"[Stage2] Early stop at epoch {epoch+1}")
            break

    if best_sd is None:
        best_sd = {k: v.detach().cpu().clone() for k, v in regressor.state_dict().items()}

    return Stage2Result(best_state_dict=best_sd, best_val_mae=float(best_mae))
