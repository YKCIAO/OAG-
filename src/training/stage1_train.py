from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class Stage1Result:
    best_state_dict: Dict[str, torch.Tensor]
    best_val_loss: float
    last_log: Dict[str, float]


@torch.no_grad()
def _eval_stage1(model, loader, loss_fn, device, cfg) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total = 0.0
    n = 0
    log_sum = {}

    for batch in loader:
        # expected: (x, ... mask) 你根据自己的 dataset 输出对齐
        # 常见：x, label, mask  或 x, ec, label, fc_mask, ec_mask
        if len(batch) == 3:
            x, age_true, mask = batch
            class_true = None
        elif len(batch) == 5:
            x, _, age_true, mask, _ = batch
            class_true = None
        else:
            # 兜底：至少要 x 和 age_true
            x = batch[0]
            age_true = batch[1]
            mask = batch[-1]
            class_true = None

        x = x.to(device)
        age_true = age_true.to(device)

        recon, z_age, z_noise, mu, logits = model(x)

        # 如果你有 age_group 分类监督，这里可自己接上 class_true
        # 否则传 dummy（loss_fn 里别用就行）
        if class_true is None:
            class_true = torch.zeros(x.size(0), dtype=torch.long, device=device)

        loss, log = loss_fn(
            recon=recon, x=x, mu=mu, age_true=age_true,
            class_pred_logits=logits, class_true=class_true,
            z_age=z_age, z_noise=z_noise, mask=mask.to(device),
            w_recon=cfg.w_recon, w_age=cfg.w_age, w_ortho=cfg.w_ortho, w_class=cfg.w_class
        )

        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        for k, v in log.items():
            log_sum[k] = log_sum.get(k, 0.0) + float(v) * bs

    avg_loss = total / max(n, 1)
    avg_log = {k: v / max(n, 1) for k, v in log_sum.items()}
    return avg_loss, avg_log


def train_stage1(model, train_loader, val_loader, optimizer, loss_fn, device, cfg) -> Stage1Result:
    best_val = float("inf")
    best_sd = None
    last_log = {}

    for epoch in range(cfg.epochs_stage1):
        model.train()
        running = 0.0
        n = 0

        for batch in train_loader:
            if len(batch) == 3:
                x, age_true, mask = batch
                class_true = None
            elif len(batch) == 5:
                x, _, age_true, mask, _ = batch
                class_true = None
            else:
                x = batch[0]
                age_true = batch[1]
                mask = batch[-1]
                class_true = None

            x = x.to(device)
            age_true = age_true.to(device)
            mask = mask.to(device)

            recon, z_age, z_noise, mu, logits = model(x)

            if class_true is None:
                class_true = torch.zeros(x.size(0), dtype=torch.long, device=device)

            loss, log = loss_fn(
                recon=recon, x=x, mu=mu, age_true=age_true,
                class_pred_logits=logits, class_true=class_true,
                z_age=z_age, z_noise=z_noise, mask=mask,
                w_recon=cfg.w_recon, w_age=cfg.w_age, w_ortho=cfg.w_ortho, w_class=cfg.w_class,
                epoch=epoch, max_epoch=cfg.epochs_stage1
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            bs = x.size(0)
            running += loss.item() * bs
            n += bs
            last_log = log

        train_loss = running / max(n, 1)
        val_loss, val_log = _eval_stage1(model, val_loader, loss_fn, device, cfg)

        if val_loss < best_val:
            best_val = val_loss
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if cfg.verbose:
            print(f"[Stage1] epoch {epoch+1}/{cfg.epochs_stage1} "
                  f"train={train_loss:.4f} val={val_loss:.4f} log={val_log}")

    if best_sd is None:
        best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return Stage1Result(best_state_dict=best_sd, best_val_loss=float(best_val), last_log=last_log)
