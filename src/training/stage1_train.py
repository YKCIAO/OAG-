from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class Stage1Result:
    best_state_dict: Dict[str, torch.Tensor]
    best_val_loss: float
    test_loss: float
    best_epoch: int
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


from typing import Optional

def train_stage1(
    model,
    train_loader,
    val_loader,
    test_loader,            # NEW
    optimizer,
    loss_fn,
    device,
    cfg
) -> Stage1Result:
    """
    Stage 1 training with:
      - train/val/test split
      - early stopping on val
      - reporting every report_every epochs (default 20)
      - final test evaluation using the best checkpoint

    Assumes _eval_stage1(model, loader, loss_fn, device, cfg) exists and returns (loss, log_dict).
    """

    best_val = float("inf")
    best_sd = None
    best_epoch = -1
    last_log = {}

    # --- config defaults (safe) ---
    report_every = getattr(cfg, "report_every", 20)      # print every 20 epochs
    early_stop_patience = getattr(cfg, "early_stop", 10) # you can set cfg.early_stop
    min_delta = getattr(cfg, "min_delta", 0.0)           # optional: require improvement by > min_delta
    warmup_epochs = getattr(cfg, "val_warmup", 0)        # optional: don't early-stop before this epoch

    patience = 0

    for epoch in range(cfg.epochs_stage1):
        # --------------------
        # Train
        # --------------------
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

        # --------------------
        # Validate (every epoch for early stop accuracy; print every 20)
        # --------------------
        val_loss, val_log = _eval_stage1(model, val_loader, loss_fn, device, cfg)

        # --------------------
        # Early stopping on val
        # --------------------
        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            # only count patience after warmup (optional)
            if epoch >= warmup_epochs:
                patience += 1

        # --------------------
        # Reporting (every report_every epochs)
        # --------------------
        if cfg.verbose and ((epoch + 1) % report_every == 0):
            print(
                f"[Stage1] epoch {epoch+1}/{cfg.epochs_stage1} "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"best_val={best_val:.4f} patience={patience}/{early_stop_patience} "
                f"log={val_log}"
            )

        # --------------------
        # Stop condition
        # --------------------
        if epoch >= warmup_epochs and patience >= early_stop_patience:
            if cfg.verbose:
                print(
                    f"[Stage1] Early stopping at epoch {epoch+1}. "
                    f"Best epoch={best_epoch+1} best_val={best_val:.4f}"
                )
            break

    # --------------------
    # Load best checkpoint before testing
    # --------------------
    if best_sd is None:
        best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)  # restore best for final test
    model.to(device)

    # --------------------
    # Final test evaluation (ONLY ONCE)
    # --------------------
    test_loss, test_log = _eval_stage1(model, test_loader, loss_fn, device, cfg)

    if cfg.verbose:
        print(f"[Stage1][TEST] loss={test_loss:.4f} log={test_log}")

    # 如果你的 Stage1Result 结构体目前没有 test 字段：
    # 你可以先只返回 best_sd/best_val/last_log；或者扩展 Stage1Result（推荐扩展）
    return Stage1Result(
        best_state_dict=best_sd,
        best_val_loss=float(best_val),
        test_loss=float(test_loss),
        best_epoch=int(best_epoch),
        last_log=last_log,
    )

