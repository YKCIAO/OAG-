# scripts/training.py
import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from src.data.datasetFC import FCDataset
from src.training.train_pipeline import TrainConfig, train_and_eval
from src.training.utils import reset_seeds


def dataset_ctor(train_x, train_y, val_x, val_y):
    train_ds = FCDataset(train_x, train_y, train=True, argument=True)
    val_ds   = FCDataset(val_x, val_y, train=False, argument=False)
    return train_ds, val_ds


def build_folds_from_group_paths(
    group_paths: Sequence[Tuple[str, str]],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert 5 group files into 5-fold CV splits.
    Each fold uses one group as validation, the other groups as training.
    Returns: list of (train_x, train_y, val_x, val_y)
    """
    xs, ys = [], []
    for fc_path, label_path in group_paths:
        x = np.load(fc_path)
        y = np.load(label_path)
        xs.append(x)
        ys.append(y)

    folds = []
    k = len(xs)
    for i in range(k):
        val_x, val_y = xs[i], ys[i]
        train_x = np.concatenate([xs[j] for j in range(k) if j != i], axis=0)
        train_y = np.concatenate([ys[j] for j in range(k) if j != i], axis=0)
        folds.append((train_x, train_y, val_x, val_y))
    return folds


def build_argparser():
    p = argparse.ArgumentParser()

    # basic
    p.add_argument("--seed", type=int, default=2574)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="D:/CodeHome/python/OAG-CAE/BN278_FC/FC_/")
    p.add_argument("--input_dir", type=str, default="D:/CodeHome/python/OAG-CAE/BN278_FC/")

    # wandb
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="Autoencoder-predictage")

    # loss weights (keep your current defaults)
    p.add_argument("--w_recon", type=float, default=0.1)
    p.add_argument("--w_age", type=float, default=0.36)
    p.add_argument("--w_ortho", type=float, default=0.23)
    p.add_argument("--w_class", type=float, default=0.74)

    return p


def main():
    args = build_argparser().parse_args()
    reset_seeds(args.seed)

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # 你的 TrainConfig 如果字段名不是 out_dir（比如 out_root），这里改成对应字段
    cfg = TrainConfig(
        seed=args.seed,
        device=args.device,
        out_dir=str(Path(args.out_dir)),
        w_recon=args.w_recon,
        w_age=args.w_age,
        w_ortho=args.w_ortho,
        w_class=args.w_class,
    )

    input_dir = Path(args.input_dir)
    # BN278_FC_i = [Ni, 278, 278]
    # label_i = [Ni, age]
    group_paths = [
        (
            str(input_dir / f"BN278_FC_{i}.npy"),
            str(input_dir / f"label{i}.npy"),
        )
        for i in range(1, 6)
    ]

    folds = build_folds_from_group_paths(group_paths)

    mean_mae = train_and_eval(
        folds=folds,
        train_dataset_ctor=dataset_ctor,
        cfg=cfg
    )

    print("Average Validation MAE:", mean_mae)


if __name__ == "__main__":
    main()
