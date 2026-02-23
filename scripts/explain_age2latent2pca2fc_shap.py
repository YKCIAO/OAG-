from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.models.OAG_CAE import OrthogonalAutoEncoder,OAEConfig
from src.models.regressors import ConvAgeRegressor, ConvAgeRegressorConfig
from src.explain.model_adapters import PCA2AgeWrapper
from src.explain.pca_shap import KernelShapConfig, run_kernelshap_on_pca, backproject_shap_to_fc
from src.explain.io import save_npy, save_expected_value, save_shap_pca_table_xlsx, save_beeswarm


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--fc_all", type=str, help="all_folds_fc_combined.npy (N,1,278,278)",
                   default='../BN278_FC/all_folds_fc_combined.npy')
    p.add_argument("--fold_sizes", type=int, nargs="+", help="e.g. 117 120 120 119 123",default=[117, 120, 120, 119, 123])
    p.add_argument("--out_root", type=str, default='../BN278_FC')

    p.add_argument("--pca_components", type=int, default=10)
    p.add_argument("--background", type=int, default=20)
    p.add_argument("--nsamples", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--encoder_template", type=str,
                   help="format string, e.g. /path/fold{fold}/best_encoder.pth",
                   default='../result/fold{fold}/fold{fold}_oag_cae_bestvalid.pth')
    p.add_argument("--regressor_template", type=str,
                   help="format string, e.g. /path/fold{fold}/best_regressor.pth",
                   default='../result/fold{fold}/fold{fold}_regressor_bestvalid.pth')
    return p


def main():
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fc = np.load(args.fc_all)  # (N,1,278,278)
    assert fc.ndim == 4 and fc.shape[1] == 1, f"Expected (N,1,278,278), got {fc.shape}"
    N, _, H, W = fc.shape

    fc_flat = fc.reshape(N, H * W)

    k = min(args.pca_components, N)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=42)
    x_pca_all = pca.fit_transform(fc_flat)

    # save PCA for reproducibility
    save_npy(str(out_root / "pca_components.npy"), pca.components_)
    save_npy(str(out_root / "pca_explained_variance_ratio.npy"), pca.explained_variance_ratio_)

    boundaries = np.cumsum([0] + args.fold_sizes).tolist()
    assert boundaries[-1] == N, f"fold_sizes sum={boundaries[-1]} != N={N}"

    shap_cfg = KernelShapConfig(background_size=args.background, nsamples=args.nsamples)

    for fold, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        fold_out = out_root / f"fold{fold}" / f"pca{args.pca_components}_kernelshap"
        fold_out.mkdir(parents=True, exist_ok=True)

        x_pca = x_pca_all[start:end]

        # load fold models
        OAG_cfg = OAEConfig(input_size=278, z_age_dim=32, z_noise_dim=32,
                            tau=1.5)
        encoder = OrthogonalAutoEncoder(cfg=OAG_cfg)
        encoder.load_state_dict(torch.load(args.encoder_template.format(fold=fold), map_location="cpu"))
        encoder.eval()
        reg_cfg = ConvAgeRegressorConfig(in_dim=32, hidden_channels=1, length=32, tau=1.5,
                                         gate_softmax_dim=2)
        regressor = ConvAgeRegressor(reg_cfg)
        regressor.load_state_dict(torch.load(args.regressor_template.format(fold=fold), map_location="cpu"))
        regressor.eval()

        wrapper = PCA2AgeWrapper(
            encoder=encoder,
            regressor=regressor,
            pca=pca,
            fc_shape=(H, W),
            device=args.device
        )

        shap_values, expected = run_kernelshap_on_pca(wrapper, x_pca, shap_cfg)

        # backproject to FC
        back = backproject_shap_to_fc(shap_values, expected, pca, (H, W))

        save_npy(str(fold_out / "shap_values_pca.npy"), shap_values)
        save_expected_value(str(fold_out / "expected_value.npy"), expected)
        save_shap_pca_table_xlsx(str(fold_out / "shap_values_pca.xlsx"), shap_values)
        save_npy(str(fold_out / "shap_fc_mean_map.npy"), back["shap_fc_mean_map"])
        save_beeswarm(str(fold_out / "beeswarm.png"), shap_values, x_pca)

        print(f"[Fold {fold}] saved to {fold_out}")

    print("DONE.")


if __name__ == "__main__":
    main()
