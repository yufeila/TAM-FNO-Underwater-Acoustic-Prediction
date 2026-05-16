from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tam_fno_mplcache")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE
while not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        raise RuntimeError("Cannot find FNO project root")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tam_fno.io_mat import MatReader
from tam_fno.models.fno2d_film import FNO2d_FiLM
from tam_fno.normalizer import load_normalizers
from tam_fno.tam_fno_config import DEFAULT_MODES1, DEFAULT_MODES2, DEFAULT_WIDTH, get_project_paths
from tam_fno.tam_fno_split import load_split_manifest
from tam_fno.tam_fno_time import build_time_feature_vector

SEASONAL_GLOBAL_INDICES = [360, 1080, 1800, 2520]
SEASON_NAMES = ["Winter (Feb)", "Spring (May)", "Summer (Aug)", "Autumn (Nov)"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate perturbation_analysis_block0 for TAM-FNO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--manuscript-dir", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--modes1", type=int, default=DEFAULT_MODES1)
    parser.add_argument("--modes2", type=int, default=DEFAULT_MODES2)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    return parser.parse_args(argv)


def default_model_path(project_root: Path) -> Path:
    return (
        project_root
        / "experiments"
        / "tam_fno"
        / "runs"
        / "modes1_32_modes2_128_epoch_100"
        / "model_tam_fno.pth"
    )


def extract_block0_triplet(
    model: FNO2d_FiLM,
    x: torch.Tensor,
    t_feat: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = model.get_grid(x.shape, x.device)
    x_in = torch.cat((x, grid), dim=-1)
    x_in = model.fc0(x_in)
    x_in = x_in.permute(0, 3, 1, 2)
    x_in = F.pad(x_in, [0, model.padding, 0, model.padding])

    x1 = model.conv0(x_in)
    x2 = model.w0(x_in)
    f_mid = x1 + x2

    film_params = model.film0(t_feat)
    gamma = film_params[:, : model.width].unsqueeze(-1).unsqueeze(-1)
    beta = film_params[:, model.width :].unsqueeze(-1).unsqueeze(-1)
    perturbation = f_mid * gamma + beta
    f_out = f_mid + perturbation

    f_mid = f_mid[..., :-model.padding, :-model.padding]
    perturbation = perturbation[..., :-model.padding, :-model.padding]
    f_out = f_out[..., :-model.padding, :-model.padding]
    return (
        f_mid[0].detach().cpu().numpy(),
        perturbation[0].detach().cpu().numpy(),
        f_out[0].detach().cpu().numpy(),
    )


@torch.no_grad()
def select_most_time_sensitive_sample(
    model: FNO2d_FiLM,
    x_test_norm: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> int:
    season_feats = [
        build_time_feature_vector(torch.tensor([idx], dtype=torch.float32)).to(device)
        for idx in SEASONAL_GLOBAL_INDICES
    ]
    best_score = -1.0
    best_idx = 0
    for start in range(0, x_test_norm.shape[0], batch_size):
        end = min(start + batch_size, x_test_norm.shape[0])
        x_batch = x_test_norm[start:end].to(device)
        preds = []
        for feat in season_feats:
            t_batch = feat.repeat(end - start, 1)
            preds.append(model(x_batch, t_batch).squeeze(-1))
        pred_stack = torch.stack(preds, dim=0)
        score = pred_stack.std(dim=0).mean(dim=(1, 2)).detach().cpu().numpy()
        local_offset = int(np.argmax(score))
        local_score = float(score[local_offset])
        if local_score > best_score:
            best_score = local_score
            best_idx = start + local_offset
    return best_idx


def maybe_copy(files: list[Path], manuscript_dir: Path | None) -> None:
    if manuscript_dir is None:
        return
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copy2(path, manuscript_dir / path.name)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = get_project_paths(PROJECT_ROOT)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else PROJECT_ROOT / "experiments" / "tam_fno" / "runs" / "modes1_32_modes2_128_epoch_100"
    )
    manuscript_dir = Path(args.manuscript_dir) if args.manuscript_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path) if args.model_path else default_model_path(PROJECT_ROOT)
    model = FNO2d_FiLM(args.modes1, args.modes2, args.width, time_dim=24).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    x_test = MatReader(str(paths.test_x_path)).read_field("test_x")
    x_norm, _ = load_normalizers(paths.normalizer_path)
    x_test_norm = x_norm.encode(x_test.unsqueeze(-1))
    manifest = load_split_manifest(paths.split_manifest_path)

    best_local_idx = select_most_time_sensitive_sample(model, x_test_norm, device, args.batch_size)
    best_global_idx = int(manifest["test_idx"][best_local_idx])

    x_sample = x_test_norm[best_local_idx : best_local_idx + 1].to(device)
    seasonal_feats = [
        build_time_feature_vector(torch.tensor([idx], dtype=torch.float32)).to(device)
        for idx in SEASONAL_GLOBAL_INDICES
    ]

    mids: list[np.ndarray] = []
    perts: list[np.ndarray] = []
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for feat in seasonal_feats:
            f_mid, perturbation, f_out = extract_block0_triplet(model, x_sample, feat)
            mids.append(f_mid)
            perts.append(perturbation)
            outs.append(f_out)

    pert_stack = np.stack(perts, axis=0)
    var_per_channel = np.var(pert_stack, axis=0).mean(axis=(1, 2))
    best_channel = int(np.argmax(var_per_channel))

    f_mid_ref = mids[0][best_channel]
    pert_vals = pert_stack[:, best_channel]
    out_vals = np.stack(outs, axis=0)[:, best_channel]
    v_mid = (np.percentile(f_mid_ref, 2), np.percentile(f_mid_ref, 98))
    max_abs_pert = max(abs(float(np.percentile(pert_vals, 2))), abs(float(np.percentile(pert_vals, 98))))
    v_pert = (-max_abs_pert, max_abs_pert)
    v_out = (np.percentile(out_vals, 2), np.percentile(out_vals, 98))

    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    fig.suptitle(
        f"Temporal Perturbation Analysis on Block 0\nHeld-out sample {best_local_idx} (global {best_global_idx}), channel {best_channel}",
        fontsize=22,
        y=0.97,
    )

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    for i, season_name in enumerate(SEASON_NAMES):
        ax = axes[0, i]
        im0 = ax.imshow(mids[i][best_channel], cmap="viridis", aspect="auto", vmin=v_mid[0], vmax=v_mid[1])
        if i == 0:
            ax.set_ylabel("Static Base $\\hat{F}(x)$", fontsize=18, labelpad=12)
        ax.set_title(season_name, fontsize=18, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 3:
            add_colorbar(im0, ax)

        ax = axes[1, i]
        im1 = ax.imshow(perts[i][best_channel], cmap="RdBu_r", aspect="auto", vmin=v_pert[0], vmax=v_pert[1])
        if i == 0:
            ax.set_ylabel("Perturbation $M(x,t)$", fontsize=18, labelpad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 3:
            add_colorbar(im1, ax)

        ax = axes[2, i]
        im2 = ax.imshow(outs[i][best_channel], cmap="viridis", aspect="auto", vmin=v_out[0], vmax=v_out[1])
        if i == 0:
            ax.set_ylabel("Modulated $F_{out}$", fontsize=18, labelpad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 3:
            add_colorbar(im2, ax)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=2.5, w_pad=2.0)
    pdf_path = out_dir / "perturbation_analysis_block0.pdf"
    png_path = out_dir / "perturbation_analysis_block0.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    maybe_copy([pdf_path, png_path], manuscript_dir)
    print(f"Selected held-out sample {best_local_idx} (global {best_global_idx})")
    print(f"Saved perturbation figure to {pdf_path}")


if __name__ == "__main__":
    main()
