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
from tam_fno.tam_fno_model import FNO2d_FiLM
from tam_fno.normalizer import load_normalizers
try:
    from tam_fno.tam_fno_config import DEFAULT_MODES1, DEFAULT_MODES2, DEFAULT_WIDTH, get_project_paths
    from tam_fno.tam_fno_split import load_split_manifest
    from tam_fno.tam_fno_time import build_time_feature_vector
except ModuleNotFoundError:
    DEFAULT_MODES1 = 32
    DEFAULT_MODES2 = 128
    DEFAULT_WIDTH = 64
    NTOTAL = 2920
    SAMPLES_PER_DAY = 8

    class _Paths:
        def __init__(self, project_root: Path):
            data_dir = project_root / "data"
            self.test_x_path = data_dir / "interp_test_x_SSP_TLshape_ndrz10.mat"
            self.normalizer_path = project_root / "normalizers" / "ssp_tl_norm_train2336_ndrz10.pt"
            self.split_manifest_path = data_dir / "split_seed2025.npz"

    def get_project_paths(project_root: Path) -> _Paths:
        return _Paths(project_root)

    def load_split_manifest(split_manifest_path: Path) -> dict[str, np.ndarray | int]:
        data = np.load(split_manifest_path)
        return {
            "seed": int(data["seed"]),
            "train_idx": data["train_idx"].astype(np.int64),
            "test_idx": data["test_idx"].astype(np.int64),
            "ntrain": int(data["ntrain"]),
            "ntest": int(data["ntest"]),
        }

    def build_time_feature_vector(global_idx: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * np.pi
        day_phase = (global_idx.remainder(float(SAMPLES_PER_DAY))) / float(SAMPLES_PER_DAY)
        year_phase = global_idx / float(NTOTAL - 1)
        freqs_day = torch.arange(1, 5, dtype=torch.float32, device=global_idx.device)
        freqs_year = torch.arange(1, 9, dtype=torch.float32, device=global_idx.device)
        feat_day = torch.cat(
            [
                torch.sin(two_pi * day_phase[:, None] * freqs_day[None, :]),
                torch.cos(two_pi * day_phase[:, None] * freqs_day[None, :]),
            ],
            dim=-1,
        )
        feat_year = torch.cat(
            [
                torch.sin(two_pi * year_phase[:, None] * freqs_year[None, :]),
                torch.cos(two_pi * year_phase[:, None] * freqs_year[None, :]),
            ],
            dim=-1,
        )
        return torch.cat([feat_day, feat_year], dim=-1)

WINTER_SUMMER_GLOBAL_INDICES = [360, 1800]
WINTER_SUMMER_NAMES = ["Winter", "Summer"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a single-column Winter/Summer perturbation figure for TAM-FNO")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--manuscript-dir", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--sample-local-idx", type=int, default=314)
    parser.add_argument("--channel", type=int, default=44)
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


def extract_block0_base_and_perturbation(
    model: FNO2d_FiLM,
    x: torch.Tensor,
    t_feat: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
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

    f_mid = f_mid[..., :-model.padding, :-model.padding]
    perturbation = perturbation[..., :-model.padding, :-model.padding]
    return f_mid[0].detach().cpu().numpy(), perturbation[0].detach().cpu().numpy()


def maybe_copy(files: list[Path], manuscript_dir: Path | None) -> None:
    if manuscript_dir is None:
        return
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copy2(path, manuscript_dir / path.name)


def add_colorbar(im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=14)


def add_row_formula(ax, formula: str) -> None:
    ax.text(
        -0.07,
        1.02,
        formula,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=16,
        clip_on=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
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

    local_idx = args.sample_local_idx
    channel = args.channel
    if not (0 <= local_idx < x_test_norm.shape[0]):
        raise ValueError(f"sample-local-idx {local_idx} out of range for test set of size {x_test_norm.shape[0]}")
    if not (0 <= channel < args.width):
        raise ValueError(f"channel {channel} out of range for width {args.width}")

    global_idx = int(manifest["test_idx"][local_idx])
    x_sample = x_test_norm[local_idx : local_idx + 1].to(device)

    mids: list[np.ndarray] = []
    perts: list[np.ndarray] = []
    with torch.no_grad():
        for idx in WINTER_SUMMER_GLOBAL_INDICES:
            t_feat = build_time_feature_vector(torch.tensor([idx], dtype=torch.float32)).to(device)
            f_mid, perturbation = extract_block0_base_and_perturbation(model, x_sample, t_feat)
            mids.append(f_mid[channel])
            perts.append(perturbation[channel])

    mid_vals = np.stack(mids, axis=0)
    pert_vals = np.stack(perts, axis=0)
    v_mid = (np.percentile(mid_vals, 2), np.percentile(mid_vals, 98))
    max_abs_pert = max(abs(float(np.percentile(pert_vals, 2))), abs(float(np.percentile(pert_vals, 98))))
    v_pert = (-max_abs_pert, max_abs_pert)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.9))
    for col, season_name in enumerate(WINTER_SUMMER_NAMES):
        ax = axes[0, col]
        im0 = ax.imshow(mids[col], cmap="viridis", aspect="auto", vmin=v_mid[0], vmax=v_mid[1])
        ax.set_title(season_name, fontsize=18, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Static Base", fontsize=18, labelpad=18)
            add_row_formula(ax, "$\\hat{F}_0(x)$")
        if col == 1:
            add_colorbar(im0, ax)

        ax = axes[1, col]
        im1 = ax.imshow(perts[col], cmap="RdBu_r", aspect="auto", vmin=v_pert[0], vmax=v_pert[1])
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Perturbation", fontsize=18, labelpad=18)
            add_row_formula(ax, "$M(x,\\mathbf{t})$")
        if col == 1:
            add_colorbar(im1, ax)

    plt.tight_layout(pad=0.55, h_pad=0.85, w_pad=0.8)
    pdf_path = out_dir / "perturbation_analysis_block0_singlecol.pdf"
    png_path = out_dir / "perturbation_analysis_block0_singlecol.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    maybe_copy([pdf_path, png_path], manuscript_dir)
    print(f"Saved single-column perturbation figure to {pdf_path}")
    print(f"Selected local test idx={local_idx}, global idx={global_idx}, channel={channel}")


if __name__ == "__main__":
    main()
