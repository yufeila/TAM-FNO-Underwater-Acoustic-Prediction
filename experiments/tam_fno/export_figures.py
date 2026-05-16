from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tam_fno_mplcache")

import matplotlib.pyplot as plt
import numpy as np
import torch

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
from tam_fno.tam_fno_config import NTOTAL, SAMPLES_PER_DAY, get_project_paths
from tam_fno.tam_fno_split import load_split_manifest
from tam_fno.tam_fno_time import build_time_feature_vector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current TAM-FNO-only paper comparison figures")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--model-film", type=str, default="")
    parser.add_argument("--vis-global-idx", type=int, default=2650)
    return parser.parse_args(argv)


def instantiate_film_model(device: torch.device) -> torch.nn.Module:
    return FNO2d_FiLM(32, 128, 64, time_dim=24).to(device)


def moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return values
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


@torch.no_grad()
def predict_film_batch(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    global_idx_batch: np.ndarray,
    y_norm,
    x_norm,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x_input = x_batch.unsqueeze(-1)
    t_input = build_time_feature_vector(torch.from_numpy(global_idx_batch.astype(np.float32)))
    x_input = x_norm.encode(x_input).to(device)
    pred = model(x_input, t_input.to(device)).squeeze(-1)
    pred = y_norm.decode(pred).detach().cpu().numpy().astype(np.float32)
    ssp_phys = x_batch.detach().cpu().numpy().astype(np.float32)
    return pred, ssp_phys


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def evaluate_film_split(
    model: torch.nn.Module,
    x_split: torch.Tensor,
    y_split: torch.Tensor,
    global_idx: np.ndarray,
    y_norm,
    x_norm,
    device: torch.device,
    batch_size: int,
    vis_local_idx: int | None,
) -> dict[str, object]:
    n = x_split.shape[0]
    rmse_values = np.zeros(n, dtype=np.float32)
    vis_payload: dict[str, np.ndarray | int] | None = None

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pred_batch, ssp_batch = predict_film_batch(
            model=model,
            x_batch=x_split[start:end],
            global_idx_batch=global_idx[start:end],
            y_norm=y_norm,
            x_norm=x_norm,
            device=device,
        )
        gt_batch = y_split[start:end].detach().cpu().numpy().astype(np.float32)
        for offset in range(end - start):
            idx = start + offset
            pred = pred_batch[offset]
            gt = gt_batch[offset]
            rmse_values[idx] = rmse(pred, gt)
            if vis_local_idx is not None and idx == vis_local_idx:
                vis_payload = {
                    "ssp": ssp_batch[offset],
                    "gt": gt,
                    "pred": pred,
                    "local_idx": idx,
                    "global_idx": int(global_idx[idx]),
                }

    return {"rmse": rmse_values, "visual": vis_payload}


def save_film_only_rmse_curve(out_path: Path, daily_curve: np.ndarray) -> None:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_ticks = np.cumsum([0] + days_in_month[:-1]) + 15
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=220)
    x_axis = np.arange(365)
    ax.plot(x_axis, moving_average(daily_curve, 7), color="#d62728", linewidth=2.8, label="Current TAM-FNO")
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(months, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("Month", fontsize=16)
    ax.set_ylabel("RMSE (dB)", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", fontsize=12, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_film_only_tl_visual(
    out_png: Path,
    out_pdf: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    ssp: np.ndarray,
    global_idx: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5))

    ssp_im = axes[0, 0].imshow(ssp, aspect="auto", cmap="viridis")
    axes[0, 0].set_title("Input SSF", fontsize=16, pad=6)
    axes[0, 0].set_ylabel("Depth", fontsize=16)
    axes[0, 0].set_xticks([])
    axes[0, 0].tick_params(axis="y", labelsize=12)
    cbar0 = fig.colorbar(ssp_im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(labelsize=11)

    vmin = min(float(gt.min()), float(pred.min()))
    vmax = max(float(gt.max()), float(pred.max()))
    gt_im = axes[0, 1].imshow(gt, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Ground Truth", fontsize=16, pad=6)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    pred_im = axes[0, 2].imshow(pred, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Current TAM-FNO", fontsize=16, pad=6)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    cbar1 = fig.colorbar(pred_im, ax=axes[0, 1:3], fraction=0.025, pad=0.02)
    cbar1.set_label("Transmission Loss (dB)", fontsize=14)
    cbar1.ax.tick_params(labelsize=11)

    diff = np.abs(pred - gt)
    axes[1, 0].text(
        0.5,
        0.5,
        f"Absolute Error\n(|Pred - GT|)\nGlobal idx={global_idx}",
        ha="center",
        va="center",
        fontsize=16,
        transform=axes[1, 0].transAxes,
    )
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    for spine in axes[1, 0].spines.values():
        spine.set_visible(False)

    err_im = axes[1, 1].imshow(diff, aspect="auto", cmap="Reds", vmin=0, vmax=float(diff.max()))
    axes[1, 1].set_title("Absolute Error", fontsize=16, pad=6)
    axes[1, 1].set_xlabel("Range", fontsize=14)
    axes[1, 1].set_ylabel("Depth", fontsize=16)
    axes[1, 1].tick_params(axis="both", labelsize=12)

    signed = pred - gt
    vmax_signed = max(abs(float(signed.min())), abs(float(signed.max())))
    signed_im = axes[1, 2].imshow(signed, aspect="auto", cmap="RdBu_r", vmin=-vmax_signed, vmax=vmax_signed)
    axes[1, 2].set_title("Signed Error", fontsize=16, pad=6)
    axes[1, 2].set_xlabel("Range", fontsize=14)
    axes[1, 2].set_yticks([])
    cbar2 = fig.colorbar(err_im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=11)
    cbar3 = fig.colorbar(signed_im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    paths = get_project_paths(PROJECT_ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_split_manifest(paths.split_manifest_path)
    train_global_idx = manifest["train_idx"]
    test_global_idx = manifest["test_idx"]
    vis_local_idx = int(np.argmin(np.abs(test_global_idx - args.vis_global_idx)))
    vis_global_idx = int(test_global_idx[vis_local_idx])

    x_norm, y_norm = load_normalizers(paths.normalizer_path)
    y_norm.mean = y_norm.mean.to(device)
    y_norm.std = y_norm.std.to(device)

    x_test = MatReader(str(paths.test_x_path)).read_field("test_x")
    y_test = MatReader(str(paths.test_y_path)).read_field("test_y")
    x_train = MatReader(str(paths.train_x_path)).read_field("train_x")
    y_train = MatReader(str(paths.train_y_path)).read_field("train_y")

    model_path = Path(args.model_film) if args.model_film else (
        PROJECT_ROOT / "experiments" / "tam_fno" / "runs" / "modes1_32_modes2_128_epoch_100" / "model_tam_fno.pth"
    )
    model = instantiate_film_model(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    rmse_all = np.full(NTOTAL, np.nan, dtype=np.float32)
    print("Evaluating current TAM-FNO on test split...")
    test_result = evaluate_film_split(
        model=model,
        x_split=x_test,
        y_split=y_test,
        global_idx=test_global_idx,
        y_norm=y_norm,
        x_norm=x_norm,
        device=device,
        batch_size=args.batch_size,
        vis_local_idx=vis_local_idx,
    )
    rmse_all[test_global_idx] = test_result["rmse"]
    print("Evaluating current TAM-FNO on train split...")
    train_result = evaluate_film_split(
        model=model,
        x_split=x_train,
        y_split=y_train,
        global_idx=train_global_idx,
        y_norm=y_norm,
        x_norm=x_norm,
        device=device,
        batch_size=args.batch_size,
        vis_local_idx=None,
    )
    rmse_all[train_global_idx] = train_result["rmse"]

    daily_curve = rmse_all.reshape(365, SAMPLES_PER_DAY).mean(axis=1)
    visual = test_result["visual"]
    if visual is None:
        raise RuntimeError("Failed to collect current-model visualization sample")

    rmse_png = out_dir / "tam_fno_current_rmse_vs_time.png"
    tl_png = out_dir / "tam_fno_current_tl_visual.png"
    tl_pdf = out_dir / "tam_fno_current_tl_visual.pdf"
    payload_json = out_dir / "tam_fno_current_payload.json"

    save_film_only_rmse_curve(rmse_png, daily_curve)
    save_film_only_tl_visual(
        tl_png,
        tl_pdf,
        gt=visual["gt"],
        pred=visual["pred"],
        ssp=visual["ssp"],
        global_idx=vis_global_idx,
    )
    payload_json.write_text(
        json.dumps(
            {
                "vis_local_idx": vis_local_idx,
                "vis_global_idx": vis_global_idx,
                "test_rmse_mean": float(np.nanmean(test_result["rmse"])),
                "full_year_rmse_mean": float(np.nanmean(rmse_all)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved current-model RMSE curve to {rmse_png}")
    print(f"Saved current-model TL visual to {tl_pdf}")
    print(f"Visual sample global idx = {vis_global_idx}")


if __name__ == "__main__":
    main()
