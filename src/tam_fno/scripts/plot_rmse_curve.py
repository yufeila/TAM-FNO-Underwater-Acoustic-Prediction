from __future__ import annotations

import argparse
import csv
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
from tam_fno.tam_fno_model import FNO2d_FiLM
from tam_fno.normalizer import load_normalizers
from tam_fno.tam_fno_config import (
    DEFAULT_MODES1,
    DEFAULT_MODES2,
    DEFAULT_WIDTH,
    NTEST,
    NTOTAL,
    SAMPLES_PER_DAY,
    TIME_FEATURE_DIM,
    get_project_paths,
)
from tam_fno.tam_fno_split import load_split_manifest
from tam_fno.tam_fno_time import build_time_feature_vector


def parse_args() -> argparse.Namespace:
    paths = get_project_paths(PROJECT_ROOT)
    default_run_dir = (
        PROJECT_ROOT
        / "experiments"
        / "tam_fno"
        / "runs"
        / "modes1_32_modes2_128_epoch_100"
    )

    parser = argparse.ArgumentParser(
        description="Evaluate TAM-FNO on the random 20% test split and plot yearly RMSE."
    )
    parser.add_argument("--checkpoint", type=Path, default=default_run_dir / "model_tam_fno.pth")
    parser.add_argument("--out-dir", type=Path, default=default_run_dir)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--modes1", type=int, default=DEFAULT_MODES1)
    parser.add_argument("--modes2", type=int, default=DEFAULT_MODES2)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--split-manifest", type=Path, default=paths.split_manifest_path)
    parser.add_argument("--normalizer", type=Path, default=paths.normalizer_path)
    parser.add_argument("--test-x", type=Path, default=paths.test_x_path)
    parser.add_argument("--test-y", type=Path, default=paths.test_y_path)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values.copy()
    half_left = window // 2
    half_right = window - 1 - half_left
    padded = np.pad(values, (half_left, half_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


@torch.no_grad()
def evaluate_rmse(
    model: torch.nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    test_global_idx: np.ndarray,
    x_norm,
    y_norm,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    n = x_test.shape[0]
    rmse_values = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = x_test[start:end].unsqueeze(-1).to(device)
        x_batch = x_norm.encode(x_batch)
        idx_batch = torch.from_numpy(test_global_idx[start:end].astype(np.float32)).to(device)
        t_batch = build_time_feature_vector(idx_batch)
        pred = model(x_batch, t_batch).squeeze(-1)
        pred = y_norm.decode(pred)
        gt = y_test[start:end].to(device)
        rmse_batch = torch.sqrt(torch.mean((pred - gt) ** 2, dim=(1, 2)))
        rmse_values[start:end] = rmse_batch.detach().cpu().numpy().astype(np.float32)

    return rmse_values


def aggregate_daily(global_idx: np.ndarray, rmse_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    day_idx = (global_idx // SAMPLES_PER_DAY).astype(np.int64)
    daily_rmse = np.full(365, np.nan, dtype=np.float32)
    daily_count = np.zeros(365, dtype=np.int64)

    for day in range(365):
        mask = day_idx == day
        daily_count[day] = int(np.count_nonzero(mask))
        if daily_count[day] > 0:
            daily_rmse[day] = float(np.mean(rmse_values[mask]))

    return daily_rmse, daily_count


def save_sample_csv(path: Path, global_idx: np.ndarray, rmse_values: np.ndarray) -> None:
    order = np.argsort(global_idx)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_test_idx", "global_idx", "day_of_year_1based", "slot_in_day", "time_day", "rmse"])
        for local_idx in order:
            idx = int(global_idx[local_idx])
            day0 = idx // SAMPLES_PER_DAY
            slot = idx % SAMPLES_PER_DAY
            writer.writerow(
                [
                    int(local_idx),
                    idx,
                    int(day0 + 1),
                    int(slot),
                    f"{day0 + slot / SAMPLES_PER_DAY:.3f}",
                    f"{float(rmse_values[local_idx]):.8f}",
                ]
            )


def save_daily_csv(path: Path, daily_rmse: np.ndarray, daily_count: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day_of_year_1based", "n_test_samples", "daily_mean_rmse"])
        for day0, (rmse, count) in enumerate(zip(daily_rmse, daily_count)):
            writer.writerow(
                [
                    day0 + 1,
                    int(count),
                    "" if np.isnan(rmse) else f"{float(rmse):.8f}",
                ]
            )


def save_curve(
    out_png: Path,
    out_pdf: Path,
    global_idx: np.ndarray,
    rmse_values: np.ndarray,
    daily_rmse: np.ndarray,
) -> None:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    month_ticks = np.cumsum(np.r_[0, days_in_month[:-1]]) + 15

    order = np.argsort(global_idx)
    sorted_idx = global_idx[order]
    sorted_rmse = rmse_values[order]
    x_time = (sorted_idx // SAMPLES_PER_DAY) + (sorted_idx % SAMPLES_PER_DAY) / SAMPLES_PER_DAY
    smoothed = moving_average(sorted_rmse.astype(np.float64), window=21)

    valid_days = np.where(~np.isnan(daily_rmse))[0]
    fig, ax = plt.subplots(figsize=(12.5, 4.8), dpi=220)
    ax.plot(x_time, sorted_rmse, color="#c7d2fe", linewidth=1.0, alpha=0.9, label="Test sample RMSE")
    ax.scatter(x_time, sorted_rmse, color="#64748b", s=10, alpha=0.45, linewidths=0)
    ax.plot(x_time, smoothed, color="#dc2626", linewidth=2.4, label="21-sample moving average")
    ax.plot(valid_days + 0.5, daily_rmse[valid_days], color="#2563eb", linewidth=1.4, alpha=0.8, label="Daily mean")

    ax.set_xlim(0, 365)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(months, fontsize=11)
    ax.set_xlabel("Month")
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("TAM-FNO RMSE on Random 20% Test Split")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    manifest = load_split_manifest(args.split_manifest)
    test_global_idx = manifest["test_idx"].astype(np.int64)
    if len(test_global_idx) != NTEST:
        raise RuntimeError(f"Expected {NTEST} test samples, got {len(test_global_idx)}")

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = FNO2d_FiLM(args.modes1, args.modes2, args.width, time_dim=TIME_FEATURE_DIM).to(device)
    model.load_state_dict(torch.load(str(args.checkpoint), map_location=device))
    model.eval()

    print("Loading normalizers and random test split...")
    x_norm, y_norm = load_normalizers(args.normalizer)
    x_norm.mean = x_norm.mean.to(device)
    x_norm.std = x_norm.std.to(device)
    y_norm.mean = y_norm.mean.to(device)
    y_norm.std = y_norm.std.to(device)

    x_test = MatReader(str(args.test_x)).read_field("test_x")
    y_test = MatReader(str(args.test_y)).read_field("test_y")
    if x_test.shape[0] != len(test_global_idx) or y_test.shape[0] != len(test_global_idx):
        raise RuntimeError(
            f"Split/data size mismatch: x={x_test.shape[0]}, y={y_test.shape[0]}, manifest={len(test_global_idx)}"
        )

    print(f"Running inference on {len(test_global_idx)} random test samples...")
    rmse_values = evaluate_rmse(
        model=model,
        x_test=x_test,
        y_test=y_test,
        test_global_idx=test_global_idx,
        x_norm=x_norm,
        y_norm=y_norm,
        device=device,
        batch_size=args.batch_size,
    )
    daily_rmse, daily_count = aggregate_daily(test_global_idx, rmse_values)

    out_png = args.out_dir / "random_test_rmse_vs_year.png"
    out_pdf = args.out_dir / "random_test_rmse_vs_year.pdf"
    sample_csv = args.out_dir / "random_test_rmse_per_sample.csv"
    daily_csv = args.out_dir / "random_test_daily_rmse.csv"
    summary_json = args.out_dir / "random_test_rmse_summary.json"

    save_curve(out_png, out_pdf, test_global_idx, rmse_values, daily_rmse)
    save_sample_csv(sample_csv, test_global_idx, rmse_values)
    save_daily_csv(daily_csv, daily_rmse, daily_count)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split_manifest": str(args.split_manifest),
        "split_seed": int(manifest["seed"]),
        "nt_total": NTOTAL,
        "n_test": int(len(test_global_idx)),
        "mean_rmse": float(np.mean(rmse_values)),
        "median_rmse": float(np.median(rmse_values)),
        "p90_rmse": float(np.percentile(rmse_values, 90)),
        "p99_rmse": float(np.percentile(rmse_values, 99)),
        "min_rmse": float(np.min(rmse_values)),
        "max_rmse": float(np.max(rmse_values)),
        "days_with_test_samples": int(np.count_nonzero(daily_count)),
        "days_without_test_samples": int(np.count_nonzero(daily_count == 0)),
        "outputs": {
            "figure_png": str(out_png),
            "figure_pdf": str(out_pdf),
            "per_sample_csv": str(sample_csv),
            "daily_csv": str(daily_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
