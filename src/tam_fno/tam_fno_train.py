from __future__ import annotations

import argparse
import os
from pathlib import Path
from timeit import default_timer

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tam_fno_mplcache")

import matplotlib.pyplot as plt
import numpy as np
import torch

from .models.fno2d_film import FNO2d_FiLM, H1Loss, LpLoss
from .normalizer import UnitGaussianNormalizer, load_normalizers, save_normalizers
from .tam_fno_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_GAMMA,
    DEFAULT_LR,
    DEFAULT_MODES1,
    DEFAULT_MODES2,
    DEFAULT_STEP_SIZE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WIDTH,
    TIME_FEATURE_DIM,
    get_project_paths,
)
from .tam_fno_data import load_preprocessed_split
from .tam_fno_split import load_split_manifest
from .tam_fno_time import build_time_feature_vector

MODEL_FILENAME = "model_tam_fno.pth"
DEFAULT_RESULT_DIR = "runs/modes1_32_modes2_128_epoch_100"


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TAM-FNO")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--modes1", type=int, default=DEFAULT_MODES1)
    parser.add_argument("--modes2", type=int, default=DEFAULT_MODES2)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--result-dir", type=str, default="")
    parser.add_argument("--split-manifest", type=str, default="")
    parser.add_argument("--normalizer-path", type=str, default="")
    parser.add_argument("--rebuild-normalizer", action="store_true")
    return parser.parse_args()


def resolve_result_dir(script_dir: Path, args: argparse.Namespace) -> Path:
    if args.result_dir:
        return Path(args.result_dir)

    result_dir = script_dir / DEFAULT_RESULT_DIR
    if args.run_tag:
        result_dir = result_dir.parent / f"{result_dir.name}_{args.run_tag}"
    return result_dir


def build_tam_fno_inputs(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    train_global_idx: np.ndarray,
    test_global_idx: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_train = x_train.unsqueeze(-1)
    x_test = x_test.unsqueeze(-1)

    train_idx_t = torch.from_numpy(train_global_idx.astype(np.float32))
    test_idx_t = torch.from_numpy(test_global_idx.astype(np.float32))
    t_train = build_time_feature_vector(train_idx_t)
    t_test = build_time_feature_vector(test_idx_t)
    return x_train, t_train, x_test, t_test


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return FNO2d_FiLM(
        args.modes1,
        args.modes2,
        args.width,
        time_dim=TIME_FEATURE_DIM,
    )


def load_or_build_normalizer_pair(
    normalizer_path: Path,
    x_train_ssp: torch.Tensor,
    y_train: torch.Tensor,
    rebuild: bool,
) -> tuple[UnitGaussianNormalizer, UnitGaussianNormalizer]:
    if normalizer_path.exists() and not rebuild:
        return load_normalizers(normalizer_path)

    x_norm = UnitGaussianNormalizer(x_train_ssp)
    y_norm = UnitGaussianNormalizer(y_train)
    save_normalizers(
        normalizer_path,
        x_norm,
        y_norm,
        meta={
            "ntrain": int(x_train_ssp.shape[0]),
            "ntest": None,
            "model": "TAM-FNO",
            "note": "Normalizer computed from TRAIN split only.",
        },
    )
    return x_norm, y_norm


def train_tam_fno(script_dir: Path) -> None:
    args = parse_train_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    paths = get_project_paths(script_dir.parents[1])
    paths.normalizers_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = (
        Path(args.split_manifest) if args.split_manifest else paths.split_manifest_path
    )
    normalizer_path = (
        Path(args.normalizer_path) if args.normalizer_path else paths.normalizer_path
    )
    result_dir = resolve_result_dir(script_dir, args)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving outputs to {result_dir}")
    print(f"Using device: {device}")
    print(f"Loading split manifest from {split_manifest_path}")
    manifest = load_split_manifest(split_manifest_path)

    print("Loading preprocessed train/test data...")
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_preprocessed_split(
        paths.train_x_path,
        paths.train_y_path,
        paths.test_x_path,
        paths.test_y_path,
    )

    x_train, t_train, x_test, t_test = build_tam_fno_inputs(
        x_train_raw,
        x_test_raw,
        manifest["train_idx"],
        manifest["test_idx"],
    )

    x_norm, y_norm = load_or_build_normalizer_pair(
        normalizer_path,
        x_train,
        y_train_raw,
        rebuild=args.rebuild_normalizer,
    )
    x_train = x_norm.encode(x_train)
    x_test = x_norm.encode(x_test)
    y_train = y_norm.encode(y_train_raw)
    y_test = y_test_raw

    if device.type == "cuda":
        x_norm.cuda()
        y_norm.cuda()
    else:
        x_norm.cpu()
        y_norm.cpu()

    train_dataset = torch.utils.data.TensorDataset(x_train, t_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, t_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = build_model(args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    rel_l2 = LpLoss(size_average=False)
    train_loss_func = H1Loss(d=2, beta=0.01)
    train_history: list[float] = []
    test_history: list[float] = []

    print("Starting training...")
    t0 = default_timer()
    for epoch in range(args.epochs):
        model.train()
        t_epoch = default_timer()
        train_l2 = 0.0
        train_opt_loss = 0.0

        for x_batch, t_batch, y_batch in train_loader:
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch, t_batch).squeeze()
            loss = train_loss_func(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_opt_loss += loss.item()

            with torch.no_grad():
                pred_phys = y_norm.decode(pred)
                y_phys = y_norm.decode(y_batch)
                batch_size = x_batch.shape[0]
                train_l2 += rel_l2(
                    pred_phys.view(batch_size, -1), y_phys.view(batch_size, -1)
                ).item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x_batch, t_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                t_batch = t_batch.to(device)
                y_batch = y_batch.to(device)

                pred = model(x_batch, t_batch).squeeze()
                pred_phys = y_norm.decode(pred)
                batch_size = x_batch.shape[0]
                test_l2 += rel_l2(
                    pred_phys.view(batch_size, -1), y_batch.view(batch_size, -1)
                ).item()

        train_l2 /= int(manifest["ntrain"])
        test_l2 /= int(manifest["ntest"])
        train_opt_loss /= int(manifest["ntrain"])
        train_history.append(train_l2)
        test_history.append(test_l2)
        print(
            f"Epoch: {epoch}, Time: {default_timer()-t_epoch:.2f}, "
            f"OptLoss: {train_opt_loss:.5f}, TrainL2: {train_l2:.5f}, TestL2: {test_l2:.5f}"
        )

    print(f"Training completed in {default_timer()-t0:.2f}s")

    model_path = result_dir / MODEL_FILENAME
    torch.save(model.state_dict(), str(model_path))
    print(f"Model saved to {model_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label="Train Loss")
    plt.plot(test_history, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Relative L2 Loss")
    plt.legend()
    plt.title("Training and Testing Loss (TAM-FNO)")
    plt.tight_layout()
    plt.savefig(str(result_dir / "loss_curve.png"))
    plt.close()

    model.eval()
    with torch.no_grad():
        x_vis, t_vis, y_vis = next(iter(test_loader))
        x_vis = x_vis.to(device)
        t_vis = t_vis.to(device)
        y_vis = y_vis.to(device)
        pred_vis = y_norm.decode(model(x_vis, t_vis).squeeze())
        ssp_vis = x_norm.decode(x_vis)[..., 0]

        fig = plt.figure(figsize=(15, 12))
        n_examples = min(3, x_vis.shape[0])
        for idx in range(n_examples):
            ssp = ssp_vis[idx].detach().cpu().numpy()
            y_true = y_vis[idx].detach().cpu().numpy()
            y_pred = pred_vis[idx].detach().cpu().numpy()
            error = y_pred - y_true

            ax = fig.add_subplot(n_examples, 4, idx * 4 + 1)
            im1 = ax.imshow(ssp, cmap="viridis", aspect="auto")
            if idx == 0:
                ax.set_title("Input SSP", fontsize=12)
            plt.colorbar(im1, ax=ax, shrink=0.8)

            ax = fig.add_subplot(n_examples, 4, idx * 4 + 2)
            im2 = ax.imshow(y_true, cmap="turbo", aspect="auto", vmin=40, vmax=110)
            if idx == 0:
                ax.set_title("Ground Truth TL", fontsize=12)
            plt.colorbar(im2, ax=ax, shrink=0.8)

            ax = fig.add_subplot(n_examples, 4, idx * 4 + 3)
            im3 = ax.imshow(y_pred, cmap="turbo", aspect="auto", vmin=40, vmax=110)
            if idx == 0:
                ax.set_title("TAM-FNO Prediction", fontsize=12)
            plt.colorbar(im3, ax=ax, shrink=0.8)

            ax = fig.add_subplot(n_examples, 4, idx * 4 + 4)
            error_max = max(abs(error.min()), abs(error.max()))
            im4 = ax.imshow(
                error, cmap="RdBu_r", aspect="auto", vmin=-error_max, vmax=error_max
            )
            if idx == 0:
                ax.set_title("Error", fontsize=12)
            plt.colorbar(im4, ax=ax, shrink=0.8)
            rmse_val = float(np.sqrt(np.mean(error**2)))
            ax.text(
                0.02,
                0.98,
                f"RMSE: {rmse_val:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.suptitle("SSP to TL Prediction Results (TAM-FNO)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(str(result_dir / "prediction_results.png"))
        plt.close()
