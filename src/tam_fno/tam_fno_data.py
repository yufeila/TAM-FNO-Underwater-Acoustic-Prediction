from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .tam_fno_config import (
    DEFAULT_SPLIT_SEED,
    NTEST,
    NTRAIN,
    NTOTAL,
    SSP_HEIGHT,
    SSP_WIDTH,
    TL_HEIGHT,
    TL_WIDTH,
)
from .tam_fno_split import ensure_split_manifest, load_split_manifest


def load_raw_mat_dataset(mat_path: Path, dataset_name: str) -> np.ndarray:
    import h5py

    with h5py.File(mat_path, "r") as handle:
        array = handle[dataset_name][()]
    axes = tuple(range(array.ndim - 1, -1, -1))
    return np.transpose(array, axes=axes).astype(np.float32)


def interpolate_ssp_to_tl_grid(ssp: np.ndarray) -> np.ndarray:
    if ssp.shape != (NTOTAL, SSP_HEIGHT, SSP_WIDTH):
        raise ValueError(f"Unexpected SSP shape {ssp.shape}")

    tensor = torch.from_numpy(ssp).unsqueeze(1)
    interp = F.interpolate(
        tensor,
        size=(TL_HEIGHT, TL_WIDTH),
        mode="bilinear",
        align_corners=True,
    )
    return interp.squeeze(1).cpu().numpy().astype(np.float32)


def preprocess_raw_data(
    raw_tl_path: Path,
    raw_ssp_path: Path,
    output_dir: Path,
    split_manifest_path: Path,
    overwrite: bool = False,
) -> dict[str, str | int | list[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_split_manifest(
        split_manifest_path,
        nt_total=NTOTAL,
        ntrain=NTRAIN,
        ntest=NTEST,
        seed=DEFAULT_SPLIT_SEED,
        overwrite=overwrite,
    )

    tl_all = load_raw_mat_dataset(raw_tl_path, "square_TL")
    ssp_all = load_raw_mat_dataset(raw_ssp_path, "square_SSP")
    if tl_all.shape != (NTOTAL, TL_HEIGHT, TL_WIDTH):
        raise ValueError(f"Unexpected TL shape {tl_all.shape}")
    if ssp_all.shape != (NTOTAL, SSP_HEIGHT, SSP_WIDTH):
        raise ValueError(f"Unexpected SSP shape {ssp_all.shape}")

    x_all = interpolate_ssp_to_tl_grid(ssp_all)
    train_idx = manifest["train_idx"]
    test_idx = manifest["test_idx"]
    x_train = x_all[train_idx]
    x_test = x_all[test_idx]
    y_train = tl_all[train_idx]
    y_test = tl_all[test_idx]

    import scipy.io as sio

    sio.savemat(output_dir / "interp_train_x_SSP_TLshape_ndrz10.mat", {"train_x": x_train})
    sio.savemat(output_dir / "interp_test_x_SSP_TLshape_ndrz10.mat", {"test_x": x_test})
    sio.savemat(output_dir / "train_y_SSP_TLshape_ndrz10.mat", {"train_y": y_train})
    sio.savemat(output_dir / "test_y_SSP_TLshape_ndrz10.mat", {"test_y": y_test})

    metadata = {
        "seed": int(manifest["seed"]),
        "nt_total": NTOTAL,
        "ntrain": NTRAIN,
        "ntest": NTEST,
        "ssp_shape": list(ssp_all.shape),
        "tl_shape": list(tl_all.shape),
        "interp_shape": list(x_all.shape),
        "ssp_to_tl_interpolation": {
            "method": "bilinear",
            "align_corners": True,
            "source_shape": [SSP_HEIGHT, SSP_WIDTH],
            "target_shape": [TL_HEIGHT, TL_WIDTH],
        },
    }
    summary_path = output_dir / "preprocess_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "split_manifest_path": str(split_manifest_path),
        "train_x_path": str(output_dir / "interp_train_x_SSP_TLshape_ndrz10.mat"),
        "test_x_path": str(output_dir / "interp_test_x_SSP_TLshape_ndrz10.mat"),
        "train_y_path": str(output_dir / "train_y_SSP_TLshape_ndrz10.mat"),
        "test_y_path": str(output_dir / "test_y_SSP_TLshape_ndrz10.mat"),
    }


def load_preprocessed_split(
    train_x_path: Path,
    train_y_path: Path,
    test_x_path: Path,
    test_y_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .io_mat import MatReader

    x_train = MatReader(str(train_x_path)).read_field("train_x")
    y_train = MatReader(str(train_y_path)).read_field("train_y")
    x_test = MatReader(str(test_x_path)).read_field("test_x")
    y_test = MatReader(str(test_y_path)).read_field("test_y")
    return x_train, y_train, x_test, y_test


def reorder_split_to_global(
    train_arr: np.ndarray,
    test_arr: np.ndarray,
    split_manifest_path: Path,
    nt_total: int = NTOTAL,
) -> np.ndarray:
    manifest = load_split_manifest(split_manifest_path)
    full = np.empty((nt_total, *train_arr.shape[1:]), dtype=train_arr.dtype)
    full[manifest["train_idx"]] = train_arr
    full[manifest["test_idx"]] = test_arr
    return full
