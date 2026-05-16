from __future__ import annotations

from pathlib import Path

import numpy as np

from .tam_fno_config import DEFAULT_SPLIT_SEED, NTEST, NTOTAL, NTRAIN


def build_split_indices(
    nt_total: int = NTOTAL,
    ntrain: int = NTRAIN,
    ntest: int = NTEST,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    if ntrain + ntest != nt_total:
        raise ValueError(f"Split sizes do not sum to nt_total: {ntrain}+{ntest}!={nt_total}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nt_total).astype(np.int64)
    return perm[:ntrain], perm[ntrain:]


def save_split_manifest(
    path: Path,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = DEFAULT_SPLIT_SEED,
    nt_total: int = NTOTAL,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        seed=np.int64(seed),
        nt_total=np.int64(nt_total),
        ntrain=np.int64(len(train_idx)),
        ntest=np.int64(len(test_idx)),
        train_idx=train_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
    )
    return path


def load_split_manifest(path: Path) -> dict[str, np.ndarray | int]:
    data = np.load(path)
    manifest: dict[str, np.ndarray | int] = {
        "seed": int(data["seed"]),
        "nt_total": int(data["nt_total"]),
        "ntrain": int(data["ntrain"]),
        "ntest": int(data["ntest"]),
        "train_idx": data["train_idx"].astype(np.int64),
        "test_idx": data["test_idx"].astype(np.int64),
    }
    return manifest


def ensure_split_manifest(
    path: Path,
    nt_total: int = NTOTAL,
    ntrain: int = NTRAIN,
    ntest: int = NTEST,
    seed: int = DEFAULT_SPLIT_SEED,
    overwrite: bool = False,
) -> dict[str, np.ndarray | int]:
    if path.exists() and not overwrite:
        manifest = load_split_manifest(path)
        if (
            manifest["nt_total"] != nt_total
            or manifest["ntrain"] != ntrain
            or manifest["ntest"] != ntest
            or manifest["seed"] != seed
        ):
            raise RuntimeError(
                f"Existing split manifest at {path} does not match requested config"
            )
        return manifest

    train_idx, test_idx = build_split_indices(nt_total, ntrain, ntest, seed)
    save_split_manifest(path, train_idx, test_idx, seed=seed, nt_total=nt_total)
    return load_split_manifest(path)
