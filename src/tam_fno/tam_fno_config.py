from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

NTRAIN = 2336
NTEST = 584
NTOTAL = 2920
SAMPLES_PER_DAY = 8

TL_HEIGHT = 199
TL_WIDTH = 800
SSP_HEIGHT = 21
SSP_WIDTH = 10

DEPTH_METERS = 200.0
RANGE_METERS = 40_000.0

TIME_DAY_HARMONICS = 4
TIME_YEAR_HARMONICS = 8
TIME_FEATURE_DIM = 2 * (TIME_DAY_HARMONICS + TIME_YEAR_HARMONICS)

DEFAULT_SPLIT_SEED = 2025

DEFAULT_BATCH_SIZE = 20
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_STEP_SIZE = 50
DEFAULT_GAMMA = 0.5

DEFAULT_MODES1 = 32
DEFAULT_MODES2 = 128
DEFAULT_WIDTH = 64


def find_project_root(start: Path | None = None) -> Path:
    probe = (start or Path(__file__)).resolve()
    if probe.is_file():
        probe = probe.parent

    for candidate in [probe, *probe.parents]:
        if (candidate / "src").exists() and (candidate / "experiments").exists():
            return candidate

    raise RuntimeError(f"Cannot find FNO project root from {start or __file__}")


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    raw_data_dir: Path
    data_dir: Path
    normalizers_dir: Path
    results_dir: Path
    split_manifest_path: Path
    normalizer_path: Path
    train_x_path: Path
    train_y_path: Path
    test_x_path: Path
    test_y_path: Path


def get_project_paths(project_root: Path | None = None) -> ProjectPaths:
    root = find_project_root(project_root)
    env_raw = os.environ.get("TAM_FNO_RAW_DATA")
    sibling_raw = root.parent / "raw_data"
    local_raw = root / "raw_data"
    if env_raw:
        raw_dir = Path(env_raw).expanduser().resolve()
    else:
        raw_dir = sibling_raw if sibling_raw.exists() or not local_raw.exists() else local_raw

    data_dir = root / "data"
    normalizers_dir = root / "normalizers"
    results_dir = root / "results"

    return ProjectPaths(
        project_root=root,
        raw_data_dir=raw_dir,
        data_dir=data_dir,
        normalizers_dir=normalizers_dir,
        results_dir=results_dir,
        split_manifest_path=data_dir / "split_seed2025.npz",
        normalizer_path=normalizers_dir / "ssp_tl_norm_train2336_ndrz10.pt",
        train_x_path=data_dir / "interp_train_x_SSP_TLshape_ndrz10.mat",
        train_y_path=data_dir / "train_y_SSP_TLshape_ndrz10.mat",
        test_x_path=data_dir / "interp_test_x_SSP_TLshape_ndrz10.mat",
        test_y_path=data_dir / "test_y_SSP_TLshape_ndrz10.mat",
    )
