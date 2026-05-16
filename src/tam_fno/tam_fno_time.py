from __future__ import annotations

import numpy as np
import torch

from .tam_fno_config import (
    NTOTAL,
    SAMPLES_PER_DAY,
    TIME_DAY_HARMONICS,
    TIME_FEATURE_DIM,
    TIME_YEAR_HARMONICS,
)


def fourier_features_1d(phase: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * np.pi
    angles = two_pi * phase[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def build_time_feature_vector(
    global_idx: torch.Tensor,
    nt_total: int = NTOTAL,
    samples_per_day: int = SAMPLES_PER_DAY,
    day_harmonics: int = TIME_DAY_HARMONICS,
    year_harmonics: int = TIME_YEAR_HARMONICS,
) -> torch.Tensor:
    day_phase = (global_idx.remainder(float(samples_per_day))) / float(samples_per_day)
    year_phase = global_idx / float(nt_total - 1)

    freqs_day = torch.arange(
        1, day_harmonics + 1, dtype=torch.float32, device=global_idx.device
    )
    freqs_year = torch.arange(
        1, year_harmonics + 1, dtype=torch.float32, device=global_idx.device
    )
    feat_day = fourier_features_1d(day_phase, freqs_day)
    feat_year = fourier_features_1d(year_phase, freqs_year)
    features = torch.cat([feat_day, feat_year], dim=-1)
    if features.shape[-1] != TIME_FEATURE_DIM:
        raise RuntimeError(
            f"Expected time feature dim {TIME_FEATURE_DIM}, got {features.shape[-1]}"
        )
    return features


def month_from_global_idx(
    global_idx: np.ndarray, samples_per_day: int = SAMPLES_PER_DAY
) -> np.ndarray:
    month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)
    cumulative = np.cumsum(month_days)
    day_idx = (global_idx // samples_per_day).astype(int)
    return np.searchsorted(cumulative, day_idx + 1) + 1


def season_from_month(month: np.ndarray) -> np.ndarray:
    season = np.empty_like(month, dtype=object)
    season[np.isin(month, [12, 1, 2])] = "DJF"
    season[np.isin(month, [3, 4, 5])] = "MAM"
    season[np.isin(month, [6, 7, 8])] = "JJA"
    season[np.isin(month, [9, 10, 11])] = "SON"
    return season


def day_from_global_idx(global_idx: np.ndarray, samples_per_day: int = SAMPLES_PER_DAY) -> np.ndarray:
    return (global_idx // samples_per_day).astype(int)
