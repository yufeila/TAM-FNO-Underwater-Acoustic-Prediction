import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from tam_fno.io_mat import MatReader
from tam_fno.normalizer import UnitGaussianNormalizer
from tam_fno.models.fno2d_film import FNO2d_FiLM

def month_from_global_idx(global_idx: np.ndarray, samples_per_day: int = 8) -> np.ndarray:
    month_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=int)
    cum = np.cumsum(month_days)
    day = (global_idx // samples_per_day).astype(int)  # 0..364
    month = np.searchsorted(cum, day + 1) + 1          # 1..12
    return month

def fourier_feats_1d(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * np.pi
    ang = two_pi * t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

def make_time_feats(global_idx: torch.Tensor, nt_total=2920) -> torch.Tensor:
    day_phase  = (global_idx.remainder(8.0)) / 8.0
    year_phase = global_idx / (nt_total - 1.0)
    freqs_day  = torch.arange(1, 1 + 4, dtype=torch.float32)
    freqs_year = torch.arange(1, 1 + 8, dtype=torch.float32)
    fd = fourier_feats_1d(day_phase, freqs_day)
    fy = fourier_feats_1d(year_phase, freqs_year)
    return torch.cat([fd, fy], dim=-1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    DATA_DIR = PROJECT_ROOT / "data"
    TEST_X_PATH  = DATA_DIR / "interp_test_x_SSP_TLshape_ndrz10.mat"
    TEST_Y_PATH  = DATA_DIR / "test_y_SSP_TLshape_ndrz10.mat"
    NORM_PATH = PROJECT_ROOT / "normalizers" / "ssp_tl_norm_train2336_ndrz10.pt"
    MODEL_PATH = PROJECT_ROOT / "experiments" / "tam_fno" / "runs" / "modes1_32_modes2_128_epoch_100" / "model_tam_fno.pth"
    
    # Load normalizers
    norm_dict = torch.load(str(NORM_PATH), map_location='cpu')
    x_normalizer = UnitGaussianNormalizer(torch.zeros(1))
    x_normalizer.mean = norm_dict['x_mean']
    x_normalizer.std = norm_dict['x_std']
    x_normalizer.eps = norm_dict['eps']
    
    y_normalizer = UnitGaussianNormalizer(torch.zeros(1))
    y_normalizer.mean = norm_dict['y_mean']
    y_normalizer.std = norm_dict['y_std']
    y_normalizer.eps = norm_dict['eps']
    
    if torch.cuda.is_available():
        x_normalizer.cuda()
        y_normalizer.cuda()
        
    # Load test data
    x_test = MatReader(str(TEST_X_PATH)).read_field('test_x')[:584].unsqueeze(-1)
    y_test = MatReader(str(TEST_Y_PATH)).read_field('test_y')[:584]
    
    # Time features
    ntrain = 2336
    ntest = 584
    test_global_idx = torch.arange(ntrain, ntrain + ntest, dtype=torch.float32)
    t_test = make_time_feats(test_global_idx)
    
    # Normalize x
    x_test = x_normalizer.encode(x_test.to(device))
    
    # Load Model
    model = FNO2d_FiLM(32, 128, 64, time_dim=24).to(device)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.eval()
    
    # Predict
    rmses = []
    month_arr = month_from_global_idx(test_global_idx.numpy())
    
    with torch.no_grad():
        for i in range(ntest):
            x = x_test[i:i+1]
            t = t_test[i:i+1].to(device)
            y_gt = y_test[i].numpy()
            
            pred = model(x, t).squeeze()
            pred = y_normalizer.decode(pred).cpu().numpy()
            
            err = pred - y_gt
            rmse = np.sqrt(np.mean(err**2))
            rmses.append(rmse)
            
    rmses = np.array(rmses)
    
    # Print Monthly Stats
    print("--- FiLM Model Monthly RMSE ---")
    for m in [10, 11, 12]:
        idx = np.where(month_arr == m)[0]
        if len(idx) > 0:
            m_rmse = rmses[idx].mean()
            print(f"Month {m} (n={len(idx)}): {m_rmse:.4f}")
            
    # Print Seasonal Stats
    djf_idx = np.where(np.isin(month_arr, [12, 1, 2]))[0]
    son_idx = np.where(np.isin(month_arr, [9, 10, 11]))[0]
    
    if len(son_idx) > 0:
        print(f"SON (n={len(son_idx)}): {rmses[son_idx].mean():.4f}")
    if len(djf_idx) > 0:
        print(f"DJF (n={len(djf_idx)}): {rmses[djf_idx].mean():.4f}")

if __name__ == '__main__':
    main()
