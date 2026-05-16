import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from tam_fno.io_mat import MatReader
from tam_fno.normalizer import UnitGaussianNormalizer
from tam_fno.tam_fno_model import FNO2d_FiLM

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

def extract_all_features(model, x, t):
    grid = model.get_grid(x.shape, x.device)
    x_in = torch.cat((x, grid), dim=-1)
    x_in = model.fc0(x_in)
    x_in = x_in.permute(0, 3, 1, 2)
    x_in = F.pad(x_in, [0, model.padding, 0, model.padding])

    # Block 0
    x1 = model.conv0(x_in)
    x2 = model.w0(x_in)
    F_mid_0 = x1 + x2
    x_out_0 = model.apply_film(F_mid_0, t, model.film0)
    x_out_0_gelu = F.gelu(x_out_0)

    # Block 1
    x1 = model.conv1(x_out_0_gelu)
    x2 = model.w1(x_out_0_gelu)
    F_mid_1 = x1 + x2
    x_out_1 = model.apply_film(F_mid_1, t, model.film1)
    x_out_1_gelu = F.gelu(x_out_1)

    # Block 2
    x1 = model.conv2(x_out_1_gelu)
    x2 = model.w2(x_out_1_gelu)
    F_mid_2 = x1 + x2
    x_out_2 = model.apply_film(F_mid_2, t, model.film2)
    x_out_2_gelu = F.gelu(x_out_2)

    # Block 3
    x1 = model.conv3(x_out_2_gelu)
    x2 = model.w3(x_out_2_gelu)
    F_mid_3 = x1 + x2
    x_out_3 = model.apply_film(F_mid_3, t, model.film3)
    
    # Strip padding
    def strip(feat):
        return feat[..., :-model.padding, :-model.padding]

    return strip(F_mid_0), strip(x_out_0), strip(x_out_1), strip(x_out_2), strip(x_out_3)

def main():
    print(">>> Starting Spatio-Temporal Decoupling Visualization (Cumulative Progression)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DATA_DIR = PROJECT_ROOT / "data"
    TEST_X_PATH  = DATA_DIR / "interp_test_x_SSP_TLshape_ndrz10.mat"
    NORM_PATH = PROJECT_ROOT / "normalizers" / "ssp_tl_norm_train2336_ndrz10.pt"
    MODEL_PATH = PROJECT_ROOT / "runs" / "modes1_32_modes2_128_epoch_100" / "model_tam_fno.pth"
    
    modes1, modes2, width, time_dim = 32, 128, 64, 24
    model = FNO2d_FiLM(modes1, modes2, width, time_dim=time_dim).to(device)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.eval()
    
    x_test_part  = MatReader(str(TEST_X_PATH)).read_field('test_x')
    norm_dict = torch.load(str(NORM_PATH), map_location='cpu')
    x_normalizer = UnitGaussianNormalizer(torch.zeros(1))
    x_normalizer.mean, x_normalizer.std = norm_dict['x_mean'], norm_dict['x_std']
    x_normalizer.eps = norm_dict['eps']
    
    x_base = x_test_part[0:1].unsqueeze(-1)
    x_base = x_normalizer.encode(x_base).to(device)
    
    time_indices = [480, 1200, 1920, 2640]
    season_names = ["Spring", "Summer", "Autumn", "Winter"]
    
    lists = {
        'm0': [], 'o0': [], 'o1': [], 'o2': [], 'o3': []
    }
    
    with torch.no_grad():
        for t_idx in time_indices:
            global_idx = torch.tensor([t_idx], dtype=torch.float32)
            t_feat = make_time_feats(global_idx).to(device)
            m0, o0, o1, o2, o3 = extract_all_features(model, x_base, t_feat)
            lists['m0'].append(m0[0].cpu().numpy())
            lists['o0'].append(o0[0].cpu().numpy())
            lists['o1'].append(o1[0].cpu().numpy())
            lists['o2'].append(o2[0].cpu().numpy())
            lists['o3'].append(o3[0].cpu().numpy())
            
    def get_best_channel(feat_list):
        stack = np.stack(feat_list)
        var = np.var(stack, axis=0)
        mean_var_per_channel = np.mean(var, axis=(1, 2))
        return np.argmax(mean_var_per_channel)

    # Automatically pick the channel that shows the most variation for each block output
    ch_m0 = get_best_channel(lists['o0']) # Use o0's variance to pick channel for m0 and o0
    ch_o1 = get_best_channel(lists['o1'])
    ch_o2 = get_best_channel(lists['o2'])
    ch_o3 = get_best_channel(lists['o3'])
    
    fig, axes = plt.subplots(5, 4, figsize=(16, 18))
    fig.suptitle('Cumulative Temporal Modulation Across FNO Blocks\n(Fixed Input SSP, Varying Seasons)', fontsize=20, y=0.98)
    
    for i in range(4):
        # Row 1: m0 (Unmodulated Base)
        ax = axes[0, i]
        ax.imshow(lists['m0'][i][ch_m0], cmap='viridis', aspect='auto')
        if i == 0: ax.set_ylabel(f"Block 0: Unmodulated $\\hat{{F}}_0$\n(Channel {ch_m0})", fontsize=14)
        ax.set_title(f"{season_names[i]}", fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Row 2: o0
        ax = axes[1, i]
        ax.imshow(lists['o0'][i][ch_m0], cmap='viridis', aspect='auto')
        if i == 0: ax.set_ylabel(f"Block 0: Modulated $F_{{out,0}}$\n(Channel {ch_m0})", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Row 3: o1
        ax = axes[2, i]
        ax.imshow(lists['o1'][i][ch_o1], cmap='viridis', aspect='auto')
        if i == 0: ax.set_ylabel(f"Block 1: Modulated $F_{{out,1}}$\n(Channel {ch_o1})", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        # Row 4: o2
        ax = axes[3, i]
        ax.imshow(lists['o2'][i][ch_o2], cmap='viridis', aspect='auto')
        if i == 0: ax.set_ylabel(f"Block 2: Modulated $F_{{out,2}}$\n(Channel {ch_o2})", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Row 5: o3
        ax = axes[4, i]
        ax.imshow(lists['o3'][i][ch_o3], cmap='viridis', aspect='auto')
        if i == 0: ax.set_ylabel(f"Block 3: Final Modulated $F_{{out,3}}$\n(Channel {ch_o3})", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = PROJECT_ROOT / "runs" / "modes1_32_modes2_128_epoch_100" / "decoupling_progression.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")

if __name__ == '__main__':
    main()
