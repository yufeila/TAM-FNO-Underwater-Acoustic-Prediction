import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

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

def main():
    print(">>> Starting Low-rank Manifold Visualization for TAM-FNO...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize TAM-FNO (formerly FNO2d_FiLM)
    modes1 = 32
    modes2 = 128
    width = 64
    time_dim = 24
    
    model = FNO2d_FiLM(modes1, modes2, width, time_dim=time_dim).to(device)
    
    # Load weights
    model_path = PROJECT_ROOT / "runs" / "modes1_32_modes2_128_epoch_100" / "model_tam_fno.pth"
    if not model_path.exists():
        print(f"Model weights not found at {model_path}!")
        print("Please ensure the model is trained and weights are available.")
        return
        
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
    
    # 2. Generate time sequence for a full year (2920 samples)
    nt_total = 2920
    global_idx = torch.arange(nt_total, dtype=torch.float32)
    t_feats = make_time_feats(global_idx, nt_total).to(device)
    
    # 3. Extract gamma parameters from the deepest block (film3)
    # The film generator outputs [batch, 2 * width], where first half is gamma
    with torch.no_grad():
        film_params = model.film3(t_feats) # Shape: (2920, 128)
        gamma = film_params[:, :width].cpu().numpy() # Shape: (2920, 64)
        
    print(f"Extracted gamma features shape: {gamma.shape}")
    
    # 4. Dimensionality Reduction (PCA)
    print("Applying PCA...")
    pca = PCA(n_components=2)
    gamma_2d = pca.fit_transform(gamma)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # 5. Visualization setup
    # Create colormap based on time of year (1-12 months approximately)
    # 2920 samples = 365 days * 8 samples/day
    days = np.arange(nt_total) / 8.0
    months = (days / 30.416).astype(int) % 12 + 1 # 1 to 12
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(gamma_2d[:, 0], gamma_2d[:, 1], c=months, cmap='hsv', alpha=0.6, s=10)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Month of the Year', fontsize=12)
    cbar.set_ticks(np.arange(1, 13))
    cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.title('Low-rank Manifold of TAM-FNO Modulator ($\\gamma$ Projection)', fontsize=14, pad=15)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add an arrow or connection to show the cyclic nature?
    # Just plotting the points sequentially with lines can be too messy, but we can plot a smoothed line
    # to show the trajectory
    from scipy.ndimage import gaussian_filter1d
    smoothed_x = gaussian_filter1d(gamma_2d[:, 0], sigma=10)
    smoothed_y = gaussian_filter1d(gamma_2d[:, 1], sigma=10)
    plt.plot(smoothed_x, smoothed_y, 'k-', alpha=0.5, linewidth=1, label='Smoothed Trajectory')
    plt.legend()
    
    # Save the figure
    save_path = PROJECT_ROOT / "runs" / "modes1_32_modes2_128_epoch_100" / "manifold_pca.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    
    # Also do t-SNE for comparison
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    gamma_tsne = tsne.fit_transform(gamma)
    
    plt.figure(figsize=(10, 8))
    scatter_tsne = plt.scatter(gamma_tsne[:, 0], gamma_tsne[:, 1], c=months, cmap='hsv', alpha=0.6, s=10)
    
    cbar_tsne = plt.colorbar(scatter_tsne)
    cbar_tsne.set_label('Month of the Year', fontsize=12)
    cbar_tsne.set_ticks(np.arange(1, 13))
    cbar_tsne.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.title('Low-rank Manifold of TAM-FNO Modulator ($\\gamma$ t-SNE)', fontsize=14, pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    smoothed_x_tsne = gaussian_filter1d(gamma_tsne[:, 0], sigma=10)
    smoothed_y_tsne = gaussian_filter1d(gamma_tsne[:, 1], sigma=10)
    plt.plot(smoothed_x_tsne, smoothed_y_tsne, 'k-', alpha=0.5, linewidth=1, label='Smoothed Trajectory')
    plt.legend()
    
    tsne_save_path = PROJECT_ROOT / "runs" / "modes1_32_modes2_128_epoch_100" / "manifold_tsne.png"
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    print(f"✅ t-SNE Visualization saved to {tsne_save_path}")

if __name__ == '__main__':
    main()
