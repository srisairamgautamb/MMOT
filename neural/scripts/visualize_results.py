
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import create_model
from data.loader import get_dataloaders

def visualize_performance(config_path, checkpoint_path, output_dir='results/plots'):
    """Load model and generate performance plots for a random validation sample."""
    import yaml
    
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load config (simple loader for now)
    # Note: In a real scenario we'd use the robust merge from train.py, 
    # but for viz simple loading is usually enough if checkpoint has metadata or we assume m4_optimized
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(config_path, 'r') as f:
        override = yaml.safe_load(f)
        
    # Manual merge for key params needed for model creation
    config['model'].update(override.get('model', {}))
    config['grid'].update(override.get('grid', {}))
    
    # Create Model
    print("Creating model...")
    model = create_model(config['model'])
    
    # Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load Data (One batch)
    _, val_loader = get_dataloaders(
        train_dir='data/train', 
        val_dir='data/val', 
        batch_size=1
    )
    
    # Get one sample
    batch = next(iter(val_loader))
    marginals = batch['marginals'].to(device)
    mask = batch['mask'].to(device)
    u_true = batch['u_star'].to(device)
    h_true = batch['h_star'].to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        u_pred, h_pred = model(marginals, mask)
        
    # Convert to numpy for plotting
    u_pred = u_pred[0].cpu().numpy()
    u_true = u_true[0].cpu().numpy()
    h_pred = h_pred[0].cpu().numpy()
    h_true = h_true[0].cpu().numpy()
    marg = marginals[0].cpu().numpy()
    
    # Get N for this sample
    N_actual = int(batch['N'][0])
    
    print(f"Sample N={N_actual}")
    print(f"Mean Abs Error (u): {np.mean(np.abs(u_pred[:N_actual+1] - u_true[:N_actual+1])):.6f}")
    
    # --- PLOT 1: Potentials (Heatmap comparison) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # U potentials
    im1 = axes[0,0].imshow(u_true[:N_actual+1].T, aspect='auto', origin='lower')
    axes[0,0].set_title(f'True Dual Potentials u* (N={N_actual})')
    axes[0,0].set_ylabel('Grid Index')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(u_pred[:N_actual+1].T, aspect='auto', origin='lower')
    axes[0,1].set_title('Predicted Potentials u_pred')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Difference
    diff = u_pred[:N_actual+1] - u_true[:N_actual+1]
    im3 = axes[1,0].imshow(diff.T, aspect='auto', origin='lower', cmap='RdBu')
    axes[1,0].set_title('Difference (Error)')
    axes[1,0].set_xlabel('Time Step')
    plt.colorbar(im3, ax=axes[1,0])

    # Marginals (Input)
    im4 = axes[1,1].imshow(marg[:N_actual+1].T, aspect='auto', origin='lower', cmap='viridis')
    axes[1,1].set_title('Input Marginals (Density)')
    axes[1,1].set_xlabel('Time Step')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/potentials_heatmap.png')
    print(f"Saved {output_dir}/potentials_heatmap.png")
    
    # --- PLOT 2: Cross-sections (Specific Time Steps) ---
    plt.figure(figsize=(12, 6))
    time_steps = [0, N_actual // 2, N_actual]
    
    for i, t in enumerate(time_steps):
        plt.subplot(1, 3, i+1)
        plt.plot(u_true[t], 'k-', label='True', linewidth=2)
        plt.plot(u_pred[t], 'r--', label='Pred', linewidth=2)
        plt.title(f'Time Step t={t}')
        if i == 0: plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/potentials_cross_section.png')
    print(f"Saved {output_dir}/potentials_cross_section.png")

if __name__ == '__main__':
    visualize_performance(
        config_path='configs/m4_optimized.yaml',
        checkpoint_path='checkpoints/final_model.pt'
    )
