import torch
import numpy as np
import sys
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root
sys.path.append('/Volumes/Hippocampus/Antigravity/MMOT')

from neural.models.architecture import create_model

def stress_test(config_path, checkpoint_path):
    print("="*60)
    print("PHASE 2C STRESS TEST: OVERFITTING & HALLUCINATION CHECK")
    print("="*60)
    
    # 1. Load Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Merge Configs
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(config_path, 'r') as f:
        overrides = yaml.safe_load(f)
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    config = deep_update(config, overrides)
    
    model = create_model(config['model'])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\n✅ Model Loaded from {checkpoint_path}")
    
    # Grid
    M = config['model']['grid_size']
    S_min, S_max = config['grid']['S_min'], config['grid']['S_max']
    grid = torch.linspace(S_min, S_max, M).to(device)
    
    # ---------------------------------------------------------
    # TEST 1: IN-DISTRIBUTION (Validation Set)
    # ---------------------------------------------------------
    print("\n[TEST 1] Standard Validation Set Check")
    # Load one val sample
    val_files = sorted(list(Path('data/val').glob('*.npz')))
    data = np.load(val_files[0])
    marginals = torch.from_numpy(data['marginals']).float().unsqueeze(0).to(device) # [1, N+1, M]
    
    with torch.no_grad():
        u_pred, h_pred = model(marginals)
        
    print(f"   Input Range: {marginals.min():.4f} to {marginals.max():.4f}")
    print(f"   Output u range: {u_pred.min():.4f} to {u_pred.max():.4f}")
    print(f"   Output h range: {h_pred.min():.4f} to {h_pred.max():.4f}")
    
    if torch.isnan(u_pred).any():
        print("   ❌ FAILURE: NaNs in prediction")
    else:
        print("   ✅ Stability: No NaNs")
        
    # ---------------------------------------------------------
    # TEST 2: EXTREME OOD INPUT (High Volatility)
    # ---------------------------------------------------------
    print("\n[TEST 2] Out-of-Distribution: EXTREME VOLATILITY")
    print("   Generating synthetic marginals with Sigma = 5.0 (Training was ~0.38)")
    
    # Create flat/wide marginals
    N = marginals.shape[1] - 1
    x = torch.linspace(S_min, S_max, M).to(device)
    sigma_extreme = 5.0
    mu_extreme = 100.0
    
    # Gaussian pdf
    dist = torch.exp(-0.5 * ((x - mu_extreme)/sigma_extreme)**2)
    dist = dist / dist.sum() # Normalize
    
    ood_marginals = dist.unsqueeze(0).unsqueeze(0).expand(1, N+1, M) # Copy same dist for all t
    
    with torch.no_grad():
        u_ood, h_ood = model(ood_marginals)
        
    print(f"   Output u range: {u_ood.min():.4f} to {u_ood.max():.4f}")
    print(f"   Output h range: {h_ood.min():.4f} to {h_ood.max():.4f}")
    
    # Check for hallucination (values shouldn't explode to millions)
    if u_ood.abs().max() > 100 or h_ood.abs().max() > 100:
        print("   ⚠️ WARNING: Output values very large. Possible Hallucination.")
    else:
        print("   ✅ Robustness: Outputs constrained even for extreme input.")
        
    # ---------------------------------------------------------
    # TEST 3: NOISE INJECTION
    # ---------------------------------------------------------
    print("\n[TEST 3] Robustness to Noise")
    noise = torch.randn_like(marginals) * 0.01
    noisy_input = torch.abs(marginals + noise)
    noisy_input = noisy_input / noisy_input.sum(dim=-1, keepdim=True) # Renormalize
    
    with torch.no_grad():
        u_noise, h_noise = model(noisy_input)
        
    diff = (u_pred - u_noise).abs().mean()
    print(f"   Perturbation Stability: MAE = {diff:.6f}")
    if diff < 0.5:
        print("   ✅ Stable: Small input change -> Small output change")
    else:
        print("   ❌ Unstable: Output changed significantly")

if __name__ == "__main__":
    stress_test('configs/production_training.yaml', 'checkpoints/best_model.pt')
