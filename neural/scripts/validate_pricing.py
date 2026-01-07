import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import yaml

# Add project root
sys.path.append('/Volumes/Hippocampus/Antigravity/MMOT')

from neural.models.architecture import create_model
from neural.inference.pricer import NeuralPricer

def validate_pricing(args):
    print("="*60)
    print("PRICING VALIDATION (Neural vs Classical)")
    print("="*60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Model
    with open('neural/configs/production_training.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('neural/configs/default.yaml', 'r') as f:
        defaults = yaml.safe_load(f)
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    final_config = deep_update(defaults, config)
    
    model = create_model(final_config['model'])
    model.to(device)
    
    ckpt_path = 'neural/checkpoints/best_model.pt'
    if Path(ckpt_path).exists():
        print(f"Loading checkpoint: {ckpt_path}")
        # weights_only=False required for numpy scalars in old checkpoints
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("⚠️ WARNING: No checkpoint found! Using random weights.")
    
    model.eval()
    
    # 2. Setup Pricer
    grid_size = final_config['model']['grid_size']
    S_min = final_config['grid']['S_min']
    S_max = final_config['grid']['S_max']
    grid = torch.linspace(S_min, S_max, grid_size).to(device)
    
    pricer = NeuralPricer(model, grid, epsilon=1.0, device=device)
    
    # 3. Validation Loop
    data_dir = Path('neural/data/val')
    files = sorted([f for f in data_dir.glob('*.npz') if not f.name.startswith('._')])
    
    if not files:
        data_dir = Path('neural/data/train') # Fallback
        files = sorted([f for f in data_dir.glob('*.npz') if not f.name.startswith('._')])
        print("Using TRAIN data for validation (Val empty)")
    
    n_tests = min(args.n_tests, len(files))
    print(f"Testing on {n_tests} instances...")
    
    errors = []
    neural_prices = []
    classical_prices = []
    drifts = []
    
    for i in range(n_tests):
        f = files[i]
        # print(f"  [{i+1}/{n_tests}] {f.name}")
        
        data = np.load(f, allow_pickle=True)
        marginals = torch.from_numpy(data['marginals']).float().to(device)
        u_star = torch.from_numpy(data['u_star']).float().to(device) # Classical u
        h_star = torch.from_numpy(data['h_star']).float().to(device) # Classical h
        params = data['params'].item()
        
        strike = params['strike']
        T = params.get('T', 1.0)
        r = params.get('r', 0.0) # Assume 0 if not present
        
        # A. Neural Price
        # This calls helper which samples paths using Neural u, h
        price_neural = pricer.price_asian_call(marginals, strike, num_paths=5000, T=T, r=r)
        
        # B. Classical Price (Using Ground Truth Potentials)
        price_classical = pricer.price_with_potentials(u_star, h_star, marginals, strike, num_paths=5000, T=T, r=r)
        
        # C. Metrics
        rel_error = abs(price_neural - price_classical) / (price_classical + 1e-6)
        
        # Drift Check (Neural)
        # We need to run sample_paths again or modify pricer to return drift?
        # Let's trust price_neural for price, but we need paths for drift.
        # pricer.sample_paths is stochastic, so let's run it once and compute both.
        # To avoid re-running, let's call sample_paths manualy.
        
        paths_neural = pricer.sample_paths(marginals, num_paths=2000)
        avg_path = paths_neural.mean(dim=0).cpu().numpy()
        drift = abs(avg_path[-1] - avg_path[0])
        
        print(f"  [{i+1}/{n_tests}] P_neu={price_neural:.4f} P_cla={price_classical:.4f} Err={rel_error*100:.2f}% Drift={drift:.4f}")
        
        errors.append(rel_error)
        neural_prices.append(price_neural)
        classical_prices.append(price_classical)
        drifts.append(drift)
        
    errors = np.array(errors) * 100
    drifts = np.array(drifts)
    
    print("-" * 60)
    print("VALIDATION RESULTS")
    print("-" * 60)
    print(f"Mean Pricing Error: {errors.mean():.2f}% +/- {errors.std():.2f}%")
    print(f"Max Pricing Error:  {errors.max():.2f}%")
    print(f"Mean Drift:         {drifts.mean():.4f}")
    
    if errors.mean() < 5.0:
        print("✅ Pricing Validation PASSED (<5% Error)")
    else:
        print("❌ Pricing Validation FAILED (>5% Error)")
    
    if drifts.mean() < 1.0:
        print("✅ Drift Validation PASSED (<1.0)")
    else:
        print("⚠️ Drift still present.")
        
    if args.output:
        import json
        out_data = {
            'mean_error': float(errors.mean()),
            'max_error': float(errors.max()),
            'mean_drift': float(drifts.mean()),
            'details': []
        }
        for i in range(len(errors)):
            out_data['details'].append({
                'error': float(errors[i]),
                'drift': float(drifts[i]),
                'neural_price': float(neural_prices[i]),
                'classical_price': float(classical_prices[i])
            })
        with open(args.output, 'w') as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved results to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tests', type=int, default=20)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    validate_pricing(args)
