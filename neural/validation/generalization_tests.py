import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import yaml
import pandas as pd

# Add project root
sys.path.append('/Volumes/Hippocampus/Antigravity/MMOT')

from neural.models.architecture import create_model
from neural.inference.pricer import NeuralPricer
from neural.data.generator import solve_instance, sample_mmot_params

def generate_test_case(regime):
    """
    Generate a specific test case based on regime.
    Regime: 'low_vol', 'high_vol', 'long_horizon', 'dense_time'
    """
    params = sample_mmot_params()
    
    # Overrides
    if regime == 'low_vol':
        params['sigma'] = 0.15
    elif regime == 'high_vol':
        params['sigma'] = 0.60 # Extrapolation (Train max 0.5)
    elif regime == 'long_horizon':
        params['T'] = 2.0 # Extrapolation (Train max 1.0)
    elif regime == 'dense_time':
        params['N'] = 20 # Train likely 10? (Generator samples random N)
    elif regime == 'default':
        pass
        
    return params

def run_generalization_tests(args):
    print("="*60)
    print("GENERALIZATION & ROBUSTNESS TESTS")
    print("="*60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Model
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
        print(f"Loading weights: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Setup Pricer
    grid_size = final_config['model']['grid_size']
    S_min = final_config['grid']['S_min']
    S_max = final_config['grid']['S_max']
    grid = torch.linspace(S_min, S_max, grid_size).to(device)
    pricer = NeuralPricer(model, grid, epsilon=1.0, device=device)
    
    # Test Regimes
    regimes = ['default', 'low_vol', 'high_vol', 'dense_time']
    results = []
    
    for regime in regimes:
        print(f"\nTesting Regime: {regime}")
        for i in range(args.samples_per_regime):
            # 1. Generate & Solve (Classical Ground Truth)
            params = generate_test_case(regime)
            try:
                # solve_instance logic from generator.py
                solution = solve_instance(params, max_iter=1000)
            except Exception as e:
                print(f"  Skipping (Solver Fail): {e}")
                continue
                
            marginals = torch.from_numpy(solution['marginals']).float().to(device)
            u_star = torch.from_numpy(solution['u_star']).float().to(device)
            h_star = torch.from_numpy(solution['h_star']).float().to(device)
            
            # 2. Neural Prediction
            with torch.no_grad():
                u_pred, h_pred = model(marginals.unsqueeze(0))
                # Remove batch dim
                u_pred = u_pred.squeeze(0)
                h_pred = h_pred.squeeze(0)
            
            # 3. Metrics
            # MAE on potentials
            mae_u = (u_pred - u_star).abs().mean().item()
            
            # Pricing Error
            strike = params['strike']
            T = params['T']
            r = 0.0 # solver assumes r=0/risk neutral Q
            
            # For Neural Price, we use NeuralPricer logic (Monte Carlo)
            price_neural = pricer.price_asian_call(marginals, strike, num_paths=2000, T=T, r=r)
            
            # For Classical, we use Pricer logic with classical potentials (to isolate model error from MC error)
            price_classical = pricer.price_with_potentials(u_star, h_star, marginals, strike, num_paths=2000, T=T, r=r)
            
            err = abs(price_neural - price_classical)
            rel_err = err / (price_classical + 1e-6)
            
            print(f"  [{i+1}] MAE_u={mae_u:.4f} P_neu={price_neural:.2f} P_cla={price_classical:.2f} Err={rel_err*100:.1f}%")
            
            results.append({
                'regime': regime,
                'mae_u': mae_u,
                'price_neural': price_neural,
                'price_classical': price_classical,
                'rel_error': rel_err
            })
            
    # Aggregation
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    print(df.groupby('regime')[['mae_u', 'rel_error']].agg(['mean', 'std']))
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_per_regime', type=int, default=3)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    run_generalization_tests(args)
