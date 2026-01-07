"""
Comprehensive real data validation with 200 instances per ticker.
"""
import torch
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from neural.models.architecture import NeuralDualSolver
from neural.tests.validation.MASTER_VALIDATION_ALL import MasterValidation

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load ORIGINAL model
print('='*80)
print('REAL DATA VALIDATION (200 instances per ticker)')
print('='*80)

ckpt = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
model = NeuralDualSolver(grid_size=150, hidden_dim=256, num_layers=3).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f'\nModel: ORIGINAL')
print(f'Epoch: {ckpt.get("epoch", 99)}')
print(f'Device: {device}\n')

# Create validator
validator = MasterValidation(device)
validator.model = model

# Run test with increased instances
tickers = ['SPY', 'AAPL', 'TSLA']
n_per = 200  # INCREASED from 40
valid_N = [2, 3, 5, 10, 20, 30, 50]

all_errors = []
all_drifts = []
ticker_results = {}

from mmot.core.solver import solve_mmot

for ticker in tickers:
    print(f'[{ticker}]')
    returns = validator._get_returns(ticker)
    
    if returns is None or len(returns) < 100:
        print(f'  Skipped (insufficient data)')
        continue
    
    errors = []
    drifts = []
    
    for i in tqdm(range(n_per), desc=f'  Testing', leave=False):
        np.random.seed(400000 + i + hash(ticker) % 10000)
        
        n_periods = np.random.choice(valid_N)
        window_size = min(252, len(returns) - 1)
        start = np.random.randint(0, max(1, len(returns) - window_size))
        window = returns[start:start + window_size]
        
        if len(window) < n_periods * 5:
            continue
        
        # Create marginals
        marginals = validator._create_marginals(window, n_periods, 150)
        grid = np.linspace(50, 200, 150)
        
        # Classical solver
        try:
            dx = grid[:, None] - grid[None, :]
            C = dx ** 2
            u_c, h_c, _ = solve_mmot(
                jnp.array(marginals), jnp.array(C), jnp.array(grid),
                max_iter=500, epsilon=1.0, damping=0.8
            )
            u_c, h_c = np.array(u_c), np.array(h_c)
        except:
            continue
        
        # Neural solver
        marg_t = torch.from_numpy(marginals).float().to(device)
        with torch.no_grad():
            u_n, h_n = model(marg_t.unsqueeze(0))
        u_n, h_n = u_n[0].cpu().numpy(), h_n[0].cpu().numpy()
        
        # Compare
        u_err = validator._np_normalized_error(u_n, u_c)
        h_err = validator._np_normalized_error(h_n, h_c)
        error = (u_err + h_err) / 2
        drift = np.abs(h_n).mean()
        
        errors.append(error)
        drifts.append(drift)
    
    if errors:
        ticker_results[ticker] = {
            'instances': len(errors),
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'drift': float(np.mean(drifts))
        }
        all_errors.extend(errors)
        all_drifts.extend(drifts)
        
        print(f'  Instances: {len(errors)}, Error: {np.mean(errors):.2f}%, Drift: {np.mean(drifts):.4f}')

# Summary
print('\n' + '='*80)
print('FINAL RESULTS')
print('='*80)
if all_errors:
    print(f'\nTotal instances: {len(all_errors)}')
    print(f'Mean error: {np.mean(all_errors):.2f}%')
    print(f'Median error: {np.median(all_errors):.2f}%')
    print(f'Mean drift: {np.mean(all_drifts):.4f}')
    
    print(f'\nBreakdown:')
    for ticker, data in ticker_results.items():
        print(f'  {ticker}: {data["mean_error"]:.2f}% ± {data.get("std", 0):.2f}% ({data["instances"]} instances)')
    
    # Save
    import json
    with open('neural/results/real_data_validation_200.json', 'w') as f:
        json.dump({
            'total_instances': len(all_errors),
            'mean_error': float(np.mean(all_errors)),
            'median_error': float(np.median(all_errors)),
            'mean_drift': float(np.mean(all_drifts)),
            'ticker_results': ticker_results
        }, f, indent=2)
    
    print(f'\n✅ Results saved to: neural/results/real_data_validation_200.json')
else:
    print('\nNo valid results')
