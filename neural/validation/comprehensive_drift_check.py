"""
COMPREHENSIVE DRIFT VERIFICATION
Complete, honest measurement on ALL validation data.
No cherry-picking, no filtering.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural.models.architecture import NeuralDualSolver


def compute_gibbs_drift(u, h, marginals, grid, epsilon=1.0):
    """Compute drift using Gibbs kernel (EXACT same as MASTER_VALIDATION)."""
    N_plus_1, M = u.shape
    N = N_plus_1 - 1
    
    total_drift = 0.0
    
    for t in range(N):
        u_tp1 = u[t+1]
        h_t = h[t]
        mu_t = marginals[t]
        
        # Gibbs kernel
        delta_S = grid[None, :] - grid[:, None]
        log_kernel = (u_tp1[None, :] + h_t[:, None] * delta_S) / epsilon
        kernel = F.softmax(log_kernel, dim=-1)
        
        # E[Y|X]
        cond_exp = torch.matmul(kernel, grid)
        
        # Drift
        drift = torch.abs(cond_exp - grid)
        weighted_drift = (drift * mu_t).sum().item()
        total_drift += weighted_drift
    
    return total_drift / N


def comprehensive_drift_check():
    """Measure drift on ALL validation data."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print("="*80)
    print("COMPREHENSIVE DRIFT VERIFICATION")
    print("="*80)
    
    # Load model
    ckpt_path = Path('checkpoints/best_model.pt')
    if not ckpt_path.exists():
        ckpt_path = Path('neural/checkpoints/best_model.pt')
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    model = NeuralDualSolver(grid_size=150, hidden_dim=256, num_layers=3).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    grid = torch.linspace(0, 1, 150).to(device)
    
    # Find ALL validation files
    val_dir = Path('neural/data/val')
    val_files = sorted(val_dir.glob('*.npz'))
    
    print(f"\nFound {len(val_files)} validation files")
    
    # Group by N
    by_n = {}
    for f in val_files:
        try:
            data = np.load(f, allow_pickle=True)
            N = len(data['marginals']) - 1
            if N not in by_n:
                by_n[N] = []
            by_n[N].append(f)
        except:
            continue
    
    print(f"\nValidation data distribution:")
    for n in sorted(by_n.keys()):
        print(f"  N={n:2d}: {len(by_n[n]):4d} files")
    
    # Compute drift for each N
    results = {}
    
    for N in sorted(by_n.keys()):
        print(f"\n{'='*80}")
        print(f"TESTING N={N} ({len(by_n[N])} instances)")
        print(f"{'='*80}")
        
        drifts = []
        errors = []
        
        for file_path in by_n[N]:
            try:
                data = np.load(file_path, allow_pickle=True)
                
                marginals = torch.from_numpy(data['marginals']).float().to(device)
                u_true = torch.from_numpy(data['u_star']).float().to(device)
                h_true = torch.from_numpy(data['h_star']).float().to(device)
                
                # Predict
                with torch.no_grad():
                    u_pred, h_pred = model(marginals.unsqueeze(0))
                    u_pred = u_pred.squeeze(0)
                    h_pred = h_pred.squeeze(0)
                
                # Compute drift
                drift = compute_gibbs_drift(
                    u_pred, h_pred, marginals, grid, epsilon=1.0
                )
                
                # Compute error
                u_err = torch.abs(u_pred - u_true).mean().item()
                h_err = torch.abs(h_pred - h_true).mean().item()
                error = (u_err + h_err) / 2
                
                drifts.append(drift)
                errors.append(error)
            except Exception as e:
                print(f"  Error on {file_path.name}: {e}")
                continue
        
        if len(drifts) == 0:
            continue
            
        # Statistics
        results[N] = {
            'count': len(drifts),
            'drift_mean': float(np.mean(drifts)),
            'drift_median': float(np.median(drifts)),
            'drift_std': float(np.std(drifts)),
            'drift_min': float(np.min(drifts)),
            'drift_max': float(np.max(drifts)),
            'error_mean': float(np.mean(errors)),
            'passes': float(np.mean([d < 0.1 for d in drifts]) * 100)
        }
        
        print(f"\nResults for N={N}:")
        print(f"  Instances: {results[N]['count']}")
        print(f"  Drift (mean ± std): {results[N]['drift_mean']:.4f} ± {results[N]['drift_std']:.4f}")
        print(f"  Drift (median): {results[N]['drift_median']:.4f}")
        print(f"  Drift (range): [{results[N]['drift_min']:.4f}, {results[N]['drift_max']:.4f}]")
        print(f"  Error (mean): {results[N]['error_mean']:.4f}")
        print(f"  Pass rate (drift<0.1): {results[N]['passes']:.1f}%")
        
        # VERDICT
        if results[N]['drift_median'] < 0.1:
            print(f"  ✅ PASS: Median drift < 0.1")
        else:
            print(f"  ❌ FAIL: Median drift >= 0.1")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    all_drifts = []
    for N in results:
        for file_path in by_n[N]:
            try:
                data = np.load(file_path, allow_pickle=True)
                marginals = torch.from_numpy(data['marginals']).float().to(device)
                with torch.no_grad():
                    u_pred, h_pred = model(marginals.unsqueeze(0))
                drift = compute_gibbs_drift(
                    u_pred[0], h_pred[0], marginals, grid, epsilon=1.0
                )
                all_drifts.append(drift)
            except:
                continue
    
    print(f"\nAcross ALL {len(all_drifts)} instances:")
    print(f"  Mean drift: {np.mean(all_drifts):.4f}")
    print(f"  Median drift: {np.median(all_drifts):.4f}")
    print(f"  Std drift: {np.std(all_drifts):.4f}")
    print(f"  Pass rate: {np.mean([d < 0.1 for d in all_drifts])*100:.1f}%")
    
    # FINAL VERDICT
    median = np.median(all_drifts)
    if median < 0.1:
        print(f"\n✅ OVERALL PASS: Median drift {median:.4f} < 0.1")
    else:
        print(f"\n❌ OVERALL FAIL: Median drift {median:.4f} >= 0.1")
    
    # Save results
    output_path = Path('neural/results/drift_comprehensive_check.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'by_n': results,
            'overall': {
                'mean': float(np.mean(all_drifts)),
                'median': float(np.median(all_drifts)),
                'std': float(np.std(all_drifts)),
                'pass_rate': float(np.mean([d < 0.1 for d in all_drifts]))
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved: {output_path}")
    
    return results


if __name__ == '__main__':
    results = comprehensive_drift_check()
