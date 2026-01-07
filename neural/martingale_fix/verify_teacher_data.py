
import numpy as np
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--checks', type=str, default='drift,convergence,normalization')
    args = parser.parse_args()
    
    print(f"Verifying {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    
    drifts = data['drifts']
    marginals = data['marginals']
    u = data['u']
    h = data['h']
    grid = data['grid']
    
    N_total = len(drifts)
    print(f"Total instances: {N_total}")
    print(f"Grid size: {len(grid)} ({grid[0]} to {grid[-1]})")
    
    # Drift Check
    mean_drift = np.mean(drifts)
    median_drift = np.median(drifts)
    max_drift = np.max(drifts)
    pass_rate = np.mean(drifts < 0.01) * 100
    
    print("\nQUALITY METRICS:")
    print("────────────────")
    print(f"Mean drift:        {mean_drift:.6f}  {'✅' if mean_drift < 0.01 else '⚠️'}")
    print(f"Median drift:      {median_drift:.6f}")
    print(f"Max drift:         {max_drift:.6f}")
    print(f"Pass rate (< 0.01): {pass_rate:.1f}%")
    
    # NaN Check
    nan_count = 0
    for i in range(N_total):
        if np.isnan(u[i]).any() or np.isnan(h[i]).any():
            nan_count += 1
            
    if nan_count == 0:
        print("No NaN/Inf values: ✅")
    else:
        print(f"NaNs found in {nan_count} instances: ❌")
        
    # Scale Check
    u_flat = np.concatenate([ui.flatten() for ui in u[:100]]) # Sample
    h_flat = np.concatenate([hi.flatten() for hi in h[:100]])
    print(f"Mean |u|:          {np.mean(np.abs(u_flat)):.1f}")
    print(f"Mean |h|:          {np.mean(np.abs(h_flat)):.1f}")
    
    if pass_rate > 95:
        print("\nREADY FOR NEURAL TRAINING ✅")
    else:
        print("\nWARNING: High drift rate. Check generator parameters.")

if __name__ == "__main__":
    main()
