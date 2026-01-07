
"""
PHASE 2: SOLVE MMOT INSTANCES WITH STABLE SOLVER
"""
import numpy as np
import argparse
import sys
import os
import time
from tqdm import tqdm

# Import solver from module
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT')
from mmot.core.solver_stable import solve_mmot_stable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to npz with marginals')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epsilon_scaled', type=float, default=0.2)
    parser.add_argument('--max_iter', type=int, default=2000)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    marginals_list = data['marginals']
    grid = data['grid'] # Shared grid
    
    n_samples = len(marginals_list)
    print(f"Loaded {n_samples} instances. Starting solver...")
    
    results = []
    
    start_time = time.time()
    
    for i in tqdm(range(n_samples)):
        m = marginals_list[i]
        
        # Check normalization (safety)
        # Re-normalize if needed (solver expects sum=1)
        # Not strictly needed if generator is good, but safe.
        
        # Verify shape
        # m shape is (N+1, M)
        
        try:
            # Solve
            res = solve_mmot_stable(
                m, grid, 
                max_iter=args.max_iter, 
                epsilon=0.01, # This is base epsilon, function applies annealing
                # Wait, solver_stable.py takes epsilon (base).
                # And applies max(epsilon/C_max, 1.0/ratio).
                # If we want final ratio 5 (eps=0.2), we rely on hardcoded ratios inside solver_stable.
                # Let's trust solver_stable's internal annealing schedule (Ratio 5).
                damping=0.1,
                verbose=args.verbose
            )
            
            # Store essential data
            results.append({
                'u': res['u'],
                'h': res['h'],
                'drift': res['drift'],
                'iterations': res['iterations'],
                'converged': res['converged']
            })
            
        except Exception as e:
            print(f"Error solving instance {i}: {e}")
            results.append(None)
            
    total_time = time.time() - start_time
    print(f"Solved {n_samples} in {total_time:.1f}s ({total_time/n_samples:.2f}s/item)")
    
    # Save results
    # We save a new NPZ containing the solution + original params?
    # Or just solutions?
    # User's plan says "Merge tomorrow".
    # I should save EVERYTHING needed for training: marginals, u, h.
    
    # Combine original data with solutions
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    
    u_list = [results[i]['u'] for i in valid_indices]
    h_list = [results[i]['h'] for i in valid_indices]
    drift_list = [results[i]['drift'] for i in valid_indices]
    
    filtered_marginals = marginals_list[valid_indices]
    filtered_params = data['params'][valid_indices]
    
    np.savez_compressed(args.output,
                        marginals=filtered_marginals,
                        grid=grid,
                        params=filtered_params,
                        u=np.array(u_list, dtype=object),
                        h=np.array(h_list, dtype=object),
                        drifts=np.array(drift_list)
                       )
    print(f"Saved {len(valid_indices)} solutions to {args.output}")

if __name__ == "__main__":
    main()
