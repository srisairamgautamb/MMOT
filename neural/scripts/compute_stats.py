import numpy as np
from pathlib import Path
import os
import argparse

def compute_stats(data_dir, num_samples=None):
    """
    Compute mean and std for u_star and h_star across the dataset.
    """
    data_path = Path(data_dir)
    files = list(data_path.glob('*.npz'))
    
    # Filter out hidden files (._*)
    files = [f for f in files if not f.name.startswith('._')]
    
    if not files:
        print(f"No files found in {data_dir}")
        return

    if num_samples:
        files = files[:num_samples]
        
    print(f"Computing stats from {len(files)} files...")
    
    u_vals = []
    h_vals = []
    
    for f in files:
        try:
            data = np.load(f)
            u_vals.append(data['u_star'].flatten())
            h_vals.append(data['h_star'].flatten())
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
            
    u_all = np.concatenate(u_vals)
    h_all = np.concatenate(h_vals)
    
    print("-" * 40)
    print(f"u_star Stats:")
    print(f"  Mean: {u_all.mean():.6f}")
    print(f"  Std:  {u_all.std():.6f}")
    print(f"  Min:  {u_all.min():.6f}")
    print(f"  Max:  {u_all.max():.6f}")
    print("-" * 40)
    print(f"h_star Stats:")
    print(f"  Mean: {h_all.mean():.6f}")
    print(f"  Std:  {h_all.std():.6f}")
    print(f"  Min:  {h_all.min():.6f}")
    print(f"  Max:  {h_all.max():.6f}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/train', help='Path to training data')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of files to sample (0 for all)')
    args = parser.parse_args()
    
    compute_stats(args.data_dir, args.num_samples if args.num_samples > 0 else None)
