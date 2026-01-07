"""
Validate generated MMOT dataset for quality assurance.

Checks:
- Data integrity (all files loadable)
- Marginal constraints (sum to 1, non-negative)
- Convex order satisfaction
- Distribution of parameters (N, T, sigma, etc.)
- Solver convergence statistics
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def validate_marginals(marginals, tol=1e-6):
    """
    Check that marginals are valid probability distributions.
    
    Args:
        marginals: [N+1, M] array
        tol: Tolerance for sum-to-one check
    
    Returns:
        is_valid, error_message
    """
    # Check non-negative
    if (marginals < 0).any():
        return False, "Negative values found"
    
    # Check sum to 1
    sums = marginals.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=tol):
        max_error = np.abs(sums - 1.0).max()
        return False, f"Sum not equal to 1 (max error: {max_error:.2e})"
    
    return True, None


def check_convex_order(mu1, mu2, grid, tol=1e-6):
    """
    Check if mu1 ≼_cx mu2 (convex order).
    
    Convex order: ∫φ(x)dμ₁ ≤ ∫φ(x)dμ₂ for all convex φ
    Simplified check: mean(μ₁) = mean(μ₂) and variance(μ₁) ≤ variance(μ₂)
    
    Args:
        mu1, mu2: Probability distributions [M]
        grid: Grid points [M]
        tol: Tolerance
    
    Returns:
        is_valid, stats
    """
    # Compute means
    mean1 = np.sum(grid * mu1)
    mean2 = np.sum(grid * mu2)
    
    # Compute variances
    var1 = np.sum((grid - mean1)**2 * mu1)
    var2 = np.sum((grid - mean2)**2 * mu2)
    
    # Check conditions
    mean_equal = np.abs(mean1 - mean2) < tol * np.abs(mean1 + mean2)
    var_increasing = var2 >= var1 - tol
    
    is_valid = mean_equal and var_increasing
    
    stats = {
        'mean1': mean1,
        'mean2': mean2,
        'var1': var1,
        'var2': var2,
        'mean_equal': mean_equal,
        'var_increasing': var_increasing
    }
    
    return is_valid, stats


def analyze_dataset(data_dir, output_dir=None):
    """
    Analyze complete dataset.
    
    Args:
        data_dir: Directory containing .npz files
        output_dir: Directory to save analysis plots (optional)
    """
    data_dir = Path(data_dir)
    files = sorted([f for f in data_dir.glob('*.npz') if not f.name.startswith('._')])
    
    if len(files) == 0:
        print(f"No .npz files found in {data_dir}")
        return
    
    print(f"Found {len(files)} instances in {data_dir}")
    print("="*70)
    
    # Statistics collectors
    stats = {
        'N_values': [],
        'T_values': [],
        'sigma_values': [],
        'S0_values': [],
        'strike_values': [],
        'runtimes': [],
        'iterations': [],
        'dual_values': [],
        'marginal_errors': [],
        'convex_order_violations': 0,
        'load_failures': 0,
        'marginal_violations': 0
    }
    
    print("Analyzing dataset...")
    for file in tqdm(files):
        try:
            # Load data
            data = np.load(file, allow_pickle=True)
            
            marginals = data['marginals']
            u_star = data['u_star']
            h_star = data['h_star']
            dual_value = data['dual_value']
            params = data['params'].item() if isinstance(data['params'], np.ndarray) else data['params']
            
            # Validate marginals
            is_valid, error_msg = validate_marginals(marginals)
            if not is_valid:
                stats['marginal_violations'] += 1
                if stats['marginal_violations'] <= 5:  # Show first 5
                    print(f"\n⚠️  Marginal violation in {file.name}: {error_msg}")
            
            # Check convex order (sample a few timesteps)
            N = marginals.shape[0] - 1
            if N > 0:
                grid = np.linspace(50, 200, marginals.shape[1])
                is_valid, _ = check_convex_order(marginals[0], marginals[N], grid)
                if not is_valid:
                    stats['convex_order_violations'] += 1
            
            # Collect statistics
            if 'N' in params:
                stats['N_values'].append(params['N'])
            if 'T' in params:
                stats['T_values'].append(params['T'])
            if 'sigma' in params:
                stats['sigma_values'].append(params['sigma'])
            if 'S0' in params:
                stats['S0_values'].append(params['S0'])
            if 'strike' in params:
                stats['strike_values'].append(params['strike'])
            
            stats['dual_values'].append(dual_value)
            
            # Marginal error
            marginal_error = np.abs(marginals.sum(axis=1) - 1.0).max()
            stats['marginal_errors'].append(marginal_error)
            
        except Exception as e:
            stats['load_failures'] += 1
            if stats['load_failures'] <= 5:  # Show first 5
                print(f"\n❌ Failed to load {file.name}: {e}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("DATASET VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\nFiles:")
    print(f"  Total: {len(files)}")
    print(f"  Loaded successfully: {len(files) - stats['load_failures']}")
    print(f"  Load failures: {stats['load_failures']}")
    
    print(f"\nQuality Checks:")
    print(f"  Marginal violations: {stats['marginal_violations']}")
    print(f"  Convex order violations: {stats['convex_order_violations']}")
    if stats['marginal_errors']:
        print(f"  Max marginal error: {np.max(stats['marginal_errors']):.2e}")
        print(f"  Mean marginal error: {np.mean(stats['marginal_errors']):.2e}")
    
    if stats['N_values']:
        print(f"\nParameter Distribution:")
        print(f"  N (time steps):")
        print(f"    Range: [{np.min(stats['N_values'])}, {np.max(stats['N_values'])}]")
        print(f"    Mean: {np.mean(stats['N_values']):.1f}")
        print(f"    Std: {np.std(stats['N_values']):.1f}")
    
    if stats['sigma_values']:
        print(f"  Volatility (σ):")
        print(f"    Range: [{np.min(stats['sigma_values']):.3f}, {np.max(stats['sigma_values']):.3f}]")
        print(f"    Mean: {np.mean(stats['sigma_values']):.3f}")
    
    if stats['dual_values']:
        print(f"\nSolver Statistics:")
        print(f"  Dual values:")
        print(f"    Range: [{np.min(stats['dual_values']):.4f}, {np.max(stats['dual_values']):.4f}]")
        print(f"    Mean: {np.mean(stats['dual_values']):.4f}")
    
    # Generate plots
    if output_dir and any(stats['N_values']):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nGenerating analysis plots in {output_dir}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # N distribution
        if stats['N_values']:
            axes[0, 0].hist(stats['N_values'], bins=20, edgecolor='black')
            axes[0, 0].set_title('Time Steps (N) Distribution')
            axes[0, 0].set_xlabel('N')
            axes[0, 0].set_ylabel('Count')
        
        # Volatility distribution
        if stats['sigma_values']:
            axes[0, 1].hist(stats['sigma_values'], bins=30, edgecolor='black')
            axes[0, 1].set_title('Volatility (σ) Distribution')
            axes[0, 1].set_xlabel('σ')
            axes[0, 1].set_ylabel('Count')
        
        # Dual values
        if stats['dual_values']:
            axes[0, 2].hist(stats['dual_values'], bins=30, edgecolor='black')
            axes[0, 2].set_title('Dual Values Distribution')
            axes[0, 2].set_xlabel('Dual Value')
            axes[0, 2].set_ylabel('Count')
        
        # Marginal errors
        if stats['marginal_errors']:
            axes[1, 0].hist(stats['marginal_errors'], bins=30, edgecolor='black')
            axes[1, 0].set_title('Marginal Errors')
            axes[1, 0].set_xlabel('Max |sum - 1|')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_yscale('log')
        
        # S0 vs Strike
        if stats['S0_values'] and stats['strike_values']:
            axes[1, 1].scatter(stats['S0_values'], stats['strike_values'], alpha=0.3)
            axes[1, 1].set_title('Spot vs Strike')
            axes[1, 1].set_xlabel('S0')
            axes[1, 1].set_ylabel('Strike')
            axes[1, 1].plot([80, 120], [80, 120], 'r--', label='ATM')
            axes[1, 1].legend()
        
        # T vs sigma
        if stats['T_values'] and stats['sigma_values']:
            sc = axes[1, 2].scatter(stats['T_values'], stats['sigma_values'], 
                                    c=stats['N_values'], alpha=0.5, cmap='viridis')
            axes[1, 2].set_title('Maturity vs Volatility')
            axes[1, 2].set_xlabel('T (years)')
            axes[1, 2].set_ylabel('σ')
            plt.colorbar(sc, ax=axes[1, 2], label='N')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_analysis.png', dpi=150)
        print(f"  Saved: {output_dir / 'dataset_analysis.png'}")
    
    print("\n" + "="*70)
    
    # Return summary
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate MMOT dataset')
    parser.add_argument('--data_dir', type=str, default='data/train',
                        help='Directory containing .npz files')
    parser.add_argument('--output_dir', type=str, default='data/analysis',
                        help='Directory to save analysis plots')
    args = parser.parse_args()
    
    analyze_dataset(args.data_dir, args.output_dir)
