"""
Jump-Diffusion (Merton Model) Data Generator

Generates MMOT instances using Merton jump-diffusion dynamics:
  dS_t = μS_t dt + σS_t dW_t + S_t dJ_t

where J_t is a compound Poisson process with:
  - Jump intensity: λ (jumps per year)
  - Jump size: Log-normal(μ_J, σ_J)

This adds realistic market features:
  - Fat tails (kurtosis >> 0)
  - Skewness (crashes vs rallies)
  - Discrete jump events
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT')
from mmot.core.solver import solve_mmot


def generate_jump_diffusion_paths(
    S0: float,
    T: float,
    N: int,
    sigma: float = 0.20,
    lambda_jump: float = 2.0,  # 2 jumps per year on average
    mu_jump: float = -0.05,  # -5% average jump (downward bias)
    sigma_jump: float = 0.10,  # 10% jump volatility
    seed: int = 42
) -> np.ndarray:
    """
    Generate stock price path with jumps.
    
    Returns:
        prices: (N+1,) array of prices at each time step
    """
    np.random.seed(seed)
    
    dt = T / N
    t_grid = np.linspace(0, T, N+1)
    
    # GBM component
    drift = 0.05  # 5% annual drift
    dW = np.random.randn(N) * np.sqrt(dt)
    
    # Jump component (Poisson process)
    n_jumps = np.random.poisson(lambda_jump * T)
    jump_times = np.sort(np.random.uniform(0, T, n_jumps))
    jump_sizes = np.random.lognormal(mu_jump, sigma_jump, n_jumps)
    
    # Construct path
    log_S = np.log(S0)
    prices = np.zeros(N+1)
    prices[0] = S0
    
    jump_idx = 0
    for i in range(N):
        t = t_grid[i]
        t_next = t_grid[i+1]
        
        # Check for jumps in this interval
        jump_multiplier = 1.0
        while jump_idx < n_jumps and jump_times[jump_idx] < t_next:
            jump_multiplier *= jump_sizes[jump_idx]
            jump_idx += 1
        
        # GBM evolution
        log_S += (drift - 0.5 * sigma**2) * dt + sigma * dW[i]
        
        # Apply jump
        log_S += np.log(jump_multiplier)
        
        prices[i+1] = np.exp(log_S)
    
    return prices


def generate_marginals_from_paths(
    paths: np.ndarray,  # (n_paths, N+1)
    grid: np.ndarray,   # (M,)
) -> np.ndarray:
    """
    Estimate marginal densities from sampled paths using histogram + smoothing.
    
    Returns:
        marginals: (N+1, M) - density at each time on grid
    """
    from scipy.ndimage import gaussian_filter1d
    
    n_paths, N_plus_1 = paths.shape
    M = len(grid)
    marginals = np.zeros((N_plus_1, M))
    
    for t in range(N_plus_1):
        data = paths[:, t]
        
        # Normalize to [0, 1] grid
        data_min, data_max = data.min(), data.max()
        data_norm = (data - data_min) / (data_max - data_min + 1e-8)
        
        # Histogram-based density
        hist, bin_edges = np.histogram(data_norm, bins=M, range=(0, 1), density=False)
        marginals[t] = hist.astype(float)
        
        # Smooth
        marginals[t] = gaussian_filter1d(marginals[t], sigma=2.0)
        
        # Normalize
        marginals[t] = marginals[t] / (marginals[t].sum() + 1e-10)
    
    return marginals


def generate_jump_diffusion_instance(
    instance_id: int,
    output_dir: Path,
    N: int = None,
    M: int = 150
) -> dict:
    """Generate one jump-diffusion MMOT instance."""
    
    # Sample parameters
    np.random.seed(instance_id)
    
    if N is None:
        N = np.random.choice([2, 3, 5, 10, 20, 30, 50],
                            p=[0.1, 0.15, 0.3, 0.25, 0.1, 0.05, 0.05])
    
    T = np.random.uniform(0.05, 1.0)
    sigma = np.random.uniform(0.15, 0.40)
    S0 = np.random.uniform(80, 120)
    
    # Jump parameters
    lambda_jump = np.random.uniform(0.5, 5.0)  # 0.5-5 jumps/year
    mu_jump = np.random.uniform(-0.10, -0.02)  # Negative bias (crashes)
    sigma_jump = np.random.uniform(0.05, 0.15)
    
    grid = jnp.linspace(0, 1, M)
    
    # Generate multiple paths for marginal estimation
    n_paths = 5000
    paths = np.zeros((n_paths, N+1))
    
    for i in range(n_paths):
        paths[i] = generate_jump_diffusion_paths(
            S0, T, N, sigma, lambda_jump, mu_jump, sigma_jump,
            seed=instance_id * 10000 + i
        )
    
    # Estimate marginals
    marginals = generate_marginals_from_paths(paths, grid.copy())
    marginals_jax = jnp.array(marginals)
    
    # Solve MMOT with classical solver
    try:
        C = jnp.zeros((M, M))  # Zero cost (pure martingale transport)
        
        u_star, h_star, converged = solve_mmot(
            marginals=marginals_jax,
            C=C,
            x_grid=grid,
            epsilon=1.0,
            max_iter=500
        )
        
        # Save
        output_path = output_dir / f'jump_diff_{instance_id:05d}.npz'
        
        metadata_dict = {
            'N': int(N),
            'T': float(T),
            'sigma': float(sigma),
            'S0': float(S0),
            'lambda_jump': float(lambda_jump),
            'mu_jump': float(mu_jump),
            'sigma_jump': float(sigma_jump),
            'type': 'jump_diffusion'
        }
        
        np.savez_compressed(
            output_path,
            marginals=marginals,
            u_star=np.array(u_star),
            h_star=np.array(h_star),
            **metadata_dict
        )
        
        return {'status': 'success', 'N': N, 'file': output_path.name}
        
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}


def generate_jump_diffusion_dataset(
    n_instances: int = 2000,
    output_dir: str = 'neural/data/augmented/jump_diffusion'
):
    """Generate full jump-diffusion dataset."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("JUMP-DIFFUSION DATA GENERATION")
    print("="*80)
    print(f"\nGenerating {n_instances} instances...")
    print(f"Output: {output_path}")
    print(f"\nFeatures:")
    print(f"  - Poisson jumps (λ = 0.5-5.0/year)")
    print(f"  - Log-normal jump sizes (μ = -10% to -2%)")
    print(f"  - Adds fat tails + skewness")
    print()
    
    results = []
    
    for i in tqdm(range(n_instances), desc="Generating"):
        result = generate_jump_diffusion_instance(i, output_path)
        results.append(result)
    
    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    
    print(f"\n✅ Complete: {success}/{n_instances} instances generated")
    print(f"📂 Saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    results = generate_jump_diffusion_dataset(n_instances=2000)
