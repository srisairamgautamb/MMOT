"""
Heston Stochastic Volatility Data Generator

Generates MMOT instances using Heston dynamics:
  dS_t = μS_t dt + √v_t S_t dW_t^S
  dv_t = κ(θ - v_t)dt + ξ√v_t dW_t^v
  
  Corr(dW_t^S, dW_t^v) = ρ

Features:
  - Time-varying volatility
  - Vol clustering (high vol → high vol)
  - Leverage effect (ρ < 0: price↓ → vol↑)
  - Realistic volatility smile
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import jax.numpy as jnp
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT')
from mmot.core.solver import solve_mmot


def generate_heston_paths(
    S0: float,
    v0: float,
    T: float,
    N: int,
    kappa: float = 2.0,   # Mean reversion speed
    theta: float = 0.04,  # Long-run variance (20% vol)
    xi: float = 0.3,      # Vol of vol
    rho: float = -0.7,    # Leverage effect (negative correlation)
    seed: int = 42
) -> tuple:
    """
    Generate Heston paths using Euler-Maruyama discretization.
    
    Returns:
        prices: (n_paths, N+1) price paths
        vols: (n_paths, N+1) volatility paths
    """
    np.random.seed(seed)
    
    dt = T / N
    mu = 0.05  # Drift
    
    # Correlated Brownian motions
    dW_S = np.random.randn(N) * np.sqrt(dt)
    dW_v_indep = np.random.randn(N) * np.sqrt(dt)
    dW_v = rho * dW_S + np.sqrt(1 - rho**2) * dW_v_indep
    
    # Initialize
    S = np.zeros(N+1)
    v = np.zeros(N+1)
    S[0] = S0
    v[0] = v0
    
    # Euler scheme with Feller condition handling
    for i in range(N):
        # Volatility (with reflection at zero)
        v[i+1] = v[i] + kappa * (theta - v[i]) * dt + xi * np.sqrt(max(v[i], 0)) * dW_v[i]
        v[i+1] = max(v[i+1], 1e-8)  # Floor at small positive
        
        # Price
        S[i+1] = S[i] * np.exp(
            (mu - 0.5 * v[i]) * dt + np.sqrt(v[i]) * dW_S[i]
        )
    
    return S, v


def generate_heston_instance(
    instance_id: int,
    output_dir: Path,
    N: int = None,
    M: int = 150
) -> dict:
    """Generate one Heston MMOT instance."""
    
    np.random.seed(instance_id)
    
    if N is None:
        N = np.random.choice([2, 3, 5, 10, 20, 30, 50],
                            p=[0.1, 0.15, 0.3, 0.25, 0.1, 0.05, 0.05])
    
    T = np.random.uniform(0.05, 1.0)
    S0 = np.random.uniform(80, 120)
    
    # Heston parameters
    v0 = np.random.uniform(0.02, 0.06)  # Initial variance (14-24% vol)
    kappa = np.random.uniform(1.0, 5.0)  # Mean reversion speed
    theta = np.random.uniform(0.02, 0.06)  # Long-run variance
    xi = np.random.uniform(0.2, 0.5)  # Vol of vol
    rho = np.random.uniform(-0.9, -0.3)  # Leverage effect
    
    grid = jnp.linspace(0, 1, M)
    
    # Generate multiple paths for marginal estimation
    n_paths = 5000
    paths = np.zeros((n_paths, N+1))
    vols = np.zeros((n_paths, N+1))
    
    for i in range(n_paths):
        S_path, v_path = generate_heston_paths(
            S0, v0, T, N, kappa, theta, xi, rho,
            seed=instance_id * 10000 + i
        )
        paths[i] = S_path
        vols[i] = v_path
    
    # Estimate marginals using histogram + smoothing
    from scipy.ndimage import gaussian_filter1d
    
    marginals = np.zeros((N+1, M))
    
    for t in range(N+1):
        data = paths[:, t]
        
        # Normalize to [0, 1]
        data_min, data_max = data.min(), data.max()
        data_norm = (data - data_min) / (data_max - data_min + 1e-8)
        
        # Histogram
        hist, bin_edges = np.histogram(data_norm, bins=M, range=(0, 1), density=False)
        marginals[t] = hist.astype(float)
        
        # Smooth
        marginals[t] = gaussian_filter1d(marginals[t], sigma=2.0)
        
        # Normalize
        marginals[t] = marginals[t] / (marginals[t].sum() + 1e-10)
    
    marginals_jax = jnp.array(marginals)
    
    # Solve MMOT
    try:
        C = jnp.zeros((M, M))
        
        u_star, h_star, converged = solve_mmot(
            marginals=marginals_jax,
            C=C,
            x_grid=grid,
            epsilon=1.0,
            max_iter=500
        )
        
        # Save
        output_path = output_dir / f'heston_{instance_id:05d}.npz'
        np.savez_compressed(
            output_path,
            marginals=marginals,
            u_star=np.array(u_star),
            h_star=np.array(h_star),
            metadata={
                'N': N,
                'T': T,
                'S0': S0,
                'v0': v0,
                'kappa': kappa,
                'theta': theta,
                'xi': xi,
                'rho': rho,
                'type': 'heston'
            }
        )
        
        return {'status': 'success', 'N': N, 'file': output_path.name}
        
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}


def generate_heston_dataset(
    n_instances: int = 2000,
    output_dir: str = 'neural/data/augmented/heston'
):
    """Generate full Heston dataset."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("HESTON STOCHASTIC VOLATILITY DATA GENERATION")
    print("="*80)
    print(f"\nGenerating {n_instances} instances...")
    print(f"Output: {output_path}")
    print(f"\nFeatures:")
    print(f"  - Stochastic volatility (κ = 1-5)")
    print(f"  - Vol of vol (ξ = 0.2-0.5)")
    print(f"  - Leverage effect (ρ = -0.9 to -0.3)")
    print(f"  - Realistic vol clustering")
    print()
    
    results = []
    
    for i in tqdm(range(n_instances), desc="Generating"):
        result = generate_heston_instance(i, output_path)
        results.append(result)
    
    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    
    print(f"\n✅ Complete: {success}/{n_instances} instances generated")
    print(f"📂 Saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    results = generate_heston_dataset(n_instances=2000)
