#!/usr/bin/env python3
"""
TEACHER DATA GENERATOR
======================
Generates MMOT instances with CLASSICAL SOLVER solutions as teacher signal.

This is the CORRECT approach:
1. Generate diverse marginals (GBM, Merton, Heston)
2. Solve MMOT classically (Martingale-Sinkhorn)
3. Save: marginals + u_classical + h_classical

Usage:
    python generate_teacher_data.py --n_instances 12000 --output teacher_data.npz
"""

import numpy as np
import jax.numpy as jnp
import sys
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import math

# Add project root to path
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT')

from mmot.core.solver import solve_mmot


# ============================================================================
# MARGINAL GENERATION
# ============================================================================

def generate_gbm_marginals(N, M, S0=100, mu=0.05, sigma=0.2, T=0.25, seed=None):
    """
    Generate marginals from Geometric Brownian Motion paths.
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    x_grid = np.linspace(50, 200, M)
    marginals = np.zeros((N + 1, M))
    
    # Initial point mass at S0
    idx = np.argmin(np.abs(x_grid - S0))
    marginals[0, idx] = 1.0
    
    # Simulate GBM distribution at each time step
    n_paths = 10000
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0
    
    for t in range(N):
        Z = np.random.randn(n_paths)
        paths[:, t+1] = paths[:, t] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    
    # Build marginals via KDE
    for t in range(1, N + 1):
        prices = paths[:, t]
        prices = np.clip(prices, 50, 200)
        
        # Gaussian KDE
        bandwidth = sigma * S0 * np.sqrt(dt * t) * 0.3
        bandwidth = max(bandwidth, 1.0)
        
        for p in prices:
            marginals[t] += np.exp(-0.5 * ((x_grid - p) / bandwidth) ** 2)
        
        marginals[t] /= marginals[t].sum() + 1e-10
    
    return marginals, x_grid


def generate_merton_marginals(N, M, S0=100, mu=0.05, sigma=0.2, T=0.25,
                              lam=5, jump_mean=-0.08, jump_std=0.1, seed=None):
    """
    Generate marginals from Merton Jump-Diffusion process.
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    x_grid = np.linspace(50, 200, M)
    marginals = np.zeros((N + 1, M))
    
    # Initial
    idx = np.argmin(np.abs(x_grid - S0))
    marginals[0, idx] = 1.0
    
    # Simulate paths
    n_paths = 10000
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0
    
    for t in range(N):
        # Diffusion
        Z = np.random.randn(n_paths)
        diffusion = np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        
        # Jumps (Poisson)
        n_jumps = np.random.poisson(lam * dt, n_paths)
        jump_sizes = np.array([
            np.sum(np.random.normal(jump_mean, jump_std, nj)) if nj > 0 else 0
            for nj in n_jumps
        ])
        jump_factor = np.exp(jump_sizes)
        
        paths[:, t+1] = paths[:, t] * diffusion * jump_factor
    
    # Build marginals via KDE
    for t in range(1, N + 1):
        prices = np.clip(paths[:, t], 50, 200)
        bandwidth = max(sigma * S0 * np.sqrt(dt * t) * 0.3, 1.0)
        
        for p in prices:
            marginals[t] += np.exp(-0.5 * ((x_grid - p) / bandwidth) ** 2)
        
        marginals[t] /= marginals[t].sum() + 1e-10
    
    return marginals, x_grid


def generate_heston_marginals(N, M, S0=100, v0=0.04, mu=0.05, kappa=2, theta=0.04,
                              xi=0.3, rho=-0.7, T=0.25, seed=None):
    """
    Generate marginals from Heston stochastic volatility model.
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    x_grid = np.linspace(50, 200, M)
    marginals = np.zeros((N + 1, M))
    
    # Initial
    idx = np.argmin(np.abs(x_grid - S0))
    marginals[0, idx] = 1.0
    
    # Simulate paths
    n_paths = 10000
    S = np.ones(n_paths) * S0
    v = np.ones(n_paths) * v0
    
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0
    
    for t in range(N):
        Z1 = np.random.randn(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_paths)
        
        # Variance (CIR process with reflection)
        v = np.maximum(v, 0)
        v_next = v + kappa * (theta - v) * dt + xi * np.sqrt(v * dt) * Z2
        v_next = np.maximum(v_next, 0)
        
        # Price
        sqrt_v = np.sqrt(np.maximum(v, 0))
        S = S * np.exp((mu - 0.5 * v) * dt + sqrt_v * np.sqrt(dt) * Z1)
        
        v = v_next
        paths[:, t+1] = S
    
    # Build marginals via KDE
    for t in range(1, N + 1):
        prices = np.clip(paths[:, t], 50, 200)
        bandwidth = max(np.sqrt(v0) * S0 * np.sqrt(dt * t) * 0.3, 1.0)
        
        for p in prices:
            marginals[t] += np.exp(-0.5 * ((x_grid - p) / bandwidth) ** 2)
        
        marginals[t] /= marginals[t].sum() + 1e-10
    
    return marginals, x_grid


# ============================================================================
# CLASSICAL SOLVER WRAPPER
# ============================================================================

def solve_instance_classical(marginals, x_grid, epsilon=1.0, max_iter=1000):
    """
    Solve a single MMOT instance using classical Martingale-Sinkhorn.
    
    Returns:
        u_star: Optimal u potentials (N+1, M)
        h_star: Optimal h potentials (N, M)
        info: Dict with convergence info
    """
    M = len(x_grid)
    dx = x_grid[:, None] - x_grid[None, :]
    C = dx ** 2  # Quadratic cost
    
    try:
        u_star, h_star, k = solve_mmot(
            jnp.array(marginals),
            jnp.array(C),
            jnp.array(x_grid),
            max_iter=max_iter,
            epsilon=epsilon,
            damping=0.8
        )
        
        u_star = np.array(u_star)
        h_star = np.array(h_star)
        iterations = int(k)
        
        # Compute drift
        drift = np.mean(np.abs(h_star))
        
        return u_star, h_star, {
            'converged': True,
            'drift': drift,
            'iterations': iterations
        }
    except Exception as e:
        return None, None, {'converged': False, 'error': str(e)}


# ============================================================================
# PARALLEL WORKER
# ============================================================================

def generate_and_solve_one(args):
    """
    Worker function: generate one instance and solve it.
    """
    idx, model_type, params, epsilon, max_iter = args
    
    # Random parameters for diversity
    np.random.seed(idx)
    
    # Vary N (time steps)
    N = np.random.choice([2, 3, 5, 10, 20, 30])
    M = 150  # Fixed grid size
    
    # Vary other parameters
    sigma = np.random.uniform(0.15, 0.35)
    T = np.random.uniform(0.1, 0.5)
    
    # Generate marginals based on model type
    if model_type == 'gbm':
        marginals, x_grid = generate_gbm_marginals(
            N=N, M=M, sigma=sigma, T=T, seed=idx
        )
    elif model_type == 'merton':
        lam = np.random.uniform(3, 10)
        jump_mean = np.random.uniform(-0.15, -0.05)
        marginals, x_grid = generate_merton_marginals(
            N=N, M=M, sigma=sigma, T=T, lam=lam, jump_mean=jump_mean, seed=idx
        )
    elif model_type == 'heston':
        kappa = np.random.uniform(1, 3)
        theta = np.random.uniform(0.03, 0.05)
        xi = np.random.uniform(0.2, 0.4)
        marginals, x_grid = generate_heston_marginals(
            N=N, M=M, kappa=kappa, theta=theta, xi=xi, T=T, seed=idx
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Solve classically
    u_star, h_star, info = solve_instance_classical(
        marginals, x_grid, epsilon=epsilon, max_iter=max_iter
    )
    
    if info['converged']:
        return {
            'idx': idx,
            'model': model_type,
            'N': N,
            'M': M,
            'marginals': marginals,
            'x_grid': x_grid,
            'u_classical': u_star,
            'h_classical': h_star,
            'drift': info['drift'],
            'converged': True
        }
    else:
        return {'idx': idx, 'converged': False, 'error': info.get('error', 'Unknown')}


# ============================================================================
# MAIN GENERATOR
# ============================================================================

def generate_teacher_dataset(n_instances, output_path, n_workers=4, epsilon=1.0):
    """
    Generate full teacher dataset with parallel processing.
    """
    print("="*70)
    print("TEACHER DATA GENERATION")
    print("="*70)
    print(f"Target: {n_instances} instances")
    print(f"Workers: {n_workers}")
    print(f"Output: {output_path}")
    print("="*70)
    
    # Split instances across model types
    n_gbm = n_instances // 3
    n_merton = n_instances // 3
    n_heston = n_instances - n_gbm - n_merton
    
    # Create task list
    tasks = []
    idx = 0
    
    for _ in range(n_gbm):
        tasks.append((idx, 'gbm', {}, epsilon, 1000))
        idx += 1
    
    for _ in range(n_merton):
        tasks.append((idx, 'merton', {}, epsilon, 1000))
        idx += 1
    
    for _ in range(n_heston):
        tasks.append((idx, 'heston', {}, epsilon, 1000))
        idx += 1
    
    print(f"\nModel distribution:")
    print(f"  GBM: {n_gbm}")
    print(f"  Merton: {n_merton}")
    print(f"  Heston: {n_heston}")
    
    # Parallel execution
    results = []
    failed = 0
    
    print(f"\nStarting processing (sequential for JAX compatibility)...")
    
    # Sequential execution (JAX doesn't work well with multiprocessing)
    for task in tqdm(tasks, desc="Generating"):
        result = generate_and_solve_one(task)
        if result['converged']:
            results.append(result)
        else:
            failed += 1
    
    print(f"\nCompleted: {len(results)} / {n_instances}")
    print(f"Failed: {failed}")
    
    if not results:
        print("ERROR: No instances generated!")
        return
    
    # Organize by N (time steps) for batching
    # Group instances with same N together
    results_by_n = {}
    for r in results:
        n = r['N']
        if n not in results_by_n:
            results_by_n[n] = []
        results_by_n[n].append(r)
    
    print(f"\nInstances by N:")
    for n, items in sorted(results_by_n.items()):
        print(f"  N={n}: {len(items)} instances")
    
    # Save to disk
    # We'll save as multiple arrays, grouped by N
    save_dict = {
        'n_instances': len(results),
        'epsilon': epsilon,
        'models': [r['model'] for r in results],
        'N_values': [r['N'] for r in results],
        'x_grid': results[0]['x_grid'],  # Same for all
    }
    
    # For variable-N data, we pad to max N
    max_N = max(r['N'] for r in results)
    M = results[0]['M']
    
    marginals_all = []
    u_all = []
    h_all = []
    
    for r in results:
        N = r['N']
        # Pad marginals to max_N + 1
        marg_padded = np.zeros((max_N + 1, M))
        marg_padded[:N+1, :] = r['marginals']
        marginals_all.append(marg_padded)
        
        # Pad u to max_N + 1
        u_padded = np.zeros((max_N + 1, M))
        u_padded[:N+1, :] = r['u_classical']
        u_all.append(u_padded)
        
        # Pad h to max_N
        h_padded = np.zeros((max_N, M))
        h_padded[:N, :] = r['h_classical']
        h_all.append(h_padded)
    
    save_dict['marginals'] = np.stack(marginals_all)
    save_dict['u_classical'] = np.stack(u_all)
    save_dict['h_classical'] = np.stack(h_all)
    save_dict['drifts'] = np.array([r['drift'] for r in results])
    
    np.savez_compressed(output_path, **save_dict)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Saved to {output_path} ({file_size:.1f} MB)")
    print(f"   Shape: marginals {save_dict['marginals'].shape}")
    print(f"   Mean drift: {np.mean(save_dict['drifts']):.6f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate MMOT teacher dataset')
    parser.add_argument('--n_instances', type=int, default=12000,
                        help='Number of instances to generate')
    parser.add_argument('--output', type=str, default='teacher_data.npz',
                        help='Output file path')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Regularization parameter')
    
    args = parser.parse_args()
    
    generate_teacher_dataset(
        n_instances=args.n_instances,
        output_path=args.output,
        n_workers=args.workers,
        epsilon=args.epsilon
    )


if __name__ == "__main__":
    main()
