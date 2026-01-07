#!/usr/bin/env python3
"""
MONEYNESS-BASED TEACHER DATA GENERATOR
======================================
Generates MMOT instances in MONEYNESS coordinates for universal stock coverage.

Moneyness = Strike / Spot Price
- 0.5 = 50% out-of-the-money (OTM)
- 1.0 = at-the-money (ATM)
- 1.5 = 50% in-the-money (ITM)

This approach works for ANY stock at ANY price:
- SPY at $683
- BRK.A at $600,000
- F at $10
- Penny stocks at $0.50

Usage:
    python generate_teacher_data_moneyness.py --n_instances 12000 --output teacher_moneyness.npz
"""

import numpy as np
import jax.numpy as jnp
import sys
import os
import argparse
from tqdm import tqdm
from scipy.stats import norm, lognorm

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT')
from mmot.core.solver import solve_mmot


# ============================================================================
# MONEYNESS GRID
# ============================================================================

def get_moneyness_grid(M=150):
    """
    Scaled moneyness grid [50, 150].
    
    This is moneyness × 100:
    - 50 = 50% OTM (strike is half of spot)
    - 100 = ATM (strike equals spot)
    - 150 = 50% ITM (strike is 1.5x spot)
    
    We use [50, 150] instead of [0.5, 1.5] for numerical stability
    with the existing solver. At inference, just divide by 100.
    """
    return np.linspace(50, 150, M)


# ============================================================================
# MARGINAL GENERATION IN MONEYNESS SPACE
# ============================================================================

def generate_gbm_marginals_moneyness(N, M, r=0.05, sigma=0.2, T=0.25, seed=None):
    """
    Generate marginals from GBM in scaled moneyness coordinates [50, 150].
    
    Under risk-neutral measure, log(S_T / S_0) ~ N((r - σ²/2)T, σ²T)
    Scaled moneyness = 100 * S_T / S_0, so centered at 100.
    """
    if seed is not None:
        np.random.seed(seed)
    
    moneyness_grid = get_moneyness_grid(M)  # [50, 150]
    
    marginals = np.zeros((N + 1, M))
    
    # t=0: Point mass at scaled moneyness = 100 (ATM)
    atm_idx = np.argmin(np.abs(moneyness_grid - 100.0))
    marginals[0, atm_idx] = 1.0
    
    dt = T / N
    n_paths = 20000
    
    # Simulate log-returns
    log_S = np.zeros(n_paths)
    
    for t in range(1, N + 1):
        Z = np.random.randn(n_paths)
        log_S += (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        
        # Convert to scaled moneyness (100 * exp(log_S))
        scaled_m = 100.0 * np.exp(log_S)
        scaled_m = np.clip(scaled_m, 50, 150)
        
        # Build density via histogram
        hist, _ = np.histogram(scaled_m, bins=M, range=(50, 150), density=True)
        marginals[t] = hist * 100 / M  # Normalize
        marginals[t] /= marginals[t].sum() + 1e-10
    
    return marginals, moneyness_grid


def generate_merton_marginals_moneyness(N, M, r=0.05, sigma=0.2, T=0.25,
                                        lam=5, jump_mean=-0.08, jump_std=0.1, seed=None):
    """
    Generate marginals from Merton jump-diffusion in scaled moneyness [50, 150].
    """
    if seed is not None:
        np.random.seed(seed)
    
    moneyness_grid = get_moneyness_grid(M)
    marginals = np.zeros((N + 1, M))
    
    # t=0: Point mass at ATM = 100
    atm_idx = np.argmin(np.abs(moneyness_grid - 100.0))
    marginals[0, atm_idx] = 1.0
    
    dt = T / N
    n_paths = 20000
    
    # Simulate paths
    log_S = np.zeros(n_paths)
    
    for t in range(1, N + 1):
        # Diffusion
        Z = np.random.randn(n_paths)
        log_S += (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        
        # Jumps
        n_jumps = np.random.poisson(lam * dt, n_paths)
        for i in range(n_paths):
            if n_jumps[i] > 0:
                log_S[i] += np.sum(np.random.normal(jump_mean, jump_std, n_jumps[i]))
        
        # Convert to scaled moneyness
        scaled_m = 100.0 * np.exp(log_S)
        scaled_m = np.clip(scaled_m, 50, 150)
        
        # Build density via histogram
        hist, _ = np.histogram(scaled_m, bins=M, range=(50, 150), density=True)
        marginals[t] = hist * 100 / M
        marginals[t] /= marginals[t].sum() + 1e-10
    
    return marginals, moneyness_grid


def generate_heston_marginals_moneyness(N, M, r=0.05, v0=0.04, kappa=2, theta=0.04,
                                        xi=0.3, rho=-0.7, T=0.25, seed=None):
    """
    Generate marginals from Heston model in scaled moneyness [50, 150].
    """
    if seed is not None:
        np.random.seed(seed)
    
    moneyness_grid = get_moneyness_grid(M)
    marginals = np.zeros((N + 1, M))
    
    # t=0: Point mass at ATM = 100
    atm_idx = np.argmin(np.abs(moneyness_grid - 100.0))
    marginals[0, atm_idx] = 1.0
    
    dt = T / N
    n_paths = 20000
    
    # Simulate paths
    log_S = np.zeros(n_paths)
    v = np.ones(n_paths) * v0
    
    for t in range(1, N + 1):
        Z1 = np.random.randn(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_paths)
        
        # Variance (CIR with truncation)
        v = np.maximum(v, 0)
        sqrt_v = np.sqrt(v)
        v_next = v + kappa * (theta - v) * dt + xi * sqrt_v * np.sqrt(dt) * Z2
        v_next = np.maximum(v_next, 0)
        
        # Price
        log_S += (r - 0.5 * v) * dt + sqrt_v * np.sqrt(dt) * Z1
        v = v_next
        
        # Convert to scaled moneyness
        scaled_m = 100.0 * np.exp(log_S)
        scaled_m = np.clip(scaled_m, 50, 150)
        
        # Build density
        hist, _ = np.histogram(scaled_m, bins=M, range=(50, 150), density=True)
        marginals[t] = hist * 100 / M
        marginals[t] /= marginals[t].sum() + 1e-10
    
    return marginals, moneyness_grid


# ============================================================================
# CLASSICAL SOLVER WRAPPER
# ============================================================================

def solve_instance_classical_moneyness(marginals, moneyness_grid, epsilon=0.1, max_iter=1000):
    """
    Solve MMOT instance in moneyness space.
    
    Cost function: c(m_s, m_t) = (m_t - m_s)² (quadratic in moneyness)
    """
    M = len(moneyness_grid)
    dm = moneyness_grid[:, None] - moneyness_grid[None, :]
    C = dm ** 2  # Quadratic cost in moneyness
    
    try:
        u_star, h_star, k = solve_mmot(
            jnp.array(marginals),
            jnp.array(C),
            jnp.array(moneyness_grid),
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
# WORKER FUNCTION
# ============================================================================

def generate_and_solve_one_moneyness(args):
    """Generate one instance and solve it in moneyness space."""
    idx, model_type, epsilon, max_iter = args
    
    np.random.seed(idx)
    
    # Vary N (time steps)
    N = np.random.choice([2, 3, 5, 10, 20, 30])
    M = 150
    
    # Vary parameters
    sigma = np.random.uniform(0.15, 0.40)  # Volatility
    T = np.random.uniform(0.05, 0.5)       # Maturity (0.5 = 6 months)
    r = np.random.uniform(0.01, 0.08)      # Risk-free rate
    
    # Generate marginals
    if model_type == 'gbm':
        marginals, moneyness_grid = generate_gbm_marginals_moneyness(
            N=N, M=M, r=r, sigma=sigma, T=T, seed=idx
        )
    elif model_type == 'merton':
        lam = np.random.uniform(3, 10)
        jump_mean = np.random.uniform(-0.15, -0.05)
        marginals, moneyness_grid = generate_merton_marginals_moneyness(
            N=N, M=M, r=r, sigma=sigma, T=T, lam=lam, jump_mean=jump_mean, seed=idx
        )
    elif model_type == 'heston':
        kappa = np.random.uniform(1, 3)
        theta = np.random.uniform(0.02, 0.06)
        xi = np.random.uniform(0.2, 0.5)
        v0 = theta
        marginals, moneyness_grid = generate_heston_marginals_moneyness(
            N=N, M=M, r=r, v0=v0, kappa=kappa, theta=theta, xi=xi, T=T, seed=idx
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    # Solve
    u_star, h_star, info = solve_instance_classical_moneyness(
        marginals, moneyness_grid, epsilon=epsilon, max_iter=max_iter
    )
    
    if info['converged']:
        return {
            'idx': idx,
            'model': model_type,
            'N': N,
            'M': M,
            'sigma': sigma,
            'T': T,
            'marginals': marginals,
            'moneyness_grid': moneyness_grid,
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

def generate_teacher_dataset_moneyness(n_instances, output_path, epsilon=0.1):
    """Generate full teacher dataset in moneyness space."""
    print("="*70)
    print("MONEYNESS-BASED TEACHER DATA GENERATION")
    print("="*70)
    print(f"Target: {n_instances} instances")
    print(f"Grid: [50, 150] scaled moneyness (works for ANY stock)")
    print(f"Output: {output_path}")
    print("="*70)
    
    # Split across model types
    n_gbm = n_instances // 3
    n_merton = n_instances // 3
    n_heston = n_instances - n_gbm - n_merton
    
    # Create task list
    tasks = []
    idx = 0
    
    for _ in range(n_gbm):
        tasks.append((idx, 'gbm', epsilon, 1000))
        idx += 1
    
    for _ in range(n_merton):
        tasks.append((idx, 'merton', epsilon, 1000))
        idx += 1
    
    for _ in range(n_heston):
        tasks.append((idx, 'heston', epsilon, 1000))
        idx += 1
    
    print(f"\nModel distribution:")
    print(f"  GBM: {n_gbm}")
    print(f"  Merton: {n_merton}")
    print(f"  Heston: {n_heston}")
    
    # Sequential processing (JAX compatibility)
    results = []
    failed = 0
    
    print(f"\nProcessing (sequential for JAX)...")
    
    for task in tqdm(tasks, desc="Generating"):
        result = generate_and_solve_one_moneyness(task)
        if result['converged']:
            results.append(result)
        else:
            failed += 1
    
    print(f"\nCompleted: {len(results)} / {n_instances}")
    print(f"Failed: {failed}")
    
    if not results:
        print("ERROR: No instances generated!")
        return
    
    # Organize data
    max_N = max(r['N'] for r in results)
    M = results[0]['M']
    moneyness_grid = results[0]['moneyness_grid']
    
    marginals_all = []
    u_all = []
    h_all = []
    
    for r in results:
        N = r['N']
        
        # Pad to max_N
        marg_padded = np.zeros((max_N + 1, M))
        marg_padded[:N+1, :] = r['marginals']
        marginals_all.append(marg_padded)
        
        u_padded = np.zeros((max_N + 1, M))
        u_padded[:N+1, :] = r['u_classical']
        u_all.append(u_padded)
        
        h_padded = np.zeros((max_N, M))
        h_padded[:N, :] = r['h_classical']
        h_all.append(h_padded)
    
    save_dict = {
        'n_instances': len(results),
        'epsilon': epsilon,
        'coordinate_system': 'moneyness',
        'grid_range': [0.5, 1.5],
        'models': np.array([r['model'] for r in results]),
        'N_values': np.array([r['N'] for r in results]),
        'moneyness_grid': moneyness_grid,
        'marginals': np.stack(marginals_all),
        'u_classical': np.stack(u_all),
        'h_classical': np.stack(h_all),
        'drifts': np.array([r['drift'] for r in results])
    }
    
    np.savez_compressed(output_path, **save_dict)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Saved to {output_path} ({file_size:.1f} MB)")
    print(f"   Shape: marginals {save_dict['marginals'].shape}")
    print(f"   Mean drift: {np.mean(save_dict['drifts']):.6f}")
    print(f"   Coordinate system: SCALED MONEYNESS [50, 150]")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate MMOT teacher dataset (MONEYNESS)')
    parser.add_argument('--n_instances', type=int, default=12000)
    parser.add_argument('--output', type=str, default='teacher_moneyness.npz')
    parser.add_argument('--epsilon', type=float, default=0.1)
    
    args = parser.parse_args()
    
    generate_teacher_dataset_moneyness(
        n_instances=args.n_instances,
        output_path=args.output,
        epsilon=args.epsilon
    )


if __name__ == "__main__":
    main()
