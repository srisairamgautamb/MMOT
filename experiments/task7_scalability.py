#!/usr/bin/env python3
"""
MMOT Scalability Analysis (Task 7)
Goal: Show runtime scales as O(NM²)

Generates: Figure 11 (Scalability Analysis)
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import time
import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from mmot.core.solver_admm import solve_mmot_admm

# ============================================================================
# BENCHMARK FUNCTION
# ============================================================================

def time_solver(N, M, n_trials=3):
    """Time the solver for given N (periods) and M (grid size)."""
    
    # Generate simple marginals
    x_grid = jnp.linspace(0.8, 1.2, M)
    
    marginals = []
    for n in range(N + 1):
        sigma = 0.03 * jnp.sqrt(1 + n * 0.2)
        pdf = jnp.exp(-0.5 * ((x_grid - 1.0) / sigma)**2)
        pdf = pdf / jnp.sum(pdf)
        marginals.append(pdf)
    
    C = (x_grid[:, None] - x_grid[None, :])**2
    
    # Warm-up run (JAX JIT compilation)
    _ = solve_mmot_admm(jnp.stack([marginals[0], marginals[1]]), C, x_grid, 
                        epsilon=0.1, max_iter=50)
    
    # Timed runs
    times = []
    for _ in range(n_trials):
        start = time.time()
        
        for i in range(N):
            _ = solve_mmot_admm(
                jnp.stack([marginals[i], marginals[i+1]]), 
                C, x_grid, 
                epsilon=0.1, 
                max_iter=100
            )
        
        elapsed = time.time() - start
        times.append(elapsed)
    
    return np.median(times)


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🚀 MMOT SCALABILITY ANALYSIS (Task 7)")
    print("="*80)
    print()
    
    # Problem sizes
    N_values = [2, 5, 10, 20]
    M_values = [25, 50, 100, 200]
    
    print(f"Testing N = {N_values}")
    print(f"Testing M = {M_values}")
    print()
    
    # Run benchmarks
    results = []
    
    for M in M_values:
        print(f"\nM = {M}:")
        for N in N_values:
            runtime = time_solver(N, M, n_trials=2)
            results.append((N, M, runtime))
            print(f"  N = {N:3d}: {runtime:.3f}s")
    
    # Convert to arrays
    results = np.array(results)
    N_arr = results[:, 0]
    M_arr = results[:, 1]
    T_arr = results[:, 2]
    
    # ========================================================================
    # VERIFY O(NM²) COMPLEXITY
    # ========================================================================
    print("\n" + "="*80)
    print("📊 COMPLEXITY ANALYSIS")
    print("="*80)
    
    # Fit: log(T) = a * log(N) + b * log(M) + c
    log_N = np.log(N_arr)
    log_M = np.log(M_arr)
    log_T = np.log(T_arr)
    
    # Design matrix for linear regression
    X = np.column_stack([np.ones(len(log_T)), log_N, log_M])
    coeffs, residuals, rank, s = np.linalg.lstsq(X, log_T, rcond=None)
    
    c0, alpha, beta = coeffs
    
    print(f"\nFitted model: T ∝ N^{alpha:.2f} × M^{beta:.2f}")
    print(f"Expected:     T ∝ N^1.00 × M^2.00")
    print()
    
    if abs(alpha - 1.0) < 0.3 and abs(beta - 2.0) < 0.5:
        print("✅ Complexity matches O(NM²) as expected!")
    else:
        print(f"⚠️ Complexity slightly differs: O(N^{alpha:.1f} M^{beta:.1f})")
    
    # ========================================================================
    # GENERATE FIGURE 11
    # ========================================================================
    print("\n" + "="*80)
    print("📊 GENERATING FIGURE 11")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Runtime vs M (for different N)
    ax1 = axes[0]
    for N in N_values:
        mask = (N_arr == N)
        ax1.loglog(M_arr[mask], T_arr[mask], 'o-', linewidth=2, markersize=8, 
                   label=f'N = {N}')
    
    # Add theoretical O(M²) line
    M_theory = np.array([25, 200])
    T_theory = (M_theory / 50)**2 * 0.1  # Scaled reference
    ax1.loglog(M_theory, T_theory, 'k--', alpha=0.5, label='O(M²) reference')
    
    ax1.set_xlabel('Grid Size (M)', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Runtime vs Grid Size', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Runtime vs N (for different M)
    ax2 = axes[1]
    for M in M_values:
        mask = (M_arr == M)
        ax2.loglog(N_arr[mask], T_arr[mask], 's-', linewidth=2, markersize=8,
                   label=f'M = {M}')
    
    # Add theoretical O(N) line
    N_theory = np.array([2, 20])
    T_theory = (N_theory / 5) * 0.1  # Scaled reference
    ax2.loglog(N_theory, T_theory, 'k--', alpha=0.5, label='O(N) reference')
    
    ax2.set_xlabel('Number of Periods (N)', fontsize=12)
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_title('Runtime vs Number of Periods', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'MMOT Solver Scalability: O(N^{alpha:.2f} M^{beta:.2f})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('figures/phase2b', exist_ok=True)
    plt.savefig('figures/phase2b/figure11_scalability.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/phase2b/figure11_scalability.pdf', bbox_inches='tight')
    print("\n✅ Saved: figures/phase2b/figure11_scalability.png")
    print("✅ Saved: figures/phase2b/figure11_scalability.pdf")
    
    plt.show()
    
    # ========================================================================
    # GENERATE TABLE DATA
    # ========================================================================
    print("\n" + "="*80)
    print("📊 SCALABILITY TABLE")
    print("="*80)
    print()
    print("| N  | M   | Runtime (s) | Complexity Factor |")
    print("|----|-----|-------------|-------------------|")
    
    base_time = results[0, 2]  # N=2, M=25 as baseline
    for N, M, T in results:
        factor = T / base_time
        theoretical = (N / 2) * (M / 25)**2
        print(f"| {int(N):2d} | {int(M):3d} | {T:11.3f} | {factor:17.1f} |")
    
    print()
    print("="*80)
    print("✅ TASK 7 COMPLETE: Figure 11 Generated")
    print("="*80)
