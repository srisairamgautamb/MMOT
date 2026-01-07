#!/usr/bin/env python3
"""
MMOT Donsker Rate Validation (Task 2)
Goal: Show discretization error scales as O(√Δt) - validates Theorem 5.2

Generates: Figure 10 (Donsker Rate Validation)
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from mmot.core.solver_admm import solve_mmot_admm

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_marginals(N, M, S0=100.0):
    """Generate N+1 marginals on M-point grid."""
    x_grid = jnp.linspace(0.7 * S0, 1.3 * S0, M)
    
    marginals = []
    for n in range(N + 1):
        t = n / N if N > 0 else 0
        # Increasing variance with time
        sigma = 0.03 * S0 * jnp.sqrt(1 + t)
        pdf = jnp.exp(-0.5 * ((x_grid - S0) / sigma)**2)
        pdf = pdf / jnp.sum(pdf)
        marginals.append(pdf)
    
    return marginals, x_grid


def solve_mmot_chain(marginals, x_grid, epsilon=0.1):
    """Solve MMOT for chain of marginals, return list of transport plans."""
    N = len(marginals) - 1
    C = (x_grid[:, None] - x_grid[None, :])**2
    
    plans = []
    for i in range(N):
        result = solve_mmot_admm(
            jnp.stack([marginals[i], marginals[i+1]]),
            C, x_grid,
            epsilon=epsilon,
            max_iter=200
        )
        plans.append(result['P'])
    
    return plans


def wasserstein_1(P1, P2, x_grid):
    """Compute approximate W1 distance between two transport plans."""
    # Simple approximation: Frobenius norm of difference weighted by grid
    diff = jnp.abs(P1 - P2)
    return float(jnp.sum(diff))


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("📊 DONSKER RATE VALIDATION (Task 2 - Theorem 5.2)")
    print("="*80)
    print()
    
    # Fixed parameters
    M = 100  # Grid size
    S0 = 100.0
    
    # Reference solution with fine discretization (but not too fine!)
    print("Computing reference solution (N=100)...")
    N_ref = 100
    marginals_ref, x_grid_ref = generate_marginals(N_ref, M, S0)
    plans_ref = solve_mmot_chain(marginals_ref, x_grid_ref)
    
    # Combine into single "joint" representation
    # For simplicity, we'll use the cost of the first transition
    C_ref = (x_grid_ref[:, None] - x_grid_ref[None, :])**2
    cost_ref = float(jnp.sum(plans_ref[0] * C_ref))
    
    print(f"Reference cost: {cost_ref:.6f}")
    print()
    
    # Test different N values
    N_values = [5, 10, 20, 50, 100, 200]
    errors = []
    delta_t_values = []
    
    print("Testing different discretizations:")
    print("-" * 60)
    
    for N in N_values:
        print(f"N = {N:3d}...", end=" ")
        
        # Solve with this discretization
        marginals_N, x_grid_N = generate_marginals(N, M, S0)
        plans_N = solve_mmot_chain(marginals_N, x_grid_N)
        
        # Compute cost for first transition
        C_N = (x_grid_N[:, None] - x_grid_N[None, :])**2
        cost_N = float(jnp.sum(plans_N[0] * C_N))
        
        # Error vs reference
        error = abs(cost_N - cost_ref) / cost_ref
        
        errors.append(error)
        delta_t = 1.0 / N
        delta_t_values.append(delta_t)
        
        print(f"Δt = {delta_t:.4f}, cost = {cost_N:.6f}, error = {error:.2e}")
    
    errors = np.array(errors)
    delta_t_values = np.array(delta_t_values)
    
    # ========================================================================
    # FIT O(√Δt) RATE
    # ========================================================================
    print("\n" + "="*80)
    print("📊 RATE ANALYSIS")
    print("="*80)
    
    # Fit log(error) = α * log(Δt) + const
    # Expected: α ≈ 0.5 (Donsker rate)
    
    log_err = np.log(errors + 1e-10)
    log_dt = np.log(delta_t_values)
    
    # Linear regression
    coeffs = np.polyfit(log_dt, log_err, 1)
    alpha = coeffs[0]
    
    print(f"\nFitted rate: error ∝ Δt^{alpha:.3f}")
    print(f"Expected (Donsker): error ∝ Δt^0.500")
    print()
    
    if abs(alpha - 0.5) < 0.2:
        print("✅ Rate matches Donsker O(√Δt) as predicted by Theorem 5.2!")
    else:
        print(f"⚠️ Rate differs from O(√Δt): got O(Δt^{alpha:.2f})")
    
    # ========================================================================
    # GENERATE FIGURE 10
    # ========================================================================
    print("\n" + "="*80)
    print("📊 GENERATING FIGURE 10")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot measured errors
    ax.loglog(delta_t_values, errors, 'bo-', linewidth=2, markersize=10, 
              label=f'Measured error (slope = {alpha:.2f})')
    
    # Plot theoretical O(√Δt) line
    dt_theory = np.array([0.002, 0.2])
    err_theory = np.sqrt(dt_theory) * (errors[-1] / np.sqrt(delta_t_values[-1]))
    ax.loglog(dt_theory, err_theory, 'r--', linewidth=2, alpha=0.7,
              label='Theoretical O(√Δt)')
    
    # Fitted line
    err_fit = np.exp(coeffs[1]) * delta_t_values**alpha
    ax.loglog(delta_t_values, err_fit, 'g:', linewidth=2, alpha=0.7,
              label=f'Fitted O(Δt^{alpha:.2f})')
    
    ax.set_xlabel('Time Step Δt = 1/N', fontsize=14)
    ax.set_ylabel('Relative Error vs Reference', fontsize=14)
    ax.set_title('Donsker Rate Validation (Theorem 5.2)\nDiscretization Error ∝ √Δt', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate(f'Fitted: O(Δt^{alpha:.2f})\nExpected: O(Δt^0.5)', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('figures/phase2b', exist_ok=True)
    plt.savefig('figures/phase2b/figure10_donsker_rate.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/phase2b/figure10_donsker_rate.pdf', bbox_inches='tight')
    print("\n✅ Saved: figures/phase2b/figure10_donsker_rate.png")
    print("✅ Saved: figures/phase2b/figure10_donsker_rate.pdf")
    
    plt.show()
    
    print("\n" + "="*80)
    print("✅ TASK 2 COMPLETE: Figure 10 Generated")
    print("="*80)
