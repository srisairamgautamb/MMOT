#!/usr/bin/env python3
"""
MMOT Robustness Test (Task 5)
Goal: Show solution is stable to noise in marginals (Theorem 6.1)

Generates: Figure 12 (Robustness to Noise)
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
# NOISE FUNCTIONS
# ============================================================================

def add_noise(marginal, noise_level, seed=None):
    """Add Gaussian noise to marginal, keeping it valid (non-negative, normalized)."""
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.randn(len(marginal)) * noise_level
    noisy = np.array(marginal) + noise
    noisy = np.maximum(noisy, 1e-10)  # Keep non-negative
    noisy = noisy / noisy.sum()  # Renormalize
    return jnp.array(noisy)


def solution_distance(P1, P2):
    """Compute distance between two transport plans."""
    return float(jnp.sum(jnp.abs(P1 - P2)))


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🛡️ ROBUSTNESS TEST (Task 5 - Theorem 6.1)")
    print("="*80)
    print()
    
    # Setup
    M = 50
    x_grid = jnp.linspace(0.8, 1.2, M)
    C = (x_grid[:, None] - x_grid[None, :])**2
    
    # Clean marginals
    mu_clean = jnp.exp(-0.5 * ((x_grid - 1.0) / 0.03)**2)
    mu_clean = mu_clean / jnp.sum(mu_clean)
    
    nu_clean = jnp.exp(-0.5 * ((x_grid - 1.0) / 0.04)**2)
    nu_clean = nu_clean / jnp.sum(nu_clean)
    
    # Solve clean problem
    print("Computing clean solution...")
    result_clean = solve_mmot_admm(jnp.stack([mu_clean, nu_clean]), C, x_grid)
    P_clean = result_clean['P']
    
    # Noise levels
    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    n_trials = 5
    
    results = []
    
    print("\nTesting noise levels...")
    print("-" * 60)
    
    for eps in noise_levels:
        distances = []
        
        for seed in range(n_trials):
            # Add noise to both marginals
            mu_noisy = add_noise(mu_clean, eps, seed=seed)
            nu_noisy = add_noise(nu_clean, eps, seed=seed + 1000)
            
            # Solve noisy problem
            result_noisy = solve_mmot_admm(jnp.stack([mu_noisy, nu_noisy]), C, x_grid)
            P_noisy = result_noisy['P']
            
            # Compute distance
            dist = solution_distance(P_clean, P_noisy)
            distances.append(dist)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        results.append((eps, mean_dist, std_dist))
        
        print(f"  Noise = {eps:.3f}: Distance = {mean_dist:.4f} ± {std_dist:.4f}")
    
    results = np.array(results)
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("📊 STABILITY ANALYSIS")
    print("="*80)
    
    # Fit linear relationship: distance ≈ C * noise
    coeffs = np.polyfit(results[:, 0], results[:, 1], 1)
    stability_constant = coeffs[0]
    
    print(f"\nStability constant: C = {stability_constant:.2f}")
    print(f"Interpretation: solution changes by ~{stability_constant:.1f}× the noise level")
    print()
    
    if stability_constant < 5:
        print("✅ EXCELLENT stability (C < 5)")
    elif stability_constant < 10:
        print("✅ GOOD stability (C < 10)")
    else:
        print("⚠️ Moderate stability")
    
    # ========================================================================
    # GENERATE FIGURE 12
    # ========================================================================
    print("\n" + "="*80)
    print("📊 GENERATING FIGURE 12")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot with error bars
    ax.errorbar(results[:, 0], results[:, 1], yerr=results[:, 2],
                fmt='bo-', linewidth=2, markersize=10, capsize=5, capthick=2,
                label='Measured stability')
    
    # Theoretical linear bound
    x_theory = np.linspace(0, max(noise_levels) * 1.1, 100)
    y_theory = stability_constant * x_theory
    ax.plot(x_theory, y_theory, 'r--', linewidth=2, alpha=0.7,
            label=f'Fitted: {stability_constant:.1f}ε')
    
    ax.set_xlabel('Noise Level (ε)', fontsize=14)
    ax.set_ylabel('Solution Distance ||P - P*||₁', fontsize=14)
    ax.set_title(f'Robustness to Marginal Noise (Theorem 6.1)\nStability Constant C = {stability_constant:.1f}',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(noise_levels) * 1.1)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    
    os.makedirs('figures/phase2b', exist_ok=True)
    plt.savefig('figures/phase2b/figure12_robustness.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/phase2b/figure12_robustness.pdf', bbox_inches='tight')
    
    print("\n✅ Saved: figures/phase2b/figure12_robustness.png")
    print("✅ Saved: figures/phase2b/figure12_robustness.pdf")
    
    print("\n" + "="*80)
    print("✅ TASK 5 COMPLETE: Figure 12 Generated")
    print("="*80)
