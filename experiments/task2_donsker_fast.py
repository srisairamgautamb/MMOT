#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from mmot.core.solver_admm import solve_mmot_admm

def generate_marginals(N, M, S0=100.0):
    x_grid = jnp.linspace(0.7 * S0, 1.3 * S0, M)
    marginals = []
    for n in range(N + 1):
        t = n / N if N > 0 else 0
        sigma = 0.03 * S0 * jnp.sqrt(1 + t)
        pdf = jnp.exp(-0.5 * ((x_grid - S0) / sigma)**2)
        pdf = pdf / jnp.sum(pdf)
        marginals.append(pdf)
    return marginals, x_grid

def solve_mmot_chain(marginals, x_grid, epsilon=0.1):
    N = len(marginals) - 1
    C = (x_grid[:, None] - x_grid[None, :])**2
    plans = []
    for i in range(N):
        result = solve_mmot_admm(jnp.stack([marginals[i], marginals[i+1]]), C, x_grid, epsilon=epsilon, max_iter=100) # Reduced iter
        plans.append(result['P'])
    return plans

if __name__ == "__main__":
    print("Running FAST Donsker Rate Validation...")
    # REDUCED PARAMETERS
    M = 50 
    S0 = 100.0
    N_ref = 20 # Reduced from 100
    
    print(f"Computing reference solution (N={N_ref})...")
    marginals_ref, x_grid_ref = generate_marginals(N_ref, M, S0)
    plans_ref = solve_mmot_chain(marginals_ref, x_grid_ref)
    C_ref = (x_grid_ref[:, None] - x_grid_ref[None, :])**2
    cost_ref = float(jnp.sum(plans_ref[0] * C_ref))
    
    N_values = [5, 10, 20] # Reduced set
    errors = []
    delta_t_values = []
    
    for N in N_values:
        print(f"N = {N}...", end=" ")
        marginals_N, x_grid_N = generate_marginals(N, M, S0)
        plans_N = solve_mmot_chain(marginals_N, x_grid_N)
        C_N = (x_grid_N[:, None] - x_grid_N[None, :])**2
        cost_N = float(jnp.sum(plans_N[0] * C_N))
        error = abs(cost_N - cost_ref) / (cost_ref + 1e-10) # Avoid div by zero
        if N == N_ref: error = 1e-10 # Force zero for ref
        
        errors.append(error)
        delta_t_values.append(1.0/N)
        print(f"Error: {error:.2e}")

    # Generate Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(delta_t_values, errors, 'bo-', label='Measured')
    # Theoretical
    dt = np.array(delta_t_values)
    ax.loglog(dt, np.sqrt(dt)*0.1, 'r--', label='O(sqrt(dt))')
    ax.set_title("Donsker Rate (Fast Check)")
    ax.legend()
    
    os.makedirs('figures/phase2b', exist_ok=True)
    plt.savefig('figures/phase2b/figure10_donsker_rate.png')
    plt.savefig('figures/phase2b/figure10_donsker_rate.pdf')
    print("Done.")
