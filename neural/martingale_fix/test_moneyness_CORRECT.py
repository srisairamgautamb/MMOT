#!/usr/bin/env python3
"""
CORRECT MONEYNESS-BASED DATA GENERATOR
======================================
Uses TRUE MONEYNESS grid [0.5, 1.5] (dimensionless ratios).

Moneyness = Strike / Spot Price
- 0.5 = 50% of spot (deep OTM)
- 1.0 = 100% of spot (ATM)  
- 1.5 = 150% of spot (deep ITM)

This works for ANY stock because moneyness is dimensionless!
"""

import numpy as np
import jax.numpy as jnp
import sys
import os

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT')
from mmot.core.solver import solve_mmot


def test_single_instance():
    """Generate and solve ONE instance to verify everything works."""
    
    print("="*60)
    print("TESTING CORRECT MONEYNESS GRID [0.5, 1.5]")
    print("="*60)
    
    # =================================================================
    # CORRECT: TRUE MONEYNESS GRID [0.5, 1.5]
    # =================================================================
    M = 150
    moneyness_grid = np.linspace(0.5, 1.5, M)
    dm = moneyness_grid[1] - moneyness_grid[0]
    
    print(f"\nGrid:")
    print(f"  Range: [{moneyness_grid[0]:.2f}, {moneyness_grid[-1]:.2f}]")
    print(f"  Spacing: {dm:.6f}")
    print(f"  Points: {M}")
    
    # =================================================================
    # MARGINALS: Log-normal in moneyness
    # =================================================================
    N = 5  # Time steps
    T = 0.25  # 3 months
    sigma = 0.25  # 25% volatility
    r = 0.05  # 5% risk-free rate
    
    dt = T / N
    
    marginals = np.zeros((N + 1, M))
    
    # t=0: Delta at m=1.0 (ATM)
    atm_idx = np.argmin(np.abs(moneyness_grid - 1.0))
    marginals[0, atm_idx] = 1.0
    
    print(f"\nMarginal parameters:")
    print(f"  N = {N}, T = {T}, sigma = {sigma}, r = {r}")
    print(f"  ATM index: {atm_idx} (m = {moneyness_grid[atm_idx]:.4f})")
    
    # Generate log-normal density for each time step
    for t in range(1, N + 1):
        time_t = t * dt
        
        # Under risk-neutral measure: log(M_t) ~ N((r - σ²/2)t, σ²t)
        mu = (r - 0.5 * sigma**2) * time_t
        var = sigma**2 * time_t
        std = np.sqrt(var)
        
        # Log-normal density
        log_m = np.log(moneyness_grid)
        density = (1.0 / (moneyness_grid * std * np.sqrt(2*np.pi))) * \
                  np.exp(-0.5 * ((log_m - mu) / std)**2)
        
        # Normalize
        density = density / (density.sum() * dm)
        marginals[t] = density
    
    print(f"\nMarginal sums:")
    for t in range(N + 1):
        total = marginals[t].sum() * dm
        print(f"  t={t}: sum = {total:.6f}, max = {marginals[t].max():.4f}")
    
    # =================================================================
    # COST MATRIX: Quadratic in moneyness
    # =================================================================
    dm_matrix = moneyness_grid[:, None] - moneyness_grid[None, :]
    C = dm_matrix ** 2
    
    print(f"\nCost matrix C = (m_i - m_j)²:")
    print(f"  Min: {C.min():.6f}")
    print(f"  Max: {C.max():.6f}")
    print(f"  Mean: {C.mean():.6f}")
    
    # =================================================================
    # SOLVE: Test different epsilon values
    # =================================================================
    print(f"\nSolving MMOT with different epsilon:")
    
    # For small cost range [0, 1], we need appropriate epsilon
    for eps in [0.001, 0.005, 0.01, 0.05, 0.1]:
        try:
            u, h, k = solve_mmot(
                jnp.array(marginals),
                jnp.array(C),
                jnp.array(moneyness_grid),
                max_iter=1000,
                epsilon=eps,
                damping=0.8
            )
            
            u_np = np.array(u)
            h_np = np.array(h)
            k = int(k)
            
            nan_u = np.isnan(u_np).sum()
            nan_h = np.isnan(h_np).sum()
            
            if nan_h > 0:
                print(f"  eps={eps:.4f}: ❌ NaN (u:{nan_u}, h:{nan_h})")
            else:
                mean_u = np.mean(np.abs(u_np))
                mean_h = np.mean(np.abs(h_np))
                
                # Compute ACTUAL drift (martingale violation)
                total_drift = 0.0
                for t in range(N):
                    # E[M_{t+1} | M_t = m] should = m for martingale
                    # With entropic transport: K(m, m') ∝ exp((u_{t+1}(m') + h_t(m)·(m'-m) - C(m,m'))/ε)
                    for m_idx in range(M):
                        m = moneyness_grid[m_idx]
                        
                        # Compute conditional expectation
                        log_K = (u_np[t+1] + h_np[t, m_idx] * (moneyness_grid - m) - C[m_idx]) / eps
                        log_K = log_K - log_K.max()  # Numerical stability
                        K = np.exp(log_K)
                        K = K / K.sum()
                        
                        E_M_next = np.sum(K * moneyness_grid)
                        drift_m = abs(E_M_next - m)
                        total_drift += drift_m * marginals[t, m_idx]
                
                avg_drift = total_drift / N
                
                print(f"  eps={eps:.4f}: ✅ iters={k:4d}, |u|={mean_u:.4f}, |h|={mean_h:.4f}, drift={avg_drift:.6f}")
                
        except Exception as e:
            print(f"  eps={eps:.4f}: ❌ Error: {str(e)[:50]}")
    
    return marginals, moneyness_grid


if __name__ == "__main__":
    test_single_instance()
