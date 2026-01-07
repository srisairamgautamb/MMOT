#!/usr/bin/env python3
"""
debug_drift_computation.py
==========================
Debug why drift = 0 for all instances (user correctly flagged this as suspicious).

Possibilities:
1. Drift computation is wrong (returning 0 incorrectly)
2. Numerical precision issue
3. Actual convergence to machine precision

Let's investigate!
"""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')

from solve_mmot_adaptive import solve_mmot_adaptive


def manual_drift_check(u, h, marginals, grid, epsilon=0.2):
    """
    Manually compute drift to verify solver output.
    
    Drift = max_{x,t} |E[Y|X=x] - x|
    
    where E[Y|X=x] = sum_y y * P(Y=y|X=x)
    and P(Y=y|X=x) = exp((u_t(x) + u_{t+1}(y) + h_t(x)*(y-x) - C(x,y))/epsilon) / Z
    """
    N = h.shape[0]
    M = len(grid)
    
    grid = np.array(grid, dtype=np.float64)  # Use float64 for accuracy
    u = np.array(u, dtype=np.float64)
    h = np.array(h, dtype=np.float64)
    
    # Cost matrix
    Delta = grid[:, None] - grid[None, :]  # (M, M)
    C = Delta ** 2
    C_scaled = C / np.max(C)
    
    max_drift = 0.0
    drift_per_step = []
    
    print(f"\n  Manual Drift Check (epsilon={epsilon}):")
    print(f"  {'Step':<6} {'Max |E[Y|X]-X|':<20} {'Mean |E[Y|X]-X|':<20}")
    print(f"  {'-'*50}")
    
    for t in range(N):
        # Log-kernel
        u_t = u[t]
        u_next = u[t+1]
        h_t = h[t]
        
        term_u = u_t[:, None] + u_next[None, :]  # (M, M)
        term_h = h_t[:, None] * Delta
        LogK = (term_u + term_h - C_scaled) / epsilon
        
        # Softmax for P(Y|X)
        LogK_stable = LogK - np.max(LogK, axis=1, keepdims=True)
        probs = np.exp(LogK_stable) / np.sum(np.exp(LogK_stable), axis=1, keepdims=True)
        
        # E[Y|X=x]
        expected_Y = np.sum(probs * grid[None, :], axis=1)
        
        # Drift = E[Y|X] - X
        drift = np.abs(expected_Y - grid)
        
        max_drift_t = np.max(drift)
        mean_drift_t = np.mean(drift)
        drift_per_step.append(max_drift_t)
        
        print(f"  t={t:<4} {max_drift_t:<20.10f} {mean_drift_t:<20.10f}")
        
        if max_drift_t > max_drift:
            max_drift = max_drift_t
    
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<6} {max_drift:<20.10f}")
    
    return max_drift, drift_per_step


def test_with_manual_verification():
    """Test solver and verify drift manually."""
    print("\n" + "="*70)
    print("DEBUGGING DRIFT COMPUTATION")
    print("="*70)
    
    M = 150
    N = 3
    grid = np.linspace(0.5, 1.5, M).astype(np.float32)
    
    # Generate GBM marginals
    sigma = 0.25
    T = 0.25
    dt = T / N
    
    marginals = np.zeros((N+1, M), dtype=np.float32)
    for t in range(N+1):
        if t == 0:
            center_idx = np.argmin(np.abs(grid - 1.0))
            marginals[t, center_idx] = 1.0
        else:
            tau = t * dt
            log_std = sigma * np.sqrt(tau)
            log_m = np.log(grid)
            pdf = np.exp(-0.5 * (log_m / log_std)**2) / (grid * log_std * np.sqrt(2*np.pi))
            marginals[t] = pdf / pdf.sum()
    
    # Solve
    print("\n  Running adaptive solver...")
    result = solve_mmot_adaptive(marginals, grid, target_drift=1e-4, verbose=False)
    
    print(f"\n  Solver reported drift: {result['drift']:.10f}")
    
    # Manual verification with SAME epsilon used in final stage
    final_epsilon = result['stage_info'][-1]['epsilon']
    print(f"\n  Final stage epsilon: {final_epsilon}")
    
    max_drift, drift_per_step = manual_drift_check(
        result['u'], result['h'], marginals, grid, epsilon=final_epsilon
    )
    
    print(f"\n  Comparison:")
    print(f"    Solver reported: {result['drift']:.10f}")
    print(f"    Manual check:    {max_drift:.10f}")
    
    if abs(result['drift'] - max_drift) < 1e-6:
        print(f"\n  ✅ Drift computation is CORRECT (values match)")
    else:
        print(f"\n  ⚠️ MISMATCH! Drift computation may have bug")
    
    # Also check with epsilon=0.2 (what we use for production)
    print(f"\n  Also checking with production epsilon=0.2:")
    max_drift_02, _ = manual_drift_check(
        result['u'], result['h'], marginals, grid, epsilon=0.2
    )
    
    print("="*70)
    
    return result, max_drift


if __name__ == '__main__':
    test_with_manual_verification()
