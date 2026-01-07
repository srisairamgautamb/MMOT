# ============================================================================
# FILE: mmot/core/solver_admm.py
# PURPOSE: MMOT Solver - Robust Alternating Projections (Grid Normalized)
# ============================================================================

"""
MMOT Solver using Alternating Projections

KEY FIX: Normalizes x_grid to [0, 1] internally to prevent numerical overflow
when using S&P 500 scale prices (~5000-9000).

The cost matrix C with raw SPX prices has values up to 17 million.
With epsilon=0.1, exp(-17M/0.1) = 0, causing the solver to collapse.
By normalizing the grid, we keep costs in a manageable range.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# ============================================================================
# SINKHORN-KNOPP (Standard Implementation)
# ============================================================================

def sinkhorn_knopp(P, mu_row, mu_col, max_iter=100, tol=1e-10):
    """
    Standard Sinkhorn-Knopp algorithm for marginal projection.
    """
    for _ in range(max_iter):
        # Row normalization
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = jnp.maximum(row_sums, 1e-100)
        P = P * (mu_row[:, None] / row_sums)
        
        # Column normalization
        col_sums = P.sum(axis=0, keepdims=True)
        col_sums = jnp.maximum(col_sums, 1e-100)
        P = P * (mu_col[None, :] / col_sums)
        
        # Check convergence
        row_err = jnp.max(jnp.abs(P.sum(axis=1) - mu_row))
        col_err = jnp.max(jnp.abs(P.sum(axis=0) - mu_col))
        if row_err < tol and col_err < tol:
            break
    
    return P


# ============================================================================
# MARTINGALE PROJECTION (Gentle Row-wise Correction)
# ============================================================================

def project_martingale_gentle(P, x_norm, strength=0.5):
    """
    Gently nudge P towards martingale constraint.
    
    x_norm should be the NORMALIZED grid (0 to 1 range).
    """
    M = P.shape[0]
    P_new = P.copy()
    
    for i in range(M):
        row = P[i]
        mass = jnp.sum(row)
        
        if mass < 1e-10:
            continue
        
        # Compute current conditional expectation
        w = row / mass
        current_mean = jnp.sum(w * x_norm)
        target_mean = x_norm[i]
        
        # Compute variance
        var = jnp.sum(w * (x_norm - current_mean)**2)
        
        if var < 1e-10:
            continue
        
        # Newton step for exponential tilting
        lam = (target_mean - current_mean) / var
        lam = lam * strength
        
        # Clip the exponent (now safe since x_norm is in [0,1])
        arg = lam * (x_norm - current_mean)
        arg = jnp.clip(arg, -10.0, 10.0)
        
        # Apply exponential tilting
        weights = row * jnp.exp(arg)
        weights = jnp.maximum(weights, 1e-100)
        new_row = weights * mass / jnp.sum(weights)
        
        P_new = P_new.at[i].set(new_row)
    
    return P_new


# ============================================================================
# MAIN SOLVER
# ============================================================================

def solve_mmot_admm(marginals, C, x_grid, epsilon=0.1, max_iter=500, tol=1e-6,
                    martingale_weight=1.0):
    """
    Solve MMOT using Alternating Projections.

    Parameters:
    -----------
    marginals : jnp.array or list
        [mu_t, mu_{t+1}] - marginal distributions at times t and t+1
    C : jnp.array (M, M)
        Cost matrix (will be normalized internally)
    x_grid : jnp.array (M,)
        Grid points (will be normalized internally)
    epsilon : float
        Entropic regularization parameter
    max_iter : int
        Maximum outer iterations
    tol : float
        Convergence tolerance
    martingale_weight : float
        Kept for API compatibility

    Returns:
    --------
    dict with keys:
        'P': Transport plan (M, M)
        'mu_t': Input marginal at time t
        'mu_next': Input marginal at time t+1
        'iterations': Number of iterations
        'converged': Whether converged
        'primal_costs': Cost history
    """
    # Handle input formats
    if isinstance(marginals, list):
        marginals = jnp.stack(marginals)

    mu_t = jnp.asarray(marginals[0], dtype=jnp.float64)
    mu_next = jnp.asarray(marginals[1], dtype=jnp.float64)
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64)
    C = jnp.asarray(C, dtype=jnp.float64)
    
    # Ensure marginals are normalized
    mu_t = mu_t / jnp.sum(mu_t)
    mu_next = mu_next / jnp.sum(mu_next)

    # =========================================================================
    # KEY FIX: Normalize the grid to [0, 1] for numerical stability
    # =========================================================================
    x_min = jnp.min(x_grid)
    x_max = jnp.max(x_grid)
    x_range = x_max - x_min
    x_norm = (x_grid - x_min) / x_range  # Now in [0, 1]
    
    # Normalize cost matrix accordingly
    # Original C = (x_i - x_j)^2
    # Normalized C_norm = ((x_i - x_min)/range - (x_j - x_min)/range)^2
    #                   = C / range^2
    C_norm = C / (x_range ** 2)
    
    # Now epsilon is relative to normalized costs (max C_norm ≈ 1)
    # So epsilon=0.1 means moderate smoothing as intended
    
    # Initialize with entropic transport (Gibbs kernel)
    K = jnp.exp(-C_norm / epsilon)
    P = K * jnp.outer(mu_t, mu_next)
    P = P / jnp.sum(P)  # Normalize
    
    # Track convergence
    costs = []

    for k in range(max_iter):
        P_old = P.copy()

        # Step 1: Project onto marginal constraints (Sinkhorn)
        P = sinkhorn_knopp(P, mu_t, mu_next, max_iter=50)
        
        # Step 2: Strong martingale correction (full Newton step)
        # Use constant strength=1.0 for proper martingale enforcement
        P = project_martingale_gentle(P, x_norm, strength=1.0)
        
        # Step 3: Project back onto marginals
        P = sinkhorn_knopp(P, mu_t, mu_next, max_iter=30)
        
        # Ensure P is non-negative and normalized
        P = jnp.maximum(P, 1e-100)
        P = P / jnp.sum(P)

        # Compute cost (using original costs for reporting)
        entropy = -jnp.sum(P * jnp.log(jnp.maximum(P, 1e-100)))
        cost = float(jnp.sum(P * C) - epsilon * entropy)
        costs.append(cost)

        # Check convergence
        diff = float(jnp.max(jnp.abs(P - P_old)))
        if diff < tol:
            return {
                'P': jnp.array(P),
                'mu_t': mu_t,
                'mu_next': mu_next,
                'iterations': k + 1,
                'converged': True,
                'primal_costs': jnp.array(costs)
            }

    # Max iterations reached
    return {
        'P': jnp.array(P),
        'mu_t': mu_t,
        'mu_next': mu_next,
        'iterations': max_iter,
        'converged': False,
        'primal_costs': jnp.array(costs)
    }
