#!/usr/bin/env python3
"""
solve_mmot_adaptive.py
======================
ADAPTIVE EPSILON MMOT SOLVER FOR PUBLICATION-QUALITY RESULTS

3-Stage Epsilon Annealing:
- Stage 1 (ε=0.5):  Fast warm-up, rough solution
- Stage 2 (ε=0.2):  Refinement, medium accuracy  
- Stage 3 (ε=0.05): High accuracy, publication quality

Target: drift < 0.0001 (vs current 0.002)

Author: MMOT Research Team
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
import time


# =============================================================================
# STABLE KERNEL COMPUTATION
# =============================================================================

@partial(jit, static_argnums=())
def compute_log_kernel(u_t, u_next, h_t, C_scaled, Delta, epsilon):
    """Compute log-kernel for given epsilon."""
    term_u = u_t[:, None] + u_next[None, :]
    term_h = h_t[:, None] * Delta
    return (term_u + term_h - C_scaled) / epsilon


@partial(jit, static_argnums=())
def logsumexp_axis0(logits):
    """Numerically stable log-sum-exp over axis 0."""
    max_logits = jnp.max(logits, axis=0, keepdims=True)
    return jnp.log(jnp.sum(jnp.exp(logits - max_logits), axis=0)) + max_logits[0]


@partial(jit, static_argnums=())
def logsumexp_axis1(logits):
    """Numerically stable log-sum-exp over axis 1."""
    max_logits = jnp.max(logits, axis=1, keepdims=True)
    return jnp.log(jnp.sum(jnp.exp(logits - max_logits), axis=1)) + max_logits[:, 0]


# =============================================================================
# SINKHORN UPDATES
# =============================================================================

@partial(jit, static_argnums=())
def sinkhorn_u_update_first(u_current, u_next, h_current, marginal, C_scaled, Delta, epsilon):
    """Update u[0]."""
    LogK = compute_log_kernel(u_current, u_next, h_current, C_scaled, Delta, epsilon)
    log_marginal = jnp.log(marginal + 1e-30)
    return epsilon * log_marginal - epsilon * logsumexp_axis1(LogK)


@partial(jit, static_argnums=())
def sinkhorn_u_update_last(u_prev, u_current, h_prev, marginal, C_scaled, Delta, epsilon):
    """Update u[N]."""
    LogK = compute_log_kernel(u_prev, u_current, h_prev, C_scaled, Delta, epsilon)
    log_marginal = jnp.log(marginal + 1e-30)
    return epsilon * log_marginal - epsilon * logsumexp_axis0(LogK)


@partial(jit, static_argnums=())
def sinkhorn_u_update_interior(u_prev, u_current, u_next, h_prev, h_current,
                                marginal, C_scaled, Delta, epsilon):
    """Update interior u[t]."""
    LogK_prev = compute_log_kernel(u_prev, u_current, h_prev, C_scaled, Delta, epsilon)
    LogK_next = compute_log_kernel(u_current, u_next, h_current, C_scaled, Delta, epsilon)
    
    log_marginal = jnp.log(marginal + 1e-30)
    lse_from_prev = logsumexp_axis0(LogK_prev)
    lse_to_next = logsumexp_axis1(LogK_next)
    
    return epsilon * log_marginal - 0.5 * epsilon * lse_from_prev - 0.5 * epsilon * lse_to_next


# =============================================================================
# MARTINGALE NEWTON STEP
# =============================================================================

@partial(jit, static_argnums=())
def newton_martingale_step(h_current, u_t, u_next, C_scaled, Delta, grid, epsilon, damping=0.8):
    """Newton step for martingale constraint: E[Y|X] = X."""
    LogK = compute_log_kernel(u_t, u_next, h_current, C_scaled, Delta, epsilon)
    
    # Stable softmax for transition probabilities
    LogK_stable = LogK - jnp.max(LogK, axis=1, keepdims=True)
    probs = jnp.exp(LogK_stable) / jnp.sum(jnp.exp(LogK_stable), axis=1, keepdims=True)
    
    # E[Y|X=x] for each x
    expected_Y = jnp.sum(probs * grid[None, :], axis=1)
    
    # Drift = E[Y|X] - X
    drift = expected_Y - grid
    
    # Variance = E[Y^2|X] - E[Y|X]^2
    expected_Y2 = jnp.sum(probs * (grid[None, :] ** 2), axis=1)
    variance = expected_Y2 - expected_Y ** 2
    variance = jnp.maximum(variance, 1e-12)
    
    # CORRECT Newton step: h_new = h + epsilon * drift / variance
    delta_h = epsilon * drift / variance
    
    return h_current + damping * delta_h


@partial(jit, static_argnums=(6,))
def solve_martingale(h_init, u_t, u_next, C_scaled, Delta, grid, epsilon, newton_iters=20):
    """Solve h using Newton iterations."""
    def body(i, h):
        return newton_martingale_step(h, u_t, u_next, C_scaled, Delta, grid, epsilon, damping=0.8)
    return jax.lax.fori_loop(0, newton_iters, body, h_init)


# =============================================================================
# DRIFT COMPUTATION
# =============================================================================

@partial(jit, static_argnums=())
def compute_drift_single(u_t, u_next, h_t, C_scaled, Delta, grid, epsilon):
    """Compute max drift for single time step."""
    LogK = compute_log_kernel(u_t, u_next, h_t, C_scaled, Delta, epsilon)
    LogK_stable = LogK - jnp.max(LogK, axis=1, keepdims=True)
    probs = jnp.exp(LogK_stable) / jnp.sum(jnp.exp(LogK_stable), axis=1, keepdims=True)
    expected_Y = jnp.sum(probs * grid[None, :], axis=1)
    drift = jnp.abs(expected_Y - grid)
    return jnp.max(drift)


def compute_drift_all(u, h, C_scaled, Delta, grid, epsilon):
    """Compute max drift across all time steps."""
    N = h.shape[0]
    max_drift = 0.0
    for t in range(N):
        drift = float(compute_drift_single(u[t], u[t+1], h[t], C_scaled, Delta, grid, epsilon))
        if drift > max_drift:
            max_drift = drift
    return max_drift


# =============================================================================
# SINGLE-STAGE SOLVER
# =============================================================================

def solve_mmot_single_epsilon(marginals, grid, epsilon, max_iter=500, 
                               u_init=None, h_init=None, newton_iters=15,
                               verbose=False):
    """
    Solve MMOT with fixed epsilon.
    
    Args:
        marginals: (N+1, M) array
        grid: (M,) array
        epsilon: regularization parameter
        max_iter: Sinkhorn iterations
        u_init, h_init: warm-start potentials (optional)
        newton_iters: Newton iterations per Sinkhorn step
        verbose: print progress
        
    Returns:
        dict with u, h, drift, iterations, time
    """
    start_time = time.time()
    
    N = marginals.shape[0] - 1
    M = marginals.shape[1]
    
    # Convert to JAX arrays
    marginals = jnp.array(marginals, dtype=jnp.float32)
    grid = jnp.array(grid, dtype=jnp.float32)
    
    # Precompute cost matrix
    Delta = grid[:, None] - grid[None, :]  # (M, M)
    C = Delta ** 2
    C_scaled = C / jnp.max(C)  # Normalize to [0, 1]
    
    # Initialize potentials
    if u_init is not None:
        u = [jnp.array(u_init[t], dtype=jnp.float32) for t in range(N+1)]
    else:
        u = [jnp.zeros(M, dtype=jnp.float32) for _ in range(N+1)]
    
    if h_init is not None:
        h = [jnp.array(h_init[t], dtype=jnp.float32) for t in range(N)]
    else:
        h = [jnp.zeros(M, dtype=jnp.float32) for _ in range(N)]
    
    # Sinkhorn iterations
    for iteration in range(max_iter):
        # Update u potentials
        u[0] = sinkhorn_u_update_first(u[0], u[1], h[0], marginals[0], C_scaled, Delta, epsilon)
        
        for t in range(1, N):
            u[t] = sinkhorn_u_update_interior(u[t-1], u[t], u[t+1], h[t-1], h[t],
                                               marginals[t], C_scaled, Delta, epsilon)
        
        u[N] = sinkhorn_u_update_last(u[N-1], u[N], h[N-1], marginals[N], C_scaled, Delta, epsilon)
        
        # Update h using Newton (martingale constraint)
        for t in range(N):
            h[t] = solve_martingale(h[t], u[t], u[t+1], C_scaled, Delta, grid, epsilon, newton_iters)
        
        # Check drift periodically
        if (iteration + 1) % 50 == 0 or iteration == max_iter - 1:
            drift = compute_drift_all(jnp.stack(u), jnp.stack(h), C_scaled, Delta, grid, epsilon)
            if verbose:
                print(f"    Iter {iteration+1:4d}: drift = {drift:.8f}")
            
            # Early stopping if drift is excellent
            if drift < 1e-5:
                if verbose:
                    print(f"    ✅ Converged at iteration {iteration+1}")
                break
    
    # Final drift
    u_final = jnp.stack(u)
    h_final = jnp.stack(h)
    final_drift = compute_drift_all(u_final, h_final, C_scaled, Delta, grid, epsilon)
    
    elapsed = time.time() - start_time
    
    return {
        'u': np.array(u_final),
        'h': np.array(h_final),
        'drift': float(final_drift),
        'iterations': iteration + 1,
        'time': elapsed,
        'epsilon': epsilon
    }


# =============================================================================
# ADAPTIVE 3-STAGE SOLVER (MAIN FUNCTION)
# =============================================================================

def solve_mmot_adaptive(marginals, grid, target_drift=1e-4, verbose=True):
    """
    Production-quality MMOT solver with adaptive epsilon annealing.
    
    3 Stages:
    - Stage 1: ε=0.5, 200 iterations (fast warm-up)
    - Stage 2: ε=0.2, 300 iterations (refinement)
    - Stage 3: ε=0.05, 500 iterations (high accuracy)
    
    Args:
        marginals: (N+1, M) array of marginal distributions
        grid: (M,) spatial grid
        target_drift: stop early if achieved
        verbose: print progress
        
    Returns:
        dict with u, h, drift, total_time, stage_info
    """
    total_start = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ADAPTIVE EPSILON MMOT SOLVER")
        print(f"{'='*60}")
        print(f"  Grid size: {len(grid)}")
        print(f"  Time steps: {len(marginals)-1}")
        print(f"  Target drift: {target_drift:.2e}")
    
    stage_info = []
    
    # =========================================================================
    # STAGE 1: Fast warm-up (ε = 0.5)
    # =========================================================================
    if verbose:
        print(f"\n  Stage 1/3: ε = 0.5 (warm-up)")
    
    result1 = solve_mmot_single_epsilon(
        marginals, grid,
        epsilon=0.5,
        max_iter=200,
        newton_iters=10,
        verbose=verbose
    )
    
    stage_info.append({
        'stage': 1, 'epsilon': 0.5,
        'drift': result1['drift'], 'time': result1['time'],
        'iterations': result1['iterations']
    })
    
    if verbose:
        print(f"    → Drift: {result1['drift']:.6f}, Time: {result1['time']:.1f}s")
    
    if result1['drift'] < target_drift:
        if verbose:
            print(f"    ✅ Target achieved in Stage 1!")
        return finalize_result(result1, stage_info, total_start)
    
    # =========================================================================
    # STAGE 2: Refinement (ε = 0.2)
    # =========================================================================
    if verbose:
        print(f"\n  Stage 2/3: ε = 0.2 (refinement)")
    
    result2 = solve_mmot_single_epsilon(
        marginals, grid,
        epsilon=0.2,
        max_iter=300,
        u_init=result1['u'],  # Warm-start!
        h_init=result1['h'],
        newton_iters=15,
        verbose=verbose
    )
    
    stage_info.append({
        'stage': 2, 'epsilon': 0.2,
        'drift': result2['drift'], 'time': result2['time'],
        'iterations': result2['iterations']
    })
    
    if verbose:
        print(f"    → Drift: {result2['drift']:.6f}, Time: {result2['time']:.1f}s")
    
    if result2['drift'] < target_drift:
        if verbose:
            print(f"    ✅ Target achieved in Stage 2!")
        return finalize_result(result2, stage_info, total_start)
    
    # =========================================================================
    # STAGE 3: High accuracy (ε = 0.05)
    # =========================================================================
    if verbose:
        print(f"\n  Stage 3/3: ε = 0.05 (high accuracy)")
    
    result3 = solve_mmot_single_epsilon(
        marginals, grid,
        epsilon=0.05,
        max_iter=500,
        u_init=result2['u'],  # Warm-start!
        h_init=result2['h'],
        newton_iters=20,
        verbose=verbose
    )
    
    stage_info.append({
        'stage': 3, 'epsilon': 0.05,
        'drift': result3['drift'], 'time': result3['time'],
        'iterations': result3['iterations']
    })
    
    if verbose:
        print(f"    → Drift: {result3['drift']:.6f}, Time: {result3['time']:.1f}s")
    
    return finalize_result(result3, stage_info, total_start)


def finalize_result(result, stage_info, total_start):
    """Add summary info to result."""
    total_time = time.time() - total_start
    
    result['total_time'] = total_time
    result['stage_info'] = stage_info
    result['converged'] = result['drift'] < 0.01
    
    return result


# =============================================================================
# TEST
# =============================================================================

def test_adaptive_solver():
    """Test the adaptive solver on sample instances."""
    print("\n" + "="*70)
    print("TESTING ADAPTIVE EPSILON SOLVER")
    print("="*70)
    
    # Create test case
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
    
    # Solve with adaptive epsilon
    result = solve_mmot_adaptive(marginals, grid, target_drift=1e-4, verbose=True)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"{'='*60}")
    print(f"  Final Drift:      {result['drift']:.8f}")
    print(f"  Total Time:       {result['total_time']:.1f}s")
    print(f"  Converged:        {result['converged']}")
    
    print(f"\n  Stage Summary:")
    for stage in result['stage_info']:
        print(f"    Stage {stage['stage']}: ε={stage['epsilon']}, "
              f"drift={stage['drift']:.6f}, time={stage['time']:.1f}s")
    
    # Grade
    if result['drift'] < 1e-4:
        print(f"\n  ✅ EXCELLENT: drift < 0.0001 (Target for publication!)")
    elif result['drift'] < 1e-3:
        print(f"\n  ✅ GOOD: drift < 0.001")
    elif result['drift'] < 1e-2:
        print(f"\n  ⚠️ ACCEPTABLE: drift < 0.01")
    else:
        print(f"\n  ❌ FAILED: drift > 0.01")
    
    print("="*70)
    
    return result


if __name__ == '__main__':
    test_adaptive_solver()
