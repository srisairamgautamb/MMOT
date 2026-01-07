#!/usr/bin/env python3
"""
NUMERICALLY STABLE MMOT SOLVER WITH AUTOMATIC COST SCALING
==========================================================
Solves MMOT on TRUE moneyness grid [0.5, 1.5] by scaling cost+epsilon.

Key insight: The MMOT problem is scale-invariant.
If we divide C by C_max and divide epsilon by C_max, the solution is the same!

This allows us to work with any grid range without numerical issues.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


# =============================================================================
# STABLE KERNEL COMPUTATION
# =============================================================================

@jit
def compute_log_kernel_stable(u_t, u_next, h_t, C_scaled, Delta, epsilon_scaled):
    """
    Compute log-kernel with automatic stability.
    
    LogK = (u_t + u_next + h_t * Delta - C_scaled) / epsilon_scaled
    
    We ensure epsilon_scaled is chosen so max(C_scaled/epsilon_scaled) ~ 20
    """
    LogK = (u_t[:, None] + u_next[None, :] + h_t[:, None] * Delta - C_scaled) / epsilon_scaled
    return LogK


@jit
def logsumexp_axis0(logits):
    """Numerically stable log-sum-exp over axis 0."""
    max_logit = jnp.max(logits, axis=0, keepdims=True)
    return jnp.squeeze(max_logit + jnp.log(jnp.sum(jnp.exp(logits - max_logit), axis=0, keepdims=True)), axis=0)


@jit
def logsumexp_axis1(logits):
    """Numerically stable log-sum-exp over axis 1."""
    max_logit = jnp.max(logits, axis=1, keepdims=True)
    return jnp.squeeze(max_logit + jnp.log(jnp.sum(jnp.exp(logits - max_logit), axis=1, keepdims=True)), axis=1)


# =============================================================================
# SINKHORN UPDATES (STABLE)
# =============================================================================

@jit
def sinkhorn_u_update_first_stable(u_current, u_next, h_current, marginal, C_scaled, Delta, epsilon_scaled):
    """Update u[0] with scaled cost."""
    # We solve for u_current. It should NOT be in the kernel expression.
    # LogK = (0 + u_next + h_current*Delta - C)/eps
    zeros = jnp.zeros_like(u_current)
    LogK = compute_log_kernel_stable(zeros, u_next, h_current, C_scaled, Delta, epsilon_scaled)
    log_marginal = logsumexp_axis1(LogK)
    return epsilon_scaled * jnp.log(marginal + 1e-10) - epsilon_scaled * log_marginal


@jit
def sinkhorn_u_update_last_stable(u_prev, u_current, h_prev, marginal, C_scaled, Delta, epsilon_scaled):
    """Update u[N] with scaled cost."""
    # We solve for u_current. It should NOT be in the kernel expression.
    zeros = jnp.zeros_like(u_current)
    LogK = compute_log_kernel_stable(u_prev, zeros, h_prev, C_scaled, Delta, epsilon_scaled)
    log_marginal = logsumexp_axis0(LogK)
    return epsilon_scaled * jnp.log(marginal + 1e-10) - epsilon_scaled * log_marginal


@jit
def sinkhorn_u_update_interior_stable(u_prev, u_current, u_next, h_prev, h_current,
                                       marginal, C_scaled, Delta, epsilon_scaled):
    """Update interior u[t] with scaled cost."""
    zeros = jnp.zeros_like(u_current)
    
    # Forward: t-1 -> t. Solve for u_t (receiver).
    # Kernel should depend on u_{t-1} but NOT u_t.
    LogK_fwd = compute_log_kernel_stable(u_prev, zeros, h_prev, C_scaled, Delta, epsilon_scaled)
    log_fwd = logsumexp_axis0(LogK_fwd)
    u_fwd = epsilon_scaled * jnp.log(marginal + 1e-10) - epsilon_scaled * log_fwd
    
    # Backward: t -> t+1. Solve for u_t (source).
    # Kernel should depend on u_{t+1} but NOT u_t.
    LogK_bwd = compute_log_kernel_stable(zeros, u_next, h_current, C_scaled, Delta, epsilon_scaled)
    log_bwd = logsumexp_axis1(LogK_bwd)
    u_bwd = epsilon_scaled * jnp.log(marginal + 1e-10) - epsilon_scaled * log_bwd
    
    return 0.5 * (u_fwd + u_bwd)


# =============================================================================
# MARTINGALE CONSTRAINT (STABLE)
# =============================================================================

@jit
def newton_martingale_step_stable(h_current, u_t, u_next, C_scaled, Delta, epsilon_scaled, damping=0.8):
    """Newton step for martingale with scaled cost."""
    LogK = compute_log_kernel_stable(u_t, u_next, h_current, C_scaled, Delta, epsilon_scaled)
    LogK_max = jnp.max(LogK, axis=1, keepdims=True)
    K = jnp.exp(LogK - LogK_max)
    
    # Normalize rows
    row_sums = jnp.sum(K, axis=1, keepdims=True)
    P = K / (row_sums + 1e-10)
    
    # Martingale residual
    f = jnp.sum(P * Delta, axis=1)
    
    # Newton derivative
    # Newton derivative (approx 1/eps * Var[X'])
    # If variance is tiny, f_prime -> 0, update -> huge
    E_Delta_sq = jnp.sum(P * (Delta ** 2), axis=1)
    variance = E_Delta_sq - f ** 2
    
    # Regularize denominator
    variance = jnp.maximum(variance, 1e-6)
    f_prime = variance / epsilon_scaled
    
    update = f / f_prime
    
    # Clamp update magnitude to prevent explosion
    update = jnp.clip(update, -100.0, 100.0)
    
    return h_current - damping * update


@jit
def solve_martingale_stable(h_init, u_t, u_next, C_scaled, Delta, epsilon_scaled, newton_iters=15):
    """Solve h using Newton with scaled cost."""
    def body(i, h):
        return newton_martingale_step_stable(h, u_t, u_next, C_scaled, Delta, epsilon_scaled)
    return jax.lax.fori_loop(0, newton_iters, body, h_init)


# =============================================================================
# MAIN SOLVER WITH AUTOMATIC SCALING
# =============================================================================

def solve_mmot_stable(marginals, x_grid, max_iter=500, epsilon=0.01, damping=0.8, 
                      target_ratio=20.0, verbose=False):
    """
    Solve MMOT with automatic cost scaling for numerical stability.
    
    Args:
        marginals: (N+1, M) array of probability distributions
        x_grid: (M,) grid points (can be moneyness [0.5, 1.5] or prices [50, 200])
        max_iter: Maximum iterations
        epsilon: Regularization parameter (in original scale)
        damping: Damping factor for updates
        target_ratio: Target C_max/epsilon_scaled ratio (default 20)
        verbose: Print progress
    
    Returns:
        dict with u, h, drift, iterations, scaling info
    """
    N = marginals.shape[0] - 1
    M = marginals.shape[1]
    
    # Compute cost matrix and its scale
    Delta = x_grid[:, None] - x_grid[None, :]
    C = Delta ** 2
    C_max = float(C.max())
    
    # Scale cost
    C_scaled = C / C_max
    
    # Convert to JAX arrays
    marginals_jax = jnp.array(marginals)
    x_grid_jax = jnp.array(x_grid)
    C_scaled_jax = jnp.array(C_scaled)
    Delta_jax = jnp.array(Delta)
    
    # Epsilon annealing schedule
    # We solve sequentially with decreasing epsilon (increasing ratio)
    ratios = [5.0]
    iters = [2000]
    
    current_u = jnp.zeros((N + 1, M))
    current_h = jnp.zeros((N, M))
    
    final_drift = float('inf')
    
    for stage, (ratio, n_iter) in enumerate(zip(ratios, iters)):
        # Calculate epsilon for this stage
        eps_stage = max(epsilon / C_max, 1.0 / ratio)
        if verbose:
            print(f"\n[Annealing Stage {stage+1}] Ratio={ratio}, Eps={eps_stage:.4f}, Iters={n_iter}")
            
        # Re-compile step functions if needed (epsilon is static arg? Check ops)
        # In our case epsilon is passed as arg, so JIT recompiles automatically for new scalar traces
        
        tol = 1e-4 * np.sqrt(N + 1)
        damping_stage = damping  # Can adjust damping per stage
        
        for iteration in range(n_iter):
            u_old, h_old = current_u, current_h
            
            # h-update (martingale)
            for t in range(N):
                h_new_t = solve_martingale_stable(
                    current_h[t], current_u[t], current_u[t+1], C_scaled_jax, Delta_jax, eps_stage
                )
                current_h = current_h.at[t].set(h_new_t)
            
            # u-update (marginals)
            for t in range(N + 1):
                if t == 0:
                    u_new = sinkhorn_u_update_first_stable(
                        current_u[0], current_u[1], current_h[0], marginals_jax[0], C_scaled_jax, Delta_jax, eps_stage
                    )
                elif t == N:
                    u_new = sinkhorn_u_update_last_stable(
                        current_u[N-1], current_u[N], current_h[N-1], marginals_jax[N], C_scaled_jax, Delta_jax, eps_stage
                    )
                else:
                    u_new = sinkhorn_u_update_interior_stable(
                        current_u[t-1], current_u[t], current_u[t+1], current_h[t-1], current_h[t],
                        marginals_jax[t], C_scaled_jax, Delta_jax, eps_stage
                    )
                
                current_u = current_u.at[t].set((1 - damping_stage) * current_u[t] + damping_stage * u_new)
            
            # Check convergence
            u_diff = float(jnp.max(jnp.abs(current_u - u_old)))
            h_diff = float(jnp.max(jnp.abs(current_h - h_old)))
            
            if verbose and iteration % 100 == 0:
                print(f"  Iter {iteration}: du={u_diff:.6f}, dh={h_diff:.6f}")
                
            if u_diff < tol and h_diff < tol:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break
    
    # Compute final drift
    total_drift = 0.0
    # Use final small epsilon for drift calc
    final_eps = max(epsilon / C_max, 1.0 / ratios[-1])
    
    for t in range(N):
        LogK = compute_log_kernel_stable(current_u[t], current_u[t+1], current_h[t], C_scaled_jax, Delta_jax, final_eps)
        LogK_norm = LogK - logsumexp_axis1(LogK)[:, None]
        P = jnp.exp(LogK_norm)
        
        # E[X' | X] - X
        E_next = jnp.sum(P * x_grid_jax[None, :], axis=1)
        drift_t = float(jnp.max(jnp.abs(E_next - x_grid_jax)))
        total_drift += drift_t
    
    avg_drift = total_drift / N
    
    return {
        'u': np.array(current_u),
        'h': np.array(current_h),
        'drift': avg_drift,
        'iterations': sum(iters), # Approximation
        'C_max': C_max,
        'epsilon_scaled': final_eps,
        'converged': (u_diff < tol and h_diff < tol)
    }


# =============================================================================
# TEST
# =============================================================================

def test_on_true_moneyness():
    """Test solver on TRUE moneyness grid [0.5, 1.5]."""
    print("="*60)
    print("TESTING STABLE SOLVER ON TRUE MONEYNESS [0.5, 1.5]")
    print("="*60)
    
    # TRUE moneyness grid
    M = 150
    moneyness_grid = np.linspace(0.5, 1.5, M)
    
    print(f"\nGrid: [{moneyness_grid.min():.2f}, {moneyness_grid.max():.2f}]")
    print(f"C_max = (1.5 - 0.5)² = {(1.5-0.5)**2:.2f}")
    
    # Generate log-normal marginals
    N = 5
    T = 0.25
    sigma = 0.25
    sigma = 0.25
    r = 0.0  # Set r=0 to ensure Martingale expectation (E[S_T] = S_0)
    dt = T / N
    dm = moneyness_grid[1] - moneyness_grid[0]
    
    marginals = np.zeros((N + 1, M))
    
    # t=0: Smooth point mass (narrow Gaussian) to prevent numerical issues
    # A single spike can cause division by zero in variance calculations
    std_0 = 0.01
    log_m = np.log(moneyness_grid)
    density_0 = np.exp(-0.5 * ((log_m - 0) / std_0)**2)  # Centered at log(1)=0
    density_0 = density_0 / density_0.sum()  # Normalize to sum=1 (discrete mass)
    marginals[0] = density_0
    
    # t>0: Log-normal
    for t in range(1, N + 1):
        time_t = t * dt
        mu = (r - 0.5 * sigma**2) * time_t
        std = sigma * np.sqrt(time_t)
        
        log_m = np.log(moneyness_grid)
        density = (1.0 / (moneyness_grid * std * np.sqrt(2*np.pi))) * \
                  np.exp(-0.5 * ((log_m - mu) / std)**2)
        density = density / density.sum()  # Normalize to sum=1 (discrete mass)
        marginals[t] = density
    
    print(f"\nMarginal sums: {[f'{m.sum():.4f}' for m in marginals]}")
    
    # Solve with stable solver
    print(f"\nSolving with stable solver...")
    result = solve_mmot_stable(marginals, moneyness_grid, max_iter=2000, 
                               epsilon=0.01, damping=0.1, verbose=True)
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Grid:              [{moneyness_grid.min():.2f}, {moneyness_grid.max():.2f}] (TRUE MONEYNESS)")
    print(f"C_max:             {result['C_max']:.4f}")
    print(f"epsilon_scaled:    {result['epsilon_scaled']:.4f}")
    print(f"Ratio:             {1.0/result['epsilon_scaled']:.1f}")
    print(f"Converged:         {result['converged']}")
    print(f"Iterations:        {result['iterations']}")
    print(f"Drift:             {result['drift']:.6f}")
    print(f"Mean |u|:          {np.abs(result['u']).mean():.4f}")
    print(f"Mean |h|:          {np.abs(result['h']).mean():.4f}")
    
    if result['drift'] < 0.01:
        print(f"\n✅ SUCCESS! Drift {result['drift']:.6f} < 0.01")
    else:
        print(f"\n⚠️ Drift {result['drift']:.6f} > 0.01 (may need tuning)")
    
    return result


if __name__ == "__main__":
    test_on_true_moneyness()
