
"""
NUMERICALLY STABLE MMOT SOLVER with Cost Scaling & Epsilon Annealing.
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
    LogK = (u_t + u_next + h_t * Delta - C_scaled) / epsilon_scaled
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
    """Update u[0]."""
    # Exclude u_current from kernel
    zeros = jnp.zeros_like(u_current)
    LogK = compute_log_kernel_stable(zeros, u_next, h_current, C_scaled, Delta, epsilon_scaled)
    log_marginal = logsumexp_axis1(LogK)
    return epsilon_scaled * jnp.log(marginal + 1e-10) - epsilon_scaled * log_marginal


@jit
def sinkhorn_u_update_last_stable(u_prev, u_current, h_prev, marginal, C_scaled, Delta, epsilon_scaled):
    """Update u[N]."""
    # Exclude u_current from kernel
    zeros = jnp.zeros_like(u_current)
    LogK = compute_log_kernel_stable(u_prev, zeros, h_prev, C_scaled, Delta, epsilon_scaled)
    log_marginal = logsumexp_axis0(LogK)
    return epsilon_scaled * jnp.log(marginal + 1e-10) - epsilon_scaled * log_marginal


@jit
def sinkhorn_u_update_interior_stable(u_prev, u_current, u_next, h_prev, h_current,
                                       marginal, C_scaled, Delta, epsilon_scaled):
    """Update interior u[t]."""
    zeros = jnp.zeros_like(u_current)
    
    # Forward: t-1 -> t. Solve for u_t (receiver).
    LogK_fwd = compute_log_kernel_stable(u_prev, zeros, h_prev, C_scaled, Delta, epsilon_scaled)
    log_fwd = logsumexp_axis0(LogK_fwd)
    u_fwd = epsilon_scaled * jnp.log(marginal + 1e-10) - epsilon_scaled * log_fwd
    
    # Backward: t -> t+1. Solve for u_t (source).
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
    
    # Newton derivative (approx 1/eps * Var[X'])
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
    """Solve h using Newton."""
    def body(i, h):
        # Using 0.8 damping inside Newton loop
        return newton_martingale_step_stable(h, u_t, u_next, C_scaled, Delta, epsilon_scaled, damping=0.8)
    return jax.lax.fori_loop(0, newton_iters, body, h_init)


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_mmot_stable(marginals, x_grid, max_iter=2000, epsilon=0.01, damping=0.1, 
                      verbose=False):
    """
    Solve MMOT with automatic cost scaling and epsilon annealing.
    """
    N = marginals.shape[0] - 1
    M = marginals.shape[1]
    
    # Grid/Cost
    Delta = x_grid[:, None] - x_grid[None, :]
    C = Delta ** 2
    C_max = float(C.max())
    
    # Scale cost
    C_scaled = C / C_max
    
    # JAX arrays
    marginals_jax = jnp.array(marginals)
    x_grid_jax = jnp.array(x_grid)
    C_scaled_jax = jnp.array(C_scaled)
    Delta_jax = jnp.array(Delta)
    
    # Annealing
    # Start with Ratio 5 (Eps=0.2 scale) which we verified works
    ratios = [5.0]
    iters_schedule = [max_iter]
    
    current_u = jnp.zeros((N + 1, M))
    current_h = jnp.zeros((N, M))
    
    for _, (ratio, n_iter) in enumerate(zip(ratios, iters_schedule)):
        eps_stage = max(epsilon / C_max, 1.0 / ratio)
        if verbose:
            print(f"[Solver] Eps Scaled: {eps_stage:.4f}")
            
        tol = 1e-4 * np.sqrt(N + 1)
        
        for iteration in range(n_iter):
            u_old, h_old = current_u, current_h
            
            # h-update
            for t in range(N):
                h_new_t = solve_martingale_stable(
                    current_h[t], current_u[t], current_u[t+1], C_scaled_jax, Delta_jax, eps_stage
                )
                current_h = current_h.at[t].set(h_new_t)
            
            # u-update
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
                
                # Global Damping
                current_u = current_u.at[t].set((1 - damping) * current_u[t] + damping * u_new)
            
            # Check convergence
            u_diff = float(jnp.max(jnp.abs(current_u - u_old)))
            h_diff = float(jnp.max(jnp.abs(current_h - h_old)))
            
            if verbose and iteration % 100 == 0:
                print(f"  Iter {iteration}: du={u_diff:.6f}, dh={h_diff:.6f}")
                
            if u_diff < tol and h_diff < tol:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break
    
    # Final Drift Calc
    final_eps = max(epsilon / C_max, 1.0 / ratios[-1])
    total_drift = 0.0
    for t in range(N):
        LogK = compute_log_kernel_stable(current_u[t], current_u[t+1], current_h[t], C_scaled_jax, Delta_jax, final_eps)
        LogK_norm = LogK - logsumexp_axis1(LogK)[:, None]
        P = jnp.exp(LogK_norm)
        E_next = jnp.sum(P * x_grid_jax[None, :], axis=1)
        drift_t = float(jnp.max(jnp.abs(E_next - x_grid_jax)))
        total_drift += drift_t
    
    avg_drift = total_drift / N
    
    return {
        'u': np.array(current_u),
        'h': np.array(current_h),
        'drift': avg_drift,
        'iterations': iteration + 1,
        'C_max': C_max,
        'epsilon_scaled': final_eps,
        'converged': (u_diff < tol and h_diff < tol)
    }
