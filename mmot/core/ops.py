# ============================================================================
# FILE: mmot/core/ops.py
# PURPOSE: Low-level JAX kernels for MMOT solver
# VERSION: 8.0 - SIMPLIFIED: Proper Sinkhorn + Newton separation
# ============================================================================
import jax
import jax.numpy as jnp
from jax import jit

# ----------------------------------------------------------------------------
# 1. NUMERICAL STABILITY PRIMITIVES
# ----------------------------------------------------------------------------
@jit
def logsumexp_axis0(logits):
    max_logit = jnp.max(logits, axis=0, keepdims=True)
    return jnp.squeeze(max_logit + jnp.log(jnp.sum(jnp.exp(logits - max_logit), axis=0, keepdims=True)), axis=0)

@jit
def logsumexp_axis1(logits):
    max_logit = jnp.max(logits, axis=1, keepdims=True)
    return jnp.squeeze(max_logit + jnp.log(jnp.sum(jnp.exp(logits - max_logit), axis=1, keepdims=True)), axis=1)

@jit
def logsumexp_axis1_keepdims(logits):
    max_logit = jnp.max(logits, axis=1, keepdims=True)
    return max_logit + jnp.log(jnp.sum(jnp.exp(logits - max_logit), axis=1, keepdims=True))

@jit
def stable_logsumexp_2d(logits):
    """Log-sum-exp over entire 2D matrix."""
    max_logit = jnp.max(logits)
    return max_logit + jnp.log(jnp.sum(jnp.exp(logits - max_logit)))

# ----------------------------------------------------------------------------
# 2. MARTINGALE PROJECTION (h-update via Newton-Raphson)
# ----------------------------------------------------------------------------
@jit
def newton_martingale_step(h_current, u_t, u_next, C, Delta, epsilon, damping=0.8):
    """
    Newton step for martingale constraint.
    Uses FULL kernel including u_t for accurate conditional.
    """
    # Full kernel for P(X_t, X_{t+1})
    LogK = (u_t[:, None] + u_next[None, :] + h_current[:, None] * Delta - C) / epsilon
    LogK_max = jnp.max(LogK, axis=1, keepdims=True)
    K = jnp.exp(LogK - LogK_max)
    
    # Normalize rows to get P(X_{t+1} | X_t)
    row_sums = jnp.sum(K, axis=1, keepdims=True)
    P = K / (row_sums + 1e-10)
    
    # Martingale residual
    f = jnp.sum(P * Delta, axis=1)
    
    # Newton derivative
    E_Delta_sq = jnp.sum(P * (Delta ** 2), axis=1)
    f_prime = (E_Delta_sq - f ** 2) / epsilon
    f_prime = jnp.where(jnp.abs(f_prime) < 1e-10, 1.0, f_prime)
    
    return h_current - damping * (f / f_prime)

@jit
def solve_martingale_constraint(h_init, u_t, u_next, C, Delta, epsilon, newton_iters=15):
    """Solve h using Newton iterations with full kernel."""
    def body(i, h):
        return newton_martingale_step(h, u_t, u_next, C, Delta, epsilon)
    return jax.lax.fori_loop(0, newton_iters, body, h_init)

# ----------------------------------------------------------------------------
# 3. MARGINAL PROJECTION (Sinkhorn-style u-update)
# ----------------------------------------------------------------------------
@jit
def sinkhorn_u_update_interior(u_prev, u_current, u_next, h_prev, h_current,
                                marginal_target, C, Delta, epsilon):
    """
    Update u[t] for interior time step (0 < t < N).
    Uses both forward and backward transitions.
    """
    # Forward: from t-1 to t
    LogK_fwd = (u_prev[:, None] + u_current[None, :] + h_prev[:, None] * Delta - C) / epsilon
    log_fwd_marginal = logsumexp_axis0(LogK_fwd)
    u_fwd = epsilon * jnp.log(marginal_target + 1e-10) - log_fwd_marginal
    
    # Backward: from t to t+1
    LogK_bwd = (u_current[:, None] + u_next[None, :] + h_current[:, None] * Delta - C) / epsilon
    log_bwd_marginal = logsumexp_axis1(LogK_bwd)
    u_bwd = epsilon * jnp.log(marginal_target + 1e-10) - log_bwd_marginal
    
    # Average
    return 0.5 * (u_fwd + u_bwd)

@jit
def sinkhorn_u_update_first(u_current, u_next, h_current, marginal_target, C, Delta, epsilon):
    """Update u[0] - only backward transition matters for marginal."""
    LogK = (u_current[:, None] + u_next[None, :] + h_current[:, None] * Delta - C) / epsilon
    log_marginal = logsumexp_axis1(LogK)
    return epsilon * jnp.log(marginal_target + 1e-10) - log_marginal

@jit
def sinkhorn_u_update_last(u_prev, u_current, h_prev, marginal_target, C, Delta, epsilon):
    """Update u[N] - only forward transition matters for marginal."""
    LogK = (u_prev[:, None] + u_current[None, :] + h_prev[:, None] * Delta - C) / epsilon
    log_marginal = logsumexp_axis0(LogK)
    return epsilon * jnp.log(marginal_target + 1e-10) - log_marginal

# Legacy compatibility
@jit
def sinkhorn_u_update_forward(u_prev, h_prev, marginal_target, C, Delta, epsilon):
    """Forward Sinkhorn update (legacy)."""
    LogK = (u_prev[:, None] + h_prev[:, None] * Delta - C) / epsilon
    log_incoming = logsumexp_axis0(LogK)
    return epsilon * jnp.log(marginal_target + 1e-10) - log_incoming

@jit
def sinkhorn_u_update_bidirectional(u_prev, u_next, h_prev, h_current,
                                    marginal_target, C, Delta, epsilon):
    """Bidirectional update (legacy)."""
    u_fwd = sinkhorn_u_update_forward(u_prev, h_prev, marginal_target, C, Delta, epsilon)
    LogK_back = (u_next[None, :] + h_current[:, None] * Delta - C) / epsilon
    log_outgoing = logsumexp_axis1(LogK_back)
    u_bwd = epsilon * jnp.log(marginal_target + 1e-10) - log_outgoing
    return 0.5 * (u_fwd + u_bwd)

# ----------------------------------------------------------------------------
# 4. DIAGNOSTICS
# ----------------------------------------------------------------------------
@jit
def compute_martingale_violation(u_t, u_next, h_t, x_grid, C, Delta, epsilon):
    """Compute max martingale violation."""
    LogP = (u_t[:, None] + u_next[None, :] + h_t[:, None] * Delta - C) / epsilon
    LogP_norm = LogP - logsumexp_axis1_keepdims(LogP)
    P = jnp.exp(LogP_norm)
    E_next = jnp.sum(P * x_grid[None, :], axis=1)
    return jnp.max(jnp.abs(E_next - x_grid))

@jit
def compute_marginal_violation(u_t, u_next, h_t, marginal_target, C, Delta, epsilon):
    """Compute marginal violation at time t."""
    LogP = (u_t[:, None] + u_next[None, :] + h_t[:, None] * Delta - C) / epsilon
    LogP_norm = LogP - stable_logsumexp_2d(LogP)
    P = jnp.exp(LogP_norm)
    marginal_computed = jnp.sum(P, axis=1)
    return jnp.sum(jnp.abs(marginal_computed - marginal_target))
