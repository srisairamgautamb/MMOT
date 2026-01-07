# ============================================================================
# FILE: mmot/core/solver.py
# VERSION: 8.0 - Joint Sinkhorn-Newton with proper constraint enforcement
# ============================================================================
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from mmot.core import ops

@partial(jit, static_argnums=(3, 4, 5))
def solve_mmot(marginals, C, x_grid, max_iter=100, epsilon=0.01, damping=0.8):
    """
    MMOT Solver v8.0: Joint Sinkhorn-Newton iteration.
    
    Key fix: Newton solver now uses FULL kernel (u_t + u_{t+1} + h_t*Delta)
    to properly account for the joint distribution.
    """
    N = marginals.shape[0] - 1
    M = marginals.shape[1]
    
    u = jnp.zeros((N + 1, M))
    h = jnp.zeros((N, M))
    Delta = x_grid[None, :] - x_grid[:, None]
    
    tol = 1e-5 * jnp.sqrt(N + 1)
    
    def cond_fun(val):
        _, _, _, _, converged, k = val
        return (~converged) & (k < max_iter)

    def body_fun(val):
        u, h, u_old, h_old, _, k = val
        
        # ================================================================
        # STEP 1: h-update (martingale) using FULL kernel
        # ================================================================
        def update_h_t(t, h_arr):
            h_update = ops.solve_martingale_constraint(
                h_arr[t], u[t], u[t+1], C, Delta, epsilon
            )
            h_new_t = (1 - damping) * h_arr[t] + damping * h_update
            return h_arr.at[t].set(h_new_t)
        
        h_new = jax.lax.fori_loop(0, N, update_h_t, h)
        
        # ================================================================
        # STEP 2: u-update (marginals) Gauss-Seidel style
        # ================================================================
        def update_u_t(t, u_arr):
            u_update = jax.lax.cond(
                t == 0,
                # First time step
                lambda: ops.sinkhorn_u_update_first(
                    u_arr[0], u_arr[1], h_new[0], marginals[0], C, Delta, epsilon
                ),
                lambda: jax.lax.cond(
                    t == N,
                    # Last time step
                    lambda: ops.sinkhorn_u_update_last(
                        u_arr[N-1], u_arr[N], h_new[N-1], marginals[N], C, Delta, epsilon
                    ),
                    # Interior
                    lambda: ops.sinkhorn_u_update_interior(
                        u_arr[t-1], u_arr[t], u_arr[t+1], h_new[t-1], h_new[t],
                        marginals[t], C, Delta, epsilon
                    )
                )
            )
            u_new_t = (1 - damping) * u_arr[t] + damping * u_update
            return u_arr.at[t].set(u_new_t)
        
        u_new = jax.lax.fori_loop(0, N+1, update_u_t, u)
        
        # ================================================================
        # Convergence check
        # ================================================================
        u_diff = jnp.max(jnp.abs(u_new - u))
        h_diff = jnp.max(jnp.abs(h_new - h))
        converged = (u_diff < tol) & (h_diff < tol)
        
        return (u_new, h_new, u_new, h_new, converged, k + 1)

    init_val = (u, h, u, h, False, 0)
    u_final, h_final, _, _, conv, k = jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    return u_final, h_final, k
