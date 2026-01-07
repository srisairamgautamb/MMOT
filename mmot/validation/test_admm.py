"""
Test ADMM Solver for MMOT
=========================
This tests the ADMM approach which has proven convergence for coupled constraints.
"""
import jax.numpy as jnp
import time
from mmot.core import solver_admm

def test_admm():
    """Test ADMM solver on basic and scale problems."""
    print("="*70)
    print("ADMM SOLVER TEST")
    print("="*70)
    
    # Test 1: N=2
    print("\n" + "-"*70)
    print("TEST 1: N=2, M=50")
    print("-"*70)
    
    N, M = 2, 50
    x_grid = jnp.linspace(-2.0, 2.0, M)
    Delta = x_grid[None, :] - x_grid[:, None]
    C = 0.5 * (Delta ** 2)
    
    # Create marginals
    marginals = []
    for t in range(N + 1):
        t_scaled = 2.0 * t / N
        sigma = jnp.sqrt(0.2 + 0.15 * t_scaled)
        pdf = jnp.exp(-0.5 * (x_grid / sigma)**2)
        pdf = pdf / jnp.sum(pdf)
        marginals.append(pdf)
    marginals = jnp.array(marginals)
    
    print(f"Initial σ: {jnp.sqrt(0.2):.3f}")
    print(f"Final σ: {jnp.sqrt(0.5):.3f}")
    
    # Solve with ADMM
    print("\nSolving with ADMM...")
    start = time.time()
    P, iters, converged = solver_admm.solve_mmot_admm(
        marginals, C, x_grid,
        max_iter=500,
        epsilon=0.1
    )
    jax.block_until_ready(P)
    elapsed = time.time() - start
    
    print(f"Iterations: {int(iters)}")
    print(f"Converged: {'✅' if converged else '❌'}")
    print(f"Time: {elapsed:.2f}s")
    
    # Check constraints
    max_mart_error = 0.0
    max_marg_error = 0.0
    
    for t in range(N):
        # Martingale check
        row_sums = jnp.sum(P[t], axis=1, keepdims=True) + 1e-10
        P_cond = P[t] / row_sums
        E_next = jnp.sum(P_cond * x_grid[None, :], axis=1)
        mart_error = jnp.max(jnp.abs(E_next - x_grid))
        max_mart_error = max(max_mart_error, float(mart_error))
        
        # Marginal check (row sums = μ_t)
        mu_t = jnp.sum(P[t], axis=1)
        marg_error_t = jnp.sum(jnp.abs(mu_t - marginals[t]))
        max_marg_error = max(max_marg_error, float(marg_error_t))
        
        # Marginal check (col sums = μ_{t+1})
        mu_t1 = jnp.sum(P[t], axis=0)
        marg_error_t1 = jnp.sum(jnp.abs(mu_t1 - marginals[t+1]))
        max_marg_error = max(max_marg_error, float(marg_error_t1))
    
    print(f"\nMax Martingale Error: {max_mart_error:.2e}")
    print(f"Max Marginal Error: {max_marg_error:.2e}")
    
    mart_ok = max_mart_error < 0.1
    marg_ok = max_marg_error < 0.1
    
    print(f"\nMartingale: {'✅ SATISFIED' if mart_ok else '❌ VIOLATED'}")
    print(f"Marginal: {'✅ SATISFIED' if marg_ok else '❌ VIOLATED'}")
    
    if mart_ok and marg_ok:
        print("\n✅ TEST 1 PASSED")
    else:
        print("\n❌ TEST 1 FAILED")
    
    # Test 2: N=10 (medium scale)
    print("\n" + "-"*70)
    print("TEST 2: N=10, M=50")
    print("-"*70)
    
    N, M = 10, 50
    
    marginals = []
    for t in range(N + 1):
        t_scaled = 2.0 * t / N
        sigma = jnp.sqrt(0.2 + 0.15 * t_scaled)
        pdf = jnp.exp(-0.5 * (x_grid / sigma)**2)
        pdf = pdf / jnp.sum(pdf)
        marginals.append(pdf)
    marginals = jnp.array(marginals)
    
    print("\nSolving with ADMM...")
    start = time.time()
    P, iters, converged = solver_admm.solve_mmot_admm(
        marginals, C, x_grid,
        max_iter=500,
        epsilon=0.1
    )
    jax.block_until_ready(P)
    elapsed = time.time() - start
    
    print(f"Iterations: {int(iters)}")
    print(f"Converged: {'✅' if converged else '❌'}")
    print(f"Time: {elapsed:.2f}s")
    
    # Check constraints
    max_mart_error = 0.0
    max_marg_error = 0.0
    
    for t in range(N):
        row_sums = jnp.sum(P[t], axis=1, keepdims=True) + 1e-10
        P_cond = P[t] / row_sums
        E_next = jnp.sum(P_cond * x_grid[None, :], axis=1)
        mart_error = jnp.max(jnp.abs(E_next - x_grid))
        max_mart_error = max(max_mart_error, float(mart_error))
        
        mu_t = jnp.sum(P[t], axis=1)
        marg_error_t = jnp.sum(jnp.abs(mu_t - marginals[t]))
        max_marg_error = max(max_marg_error, float(marg_error_t))
    
    print(f"\nMax Martingale Error: {max_mart_error:.2e}")
    print(f"Max Marginal Error: {max_marg_error:.2e}")
    
    mart_ok = max_mart_error < 0.1
    marg_ok = max_marg_error < 0.1
    
    print(f"\nMartingale: {'✅ SATISFIED' if mart_ok else '❌ VIOLATED'}")
    print(f"Marginal: {'✅ SATISFIED' if marg_ok else '❌ VIOLATED'}")
    
    if mart_ok and marg_ok:
        print("\n✅ TEST 2 PASSED")
    else:
        print("\n❌ TEST 2 FAILED")
    
    print("\n" + "="*70)
    print("ADMM TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    import jax
    test_admm()
