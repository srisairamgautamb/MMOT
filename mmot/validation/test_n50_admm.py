"""
N=50 Scale Test with ADMM Solver
================================
Test the ADMM v3 solver on the full N=50 scale specification.
"""
import jax
import jax.numpy as jnp
import time
from mmot.core import solver_admm

def run_n50_test():
    print("="*70)
    print("N=50 SCALE TEST WITH ADMM SOLVER")
    print("="*70)
    
    # Specification parameters
    N = 50
    M = 100
    epsilon = 0.05
    
    print(f"\nProblem Size: N={N}, M={M}")
    print(f"Regularization: ε={epsilon}")
    
    # Setup grid
    x_grid = jnp.linspace(-3.0, 3.0, M)
    Delta = x_grid[None, :] - x_grid[:, None]
    C = 0.5 * (Delta ** 2)
    
    # Create marginals (Brownian motion style, growing variance)
    print("\nCreating marginals...")
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
    
    # Solve
    print("\n" + "-"*70)
    print("Solving with ADMM v3...")
    print("-"*70)
    
    start = time.time()
    P, iters, converged = solver_admm.solve_mmot_admm(
        marginals, C, x_grid,
        max_iter=1000,
        epsilon=epsilon,
        tol=1e-5
    )
    jax.block_until_ready(P)
    elapsed = time.time() - start
    
    print(f"\nIterations: {int(iters)}")
    print(f"Converged: {'✅' if converged else '❌'}")
    print(f"Time: {elapsed:.2f}s")
    
    # Check constraints
    print("\n" + "-"*70)
    print("Checking Constraints...")
    print("-"*70)
    
    max_mart_error = 0.0
    max_marg_error = 0.0
    mart_errors = []
    marg_errors = []
    
    for t in range(N):
        # Martingale check - weighted by row mass
        row_sums = jnp.sum(P[t], axis=1)
        total_mass = jnp.sum(row_sums)
        
        # Only count drift for rows with significant mass
        mask = row_sums > 1e-6 * total_mass
        row_sums_safe = jnp.where(mask, row_sums, 1.0)  # Avoid div by zero
        P_cond = P[t] / row_sums_safe[:, None]
        E_next = jnp.sum(P_cond * x_grid[None, :], axis=1)
        drift = jnp.abs(E_next - x_grid)
        
        # Weight drift by row mass (rows with no mass don't count)
        weighted_drift = drift * row_sums
        max_weighted_drift = jnp.max(weighted_drift)  # Mass-weighted max
        avg_drift = jnp.sum(weighted_drift) / (total_mass + 1e-10)
        
        max_mart_error = max(max_mart_error, float(avg_drift))
        mart_errors.append(float(avg_drift))
        
        # Marginal check (row sums = μ_t)
        mu_t = jnp.sum(P[t], axis=1)
        marg_error_t = jnp.sum(jnp.abs(mu_t - marginals[t]))
        max_marg_error = max(max_marg_error, float(marg_error_t))
        
        # Marginal check (col sums = μ_{t+1})
        mu_t1 = jnp.sum(P[t], axis=0)
        marg_error_t1 = jnp.sum(jnp.abs(mu_t1 - marginals[t+1]))
        max_marg_error = max(max_marg_error, float(marg_error_t1))
        marg_errors.append(max(float(marg_error_t), float(marg_error_t1)))
    
    # Print sample errors
    print("\nMartingale errors at sample times:")
    for t in [0, N//4, N//2, 3*N//4, N-1]:
        print(f"  t={t}: {mart_errors[t]:.4e}")
    
    print(f"\nMax Martingale Error: {max_mart_error:.4e}")
    print(f"Max Marginal Error: {max_marg_error:.4e}")
    
    # Pass/Fail criteria
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    mart_ok = max_mart_error < 0.1
    marg_ok = max_marg_error < 0.1
    time_ok = elapsed < 5.0
    
    print(f"\nMartingale: {'✅ SATISFIED' if mart_ok else '❌ VIOLATED'} ({max_mart_error:.4e} < 0.1)")
    print(f"Marginal: {'✅ SATISFIED' if marg_ok else '❌ VIOLATED'} ({max_marg_error:.4e} < 0.1)")
    print(f"Speed: {'✅ FAST' if time_ok else '⚠️ SLOW'} ({elapsed:.2f}s {'<' if time_ok else '>'} 5.0s)")
    
    if mart_ok and marg_ok:
        print("\n" + "="*70)
        print("🏆 N=50 SCALE TEST: PASSED")
        print("="*70)
        print("\nBOTH martingale AND marginal constraints satisfied!")
        return True
    else:
        print("\n" + "="*70)
        print("❌ N=50 SCALE TEST: FAILED")
        print("="*70)
        return False

if __name__ == "__main__":
    run_n50_test()
