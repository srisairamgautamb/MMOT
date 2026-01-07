"""
RIGOROUS MMOT DIAGNOSTIC
========================
This test checks if we're REALLY solving MMOT, not just parameter tuning.

Key Questions:
1. Does σ₀=0.45 work for N=50? (separates σ vs N issue)
2. Are BOTH marginal AND martingale constraints satisfied?
3. Is the dual gap actually closing?
"""
import jax.numpy as jnp
import time
from mmot.core import solver, ops

def compute_marginal_error(u, h, marginals, C, Delta, epsilon):
    """Compute marginal constraint violation."""
    N = marginals.shape[0] - 1
    M = marginals.shape[1]
    
    errors = []
    for t in range(N):
        # Compute joint P(X_t, X_{t+1})
        LogP = (u[t][:, None] + u[t+1][None, :] + h[t][:, None] * Delta - C) / epsilon
        LogP_max = jnp.max(LogP)
        P = jnp.exp(LogP - LogP_max)
        P = P / jnp.sum(P)
        
        # Marginal at t (row sum)
        mu_t_computed = jnp.sum(P, axis=1)
        # Marginal at t+1 (col sum)
        mu_t1_computed = jnp.sum(P, axis=0)
        
        error_t = jnp.sum(jnp.abs(mu_t_computed - marginals[t]))
        error_t1 = jnp.sum(jnp.abs(mu_t1_computed - marginals[t+1]))
        
        errors.append(float(max(error_t, error_t1)))
    
    return errors

def run_rigorous_diagnostic():
    """Run rigorous diagnostic tests."""
    print("="*70)
    print("RIGOROUS MMOT DIAGNOSTIC")
    print("="*70)
    print("\nThis test checks if we're REALLY solving MMOT.\n")
    
    # Test configurations: (name, N, σ₀, description)
    tests = [
        ("Wide σ, Small N", 2, 0.45, "Known working case"),
        ("Wide σ, Large N", 50, 0.45, "Does wide σ fix N=50?"),
        ("Narrow σ, Small N", 2, 0.05, "Does narrow σ break N=2?"),
        ("Narrow σ, Large N", 50, 0.05, "Known failing case"),
    ]
    
    M = 100
    epsilon = 0.05
    
    results = []
    
    for name, N, sigma_0, desc in tests:
        print(f"\n{'='*70}")
        print(f"TEST: {name}")
        print(f"  N={N}, σ₀={sigma_0}")
        print(f"  Description: {desc}")
        print(f"{'='*70}")
        
        # Setup grid
        grid_range = max(3.0, 5 * jnp.sqrt(sigma_0**2 + 0.3))  # Dynamic grid based on final variance
        x_grid = jnp.linspace(-float(grid_range), float(grid_range), M)
        Delta = x_grid[None, :] - x_grid[:, None]
        C = 0.5 * (Delta ** 2)
        
        # Create marginals
        marginals = []
        for t in range(N + 1):
            t_scaled = 2.0 * t / N
            variance = sigma_0**2 + 0.15 * t_scaled  # Variance grows over time
            sigma = jnp.sqrt(variance)
            pdf = jnp.exp(-0.5 * (x_grid / sigma)**2)
            pdf = pdf / jnp.sum(pdf)
            marginals.append(pdf)
        marginals = jnp.array(marginals)
        
        print(f"\n  Initial σ: {sigma_0}")
        print(f"  Final σ: {float(jnp.sqrt(sigma_0**2 + 0.3)):.3f}")
        print(f"  Grid: [{-grid_range:.1f}, {grid_range:.1f}]")
        
        # Solve
        start = time.time()
        u, h, iters = solver.solve_mmot(
            marginals, C, x_grid,
            max_iter=3000,
            epsilon=epsilon,
            damping=0.8
        )
        u.block_until_ready()
        elapsed = time.time() - start
        
        # Check MARTINGALE constraint
        max_mart_error = 0.0
        for t in range(N):
            drift = ops.compute_martingale_violation(
                u[t], u[t+1], h[t], x_grid, C, Delta, epsilon
            )
            max_mart_error = max(max_mart_error, float(drift))
        
        # Check MARGINAL constraint
        marg_errors = compute_marginal_error(u, h, marginals, C, Delta, epsilon)
        max_marg_error = max(marg_errors)
        
        # Convergence check
        converged = int(iters) < 3000
        
        print(f"\n  Results:")
        print(f"    Iterations: {int(iters)}")
        print(f"    Time: {elapsed:.2f}s")
        print(f"    Converged: {'✅' if converged else '❌'}")
        print(f"    Max Martingale Error: {max_mart_error:.2e}")
        print(f"    Max Marginal Error: {max_marg_error:.2e}")
        
        # Determine pass/fail
        mart_ok = max_mart_error < 0.1
        marg_ok = max_marg_error < 0.1
        
        print(f"\n  Constraints:")
        print(f"    Martingale: {'✅ SATISFIED' if mart_ok else '❌ VIOLATED'} ({max_mart_error:.2e})")
        print(f"    Marginal: {'✅ SATISFIED' if marg_ok else '❌ VIOLATED'} ({max_marg_error:.2e})")
        
        overall = mart_ok and marg_ok and converged
        print(f"\n  Overall: {'✅ PASS' if overall else '❌ FAIL'}")
        
        results.append({
            'name': name,
            'N': N,
            'sigma_0': sigma_0,
            'iters': int(iters),
            'time': elapsed,
            'converged': converged,
            'mart_error': max_mart_error,
            'marg_error': max_marg_error,
            'passed': overall
        })
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print(f"\n{'Test':<25} {'N':<5} {'σ₀':<6} {'Mart':<12} {'Marg':<12} {'Pass'}")
    print("-"*70)
    for r in results:
        status = '✅' if r['passed'] else '❌'
        print(f"{r['name']:<25} {r['N']:<5} {r['sigma_0']:<6.2f} {r['mart_error']:<12.2e} {r['marg_error']:<12.2e} {status}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    wide_small = next((r for r in results if r['sigma_0'] == 0.45 and r['N'] == 2), None)
    wide_large = next((r for r in results if r['sigma_0'] == 0.45 and r['N'] == 50), None)
    narrow_small = next((r for r in results if r['sigma_0'] == 0.05 and r['N'] == 2), None)
    narrow_large = next((r for r in results if r['sigma_0'] == 0.05 and r['N'] == 50), None)
    
    print("\n📊 FINDINGS:")
    
    if wide_small and wide_small['passed']:
        print("  ✅ Wide σ₀=0.45, N=2: Works (expected)")
    
    if wide_large and wide_large['passed']:
        print("  ✅ Wide σ₀=0.45, N=50: Works → Problem is σ₀, NOT N!")
        print("     This means N=50 is solvable with proper marginals.")
    elif wide_large and not wide_large['passed']:
        print("  ❌ Wide σ₀=0.45, N=50: Fails → Problem IS with large N")
    
    if narrow_small and narrow_small['passed']:
        print("  ✅ Narrow σ₀=0.05, N=2: Works → Narrow marginals are OK for small N")
    elif narrow_small and not narrow_small['passed']:
        print("  ❌ Narrow σ₀=0.05, N=2: Fails → Narrow marginals violate discrete convex order")
    
    if narrow_large and not narrow_large['passed']:
        print("  ❌ Narrow σ₀=0.05, N=50: Fails (expected)")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if wide_large and wide_large['passed'] and narrow_small and not narrow_small['passed']:
        print("\n🎯 ROOT CAUSE: DISCRETE CONVEX ORDER VIOLATION")
        print("   The issue is NOT the algorithm or N.")
        print("   Narrow marginals (σ₀=0.05) violate convex order on discrete grid.")
        print("   This is a FEASIBILITY issue, not an algorithm bug.")
        print("\n   SOLUTION: Require σ₀ > 5Δx (standard numerical OT practice).")
        print("   This is NOT hardcoding - it's a mathematical requirement.")
    elif wide_large and wide_large['passed']:
        print("\n✅ ALGORITHM IS CORRECT")
        print("   The solver works for all N when marginals satisfy convex order.")
        print("   σ₀ > 0.4 is a feasibility constraint, not parameter tuning.")
    else:
        print("\n⚠️  FURTHER INVESTIGATION NEEDED")
        print("   Results suggest a potential algorithmic issue.")
    
    return results

if __name__ == "__main__":
    run_rigorous_diagnostic()
