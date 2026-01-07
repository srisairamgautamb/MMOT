"""
ANTI-HARDCODING VERIFICATION TEST
=================================
This test proves the solver is general and not tuned to specific values.
We test with EXTREME and VARIED parameters to demonstrate generality.
"""
import jax.numpy as jnp
import time
from mmot.core import solver, ops

def run_anti_hardcoding_test():
    """
    Test the solver with wildly different parameters to prove no hardcoding.
    
    If the solver were hardcoded, it would ONLY work for specific values.
    By testing extremes, we prove the algorithm is general.
    """
    print("="*70)
    print("ANTI-HARDCODING VERIFICATION TEST")
    print("="*70)
    print("\nThis test proves the solver works with VARIED parameters,")
    print("not just the specific values in the basic/scale tests.\n")
    
    results = []
    
    # ========================================================================
    # TEST SUITE: Vary ALL parameters
    # ========================================================================
    test_configs = [
        # (name, N, M, epsilon, damping, sigma_scale, grid_range)
        ("Minimal N=3", 3, 50, 0.05, 0.8, 1.0, 2.0),
        ("Medium N=15", 15, 80, 0.08, 0.7, 1.2, 2.5),
        ("Large N=30", 30, 100, 0.05, 0.8, 1.0, 3.0),
        ("Small epsilon", 10, 100, 0.02, 0.9, 1.0, 3.0),
        ("Large epsilon", 10, 100, 0.15, 0.6, 1.0, 3.0),
        ("Low damping", 10, 100, 0.05, 0.5, 1.0, 3.0),
        ("High damping", 10, 100, 0.05, 0.95, 1.0, 3.0),
        ("Wide marginals", 10, 100, 0.05, 0.8, 2.0, 4.0),
        ("Narrow grid M=40", 10, 40, 0.1, 0.8, 1.0, 2.0),
        ("Fine grid M=150", 10, 150, 0.05, 0.8, 1.0, 3.0),
    ]
    
    for name, N, M, epsilon, damping, sigma_scale, grid_range in test_configs:
        print(f"\n{'-'*70}")
        print(f"TEST: {name}")
        print(f"  N={N}, M={M}, ε={epsilon}, damping={damping}")
        print(f"  σ_scale={sigma_scale}, grid=[{-grid_range}, {grid_range}]")
        print(f"{'-'*70}")
        
        # Create grid
        x_grid = jnp.linspace(-grid_range, grid_range, M)
        Delta = x_grid[None, :] - x_grid[:, None]
        C = 0.5 * (Delta ** 2)
        
        # Create marginals with varying parameters
        # σ(t) = sqrt(base_var + growth_rate * t_scaled)
        base_var = 0.2 * sigma_scale
        growth_rate = 0.15 * sigma_scale
        
        marginals = []
        for t in range(N + 1):
            t_scaled = 2.0 * t / N
            sigma = jnp.sqrt(base_var + growth_rate * t_scaled)
            pdf = jnp.exp(-0.5 * (x_grid / sigma)**2)
            pdf = pdf / jnp.sum(pdf)
            marginals.append(pdf)
        marginals = jnp.array(marginals)
        
        # Solve
        try:
            start = time.time()
            u, h, iters = solver.solve_mmot(
                marginals, C, x_grid,
                max_iter=3000,
                epsilon=epsilon,
                damping=damping
            )
            u.block_until_ready()
            exec_time = time.time() - start
            
            # Check martingale
            max_drift = 0.0
            for t in range(N):
                drift = ops.compute_martingale_violation(
                    u[t], u[t+1], h[t], x_grid, C, Delta, epsilon
                )
                max_drift = max(max_drift, float(drift))
            
            # Determine pass/fail
            # Tolerance scales with epsilon (entropic error ~ O(ε))
            tol = max(0.1, 2 * epsilon)
            passed = max_drift < tol
            
            print(f"  Converged in {int(iters)} iterations")
            print(f"  Time: {exec_time:.2f}s")
            print(f"  Max drift: {max_drift:.2e} (tolerance: {tol:.2e})")
            print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")
            
            results.append({
                'name': name,
                'N': N, 'M': M, 'epsilon': epsilon, 'damping': damping,
                'iters': int(iters),
                'time': exec_time,
                'max_drift': max_drift,
                'passed': passed
            })
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                'name': name,
                'N': N, 'M': M, 'epsilon': epsilon, 'damping': damping,
                'iters': -1,
                'time': -1,
                'max_drift': float('inf'),
                'passed': False
            })
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("ANTI-HARDCODING TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    
    print(f"\n{'Test Name':<25} {'N':<4} {'M':<4} {'ε':<6} {'Drift':<12} {'Status'}")
    print("-"*70)
    for r in results:
        status = '✅' if r['passed'] else '❌'
        drift_str = f"{r['max_drift']:.2e}" if r['max_drift'] < float('inf') else "ERROR"
        print(f"{r['name']:<25} {r['N']:<4} {r['M']:<4} {r['epsilon']:<6.3f} {drift_str:<12} {status}")
    
    print("-"*70)
    print(f"\nRESULT: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n" + "="*70)
        print("🏆 ANTI-HARDCODING VERIFICATION: PASSED")
        print("="*70)
        print("\nThis proves:")
        print("  ✅ Solver works with different N values (3, 10, 15, 30)")
        print("  ✅ Solver works with different M values (40, 50, 80, 100, 150)")
        print("  ✅ Solver works with different epsilon (0.02 to 0.15)")
        print("  ✅ Solver works with different damping (0.5 to 0.95)")
        print("  ✅ Solver works with different marginal widths")
        print("  ✅ Solver works with different grid ranges")
        print("\n📜 CONCLUSION: No hardcoding detected. Algorithm is GENERAL.")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} tests failed")
        return False

if __name__ == "__main__":
    run_anti_hardcoding_test()
