"""Find the N threshold where the solver breaks"""
import jax.numpy as jnp
import time
from mmot.core import solver, ops

def test_threshold():
    """Test N = 2, 5, 10, 15, 20, 30, 50"""
    
    M = 100
    T = 1.0
    x_grid = jnp.linspace(-1.5, 1.5, M)
    Delta = x_grid[None, :] - x_grid[:, None]
    C = 0.5 * (Delta ** 2)
    
    print("="*70)
    print("THRESHOLD TEST: Finding where N breaks")
    print("="*70)
    
    results = []
    
    for N in [2, 5, 10, 15, 20, 30, 50]:
        print(f"\n{'='*70}")
        print(f"Testing N={N}")
        print(f"{'='*70}")
        
        dt = T / N
        
        # Create marginals
        marginals = []
        for t in range(N + 1):
            time_val = t * dt
            if time_val == 0:
                sigma = 0.05
            else:
                sigma = jnp.sqrt(0.05**2 + 0.04 * time_val)
            pdf = jnp.exp(-0.5 * (x_grid / sigma)**2)
            marginals.append(pdf / jnp.sum(pdf))
        marginals = jnp.array(marginals)
        
        # Solve
        start = time.time()
        u, h, iters = solver.solve_mmot(
            marginals, C, x_grid,
            max_iter=2000,
            epsilon=0.05,
            damping=0.8
        )
        u.block_until_ready()
        exec_time = time.time() - start
        
        # Check drift at multiple points
        drifts = []
        for t in range(N):
            drift = ops.compute_martingale_violation(
                u[t], u[t+1], h[t], x_grid, C, Delta, 0.05
            )
            drifts.append(float(drift))
        
        max_drift = max(drifts)
        avg_drift = sum(drifts) / len(drifts)
        
        print(f"Iterations: {int(iters)}")
        print(f"Time: {exec_time:.2f}s")
        print(f"Max drift: {max_drift:.2e}")
        print(f"Avg drift: {avg_drift:.2e}")
        print(f"Drift at t=0: {drifts[0]:.2e}")
        print(f"Drift at t=N/2: {drifts[N//2]:.2e}")
        print(f"Drift at t=N-1: {drifts[N-1]:.2e}")
        
        success = max_drift < 0.1
        print(f"Status: {'✅ PASS' if success else '❌ FAIL'}")
        
        results.append({
            'N': N,
            'iters': int(iters),
            'time': exec_time,
            'max_drift': max_drift,
            'avg_drift': avg_drift,
            'success': success
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'N':<5} {'Iters':<8} {'Time':<8} {'Max Drift':<12} {'Status'}")
    print("-"*70)
    for r in results:
        status = '✅' if r['success'] else '❌'
        print(f"{r['N']:<5} {r['iters']:<8} {r['time']:<8.2f} {r['max_drift']:<12.2e} {status}")
    
    # Find threshold
    working = [r['N'] for r in results if r['success']]
    if working:
        print(f"\n✅ Algorithm works up to N={max(working)}")
    else:
        print("\n❌ Algorithm fails even at N=2")
    
    return results

if __name__ == "__main__":
    test_threshold()
