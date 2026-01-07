#!/usr/bin/env python3
"""
test_adaptive_multiple.py
=========================
Test the adaptive solver on multiple instances to verify robustness.
"""

import numpy as np
import sys
import time

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
from solve_mmot_adaptive import solve_mmot_adaptive


def generate_random_gbm_marginals(M, N, grid, sigma=None):
    """Generate random GBM marginals."""
    if sigma is None:
        sigma = np.random.uniform(0.1, 0.5)
    
    T = np.random.uniform(0.1, 0.5)
    dt = T / N
    
    marginals = np.zeros((N+1, M), dtype=np.float32)
    for t in range(N+1):
        if t == 0:
            center_idx = np.argmin(np.abs(grid - 1.0))
            marginals[t, center_idx] = 1.0
        else:
            tau = t * dt
            log_std = max(sigma * np.sqrt(tau), 0.01)
            log_m = np.log(np.clip(grid, 0.01, None))
            pdf = np.exp(-0.5 * (log_m / log_std)**2) / (grid * log_std * np.sqrt(2*np.pi))
            pdf = np.clip(pdf, 0, None)
            if pdf.sum() > 0:
                marginals[t] = pdf / pdf.sum()
            else:
                center_idx = np.argmin(np.abs(grid - 1.0))
                marginals[t, center_idx] = 1.0
    
    return marginals, sigma, T


def test_multiple_instances(n_instances=10):
    """Test on multiple random instances."""
    print("\n" + "="*70)
    print("TESTING ADAPTIVE SOLVER ON MULTIPLE INSTANCES")
    print("="*70)
    
    M = 150
    grid = np.linspace(0.5, 1.5, M).astype(np.float32)
    
    drifts = []
    times = []
    
    print(f"\n{'Instance':<10} {'N':<5} {'σ':<8} {'Drift':<15} {'Time':<10} {'Status'}")
    print("-"*60)
    
    for i in range(n_instances):
        N = np.random.choice([2, 3, 4, 5])
        marginals, sigma, T = generate_random_gbm_marginals(M, N, grid)
        
        # Solve with adaptive epsilon
        result = solve_mmot_adaptive(marginals, grid, target_drift=1e-4, verbose=False)
        
        drifts.append(result['drift'])
        times.append(result['total_time'])
        
        status = "✅" if result['drift'] < 0.0001 else ("⚠️" if result['drift'] < 0.01 else "❌")
        print(f"{i+1:<10} {N:<5} {sigma:<8.3f} {result['drift']:<15.8f} {result['total_time']:<10.1f}s {status}")
    
    # Summary
    print("-"*60)
    print(f"\nSUMMARY ({n_instances} instances):")
    print(f"  Mean drift:     {np.mean(drifts):.8f}")
    print(f"  Max drift:      {max(drifts):.8f}")
    print(f"  Min drift:      {min(drifts):.8f}")
    print(f"  Mean time:      {np.mean(times):.1f}s")
    print(f"  Total time:     {sum(times):.1f}s")
    
    excellent_rate = sum(1 for d in drifts if d < 0.0001) / len(drifts) * 100
    acceptable_rate = sum(1 for d in drifts if d < 0.01) / len(drifts) * 100
    
    print(f"\n  < 0.0001 (EXCELLENT): {excellent_rate:.0f}%")
    print(f"  < 0.01 (ACCEPTABLE):  {acceptable_rate:.0f}%")
    
    if excellent_rate >= 90:
        print(f"\n🎉 EXCELLENT! {excellent_rate:.0f}% of instances achieve drift < 0.0001")
    elif acceptable_rate >= 90:
        print(f"\n✅ GOOD! {acceptable_rate:.0f}% of instances achieve drift < 0.01")
    else:
        print(f"\n⚠️ NEEDS IMPROVEMENT")
    
    print("="*70)
    
    return drifts, times


if __name__ == '__main__':
    test_multiple_instances(n_instances=10)
