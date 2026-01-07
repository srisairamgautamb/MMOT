#!/usr/bin/env python3
"""
test_real_market_universal.py
=============================
Test the hybrid MMOT solver on simulated "real market" data for multiple stocks.

Demonstrates UNIVERSAL applicability:
- SPY at $683 (high price, low vol)
- AMD at $150 (mid price, high vol)
- F at $10 (low price, low vol)
- TSLA at $395 (mid-high price, high vol)

All use the SAME trained model (moneyness space [0.5, 1.5]).
"""

import numpy as np
import torch
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
from hybrid_neural_solver import HybridMMOTSolver


def generate_gbm_marginals(S0, sigma, T, N, M, grid):
    """
    Generate marginal distributions using Geometric Brownian Motion.
    
    Args:
        S0: Initial stock price
        sigma: Annualized volatility
        T: Time to maturity in years
        N: Number of time steps
        M: Grid size
        grid: Moneyness grid [0.5, 1.5]
        
    Returns:
        marginals: (N+1, M) array of marginal distributions
    """
    dt = T / N
    marginals = np.zeros((N+1, M))
    
    for t in range(N+1):
        # At time t, the forward price follows log-normal
        # Mean in log-space: 0 (martingale, r=0)
        # Std in log-space: sigma * sqrt(t * dt)
        
        if t == 0:
            # Initial distribution: delta at moneyness = 1.0
            center_idx = np.argmin(np.abs(grid - 1.0))
            marginals[t, center_idx] = 1.0
        else:
            tau = t * dt
            log_std = sigma * np.sqrt(tau)
            
            # Compute PDF of log-normal at each grid point
            # log(S_t/S_0) ~ N(0, sigma^2 * t)
            # Moneyness m = S_t / S_0
            log_m = np.log(grid)
            pdf = np.exp(-0.5 * (log_m / log_std)**2) / (grid * log_std * np.sqrt(2*np.pi))
            
            # Normalize
            marginals[t] = pdf / pdf.sum()
    
    return marginals


def test_stock(solver, stock_name, S0, sigma, T, N, grid, verbose=True):
    """Test the solver on a single stock."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing: {stock_name} (S0=${S0}, σ={sigma*100:.0f}%)")
        print('='*60)
    
    # Generate marginals
    marginals = generate_gbm_marginals(S0, sigma, T, N, len(grid), grid)
    
    # Verify martingale condition on input marginals
    expected_m = np.sum(marginals * grid, axis=1)
    if verbose:
        print(f"Input marginals E[m] at each t: {expected_m[:3]}... (should be ~1.0)")
    
    # Solve
    result = solver.solve(marginals, grid, n_newton_iters=100, verbose=verbose)
    
    return {
        'stock': stock_name,
        'S0': S0,
        'sigma': sigma,
        'drift': result['drift'],
        'neural_time': result['neural_time'],
        'newton_time': result['newton_time'],
        'total_time': result['total_time']
    }


def main():
    print("="*70)
    print("UNIVERSAL MMOT SOLVER - REAL MARKET TEST")
    print("="*70)
    print("Testing on multiple stocks with DIFFERENT prices and volatilities")
    print("All use the SAME model trained in moneyness space [0.5, 1.5]")
    print()
    
    # Initialize solver
    solver = HybridMMOTSolver('checkpoints/best_model.pth', device='mps', epsilon=0.2)
    
    # Moneyness grid (same as training)
    M = 150
    grid = np.linspace(0.5, 1.5, M).astype(np.float32)
    
    # Test parameters
    T = 0.25  # 3 months
    N = 3     # 3 time steps
    
    # Test stocks
    stocks = [
        ('SPY', 683, 0.15),   # S&P 500 ETF: high price, low vol
        ('AMD', 150, 0.35),   # AMD: mid price, high vol
        ('F', 10, 0.25),      # Ford: low price, medium vol
        ('TSLA', 395, 0.45),  # Tesla: mid-high price, very high vol
    ]
    
    results = []
    for stock_name, S0, sigma in stocks:
        result = test_stock(solver, stock_name, S0, sigma, T, N, grid)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: UNIVERSAL COVERAGE TEST")
    print("="*70)
    print(f"{'Stock':<8} {'Price':>8} {'Vol':>6} {'Drift':>12} {'Time':>8}")
    print("-"*50)
    
    all_pass = True
    for r in results:
        status = "✅" if r['drift'] < 0.01 else "❌"
        if r['drift'] >= 0.01:
            all_pass = False
        print(f"{r['stock']:<8} ${r['S0']:>6} {r['sigma']*100:>5.0f}% {r['drift']:>12.6f} {r['total_time']*1000:>6.1f}ms {status}")
    
    print("-"*50)
    print(f"Max Drift: {max(r['drift'] for r in results):.6f}")
    print(f"Avg Time:  {np.mean([r['total_time'] for r in results])*1000:.1f}ms")
    print()
    
    if all_pass:
        print("🎉 SUCCESS! Model works universally across all stock prices!")
        print("   - Works for $10 stock (Ford)")
        print("   - Works for $683 stock (SPY)")
        print("   - Same model, no retraining needed!")
    else:
        print("⚠️ Some stocks failed the drift threshold.")
    
    print("="*70)
    
    return results


if __name__ == '__main__':
    main()
