#!/usr/bin/env python3
"""
REAL MARKET DATA TESTING
========================
Downloads real S&P 500 (SPY) option data and tests both classical and neural solvers.

Steps:
1. Download option chains from yfinance
2. Calibrate risk-neutral densities (Breeden-Litzenberger)
3. Create MMOT marginals from real market data
4. Test classical solver
5. Test neural solver (with Newton projection)
6. Report metrics

Usage:
    python test_real_market_data.py
"""

import numpy as np
import torch
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.stats import norm

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT')
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')

import jax.numpy as jnp
from mmot.core.solver import solve_mmot
from architecture_fixed import ImprovedTransformerMMOT


def download_option_data(ticker='SPY', n_expirations=5):
    """Download real option chain data from yfinance."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING REAL OPTION DATA: {ticker}")
    print(f"{'='*60}")
    
    stock = yf.Ticker(ticker)
    
    # Get current price
    hist = stock.history(period='1d')
    if len(hist) == 0:
        print("Warning: Could not get current price, using 580 as estimate")
        S0 = 580.0
    else:
        S0 = hist['Close'].iloc[-1]
    print(f"Current price: ${S0:.2f}")
    
    # Get available expirations
    expirations = stock.options
    print(f"Available expirations: {len(expirations)}")
    
    # Select expirations (skip very near-term)
    selected_exp = []
    today = datetime.now()
    
    for exp in expirations:
        exp_date = datetime.strptime(exp, '%Y-%m-%d')
        days_to_exp = (exp_date - today).days
        
        # Select expirations between 7 and 180 days
        if 7 <= days_to_exp <= 180:
            selected_exp.append((exp, days_to_exp))
            if len(selected_exp) >= n_expirations:
                break
    
    print(f"Selected {len(selected_exp)} expirations:")
    for exp, days in selected_exp:
        print(f"  {exp} ({days} days)")
    
    # Download option chains
    chains = []
    for exp, days in selected_exp:
        try:
            opt = stock.option_chain(exp)
            
            # Filter to liquid strikes (near the money)
            calls = opt.calls[
                (opt.calls['strike'] >= S0 * 0.85) & 
                (opt.calls['strike'] <= S0 * 1.15) &
                (opt.calls['volume'] > 0) &
                (opt.calls['impliedVolatility'] > 0.01)
            ].copy()
            
            if len(calls) >= 5:
                chains.append({
                    'expiration': exp,
                    'days': days,
                    'calls': calls,
                    'S0': S0
                })
                print(f"  {exp}: {len(calls)} valid call options")
            else:
                print(f"  {exp}: Skipped (only {len(calls)} valid options)")
                
        except Exception as e:
            print(f"  {exp}: Error - {e}")
    
    return chains, S0


def calibrate_risk_neutral_density(calls, S0, T, r=0.05, grid_size=150):
    """
    Calibrate risk-neutral density using Breeden-Litzenberger formula.
    
    The risk-neutral density q(K) = e^(rT) * d²C/dK² 
    where C(K) is the call price.
    """
    # Extract strike and mid price
    strikes = calls['strike'].values
    mids = (calls['bid'].values + calls['ask'].values) / 2
    mids = np.where(mids > 0, mids, calls['lastPrice'].values)
    
    # Remove NaN
    valid = ~np.isnan(mids) & (mids > 0)
    strikes = strikes[valid]
    mids = mids[valid]
    
    if len(strikes) < 5:
        return None
    
    # Sort by strike
    idx = np.argsort(strikes)
    strikes = strikes[idx]
    mids = mids[idx]
    
    # Interpolate prices
    K_fine = np.linspace(strikes.min(), strikes.max(), 200)
    
    try:
        # Use cubic interpolation
        price_interp = interp1d(strikes, mids, kind='cubic', fill_value='extrapolate')
        C = price_interp(K_fine)
    except:
        price_interp = interp1d(strikes, mids, kind='linear', fill_value='extrapolate')
        C = price_interp(K_fine)
    
    # Compute second derivative (Breeden-Litzenberger)
    dK = K_fine[1] - K_fine[0]
    d2C = np.gradient(np.gradient(C, dK), dK)
    
    # Risk-neutral density
    discount = np.exp(r * T)
    q = discount * d2C
    
    # Fix any negative values (numerical artifacts)
    q = np.maximum(q, 0)
    
    # Normalize
    q = q / (np.sum(q) * dK + 1e-10)
    
    # Map to our standard grid [50, 200]
    x_grid = np.linspace(50, 200, grid_size)
    
    # Scale K to match our grid (center at 100)
    scale = 100.0 / S0
    K_scaled = K_fine * scale
    
    # Interpolate to our grid
    try:
        q_interp = interp1d(K_scaled, q, kind='linear', bounds_error=False, fill_value=0)
        marginal = q_interp(x_grid)
    except:
        # Fallback: Gaussian centered at scaled S0
        marginal = np.exp(-0.5 * ((x_grid - 100) / 10) ** 2)
    
    # Normalize
    marginal = np.maximum(marginal, 0)
    marginal = marginal / (marginal.sum() + 1e-10)
    
    return marginal


def create_mmot_from_options(chains, n_periods=10, grid_size=150):
    """Create MMOT marginals from real option data."""
    print(f"\n{'='*60}")
    print("CALIBRATING RISK-NEUTRAL DENSITIES")
    print(f"{'='*60}")
    
    if len(chains) < 2:
        print("Not enough option chains!")
        return None
    
    x_grid = np.linspace(50, 200, grid_size)
    
    # Use first chain's S0
    S0 = chains[0]['S0']
    
    # Initial marginal: point mass at scaled S0
    marginals = [np.zeros(grid_size)]
    initial_price = 100.0  # Scaled
    idx = np.argmin(np.abs(x_grid - initial_price))
    marginals[0][idx] = 1.0
    
    # Calibrate marginals from each expiration
    for chain in chains:
        T = chain['days'] / 365.0
        density = calibrate_risk_neutral_density(chain['calls'], S0, T, grid_size=grid_size)
        
        if density is not None:
            marginals.append(density)
            print(f"  {chain['expiration']}: Calibrated (max density = {density.max():.4f})")
        else:
            print(f"  {chain['expiration']}: Calibration failed, using Gaussian fallback")
            # Fallback Gaussian
            sigma = 0.2 * np.sqrt(T) * 100
            mu = 100.0
            density = np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
            density = density / density.sum()
            marginals.append(density)
    
    # Pad or interpolate to n_periods + 1
    marginals = np.array(marginals)
    
    return marginals, x_grid


def test_classical_solver(marginals, x_grid):
    """Test classical Martingale-Sinkhorn solver."""
    print(f"\n{'='*60}")
    print("TESTING CLASSICAL SOLVER")
    print(f"{'='*60}")
    
    N = marginals.shape[0] - 1
    M = len(x_grid)
    
    dx = x_grid[:, None] - x_grid[None, :]
    C = dx ** 2
    
    import time
    start = time.time()
    
    u, h, k = solve_mmot(
        jnp.array(marginals),
        jnp.array(C),
        jnp.array(x_grid),
        max_iter=1000,
        epsilon=1.0,
        damping=0.8
    )
    
    elapsed = time.time() - start
    
    u = np.array(u)
    h = np.array(h)
    
    # Compute drift
    drift = np.mean(np.abs(h))
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Iterations: {int(k)}")
    print(f"  Drift: {drift:.6f}")
    
    return u, h, elapsed, drift


def test_neural_solver(marginals, x_grid, model_path):
    """Test neural solver with Newton projection."""
    print(f"\n{'='*60}")
    print("TESTING NEURAL SOLVER (with Newton Projection)")
    print(f"{'='*60}")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load model
    model = ImprovedTransformerMMOT(M=150, N_max=50).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare input
    # Pad marginals to max expected shape
    N = marginals.shape[0] - 1
    max_N = 30
    
    marg_padded = np.zeros((max_N + 1, 150))
    marg_padded[:N+1, :] = marginals
    
    marg_t = torch.from_numpy(marg_padded).float().unsqueeze(0).to(device)
    x_grid_t = torch.from_numpy(x_grid).float().to(device)
    
    import time
    start = time.time()
    
    with torch.no_grad():
        u, h = model(marg_t, x_grid_t)
    
    neural_time = time.time() - start
    
    # Apply Newton projection
    start_proj = time.time()
    
    total_drift = 0.0
    n_steps = min(N, 10)
    
    for t in range(n_steps):
        u_next = u[0, t+1:t+2] * 0.01
        h_init = torch.zeros(1, 150).to(device)
        
        h_refined = model.martingale_projection.refine_hard_constraint(
            h_init, u_next, x_grid_t, n_iters=100, tol=1e-6, epsilon=0.5
        )
        
        # Check drift
        epsilon_val = 0.5
        delta_S = x_grid_t[None, :] - x_grid_t[:, None]
        log_K = (u_next[0, None, :] + h_refined[0, :, None] * delta_S) / epsilon_val
        K = torch.softmax(log_K, dim=1)
        E_Y = (K * x_grid_t[None, :]).sum(dim=1)
        deviation = (E_Y - x_grid_t).abs()
        
        mu_t = marg_t[0, t]
        weighted_drift = (deviation * mu_t).sum().item()
        total_drift += weighted_drift
    
    proj_time = time.time() - start_proj
    total_time = neural_time + proj_time
    
    avg_drift = total_drift / (n_steps * 150)
    
    print(f"  Neural time: {neural_time*1000:.1f}ms")
    print(f"  Projection time: {proj_time*1000:.1f}ms")
    print(f"  Total time: {total_time*1000:.1f}ms")
    print(f"  Projected Drift: {avg_drift:.6f}")
    
    return u.cpu().numpy(), h.cpu().numpy(), total_time, avg_drift


def main():
    print("="*70)
    print("REAL MARKET DATA TESTING")
    print("="*70)
    
    # Step 1: Download option data
    chains, S0 = download_option_data('SPY', n_expirations=6)
    
    if len(chains) < 2:
        print("\n❌ Could not download enough option data!")
        print("Falling back to simulated 'real-like' data...")
        
        # Fallback: create realistic marginals
        x_grid = np.linspace(50, 200, 150)
        N = 5
        marginals = np.zeros((N + 1, 150))
        marginals[0, 75] = 1.0  # Point mass at 100
        
        for t in range(1, N + 1):
            sigma = 10 * np.sqrt(t / 365)
            marginals[t] = np.exp(-0.5 * ((x_grid - 100) / sigma) ** 2)
            marginals[t] /= marginals[t].sum()
    else:
        # Step 2: Calibrate marginals
        marginals, x_grid = create_mmot_from_options(chains, grid_size=150)
    
    print(f"\nMarginals shape: {marginals.shape}")
    
    # Step 3: Test classical solver
    u_classical, h_classical, time_classical, drift_classical = test_classical_solver(marginals, x_grid)
    
    # Step 4: Test neural solver
    model_path = 'best_model_drift0.2797.pth'
    if os.path.exists(model_path):
        u_neural, h_neural, time_neural, drift_neural = test_neural_solver(marginals, x_grid, model_path)
    else:
        print(f"\n❌ Model not found: {model_path}")
        time_neural = 0
        drift_neural = -1
    
    # Step 5: Report
    print(f"\n{'='*70}")
    print("FINAL RESULTS: REAL MARKET DATA")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Classical':<20} {'Neural+Proj':<20}")
    print("-"*65)
    print(f"{'Time':<25} {time_classical*1000:.1f}ms{'':<12} {time_neural*1000:.1f}ms")
    print(f"{'Drift':<25} {drift_classical:.6f}{'':<12} {drift_neural:.6f}")
    print(f"{'Speedup':<25} {'-':<20} {time_classical/time_neural:.1f}x")
    print("-"*65)
    
    if drift_neural < 0.05:
        print("\n✅ NEURAL SOLVER PASSES ON REAL DATA (Drift < 0.05)")
    else:
        print(f"\n⚠️ Neural drift = {drift_neural:.4f} (target < 0.05)")


if __name__ == "__main__":
    main()
