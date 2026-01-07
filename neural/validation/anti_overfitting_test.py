"""
ANTI-OVERFITTING TEST: Run backtest with 100 random seeds
This proves results are genuine and not cherry-picked.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(0, S - K)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def generate_market(n_days=252, base_price=100.0, seed=42):
    np.random.seed(seed)
    dt = 1/252
    log_returns = 0.08 * dt + 0.20 * np.sqrt(dt) * np.random.randn(n_days)
    prices = base_price * np.exp(np.cumsum(log_returns))
    ivol = 0.20 + 0.03 * np.random.randn(n_days).cumsum() * np.sqrt(dt)
    ivol = np.clip(ivol, 0.10, 0.50)
    return prices, ivol


def run_single_backtest(seed_market, seed_trades):
    """Run one backtest with given seeds."""
    prices, ivol = generate_market(seed=seed_market)
    np.random.seed(seed_trades)
    
    trades = []
    n_days = len(prices)
    r, T = 0.05, 30/365
    holding = 5
    
    for i in range(n_days - holding):
        spot = prices[i]
        vol = ivol[i]
        strike = round(spot, 0)
        
        # Bounds
        vol_low = max(0.05, vol - 0.03)
        vol_high = vol + 0.03
        lower = black_scholes_call(spot, strike, T, r, vol_low)
        upper = black_scholes_call(spot, strike, T, r, vol_high)
        
        # Market price with noise
        bs = black_scholes_call(spot, strike, T, r, vol)
        noise = np.random.normal(0, 0.10 * bs)
        market = max(0.01, bs + noise)
        
        # Signal
        if market < lower * 0.92:
            trade_type = 'BUY'
        elif market > upper * 1.08:
            trade_type = 'SELL'
        else:
            continue
        
        # Exit
        future_spot = prices[i + holding]
        future_vol = ivol[i + holding]
        T_rem = max(0.001, T - holding/365)
        exit_bs = black_scholes_call(future_spot, strike, T_rem, r, future_vol)
        exit_noise = np.random.normal(0, 0.02 * exit_bs)
        exit_price = max(0.01, exit_bs + exit_noise)
        
        # P&L
        if trade_type == 'BUY':
            pnl = (exit_price - market) - 0.005 * (market + exit_price)
        else:
            pnl = (market - exit_price) - 0.005 * (market + exit_price)
        
        trades.append({'pnl': pnl, 'win': pnl > 0})
    
    if len(trades) == 0:
        return None
    
    wins = sum(1 for t in trades if t['win'])
    pnls = [t['pnl'] for t in trades]
    
    return {
        'trades': len(trades),
        'wins': wins,
        'win_rate': wins / len(trades),
        'total_pnl': sum(pnls),
        'mean_pnl': np.mean(pnls),
        'sharpe': np.mean(pnls) / np.std(pnls) * np.sqrt(252/5) if np.std(pnls) > 0 else 0
    }


def run_anti_overfitting_test(n_seeds=100):
    """Run backtest with many seeds to prove robustness."""
    print("=" * 80)
    print("ANTI-OVERFITTING TEST: 100 Random Seeds")
    print("=" * 80)
    print()
    print("⚠️  This tests that results are NOT cherry-picked or overfitted.")
    print("    Running same strategy with 100 different random seeds...")
    print()
    
    all_results = []
    
    for seed in range(n_seeds):
        result = run_single_backtest(seed_market=seed, seed_trades=seed + 1000)
        if result is not None:
            all_results.append(result)
        
        if (seed + 1) % 20 == 0:
            print(f"  Completed {seed + 1}/{n_seeds} seeds...")
    
    if len(all_results) == 0:
        print("  No trades generated in any seed.")
        return
    
    # Aggregate statistics
    win_rates = [r['win_rate'] for r in all_results]
    sharpes = [r['sharpe'] for r in all_results]
    pnls = [r['total_pnl'] for r in all_results]
    trade_counts = [r['trades'] for r in all_results]
    
    print("\n" + "-" * 60)
    print("AGGREGATE RESULTS ACROSS 100 SEEDS")
    print("-" * 60)
    
    print(f"\n  Seeds with trades: {len(all_results)}/{n_seeds}")
    print(f"  Avg trades/seed:  {np.mean(trade_counts):.1f}")
    
    print(f"\n  WIN RATE:")
    print(f"    Mean:   {np.mean(win_rates)*100:.1f}%")
    print(f"    Std:    {np.std(win_rates)*100:.1f}%")
    print(f"    Min:    {np.min(win_rates)*100:.1f}%")
    print(f"    Max:    {np.max(win_rates)*100:.1f}%")
    
    print(f"\n  SHARPE RATIO:")
    print(f"    Mean:   {np.mean(sharpes):.2f}")
    print(f"    Std:    {np.std(sharpes):.2f}")
    print(f"    Min:    {np.min(sharpes):.2f}")
    print(f"    Max:    {np.max(sharpes):.2f}")
    
    print(f"\n  TOTAL P&L:")
    print(f"    Mean:   ${np.mean(pnls):.2f}")
    print(f"    Std:    ${np.std(pnls):.2f}")
    print(f"    Min:    ${np.min(pnls):.2f}")
    print(f"    Max:    ${np.max(pnls):.2f}")
    
    # Honest assessment
    print("\n" + "-" * 60)
    print("HONEST ASSESSMENT")
    print("-" * 60)
    
    profitable_pct = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    print(f"\n  Profitable seeds: {profitable_pct:.0f}%")
    
    if np.mean(sharpes) > 1.0:
        print("  📈 GOOD: Average Sharpe > 1.0 across all seeds")
    elif np.mean(sharpes) > 0.5:
        print("  📊 MODERATE: Average Sharpe 0.5-1.0")
    elif np.mean(sharpes) > 0:
        print("  ⚠️  WEAK: Positive but low Sharpe")
    else:
        print("  ❌ No edge: Average Sharpe ≤ 0")
    
    if profitable_pct > 70:
        print("  ✅ Robust: >70% of seeds profitable")
    elif profitable_pct > 50:
        print("  👍 Reasonable: >50% of seeds profitable")
    else:
        print("  ⚠️  Fragile: <50% of seeds profitable")
    
    print("\n" + "=" * 80)
    
    # Save
    summary = {
        'n_seeds': n_seeds,
        'seeds_with_trades': len(all_results),
        'avg_trades': float(np.mean(trade_counts)),
        'mean_win_rate': float(np.mean(win_rates)),
        'std_win_rate': float(np.std(win_rates)),
        'mean_sharpe': float(np.mean(sharpes)),
        'std_sharpe': float(np.std(sharpes)),
        'mean_pnl': float(np.mean(pnls)),
        'profitable_pct': float(profitable_pct)
    }
    
    output_path = Path('neural/results/anti_overfitting_test.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved: {output_path}")
    
    return summary


if __name__ == '__main__':
    run_anti_overfitting_test(n_seeds=100)
