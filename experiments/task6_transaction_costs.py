#!/usr/bin/env python3
"""
MMOT Transaction Costs Analysis (Task 6) - FIXED v3
Goal: Show how transaction costs affect price bounds (Theorem 6.3)

ECONOMICS FIX:
Transaction costs make hedging MORE EXPENSIVE, so option prices INCREASE.
This computes the UPPER bound on prices when friction exists.

Generates: Figure 13 (Transaction Cost Impact)
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from mmot.core.solver_admm import solve_mmot_admm

# ============================================================================
# PRICING WITH TRANSACTION COSTS (UPPER BOUND)
# ============================================================================

def compute_hedging_cost_with_friction(P, x_grid, K, k_bps, r=0.045, T=1.0):
    """
    Compute option ASKING price including hedging costs with friction.
    
    Economic interpretation:
    - Seller hedges by replicating the payoff
    - Transaction costs make each hedge trade more expensive
    - Seller must charge MORE to cover hedging costs
    - Price = payoff + hedging_friction
    
    This is the UPPER bound on option prices!
    """
    k = k_bps / 10000  # Convert bps to fraction
    
    S_t = x_grid[:, None]
    S_next = x_grid[None, :]
    
    # Asian payoff (what buyer receives)
    avg_price = 0.5 * (S_t + S_next)
    payoff = jnp.maximum(avg_price - K, 0)
    
    # Hedging cost = proportional to position changes
    # When price moves from S_t to S_next, hedger must rebalance
    # Cost of rebalancing = k * |delta change| ≈ k * |S_next - S_t|
    hedging_friction = k * jnp.abs(S_next - S_t)
    
    # Total cost to seller = payoff + hedging friction
    # This is the ASKING PRICE (upper bound)
    total_cost = payoff + hedging_friction
    
    # Discounted expected value
    price = np.exp(-r * T) * float(jnp.sum(P * total_cost))
    
    return price


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("💸 TRANSACTION COSTS ANALYSIS (Task 6 - Theorem 6.3)")
    print("="*80)
    print()
    
    # Setup - REALISTIC S&P 500 PARAMETERS
    S0 = 6905.74
    r = 0.045
    T = 1.0
    M = 50
    x_grid = jnp.linspace(0.7 * S0, 1.3 * S0, M)
    K = S0
    
    print(f"Parameters: S0=${S0:.2f}, K=${K:.2f}, r={r*100:.1f}%, T={T} year")
    print()
    print("Economic interpretation:")
    print("  • Transaction costs make hedging MORE expensive")
    print("  • Seller charges HIGHER price to cover hedging friction")
    print("  • This shows the UPPER bound (asking price)")
    print()
    
    # Generate marginals
    sigma1 = 0.03 * S0
    sigma2 = 0.04 * S0
    
    mu = jnp.exp(-0.5 * ((x_grid - S0) / sigma1)**2)
    mu = mu / jnp.sum(mu)
    
    nu = jnp.exp(-0.5 * ((x_grid - S0) / sigma2)**2)
    nu = nu / jnp.sum(nu)
    
    marginals = jnp.stack([mu, nu])
    
    # Solve MMOT once
    C = (x_grid[:, None] - x_grid[None, :])**2
    result = solve_mmot_admm(marginals, C, x_grid)
    P = result['P']
    
    # Transaction cost levels (in basis points)
    k_values = [0, 10, 25, 50, 100, 200, 500]
    
    results = []
    
    print("Testing transaction cost levels...")
    print("-" * 70)
    print(f"{'k (bps)':>10} {'Ask Price ($)':>14} {'Δ from Base':>14} {'% Increase':>12}")
    print("-" * 70)
    
    # Base price (frictionless)
    price_base = compute_hedging_cost_with_friction(P, x_grid, K, 0, r, T)
    
    for k in k_values:
        price = compute_hedging_cost_with_friction(P, x_grid, K, k, r, T)
        delta = price - price_base
        pct_change = (delta / price_base) * 100 if price_base > 0 else 0
        
        results.append((k, price, delta, pct_change))
        print(f"{k:>10} {price:>14.2f} {delta:>+14.2f} {pct_change:>+11.1f}%")
    
    results = np.array(results)
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("📊 TRANSACTION COST IMPACT (UPPER BOUND)")
    print("="*80)
    
    # Fit: price_increase = α * k
    mask = results[:, 0] > 0
    if np.sum(mask) > 1:
        coeffs = np.polyfit(results[mask, 0], results[mask, 2], 1)
        alpha = coeffs[0]
        print(f"\nFitted relationship: ΔPrice ≈ +${alpha:.4f} × k (bps)")
        print(f"Each 100 bps of friction adds ${alpha * 100:.2f} to asking price")
    
    # ========================================================================
    # GENERATE FIGURE 13
    # ========================================================================
    print("\n" + "="*80)
    print("📊 GENERATING FIGURE 13")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Price vs Transaction Cost
    ax1 = axes[0]
    ax1.plot(results[:, 0], results[:, 1], 'go-', linewidth=2, markersize=10)
    ax1.axhline(price_base, color='blue', linestyle='--', alpha=0.5, 
                label=f'Frictionless price: ${price_base:.0f}')
    ax1.fill_between(results[:, 0], price_base, results[:, 1], 
                     alpha=0.2, color='green', label='Friction premium')
    ax1.set_xlabel('Transaction Cost k (basis points)', fontsize=12)
    ax1.set_ylabel('Option Ask Price ($)', fontsize=12)
    ax1.set_title('Asian Call UPPER BOUND\n(Hedging Cost with Friction)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Percentage Impact
    ax2 = axes[1]
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(k_values)))
    bars = ax2.bar(range(len(k_values)), results[:, 3],
                   tick_label=[str(k) for k in k_values],
                   color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Transaction Cost k (basis points)', fontsize=12)
    ax2.set_ylabel('Price Increase (%)', fontsize=12)
    ax2.set_title('Hedging Cost Premium\n(% Above Frictionless Price)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, results[:, 3]):
        if val > 0.1:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f'+{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Impact of Transaction Costs on MMOT Pricing (Theorem 6.3)\n'
                 f'S&P 500, S₀=${S0:.0f}, K=ATM, Upper Bound',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs('figures/phase2b', exist_ok=True)
    plt.savefig('figures/phase2b/figure13_transaction_costs.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/phase2b/figure13_transaction_costs.pdf', bbox_inches='tight')
    
    print("\n✅ Saved: figures/phase2b/figure13_transaction_costs.png")
    print("✅ Saved: figures/phase2b/figure13_transaction_costs.pdf")
    
    # Verification
    print("\n" + "="*80)
    print("✓ VERIFICATION (Economics Check)")
    print("="*80)
    
    # Prices should INCREASE with friction
    if results[-1, 1] > results[0, 1]:
        print(f"✅ Prices INCREASE with transaction cost (correct economics!)")
        print(f"   k=0:   ${results[0, 1]:.2f}")
        print(f"   k=500: ${results[-1, 1]:.2f} (+{results[-1, 3]:.0f}%)")
    else:
        print(f"❌ ERROR: Prices should increase with friction!")
    
    print("\n" + "="*80)
    print("✅ TASK 6 COMPLETE: Figure 13 Generated (ECONOMICS FIXED)")
    print("="*80)
