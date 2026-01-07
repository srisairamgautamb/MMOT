"""
Visualization utilities for MMOT.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
from typing import List, Dict

plt.style.use('seaborn-v0_8-paper')
sns.set_context("notebook", font_scale=1.2)

def plot_marginals(
    marginals: List[Dict], 
    S0: float, 
    save_path: str = None
):
    """
    Plots the calibrated marginal densities.
    """
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("viridis", len(marginals))
    
    for i, m in enumerate(marginals):
        t_label = m['t']
        plt.plot(m['x'], m['pdf'], label=f"T={t_label}", color=colors[i], linewidth=2)
        plt.fill_between(m['x'], m['pdf'], alpha=0.1, color=colors[i])
        
    plt.axvline(S0, color='red', linestyle='--', label=f'Spot ({S0:.2f})')
    plt.title("Calibrated S&P 500 Risk-Neutral Densities")
    plt.xlabel("Price")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_transport_plan(
    P_matrix: np.ndarray, 
    x_grid: np.ndarray, 
    t_idx: int, 
    save_path: str = None
):
    """
    Plots the transport plan heat map for transition t -> t+1.
    """
    plt.figure(figsize=(10, 8))
    
    # Log scale or power scale for visibility given sparsity?
    # Usually Sinkhorn plans are fairly diffuse initially but sharpen up.
    # We plot P_matrix directly.
    
    # Use x_grid for axis labels
    # We can perform a rough heatmap
    
    # Simplify axis labels (show every 10th ticks)
    n = len(x_grid)
    tick_step = max(n // 10, 1)
    
    ax = sns.heatmap(
        P_matrix, 
        cmap="magma", 
        xticklabels=tick_step, 
        yticklabels=tick_step,
        cbar_kws={'label': 'Probability Mass'}
    )
    
    # Format ticks to show price values
    ax.set_xticklabels([f"{x_grid[i]:.0f}" for i in range(0, n, tick_step)])
    ax.set_yticklabels([f"{x_grid[i]:.0f}" for i in range(0, n, tick_step)])
    
    plt.title(f"Optimal Transport Plan $\pi^*_{{{t_idx},{t_idx+1}}}$")
    plt.xlabel(f"Price at $T_{{{t_idx+1}}}$")
    plt.ylabel(f"Price at $T_{{{t_idx}}}$")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_convergence(
    primal_vals: List[float], 
    dual_vals: List[float] = None, 
    save_path: str = None
):
    """
    Plots convergence of primal (and optional dual) objective.
    """
    plt.figure(figsize=(10, 5))
    
    plt.plot(primal_vals, label="Primal Cost", linewidth=2, color='blue')
    if dual_vals:
        plt.plot(dual_vals, label="Dual Objective", linewidth=2, color='green', linestyle='--')
        
    plt.title("Solver Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_pricing_bounds(
    maturities: List[str],
    mmot_lower: List[float],
    mmot_upper: List[float],
    market_prices: List[float] = None,
    option_type: str = "Asian Call",
    save_path: str = None
):
    """
    Comparison of MMOT bounds vs Market (if applicable).
    """
    x = np.arange(len(maturities))
    
    plt.figure(figsize=(10, 6))
    
    plt.fill_between(x, mmot_lower, mmot_upper, color='gray', alpha=0.3, label="MMOT No-Arbitrage Band")
    plt.plot(x, mmot_lower, 'k--', linewidth=1)
    plt.plot(x, mmot_upper, 'k--', linewidth=1)
    
    if market_prices:
        plt.plot(x, market_prices, 'ro-', linewidth=2, label="Market / Model Price")
        
    plt.xticks(x, maturities, rotation=45)
    plt.title(f"Model-Free Pricing Bounds: {option_type}")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_transport_3d(
    P_matrix: np.ndarray, 
    x_grid: np.ndarray, 
    t_idx: int, 
    save_path: str = None
):
    """
    3D Surface plot of the transport plan.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    Y = x_grid  # T_t
    X = x_grid  # T_{t+1}
    X, Y = np.meshgrid(X, Y)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, P_matrix, cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax.set_xlabel(f"Price at $T_{{{t_idx+1}}}$")
    ax.set_ylabel(f"Price at $T_{{{t_idx}}}$")
    ax.set_zlabel("Probability Mass")
    ax.set_title(f"3D Transport Plan $\pi^*_{{{t_idx},{t_idx+1}}}$")
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
