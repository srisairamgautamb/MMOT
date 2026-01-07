"""
Data loader utility for MMOT.
Fetches S&P 500 option data via yfinance and calibration utilities.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import jax.numpy as jnp
from typing import Tuple, List, Dict, Optional

def fetch_sp500_data(ticker_symbol: str = "SPY") -> Tuple[float, List[str], object]:
    """
    Fetches current price and expiration dates for the ticker.
    
    Args:
        ticker_symbol: Symbol to fetch (default "SPY")
        
    Returns:
        current_price: Latest close price
        expirations: List of expiration dates strings
        ticker_obj: yfinance Ticker object
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # Try to get current price safely
    hist = ticker.history(period="1d")
    if hist.empty:
        raise ValueError(f"Could not fetch history for {ticker_symbol}")
    
    current_price = hist['Close'].iloc[-1]
    expirations = ticker.options
    
    return current_price, list(expirations), ticker

def get_option_chain(ticker_obj, expiration: str) -> pd.DataFrame:
    """
    Fetches call option chain for a specific expiration.
    
    Args:
        ticker_obj: yfinance Ticker object
        expiration: Expiration date string (YYYY-MM-DD)
        
    Returns:
        df: DataFrame with columns ['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']
    """
    opts = ticker_obj.option_chain(expiration)
    calls = opts.calls
    
    # Filter for reasonable liquidity if possible, but keep it simple for now
    df = calls[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']].copy()
    
    # Use mid-price if bid/ask are valid, else lastPrice
    df['mid'] = 0.5 * (df['bid'] + df['ask'])
    mask = (df['bid'] > 0) & (df['ask'] > 0)
    df.loc[~mask, 'mid'] = df.loc[~mask, 'lastPrice']
    
    return df.sort_values('strike')

def calibrate_density(
    strikes: np.ndarray, 
    prices: np.ndarray, 
    S0: float, 
    grid_min: float, 
    grid_max: float, 
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrates risk-neutral density from Call prices using Breeden-Litzenberger.
    Uses smoothing spline to interpolate prices and take 2nd derivative.
    
    p(K) = exp(rT) * d^2C/dK^2
    We assume r=0 for simplicity in this visualization context, or effectively absorb it.
    
    Args:
        strikes: Array of strike prices
        prices: Array of call option prices
        S0: Current spot price (for grid centering)
        grid_min: Min strike for output grid
        grid_max: Max strike for output grid
        n_points: Number of grid points
        
    Returns:
        grid_x: Strike grid
        density: PDF values on the grid (normalized)
    """
    # Create output grid
    grid_x = np.linspace(grid_min, grid_max, n_points)
    
    # Filter strikes to be within reasonable range and strictly increasing
    # Sometimes data has noise
    sorted_idx = np.argsort(strikes)
    k_sorted = strikes[sorted_idx]
    c_sorted = prices[sorted_idx]
    
    # Remove duplicates
    valid_mask = np.concatenate(([True], np.diff(k_sorted) > 0))
    k_unique = k_sorted[valid_mask]
    c_unique = c_sorted[valid_mask]

    # Use UnivariateSpline with smoothing
    # k=3 (cubic), s is smoothing factor. 
    # We want to ensure convexity mostly, but raw spline might not guarantee it.
    # We'll take the max(deriv2, 0) to ensure non-negative density.
    
    # Heuristic for smoothing factor s: depends on number of points and variance
    # Default s=None allows scipy to estimate. Let's try a small amount of smoothing.
    spline = UnivariateSpline(k_unique, c_unique, k=3, s=0.01 * len(k_unique))
    
    # Second derivative
    deriv2 = spline.derivative(n=2)
    
    pdf_vals = deriv2(grid_x)
    
    # Enforce non-negativity and normalize
    pdf_vals = np.maximum(pdf_vals, 0.0)
    
    # Normalize
    dx = grid_x[1] - grid_x[0]
    total_mass = np.sum(pdf_vals) * dx
    
    if total_mass > 1e-6:
        pdf_vals /= total_mass
    else:
        # Fallback to gaussian if calibration fails completely
        print("Warning: Calibration failed (mass ~ 0), using fallback Gaussian")
        sigma = 0.2
        pdf_vals = np.exp(-0.5 * ((grid_x - S0) / (S0 * sigma))**2)
        pdf_vals /= np.sum(pdf_vals) * dx

    return grid_x, pdf_vals

def get_calibrated_marginals(
    ticker: str = "SPY", 
    n_steps: int = 3,
    n_grid: int = 100
) -> Dict:
    """
    High-level function to get 'N=n_steps' marginals for the demo.
    Selects T1, T2, ... Tn from available expirations.
    """
    current_price, expirations, ticker_obj = fetch_sp500_data(ticker)
    
    # Select expirations: e.g., 1 month, 2 months, 3 months out
    # Skip very near term (noise)
    selected_exps = expirations[1:n_steps+1] if len(expirations) > n_steps else expirations[:n_steps]
    
    marginals = []
    
    # Define common grid relative to spot
    # e.g. +/- 30%
    grid_min = current_price * 0.7
    grid_max = current_price * 1.3
    
    for exp in selected_exps:
        chain = get_option_chain(ticker_obj, exp)
        
        # Filter OTM/ATM for better signal? Or just use all.
        # Deep ITM calls have low liquidity usually.
        # Let's focus on a window around spot.
        mask = (chain['strike'] > current_price * 0.5) & (chain['strike'] < current_price * 1.5)
        subset = chain[mask]
        
        if len(subset) < 10:
             print(f"Warning: Not enough data for {exp}, skipping calibration")
             continue
             
        strikes = subset['strike'].values
        prices = subset['mid'].values
        
        x_grid, pdf = calibrate_density(strikes, prices, current_price, grid_min, grid_max, n_grid)
        marginals.append({
            't': exp,
            'x': x_grid,
            'pdf': pdf
        })
        
    return {
        'S0': current_price,
        'grid_min': grid_min,
        'grid_max': grid_max,
        'marginals': marginals
    }
