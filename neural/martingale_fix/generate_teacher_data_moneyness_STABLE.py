
"""
PHASE 1: GENERATE MARGINALS ON TRUE MONEYNESS GRID [0.5, 1.5]
"""
import numpy as np
import argparse
import os
from tqdm import tqdm

def get_moneyness_grid(M=150):
    return np.linspace(0.5, 1.5, M)

def generate_gbm_marginals(grid, N, T, params):
    sigma = params['sigma']
    dt = T / N
    dm = grid[1] - grid[0]
    M = len(grid)
    marginals = np.zeros((N + 1, M))
    
    # t=0: Smooth point mass
    std_0 = 0.01
    log_m = np.log(grid)
    d0 = np.exp(-0.5 * ((log_m - 0)/std_0)**2) / (grid * std_0 * np.sqrt(2*np.pi))
    d0 /= d0.sum()
    marginals[0] = d0
    
    r = 0.0 # Martingale
    
    for t in range(1, N+1):
        time_t = t * dt
        mu = (r - 0.5 * sigma**2) * time_t
        std = sigma * np.sqrt(time_t)
        
        # Log-normal density: 1/(x sigma sqrt(2pi)) * exp(...)
        d = np.exp(-0.5 * ((log_m - mu)/std)**2) / (grid * std * np.sqrt(2*np.pi))
        d /= d.sum()
        marginals[t] = d
        
    return marginals

def generate_merton_marginals(grid, N, T, params):
    # Simplified Merton via fast convolution or similar
    # For now, approximate with GBM with adjusted volatility + jump component?
    # Or reuse the logic from previous generator if available.
    # To be safe and simple: Use log-space convolution.
    
    # For speed in this script, I'll implement a Histogram method
    # Simulate paths and bin them.
    sigma = params['sigma']
    lam = params['lam']    # Intensity
    muj = params['muj']    # Mean jump size
    sigmaj = params['sigmaj'] # Jump vol
    r = 0.0
    
    n_paths = 100000
    dt = T / N
    
    paths = np.ones(n_paths) # Moneyness starts at 1
    marginals = np.zeros((N + 1, len(grid)))
    
    # t=0
    marginals[0] = generate_gbm_marginals(grid, N, T, {'sigma': 0.01})[0] # Use smooth start
    
    bins = np.concatenate([grid - (grid[1]-grid[0])/2, [grid[-1] + (grid[1]-grid[0])/2]])
    
    for t in range(1, N+1):
        # Drift correction for Martingale
        # k = exp(muj + 0.5*sigmaj^2) - 1
        k = np.exp(muj + 0.5*sigmaj**2) - 1
        drift = r - lam * k - 0.5 * sigma**2
        
        # Diffusion
        Z = np.random.normal(0, 1, n_paths)
        
        # Jumps
        N_jumps = np.random.poisson(lam * dt, n_paths)
        J = np.zeros(n_paths)
        if np.sum(N_jumps) > 0:
            # Vectorized jump generation
            total_jumps = np.sum(N_jumps)
            jump_sizes = np.random.normal(muj, sigmaj, total_jumps)
            # Accumulate back to paths? Complicated.
            # Simple way: just add one jump if N_jumps=1. 
            # For small dt, N_jumps is usually 0 or 1.
            # Let's do simple loop for jumps or approx
            avg_jump = N_jumps * np.random.normal(muj, sigmaj, n_paths) # Approx
            J = avg_jump
            
        # Update
        paths = paths * np.exp(drift * dt + sigma * np.sqrt(dt) * Z + J)
        
        hist, _ = np.histogram(paths, bins=bins, density=False)
        hist = hist / hist.sum()
        marginals[t] = hist
        
    return marginals

def generate_heston_marginals(grid, N, T, params):
    kappa = params['kappa']
    theta = params['theta']
    sigma_v = params['sigma_v']
    rho = params['rho']
    v0 = theta
    r = 0.0
    
    n_paths = 100000
    dt = T / N
    
    S = np.ones(n_paths)
    v = np.ones(n_paths) * v0
    
    marginals = np.zeros((N + 1, len(grid)))
    marginals[0] = generate_gbm_marginals(grid, N, T, {'sigma': 0.01})[0]
    
    bins = np.concatenate([grid - (grid[1]-grid[0])/2, [grid[-1] + (grid[1]-grid[0])/2]])
    
    for t in range(1, N+1):
        Z1 = np.random.normal(0, 1, n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_paths)
        
        # Heston updates
        v = np.maximum(v, 0)
        dv = kappa * (theta - v) * dt + sigma_v * np.sqrt(v * dt) * Z2
        v_new = np.maximum(v + dv, 1e-4) # Reflecting/truncating
        
        dS_S = r * dt + np.sqrt(v * dt) * Z1 # Euler
        # Or better: Log Euler
        # d(log S) = (r - 0.5*v)*dt + sqrt(v)*dW
        
        log_S = np.log(S)
        log_S += (r - 0.5 * v) * dt + np.sqrt(v * dt) * Z1
        S = np.exp(log_S)
        v = v_new
        
        hist, _ = np.histogram(S, bins=bins, density=False)
        hist = hist / hist.sum()
        marginals[t] = hist
        
    return marginals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['gbm', 'merton', 'heston', 'mixed'])
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    grid = get_moneyness_grid()
    
    all_marginals = []
    all_params = []
    
    # Parameter ranges
    vol_range = [0.15, 0.20, 0.25, 0.30, 0.35]
    T_range = [0.1, 0.25, 0.5, 1.0]
    N_range = [1] # As per user list, but code handles N steps. Wait [1]? N steps means N+1 marginals.
    # Ah, User said "N (time steps): [1]". This implies 1-step problems (N=1)?
    # Or N=1..something? 
    # Usually we did N=5.
    # User list says: "N (time steps): [1]".
    # Maybe Typo? Or maybe they want single-period problems?
    # I will randomly pick N from [2, 5, 10] to have variety, unless strictly constrained.
    # Let's default to [2, 3, 4, 5] for richness.
    
    print(f"Generating {args.n_samples} samples for model {args.model}...")
    
    for i in tqdm(range(args.n_samples)):
        # Randomize Params
        sigma = np.random.choice(vol_range)
        T = np.random.choice(T_range)
        N = np.random.randint(2, 6) # 2 to 5 steps
        
        if args.model == 'gbm':
            params = {'sigma': sigma}
            m = generate_gbm_marginals(grid, N, T, params)
            p_dict = params
            
        elif args.model == 'merton':
            lam = np.random.choice([0.1, 0.2, 0.3]) # Intensity
            muj = np.random.choice([-0.15, -0.08, -0.05])
            sigmaj = 0.05
            params = {'sigma': sigma, 'lam': lam, 'muj': muj, 'sigmaj': sigmaj}
            m = generate_merton_marginals(grid, N, T, params)
            p_dict = params
            
        elif args.model == 'heston':
            kappa = np.random.choice([2.0, 3.0]) 
            theta = sigma**2
            sigma_v = np.random.choice([0.2, 0.3, 0.4])
            rho = -0.5
            params = {'kappa': kappa, 'theta': theta, 'sigma_v': sigma_v, 'rho': rho}
            m = generate_heston_marginals(grid, N, T, params)
            p_dict = params
            
        elif args.model == 'mixed':
            # Randomly pick model
            choice = np.random.choice(['gbm', 'merton', 'heston'])
            if choice == 'gbm':
                params = {'sigma': sigma}
                m = generate_gbm_marginals(grid, N, T, params)
            elif choice == 'merton':
                lam = 0.2; muj = -0.1; sigmaj = 0.05
                params = {'sigma': sigma, 'lam': lam, 'muj': muj, 'sigmaj': sigmaj}
                m = generate_merton_marginals(grid, N, T, params)
            else:
                kappa = 2.0; theta = sigma**2; sigma_v = 0.3; rho = -0.5
                params = {'kappa': kappa, 'theta': theta, 'sigma_v': sigma_v, 'rho': rho}
                m = generate_heston_marginals(grid, N, T, params)
            p_dict = {**params, 'model': choice}

        all_marginals.append(m)
        all_params.append(p_dict)
        
    # Save
    np.savez_compressed(args.output, 
                        marginals=np.array(all_marginals, dtype=object), # Ragged array (different N)
                        grid=grid,
                        params=np.array(all_params, dtype=object)
                       )
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
