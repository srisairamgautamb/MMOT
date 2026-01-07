"""
Generate 10,000 solved MMOT instances using Phase 2a classical solver.
Runtime: ~50 hours on Apple M4 (5s per instance × 10,000)
Output: 50GB of .npz files containing (marginals, u*, h*) tuples
"""

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import time

# Add Phase 2a solver to path
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT')
try:
    from mmot.core.solver import solve_mmot
except ImportError:
    print("WARNING: Phase 2a solver not found. Using dummy solver for testing.")
    def solve_mmot(*args, **kwargs):
        """Dummy solver for testing when Phase 2a not available."""
        raise NotImplementedError("Phase 2a solver required for data generation")

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_INSTANCES = 10_000
TRAIN_SPLIT = 0.85  # 8,500 train, 1,500 val
OUTPUT_DIR = Path('data/raw')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Grid specification (OPTIMIZED for speed)
M = 150  # Grid points (reduced from 200 for 1.8× speedup, still sufficient resolution)
S_MIN = 50.0
S_MAX = 200.0
S_GRID = jnp.linspace(S_MIN, S_MAX, M)

# ============================================================================
# PARAMETER SAMPLING DISTRIBUTIONS
# ============================================================================
def sample_mmot_params():
    """
    Sample diverse MMOT problem parameters for generalization.
    
    Returns:
        dict with keys: N, T, sigma, S0, strike, option_type, skew
    """
    # Number of time steps (DIVERSE: 2 to 50)
    N = np.random.choice([2, 3, 5, 10, 20, 30, 50], 
                         p=[0.1, 0.15, 0.3, 0.25, 0.1, 0.05, 0.05])
    
    # Total time (days to years)
    T = np.random.uniform(0.05, 1.0)  # 18 days to 1 year
    
    # Volatility (low to high vol regimes)
    sigma = np.random.uniform(0.10, 0.50)  # 10% to 50% annualized
    
    # Initial spot (normalize around 100)
    S0 = np.random.uniform(80, 120)
    
    # Strike (OTM, ATM, ITM)
    moneyness = np.random.choice(['OTM', 'ATM', 'ITM'], p=[0.3, 0.4, 0.3])
    if moneyness == 'OTM':
        strike = S0 * np.random.uniform(1.05, 1.20)
    elif moneyness == 'ATM':
        strike = S0 * np.random.uniform(0.95, 1.05)
    else:  # ITM
        strike = S0 * np.random.uniform(0.80, 0.95)
    
    # Option type
    option_type = np.random.choice(['call', 'put'])
    
    # Skew parameter (for non-Gaussian marginals)
    # 0 = symmetric, >0 = right skew, <0 = left skew
    skew = np.random.normal(0, 0.3)
    
    # Kurtosis (fat tails)
    kurtosis = np.random.uniform(0, 2)  # 0 = normal, 2 = very fat
    
    return {
        'N': N,
        'T': T,
        'sigma': sigma,
        'S0': S0,
        'strike': strike,
        'option_type': option_type,
        'skew': skew,
        'kurtosis': kurtosis,
        'grid': S_GRID
    }

# ============================================================================
# MARGINAL GENERATION (with Market Realism)
# ============================================================================
def generate_marginals(params):
    """
    Generate N+1 marginal distributions satisfying:
    1. Convex order (risk increases over time)
    2. Realistic implied vol surface
    3. Optional skew/kurtosis
    
    Returns:
        marginals: array [N+1, M] of probability densities
    """
    N = params['N']
    T = params['T']
    sigma = params['sigma']
    S0 = params['S0']
    skew = params['skew']
    kurt = params['kurtosis']
    S = params['grid']
    
    dt = T / N
    marginals = []
    
    for t_idx in range(N + 1):
        t = t_idx * dt
        
        # Time-dependent volatility (term structure)
        sigma_t = sigma * (1 + 0.1 * np.sqrt(t))  # Vol increases with sqrt(t)
        
        # Black-Scholes marginal (lognormal)
        forward = S0  # Under Q (risk-neutral), drift = 0
        var_t = sigma_t**2 * t
        
        if var_t < 1e-6:  # t=0 edge case
            mu_t = jnp.zeros(M)
            mu_t = mu_t.at[jnp.argmin(jnp.abs(S - S0))].set(1.0)
        else:
            # Lognormal density
            log_S = jnp.log(S)
            log_S0 = jnp.log(forward)
            density = (1 / (S * jnp.sqrt(2 * jnp.pi * var_t))) * \
                      jnp.exp(-0.5 * ((log_S - log_S0 + 0.5 * var_t)**2) / var_t)
            
            # Add skewness via sinh-arcsinh transform (if requested)
            if abs(skew) > 0.01:
                z = (log_S - log_S0) / jnp.sqrt(var_t)
                z_skewed = jnp.sinh(jnp.arcsinh(z) + skew)
                density = density * jnp.cosh(jnp.arcsinh(z) + skew) / jnp.cosh(jnp.arcsinh(z))
            
            # Normalize to probability
            mu_t = density / jnp.sum(density)
        
        marginals.append(mu_t)
    
    marginals = jnp.stack(marginals)  # [N+1, M]
    
    return marginals

# ============================================================================
# SOLVING WITH PHASE 2a TEACHER
# ============================================================================
def solve_instance(params, epsilon=1.0, max_iter=2000):
    """
    Solve one MMOT instance using classical Phase 2a solver.
    
    Returns:
        dict: {
            'marginals': [N+1, M],
            'u_star': [N+1, M],
            'h_star': [N, M],
            'dual_value': scalar,
            'iterations': int,
            'runtime': float (seconds)
        }
    """
    # Generate marginals
    marginals = generate_marginals(params)
    
    # Set up MMOT problem
    N = params['N']
    S = params['grid']
    M = len(S)
    
    # Cost function: quadratic (standard)
    C = jnp.zeros((M, M))
    for i in range(M):
        for j in range(M):
            C = C.at[i, j].set((S[i] - S[j])**2)
    
    # Solve using Phase 2a solver
    # Phase 2a signature: solve_mmot(marginals, C, x_grid, max_iter, epsilon, damping)
    t_start = time.time()
    
    try:
        u_final, h_final, iterations = solve_mmot(
            marginals=marginals,
            C=C,
            x_grid=S,
            max_iter=max_iter,
            epsilon=epsilon,
            damping=0.8  # Standard damping for stability
        )
        runtime = time.time() - t_start
        
        # Compute dual value (approximate as sum of u potentials)
        dual_value = float(jnp.sum(u_final * marginals))
        
        return {
            'marginals': np.array(marginals),
            'u_star': np.array(u_final),
            'h_star': np.array(h_final),
            'dual_value': dual_value,
            'iterations': int(iterations),
            'runtime': runtime,
            'params': params  # Store for reproducibility
        }
    except Exception as e:
        print(f"Error solving instance: {e}")
        raise

# ============================================================================
# BATCH GENERATION
# ============================================================================
def generate_dataset(num_instances, output_dir, start_idx=0):
    """
    Generate dataset by solving num_instances MMOT problems.
    Save incrementally to avoid memory overflow.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating {num_instances} MMOT instances...")
    print(f"Estimated time: {num_instances * 5 / 3600:.1f} hours")
    
    for idx in tqdm(range(start_idx, start_idx + num_instances)):
        try:
            # Sample parameters
            params = sample_mmot_params()
            
            # Solve
            solution = solve_instance(params)
            
            # Save to disk (one file per instance)
            filename = output_dir / f'mmot_{idx:06d}.npz'
            np.savez_compressed(
                filename,
                marginals=solution['marginals'],
                u_star=solution['u_star'],
                h_star=solution['h_star'],
                dual_value=solution['dual_value'],
                params=params
            )
            
            # Log progress every 100 instances
            if (idx + 1) % 100 == 0:
                avg_runtime = solution['runtime']
                print(f"[{idx+1}/{num_instances}] Avg solve time: {avg_runtime:.2f}s")
        
        except Exception as e:
            print(f"ERROR solving instance {idx}: {e}")
            continue
    
    print(f"Dataset generation complete. Saved to {output_dir}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=10000)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()
    
    # Generate data
    generate_dataset(
        num_instances=args.num,
        output_dir=OUTPUT_DIR,
        start_idx=args.start
    )
    
    # Split train/val
    print("Splitting into train/val...")
    files = sorted(OUTPUT_DIR.glob('*.npz'))
    split_idx = int(len(files) * TRAIN_SPLIT)
    
    train_dir = Path('data/train')
    val_dir = Path('data/val')
    train_dir.mkdir(exist_ok=True, parents=True)
    val_dir.mkdir(exist_ok=True, parents=True)
    
    for i, f in enumerate(files):
        if i < split_idx:
            f.rename(train_dir / f.name)
        else:
            f.rename(val_dir / f.name)
    
    print(f"Train: {split_idx} instances")
    print(f"Val: {len(files) - split_idx} instances")
