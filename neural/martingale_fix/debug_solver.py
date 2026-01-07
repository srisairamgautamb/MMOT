
import numpy as np
import scipy.special

def logsumexp(a, axis=None, keepdims=False):
    return scipy.special.logsumexp(a, axis=axis, keepdims=keepdims)

def debug_step():
    print("DEBUGGING SOLVER STEP-BY-STEP")
    
    # 1. Setup Grid and Cost
    M = 150
    moneyness_grid = np.linspace(0.5, 1.5, M)
    dm = moneyness_grid[1] - moneyness_grid[0]
    
    Delta = moneyness_grid[:, None] - moneyness_grid[None, :]
    C = Delta ** 2
    C_max = C.max()
    print(f"C_max: {C_max}")
    
    # Scale
    C_scaled = C / C_max
    epsilon_scaled = 0.05  # As in the failed run
    
    # 2. Setup Marginals (N=5)
    # Just check t=0 (narrow) and t=1 (wider)
    std_0 = 0.01
    log_m = np.log(moneyness_grid)
    d0 = np.exp(-0.5 * ((log_m - 0)/std_0)**2)
    d0 /= d0.sum()
    
    std_1 = 0.25 * np.sqrt(0.25/5)
    d1 = np.exp(-0.5 * ((log_m - 0)/std_1)**2)
    d1 /= d1.sum()
    
    print(f"Marginal 0 min/max: {d0.min()}, {d0.max()}")
    print(f"Marginal 1 min/max: {d1.min()}, {d1.max()}")
    
    # 3. Init Potentials
    N = 1 # Just one step for debug
    u = np.zeros((N+1, M))
    h = np.zeros((N, M))
    
    # 4. h-update (Newton)
    t = 0
    u_t = u[t]
    u_next = u[t+1]
    h_curr = h[t]
    
    print("\n--- h-update debug ---")
    # LogK = (u + u + h*Delta - C) / eps
    # h*Delta should be 0 since h=0
    LogK = (u_t[:, None] + u_next[None, :] + h_curr[:, None] * Delta - C_scaled) / epsilon_scaled
    print(f"LogK min/max: {LogK.min()}, {LogK.max()}")
    
    LogK_max = LogK.max(axis=1, keepdims=True)
    K = np.exp(LogK - LogK_max)
    row_sums = K.sum(axis=1, keepdims=True)
    P = K / (row_sums + 1e-10)
    
    f = np.sum(P * Delta, axis=1)
    print(f"f (martingale residual) min/max: {f.min()}, {f.max()}")
    
    E_Delta_sq = np.sum(P * (Delta ** 2), axis=1)
    f_prime = (E_Delta_sq - f ** 2) / epsilon_scaled
    print(f"f_prime min/max: {f_prime.min()}, {f_prime.max()}")
    
    # Update h
    # h_new = h - damping * f / f_prime
    damping = 0.8
    update = damping * (f / f_prime)
    print(f"h update min/max: {update.min()}, {update.max()}")
    
    h[t] = h_curr - update
    
    # 5. u-update checks
    print("\n--- u-update debug ---")
    # u[0]
    LogK = (u[0][:, None] + u[1][None, :] + h[0][:, None] * Delta - C_scaled) / epsilon_scaled
    print(f"LogK (u0) min/max: {LogK.min()}, {LogK.max()}")
    
    log_marg = logsumexp(LogK, axis=1)
    print(f"log_marg min/max: {log_marg.min()}, {log_marg.max()}")
    
    target = epsilon_scaled * np.log(d0 + 1e-10)
    print(f"target (eps*log(mu)) min/max: {target.min()}, {target.max()}")
    
    u_calc = target - log_marg
    print(f"Calculated u[0] min/max: {u_calc.min()}, {u_calc.max()}")

if __name__ == "__main__":
    debug_step()
