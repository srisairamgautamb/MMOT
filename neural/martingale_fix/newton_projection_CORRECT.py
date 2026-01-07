#!/usr/bin/env python3
"""
newton_projection_CORRECT.py
============================
CORRECTED Newton-Raphson projection for martingale constraint.

Key insight: The martingale constraint is:
    F(h) = E[Y|X=x] - x = 0

The derivative is:
    dF/dh = -Var[Y|X=x] / epsilon  (NEGATIVE!)

The Newton update is:
    h_new = h - F(h) / F'(h)
          = h - drift / (-variance/epsilon)
          = h + epsilon * drift / variance

This is the CORRECT formula that will CONVERGE.
"""

import torch
import torch.nn.functional as F
import numpy as np


def newton_projection_correct(u_t, u_next, h_init, grid, epsilon=0.2, 
                               max_iter=500, tol=1e-6, verbose=False):
    """
    CORRECT Newton-Raphson projection for martingale constraint.
    
    Solves: Find h such that E[Y|X=x] = x for all x
    
    Args:
        u_t: (M,) u potential at time t
        u_next: (M,) u potential at time t+1
        h_init: (M,) initial h guess
        grid: (M,) spatial grid
        epsilon: regularization parameter
        max_iter: maximum iterations
        tol: convergence tolerance
        verbose: print iteration info
        
    Returns:
        h_refined: (M,) refined h satisfying martingale
        converged: bool
    """
    M = len(grid)
    device = grid.device
    x = grid
    
    # Precompute Delta matrix: Delta[i,j] = x_i - x_j (i=row=X, j=col=Y)
    x_i = x.unsqueeze(1)  # (M, 1)
    x_j = x.unsqueeze(0)  # (1, M)
    Delta = x_i - x_j  # (M, M)
    
    # Cost matrix (scaled)
    C = Delta ** 2
    C_max = C.max().item()
    C_scaled = C / C_max
    
    h = h_init.clone()
    
    for iteration in range(max_iter):
        # Build log kernel
        term_u = u_t.unsqueeze(1) + u_next.unsqueeze(0)  # (M, M)
        term_h = h.unsqueeze(1) * Delta  # (M, M)
        
        LogK = (term_u + term_h - C_scaled) / epsilon
        
        # Numerical stability: subtract max per row
        LogK_stable = LogK - LogK.max(dim=1, keepdim=True)[0]
        
        # Transition probabilities P(y|x)
        probs = F.softmax(LogK_stable, dim=1)  # (M, M)
        
        # E[Y | X=x]
        expected_y = torch.sum(probs * x.unsqueeze(0), dim=1)  # (M,)
        
        # Drift = E[Y|X=x] - x (this is F(h) = residual)
        drift = expected_y - x  # (M,)
        
        residual_max = drift.abs().max().item()
        
        if verbose and (iteration % 50 == 0 or iteration < 5):
            print(f"Iter {iteration:4d}: max_residual={residual_max:.8f}")
        
        # Check convergence
        if residual_max < tol:
            if verbose:
                print(f"✅ CONVERGED at iteration {iteration}, residual={residual_max:.8f}")
            return h, True
        
        # Compute variance = E[Y^2|X] - E[Y|X]^2
        expected_y2 = torch.sum(probs * (x.unsqueeze(0) ** 2), dim=1)  # (M,)
        variance = expected_y2 - expected_y ** 2  # (M,)
        
        # Clamp variance to avoid division by zero
        variance = torch.clamp(variance, min=1e-8)
        
        # CORRECT Newton step:
        # F(h) = drift
        # F'(h) = -variance / epsilon (derivative of E[Y|X] w.r.t. h is NEGATIVE)
        # Newton: h_new = h - F / F' = h - drift / (-variance/epsilon) = h + epsilon * drift / variance
        
        step = epsilon * drift / variance
        
        # Damping for early iterations
        alpha = 0.1 if iteration < 5 else (0.3 if iteration < 20 else 0.5)
        
        h = h + alpha * step
    
    if verbose:
        print(f"❌ FAILED after {max_iter} iterations, residual={residual_max:.6f}")
    
    return h, False


def test_newton_projection():
    """Test the corrected Newton projection."""
    import sys
    sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
    from architecture_fixed import ImprovedTransformerMMOT
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    data = np.load('data/validation_solved.npz', allow_pickle=True)
    marginals = data['marginals']
    grid = data['grid']
    u_teacher = data['u']
    h_teacher = data['h']
    
    grid_torch = torch.from_numpy(grid.astype(np.float32)).to(device)
    M = len(grid)
    
    # Load model
    model = ImprovedTransformerMMOT(M=M, d_model=128, n_heads=4, n_layers=4).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    model.eval()
    
    print("\n" + "="*70)
    print("TESTING CORRECTED NEWTON PROJECTION")
    print("="*70)
    
    # Test on first 5 instances
    all_drifts_before = []
    all_drifts_after = []
    
    for idx in range(5):
        m = torch.from_numpy(marginals[idx].astype(np.float32)).unsqueeze(0).to(device)
        N = m.shape[1] - 1
        
        with torch.no_grad():
            u_pred, h_pred = model(m, grid_torch)
        u_pred = u_pred[0, :N+1]
        h_pred = h_pred[0, :N]
        
        # Test t=0
        t = 0
        u_t = u_pred[t]
        u_next = u_pred[t+1]
        h_init = h_pred[t]
        
        # Run corrected Newton
        verbose = (idx == 0)  # Verbose only for first instance
        h_refined, converged = newton_projection_correct(
            u_t, u_next, h_init, grid_torch, epsilon=0.2, max_iter=500, verbose=verbose
        )
        
        # Compute drifts
        def compute_drift(u_t, u_next, h, grid, epsilon):
            x = grid
            Delta = x.unsqueeze(1) - x.unsqueeze(0)
            C = Delta ** 2
            C_scaled = C / C.max()
            term_u = u_t.unsqueeze(1) + u_next.unsqueeze(0)
            term_h = h.unsqueeze(1) * Delta
            LogK = (term_u + term_h - C_scaled) / epsilon
            probs = F.softmax(LogK, dim=1)
            expected_y = torch.sum(probs * x.unsqueeze(0), dim=1)
            return (expected_y - x).abs().max().item()
        
        drift_before = compute_drift(u_t, u_next, h_init, grid_torch, 0.2)
        drift_after = compute_drift(u_t, u_next, h_refined, grid_torch, 0.2)
        
        all_drifts_before.append(drift_before)
        all_drifts_after.append(drift_after)
        
        status = "✅" if converged else "❌"
        print(f"Instance {idx}: Before={drift_before:.6f}, After={drift_after:.6f} {status}")
    
    print("\n" + "-"*70)
    print(f"SUMMARY (5 instances):")
    print(f"  Mean drift BEFORE Newton: {np.mean(all_drifts_before):.6f}")
    print(f"  Mean drift AFTER Newton:  {np.mean(all_drifts_after):.6f}")
    print(f"  Max drift AFTER Newton:   {max(all_drifts_after):.6f}")
    
    if max(all_drifts_after) < 0.01:
        print("\n🎉 SUCCESS! Newton projection achieves drift < 0.01!")
    elif max(all_drifts_after) < 0.001:
        print("\n🎉 EXCELLENT! Newton projection achieves drift < 0.001!")
    else:
        print(f"\n⚠️ Need more work. Max drift = {max(all_drifts_after):.6f}")


if __name__ == '__main__':
    test_newton_projection()
