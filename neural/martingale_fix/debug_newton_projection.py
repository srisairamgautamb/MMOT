#!/usr/bin/env python3
"""
debug_newton_projection.py
==========================
Diagnostic script to analyze why Newton projection isn't converging.

Expected: After 500 iterations, residual < 0.0001
Actual:   After 500 iterations, residual = 0.027 (TOO HIGH!)

This script will print detailed convergence info to diagnose the issue.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
from architecture_fixed import ImprovedTransformerMMOT


def compute_drift_from_h(u_t, u_next, h_t, grid, epsilon):
    """Compute the drift (martingale residual) for given potentials."""
    M = len(grid)
    x = grid
    
    # Build Delta matrix - MUST MATCH SOLVER DEFINITION
    x_i = x.unsqueeze(1)  # (M, 1)
    x_j = x.unsqueeze(0)  # (1, M)
    Delta = x_i - x_j  # (M, M) - this is x - y, where i=x, j=y
    
    # Cost matrix
    C = Delta ** 2
    C_max = C.max().item()
    C_scaled = C / C_max
    
    # Log kernel
    term_u = u_t.unsqueeze(1) + u_next.unsqueeze(0)  # (M, M)
    term_h = h_t.unsqueeze(1) * Delta  # (M, M)
    
    LogK = (term_u + term_h - C_scaled) / epsilon
    
    # Transition probabilities P(y|x)
    probs = F.softmax(LogK, dim=1)  # (M, M), rows sum to 1
    
    # Expected next position E[Y | X=x]
    expected_y = torch.sum(probs * x.unsqueeze(0), dim=1)  # (M,)
    
    # Drift = E[Y|X=x] - x (should be 0 for martingale)
    drift = expected_y - x  # (M,)
    
    return drift, probs


def newton_projection_debug(u_t, u_next, h_init, grid, epsilon=0.2, max_iter=500, tol=1e-6):
    """
    Newton-Raphson projection with full debugging.
    
    Solves: Find h such that E[Y|X=x] = x for all x
    """
    print("\n" + "="*70)
    print("NEWTON PROJECTION DEBUG")
    print("="*70)
    print(f"Grid: [{grid.min():.3f}, {grid.max():.3f}], M={len(grid)}")
    print(f"Epsilon: {epsilon}")
    print(f"Max iterations: {max_iter}")
    print(f"Tolerance: {tol}")
    print(f"u_t range: [{u_t.min():.4f}, {u_t.max():.4f}]")
    print(f"u_next range: [{u_next.min():.4f}, {u_next.max():.4f}]")
    print(f"h_init range: [{h_init.min():.4f}, {h_init.max():.4f}]")
    print("-"*70)
    
    M = len(grid)
    h = h_init.clone()
    x = grid
    
    # Delta matrix
    x_i = x.unsqueeze(1)
    x_j = x.unsqueeze(0)
    Delta = x_i - x_j  # (M, M)
    
    C = Delta ** 2
    C_max = C.max().item()
    C_scaled = C / C_max
    
    history = []
    
    for iteration in range(max_iter):
        # Compute current drift (residual)
        drift, probs = compute_drift_from_h(u_t, u_next, h, grid, epsilon)
        residual_max = drift.abs().max().item()
        residual_mean = drift.abs().mean().item()
        
        history.append(residual_max)
        
        # Print progress
        if iteration % 50 == 0 or iteration < 10 or residual_max < tol:
            print(f"Iter {iteration:4d}: max_residual={residual_max:.6f}, mean={residual_mean:.6f}, h_range=[{h.min():.2f},{h.max():.2f}]")
        
        # Check convergence
        if residual_max < tol:
            print(f"\n✅ CONVERGED at iteration {iteration}")
            print(f"   Final residual: {residual_max:.8f}")
            return h, True, history
        
        # Compute Jacobian using finite differences (for debugging)
        # This is dF/dh where F(h) = E[Y|X=x] - x
        
        # For the softmax-based transition:
        # d(E[Y|X]) / dh = Var[Y|X] / epsilon * Delta
        # This is the analytical derivative
        
        # Rebuild kernel for Jacobian
        term_u = u_t.unsqueeze(1) + u_next.unsqueeze(0)
        term_h = h.unsqueeze(1) * Delta
        LogK = (term_u + term_h - C_scaled) / epsilon
        probs = F.softmax(LogK, dim=1)  # P(y|x)
        
        # E[Y|X=x]
        expected_y = torch.sum(probs * x.unsqueeze(0), dim=1)
        
        # E[Y^2|X=x]
        expected_y2 = torch.sum(probs * (x.unsqueeze(0) ** 2), dim=1)
        
        # Var[Y|X=x] = E[Y^2] - E[Y]^2
        var_y = expected_y2 - expected_y ** 2  # (M,)
        
        # Jacobian diagonal: d(drift_x)/d(h_x) = Var[Y|X=x] / epsilon
        # For the linear-h formulation, derivative is simpler
        jacobian_diag = var_y / epsilon  # (M,)
        
        # Regularize to avoid division by zero
        jacobian_diag = torch.clamp(jacobian_diag, min=1e-6)
        
        # Newton step: h_new = h - F(h) / F'(h)
        delta_h = drift / jacobian_diag
        
        # Damping for stability
        alpha = 0.5 if iteration < 10 else 1.0
        
        h = h - alpha * delta_h
    
    print(f"\n❌ FAILED TO CONVERGE after {max_iter} iterations")
    print(f"   Final residual: {residual_max:.6f}")
    return h, False, history


def main():
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
    
    # Test on instance 0
    idx = 0
    m = torch.from_numpy(marginals[idx].astype(np.float32)).unsqueeze(0).to(device)
    N = m.shape[1] - 1
    
    # Get neural prediction
    with torch.no_grad():
        u_pred, h_pred = model(m, grid_torch)
    u_pred = u_pred[0, :N+1]  # (N+1, M)
    h_pred = h_pred[0, :N]    # (N, M)
    
    print(f"\n{'='*70}")
    print(f"TESTING INSTANCE {idx}")
    print(f"{'='*70}")
    print(f"N = {N} time steps")
    
    # Test Newton for t=0
    t = 0
    u_t = u_pred[t]
    u_next = u_pred[t+1]
    h_init = h_pred[t]
    
    # Compute initial drift
    drift_init, _ = compute_drift_from_h(u_t, u_next, h_init, grid_torch, epsilon=0.2)
    print(f"\nInitial drift (before Newton): {drift_init.abs().max().item():.6f}")
    
    # Run Newton with debug
    h_refined, converged, history = newton_projection_debug(
        u_t, u_next, h_init, grid_torch, epsilon=0.2, max_iter=500
    )
    
    # Compute final drift
    drift_final, _ = compute_drift_from_h(u_t, u_next, h_refined, grid_torch, epsilon=0.2)
    print(f"\nFinal drift (after Newton): {drift_final.abs().max().item():.6f}")
    
    # Now test with TEACHER potentials (ground truth)
    print("\n" + "="*70)
    print("COMPARISON WITH TEACHER DATA")
    print("="*70)
    
    u_teacher_t = torch.from_numpy(u_teacher[idx][t].astype(np.float32)).to(device)
    u_teacher_next = torch.from_numpy(u_teacher[idx][t+1].astype(np.float32)).to(device)
    h_teacher_t = torch.from_numpy(h_teacher[idx][t].astype(np.float32)).to(device)
    
    drift_teacher, _ = compute_drift_from_h(u_teacher_t, u_teacher_next, h_teacher_t, grid_torch, epsilon=0.2)
    print(f"Teacher drift: {drift_teacher.abs().max().item():.6f}")
    
    # Run Newton starting from teacher h
    print("\nRunning Newton starting from TEACHER h:")
    h_refined_teacher, converged_teacher, _ = newton_projection_debug(
        u_teacher_t, u_teacher_next, h_teacher_t, grid_torch, epsilon=0.2, max_iter=100
    )
    
    drift_final_teacher, _ = compute_drift_from_h(u_teacher_t, u_teacher_next, h_refined_teacher, grid_torch, epsilon=0.2)
    print(f"Teacher drift after Newton: {drift_final_teacher.abs().max().item():.6f}")


if __name__ == '__main__':
    main()
