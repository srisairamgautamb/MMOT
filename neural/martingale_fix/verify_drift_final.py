#!/usr/bin/env python3
"""
verify_drift_final.py
=====================
Final verification script for MMOT Neural Solver.

Tests the trained model on validation data and computes EXACT drift
WITHOUT any Newton projection or post-processing.

Success Criteria: Max Drift < 0.01

Author: MMOT Research Team
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import argparse

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
from architecture_fixed import ImprovedTransformerMMOT


def compute_drift_comprehensive(u, h, grid, N, epsilon=0.2):
    """
    Compute drift for all time steps.
    
    Returns:
        drifts: list of max absolute drift per time step
        mean_drift: mean of max drifts
        max_drift: overall maximum drift
    """
    M = len(grid)
    device = u.device
    
    x = grid
    x_i = x.unsqueeze(1)
    x_j = x.unsqueeze(0)
    Delta = x_i - x_j  # (M, M)
    
    C = Delta ** 2
    C_max = C.max().item()
    C_scaled = C / C_max
    
    drifts = []
    drift_vectors = []
    
    for t in range(N):
        u_t = u[t]
        u_next = u[t+1]
        h_t = h[t]
        
        term_u = u_t.unsqueeze(1) + u_next.unsqueeze(0)
        term_h = h_t.unsqueeze(1) * Delta
        
        LogK = (term_u + term_h - C_scaled) / epsilon
        probs = F.softmax(LogK, dim=1)
        
        expected_y = torch.sum(probs * x.unsqueeze(0), dim=1)
        drift = expected_y - x
        
        drifts.append(drift.abs().max().item())
        drift_vectors.append(drift.cpu().numpy())
    
    return {
        'drifts_per_t': drifts,
        'mean_drift': np.mean(drifts),
        'max_drift': max(drifts),
        'drift_vectors': drift_vectors
    }


def main():
    parser = argparse.ArgumentParser(description='Final Drift Verification')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--data', type=str, default='data/validation_solved.npz')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--target', type=float, default=0.01)
    args = parser.parse_args()
    
    device = args.device if torch.backends.mps.is_available() or args.device != 'mps' else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    marginals = data['marginals']
    grid = data['grid']
    u_teacher = data['u']
    h_teacher = data['h']
    
    grid_torch = torch.from_numpy(grid.astype(np.float32)).to(device)
    M = len(grid)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = ImprovedTransformerMMOT(M=M, d_model=128, n_heads=4, n_layers=4).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Test on all validation instances
    print(f"\nTesting on {len(marginals)} validation instances...")
    print("="*60)
    
    all_max_drifts = []
    all_mean_drifts = []
    teacher_drifts = []
    
    with torch.no_grad():
        for i in range(len(marginals)):
            m = torch.from_numpy(marginals[i].astype(np.float32)).unsqueeze(0).to(device)
            N = m.shape[1] - 1
            
            # Model prediction
            u_pred, h_pred = model(m, grid_torch)
            u_pred = u_pred[0, :N+1]
            h_pred = h_pred[0, :N]
            
            # Compute drift
            result = compute_drift_comprehensive(u_pred, h_pred, grid_torch, N)
            all_max_drifts.append(result['max_drift'])
            all_mean_drifts.append(result['mean_drift'])
            
            # Also compute teacher drift for reference
            u_t = torch.from_numpy(u_teacher[i].astype(np.float32)).to(device)
            h_t = torch.from_numpy(h_teacher[i].astype(np.float32)).to(device)
            teacher_result = compute_drift_comprehensive(u_t, h_t, grid_torch, N)
            teacher_drifts.append(teacher_result['max_drift'])
            
            if i < 5 or i == len(marginals) - 1:
                print(f"Instance {i}: Model Drift={result['max_drift']:.6f}, "
                      f"Teacher Drift={teacher_result['max_drift']:.6f}")
    
    print("="*60)
    
    # Statistics
    model_max = max(all_max_drifts)
    model_mean = np.mean(all_max_drifts)
    model_median = np.median(all_max_drifts)
    
    teacher_max = max(teacher_drifts)
    teacher_mean = np.mean(teacher_drifts)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Model Performance:")
    print(f"  Max Drift:    {model_max:.6f}")
    print(f"  Mean Drift:   {model_mean:.6f}")
    print(f"  Median Drift: {model_median:.6f}")
    print(f"  Pass Rate:    {100 * sum(1 for d in all_max_drifts if d < args.target) / len(all_max_drifts):.1f}%")
    print()
    print(f"Teacher (Reference):")
    print(f"  Max Drift:    {teacher_max:.6f}")
    print(f"  Mean Drift:   {teacher_mean:.6f}")
    print()
    print(f"Target: {args.target}")
    print("="*60)
    
    if model_max < args.target:
        print("🎉 SUCCESS! Model meets target WITHOUT projection!")
    elif model_mean < args.target:
        print("⚠️ PARTIAL: Mean drift OK, but some outliers exist.")
    else:
        print("❌ NOT YET: Model needs more training.")
    
    print("="*60)


if __name__ == '__main__':
    main()
