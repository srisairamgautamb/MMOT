#!/usr/bin/env python3
"""
Systematic Ablation Study for Neural MMOT

Tests each architectural component to understand its contribution.
This provides scientific justification for design choices.
"""

import sys
import os
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time

from neural.models.architecture import NeuralDualSolver
from neural.data.loader import MMOTDataset
from neural.training.loss import MMOTLoss


def train_config(config, train_loader, val_loader, device, epochs=30):
    """Train a model with given config and return metrics."""
    
    model = NeuralDualSolver(
        grid_size=config.get('grid_size', 150),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 3),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.0)
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    grid = torch.linspace(0, 1, 150).to(device)
    lambda_mart = config.get('lambda_martingale', 5.0)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            marginals = batch['marginals'].to(device)
            u_true = batch['u_star'].to(device)
            h_true = batch['h_star'].to(device)
            
            optimizer.zero_grad()
            u_pred, h_pred = model(marginals)
            
            # Simple distillation loss
            loss_distill = F.mse_loss(u_pred, u_true) + F.mse_loss(h_pred, h_true)
            
            # Simple martingale loss (drift)
            drift_loss = compute_batch_drift_loss(u_pred, h_pred, marginals, grid)
            
            loss = loss_distill + lambda_mart * drift_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                marginals = batch['marginals'].to(device)
                u_true = batch['u_star'].to(device)
                h_true = batch['h_star'].to(device)
                u_pred, h_pred = model(marginals)
                val_loss = F.mse_loss(u_pred, u_true) + F.mse_loss(h_pred, h_true)
                val_losses.append(val_loss.item())
        
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # Final evaluation
    return evaluate_model(model, val_loader, grid, device)


def compute_batch_drift_loss(u, h, marginals, grid):
    """Compute average drift loss for a batch."""
    B, N_plus_1, M = u.shape
    N = N_plus_1 - 1
    epsilon = 1.0
    
    total_drift = 0.0
    for t in range(min(N, 3)):  # Just first 3 steps for speed
        u_tp1 = u[:, t+1]  # (B, M)
        h_t = h[:, t]      # (B, M)
        
        # Gibbs kernel: K(x,y) ∝ exp((u(y) + h(x)*(y-x))/ε)
        delta_S = grid[None, :] - grid[:, None]  # (M, M)
        log_kernel = (u_tp1[:, None, :] + h_t[:, :, None] * delta_S[None]) / epsilon
        kernel = F.softmax(log_kernel, dim=-1)  # (B, M, M)
        
        # Conditional expectation E[Y|X]
        cond_exp = torch.matmul(kernel, grid)  # (B, M)
        
        # Drift = |E[Y|X] - X|
        drift = torch.abs(cond_exp - grid[None, :])  # (B, M)
        
        # Weight by marginal
        mu_t = marginals[:, t]  # (B, M)
        weighted_drift = (drift * mu_t).sum(dim=-1).mean()
        total_drift += weighted_drift
    
    return total_drift / min(N, 3)


def evaluate_model(model, loader, grid, device):
    """Evaluate model and return metrics."""
    model.eval()
    
    errors = []
    drifts = []
    
    with torch.no_grad():
        for batch in loader:
            marginals = batch['marginals'].to(device)
            u_true = batch['u_star'].to(device)
            h_true = batch['h_star'].to(device)
            
            u_pred, h_pred = model(marginals)
            
            # Compute error
            u_scale = u_true.abs().mean() + 1e-8
            error = (u_pred - u_true).abs().mean() / u_scale * 100
            errors.append(error.item())
            
            # Compute drift
            for i in range(marginals.shape[0]):
                drift = compute_drift(u_pred[i], h_pred[i], marginals[i], grid)
                drifts.append(drift)
    
    return {
        'error': np.mean(errors),
        'error_std': np.std(errors),
        'drift': np.mean(drifts),
        'drift_std': np.std(drifts)
    }


def compute_drift(u, h, marginals, grid):
    """Compute martingale drift for one instance."""
    N = h.shape[0]
    M = len(grid)
    epsilon = 1.0
    
    total_drift = 0.0
    for t in range(N):
        u_tp1 = u[t+1]
        h_t = h[t]
        
        delta_S = grid[None, :] - grid[:, None]
        log_kernel = (u_tp1[None, :] + h_t[:, None] * delta_S) / epsilon
        kernel = F.softmax(log_kernel, dim=1)
        cond_exp = torch.matmul(kernel, grid)
        drift_x = torch.abs(cond_exp - grid)
        
        mu_t = marginals[t]
        weighted_drift = (drift_x * mu_t).sum().item()
        total_drift += weighted_drift
    
    return total_drift / N


def run_ablation_study():
    """Run complete ablation study."""
    
    print("=" * 80)
    print("SYSTEMATIC ABLATION STUDY")
    print("=" * 80)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    train_dataset = MMOTDataset('neural/data/train')
    val_dataset = MMOTDataset('neural/data/val')
    
    # Use subset for faster ablation
    train_subset = torch.utils.data.Subset(train_dataset, range(min(2000, len(train_dataset))))
    val_subset = torch.utils.data.Subset(val_dataset, range(min(500, len(val_dataset))))
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32)
    
    print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")
    
    results = {}
    
    # BASELINE
    print("\n" + "=" * 80)
    print("BASELINE MODEL (hidden=256, layers=3, λ=5.0)")
    print("=" * 80)
    
    baseline_config = {
        'hidden_dim': 256,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.0,
        'lambda_martingale': 5.0,
        'learning_rate': 1e-4
    }
    
    start = time.time()
    results['baseline'] = train_config(baseline_config, train_loader, val_loader, device)
    print(f"  Time: {time.time()-start:.1f}s")
    print(f"  Error: {results['baseline']['error']:.2f}%")
    print(f"  Drift: {results['baseline']['drift']:.4f}")
    
    # ABLATION 1: Model Capacity
    print("\n" + "=" * 80)
    print("ABLATION 1: MODEL CAPACITY STUDY")
    print("=" * 80)
    
    for hidden in [128, 256, 384, 512]:
        config = baseline_config.copy()
        config['hidden_dim'] = hidden
        config['num_heads'] = max(4, hidden // 64)
        
        print(f"\n  Testing hidden_dim={hidden}...")
        start = time.time()
        results[f'hidden_{hidden}'] = train_config(config, train_loader, val_loader, device)
        print(f"    Time: {time.time()-start:.1f}s")
        print(f"    Error: {results[f'hidden_{hidden}']['error']:.2f}%")
        print(f"    Drift: {results[f'hidden_{hidden}']['drift']:.4f}")
    
    # ABLATION 2: Number of Layers
    print("\n" + "=" * 80)
    print("ABLATION 2: DEPTH STUDY")
    print("=" * 80)
    
    for layers in [2, 3, 4, 5, 6]:
        config = baseline_config.copy()
        config['num_layers'] = layers
        
        print(f"\n  Testing num_layers={layers}...")
        start = time.time()
        results[f'layers_{layers}'] = train_config(config, train_loader, val_loader, device)
        print(f"    Time: {time.time()-start:.1f}s")
        print(f"    Error: {results[f'layers_{layers}']['error']:.2f}%")
        print(f"    Drift: {results[f'layers_{layers}']['drift']:.4f}")
    
    # ABLATION 3: Lambda Martingale
    print("\n" + "=" * 80)
    print("ABLATION 3: MARTINGALE LOSS WEIGHT")
    print("=" * 80)
    
    for lam in [1.0, 2.5, 5.0, 10.0, 20.0]:
        config = baseline_config.copy()
        config['lambda_martingale'] = lam
        
        print(f"\n  Testing λ_martingale={lam}...")
        start = time.time()
        results[f'lambda_{lam}'] = train_config(config, train_loader, val_loader, device)
        print(f"    Time: {time.time()-start:.1f}s")
        print(f"    Error: {results[f'lambda_{lam}']['error']:.2f}%")
        print(f"    Drift: {results[f'lambda_{lam}']['drift']:.4f}")
    
    # ABLATION 4: Dropout
    print("\n" + "=" * 80)
    print("ABLATION 4: REGULARIZATION")
    print("=" * 80)
    
    for dropout in [0.0, 0.1, 0.2, 0.3]:
        config = baseline_config.copy()
        config['dropout'] = dropout
        
        print(f"\n  Testing dropout={dropout}...")
        start = time.time()
        results[f'dropout_{dropout}'] = train_config(config, train_loader, val_loader, device)
        print(f"    Time: {time.time()-start:.1f}s")
        print(f"    Error: {results[f'dropout_{dropout}']['error']:.2f}%")
        print(f"    Drift: {results[f'dropout_{dropout}']['drift']:.4f}")
    
    # ABLATION 5: Learning Rate
    print("\n" + "=" * 80)
    print("ABLATION 5: LEARNING RATE")
    print("=" * 80)
    
    for lr in [5e-5, 1e-4, 3e-4, 1e-3]:
        config = baseline_config.copy()
        config['learning_rate'] = lr
        
        print(f"\n  Testing lr={lr}...")
        start = time.time()
        results[f'lr_{lr}'] = train_config(config, train_loader, val_loader, device)
        print(f"    Time: {time.time()-start:.1f}s")
        print(f"    Error: {results[f'lr_{lr}']['error']:.2f}%")
        print(f"    Drift: {results[f'lr_{lr}']['drift']:.4f}")
    
    # SUMMARY
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    
    # Find best configurations
    print("\n" + "-" * 80)
    print("CAPACITY STUDY:")
    print("-" * 80)
    for hidden in [128, 256, 384, 512]:
        r = results[f'hidden_{hidden}']
        marker = "★" if r['drift'] < 0.1 and r['error'] < 1.0 else " "
        print(f"  {marker} hidden={hidden:3d}: Error={r['error']:.2f}%, Drift={r['drift']:.4f}")
    
    print("\n" + "-" * 80)
    print("DEPTH STUDY:")
    print("-" * 80)
    for layers in [2, 3, 4, 5, 6]:
        r = results[f'layers_{layers}']
        marker = "★" if r['drift'] < 0.1 and r['error'] < 1.0 else " "
        print(f"  {marker} layers={layers}: Error={r['error']:.2f}%, Drift={r['drift']:.4f}")
    
    print("\n" + "-" * 80)
    print("MARTINGALE WEIGHT STUDY:")
    print("-" * 80)
    for lam in [1.0, 2.5, 5.0, 10.0, 20.0]:
        r = results[f'lambda_{lam}']
        marker = "★" if r['drift'] < 0.1 and r['error'] < 1.0 else " "
        print(f"  {marker} λ={lam:4.1f}: Error={r['error']:.2f}%, Drift={r['drift']:.4f}")
    
    print("\n" + "-" * 80)
    print("REGULARIZATION STUDY:")
    print("-" * 80)
    for dropout in [0.0, 0.1, 0.2, 0.3]:
        r = results[f'dropout_{dropout}']
        marker = "★" if r['drift'] < 0.1 and r['error'] < 1.0 else " "
        print(f"  {marker} dropout={dropout:.1f}: Error={r['error']:.2f}%, Drift={r['drift']:.4f}")
    
    print("\n" + "-" * 80)
    print("LEARNING RATE STUDY:")
    print("-" * 80)
    for lr in [5e-5, 1e-4, 3e-4, 1e-3]:
        r = results[f'lr_{lr}']
        marker = "★" if r['drift'] < 0.1 and r['error'] < 1.0 else " "
        print(f"  {marker} lr={lr:.0e}: Error={r['error']:.2f}%, Drift={r['drift']:.4f}")
    
    # Find optimal config
    valid_configs = {k: v for k, v in results.items() if v['drift'] < 0.15}
    if valid_configs:
        best_key = min(valid_configs.keys(), key=lambda k: valid_configs[k]['error'])
        best = valid_configs[best_key]
        print("\n" + "=" * 80)
        print(f"OPTIMAL CONFIGURATION: {best_key}")
        print("=" * 80)
        print(f"  Error: {best['error']:.2f}%")
        print(f"  Drift: {best['drift']:.4f}")
    
    # Save results
    output_path = Path('neural/results/ablation_study.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: float(vv) for kk, vv in v.items()}
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n✅ Results saved: {output_path}")
    
    return results


if __name__ == '__main__':
    results = run_ablation_study()
