"""
Comprehensive Drift Analysis Script

Measures THREE types of drift to properly understand model performance:
1. Teacher Drift - Ground truth (should be ~0)
2. Student Drift - Neural approximation error
3. Approximation Gap - Student - Teacher

Author: Neural MMOT Team
Date: January 2026
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural.models.architecture import NeuralDualSolver


def compute_gibbs_drift(u, h, marginals, grid, epsilon=1.0):
    """
    Compute drift using Gibbs kernel reconstruction.
    
    This measures: |E[Y|X] - X| (martingale violation)
    
    Args:
        u: (N+1, M) potentials - can be teacher or student
        h: (N, M) martingale adjustments
        marginals: (N+1, M) probability distributions
        grid: (M,) price grid
        epsilon: entropic regularization (MUST match training!)
        
    Returns:
        mean_drift: scalar (lower is better)
    """
    device = u.device
    N = len(h)
    M = len(grid)
    total_drift = 0.0
    
    S = grid.to(device)
    
    for t in range(N):
        # Build cost matrix: delta_S[i,j] = S[j] - S[i] = y - x
        delta_S = S[None, :] - S[:, None]  # [M, M]
        
        # Get potentials
        u_tp1 = u[t+1]  # (M,) - varies over y (next state)
        h_t = h[t]      # (M,) - varies over x (current state)
        
        # Build Gibbs kernel: P(y|x) ∝ exp((u(y) + h(x)·(y-x)) / ε)
        # u_tp1[None, :] broadcasts u(y) over rows (x)
        # h_t[:, None] broadcasts h(x) over columns (y)
        log_kernel = (
            u_tp1[None, :] +           # [1, M] -> broadcast over x
            h_t[:, None] * delta_S     # [M, 1] * [M, M] -> h(x) * (y-x)
        ) / epsilon
        
        # Normalize: P(y|x) sums to 1 over y for each x
        kernel = F.softmax(log_kernel, dim=1)  # [M, M], sum over cols = 1
        
        # Conditional expectation: E[Y | X=x] = sum_y P(y|x) * y
        cond_exp = torch.matmul(kernel, S)  # [M]
        
        # Drift: |E[Y|X] - X| weighted by P(X=x)
        mu_t = marginals[t].to(device)
        drift = torch.abs(cond_exp - S) * mu_t
        total_drift += drift.sum().item()
    
    return total_drift / N


def comprehensive_drift_analysis(model, data_dir, grid, epsilon=1.0, 
                                  device='cpu', num_samples=50):
    """
    Perform comprehensive drift analysis comparing teacher vs student.
    
    Args:
        model: Trained NeuralDualSolver
        data_dir: Directory containing .npz files with u_star, h_star, marginals
        grid: Price grid tensor
        epsilon: Entropic regularization
        device: Compute device
        num_samples: Number of samples to analyze
        
    Returns:
        dict with teacher_drift, student_drift, approximation_gap, dual_magnitude
    """
    results = {
        'teacher_drift': [],
        'student_drift': [],
        'dual_magnitude': [],
        'u_error': [],
        'h_error': []
    }
    
    data_path = Path(data_dir)
    files = sorted(data_path.glob('*.npz'))[:num_samples]
    
    if not files:
        print(f"  ERROR: No .npz files found in {data_dir}")
        return None
    
    model = model.to(device)
    model.eval()
    grid = grid.to(device)
    
    for file in tqdm(files, desc="  Analyzing drift"):
        try:
            data = np.load(file, allow_pickle=True)
            
            # Load data - note: field names are u_star/h_star, not u_potentials/h_potentials
            marginals = torch.from_numpy(data['marginals']).float()
            u_true = torch.from_numpy(data['u_star']).float().to(device)
            h_true = torch.from_numpy(data['h_star']).float().to(device)
            
            # Get neural predictions
            with torch.no_grad():
                u_pred, h_pred = model(marginals.unsqueeze(0).to(device))
            u_pred = u_pred[0]  # Remove batch dim
            h_pred = h_pred[0]
            
            # METRIC 1: Teacher Drift (ground truth - should be ~0)
            teacher_drift = compute_gibbs_drift(
                u_true, h_true, marginals, grid, epsilon
            )
            results['teacher_drift'].append(teacher_drift)
            
            # METRIC 2: Student Drift (neural approximation error)
            student_drift = compute_gibbs_drift(
                u_pred, h_pred, marginals, grid, epsilon
            )
            results['student_drift'].append(student_drift)
            
            # METRIC 3: Dual Magnitude (|h| mean)
            dual_mag = h_pred.abs().mean().item()
            results['dual_magnitude'].append(dual_mag)
            
            # METRIC 4: Direct potential errors
            u_error = (u_pred - u_true).abs().mean().item()
            h_error = (h_pred - h_true).abs().mean().item()
            results['u_error'].append(u_error)
            results['h_error'].append(h_error)
            
        except Exception as e:
            # Skip problematic files
            continue
    
    if not results['teacher_drift']:
        print("  ERROR: No valid samples processed")
        return None
    
    # Compute summary statistics
    summary = {
        'teacher_drift_mean': float(np.mean(results['teacher_drift'])),
        'teacher_drift_std': float(np.std(results['teacher_drift'])),
        'student_drift_mean': float(np.mean(results['student_drift'])),
        'student_drift_std': float(np.std(results['student_drift'])),
        'approximation_gap': float(np.mean(results['student_drift']) - 
                                   np.mean(results['teacher_drift'])),
        'dual_magnitude_mean': float(np.mean(results['dual_magnitude'])),
        'u_error_mean': float(np.mean(results['u_error'])),
        'h_error_mean': float(np.mean(results['h_error'])),
        'samples_analyzed': len(results['teacher_drift'])
    }
    
    return summary


def print_drift_report(summary, epsilon):
    """Print formatted drift analysis report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE DRIFT ANALYSIS REPORT")
    print(f"Epsilon: {epsilon}")
    print("="*80)
    
    print(f"\n📊 Teacher Drift (Ground Truth):")
    print(f"   Mean: {summary['teacher_drift_mean']:.6f}")
    print(f"   Std:  {summary['teacher_drift_std']:.6f}")
    print(f"   → Expected: <0.01 (classical solver should be near-perfect)")
    status = "✅ PASS" if summary['teacher_drift_mean'] < 0.05 else "❌ FAIL"
    print(f"   → Status: {status}")
    
    print(f"\n📊 Student Drift (Neural Approximation):")
    print(f"   Mean: {summary['student_drift_mean']:.6f}")
    print(f"   Std:  {summary['student_drift_std']:.6f}")
    print(f"   → This measures how well NN approximates teacher")
    print(f"   → Target: <0.2 after epsilon fix")
    status = "✅ PASS" if summary['student_drift_mean'] < 0.2 else "⚠️ NEEDS WORK"
    print(f"   → Status: {status}")
    
    print(f"\n📊 Approximation Gap (Student - Teacher):")
    print(f"   Gap: {summary['approximation_gap']:.6f}")
    print(f"   → This is the TRUE measure of your model's constraint error")
    print(f"   → Target: <0.2")
    
    print(f"\n📊 Dual Magnitude (|h| mean):")
    print(f"   Mean: {summary['dual_magnitude_mean']:.4f}")
    print(f"   → Alternative stability metric")
    print(f"   → Target: <0.5")
    
    print(f"\n📊 Potential Errors:")
    print(f"   |u_pred - u_true|: {summary['u_error_mean']:.4f}")
    print(f"   |h_pred - h_true|: {summary['h_error_mean']:.4f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    
    if summary['teacher_drift_mean'] < 0.05:
        print("✅ Teacher potentials satisfy martingale (as expected)")
    else:
        print("⚠️ Teacher drift unexpectedly high - check data generation")
    
    if summary['approximation_gap'] < 0.2:
        print("✅ Neural network approximation is within acceptable bounds")
    else:
        print("⚠️ Approximation gap too large - needs more training or tuning")
    
    print("="*80 + "\n")


def main():
    """Run comprehensive drift analysis."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Comprehensive Drift Analysis')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='neural/data/val',  # Use validation data
                       help='Directory with test data')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Entropic regularization (MUST match training)')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to analyze')
    parser.add_argument('--save-json', type=str, default=None,
                       help='Save results to JSON file')
    args = parser.parse_args()
    
    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load config
    config_path = Path('neural/configs/default.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create model config with proper structure
    model_config = {
        'grid_size': config['model']['grid_size'],
        'hidden_dim': config['model']['hidden_dim'],
        'num_layers': config['model']['num_layers'],
        'num_heads': config['model']['num_heads'],
        'dropout': config['model']['dropout']
    }
    
    # Create grid
    grid_size = model_config['grid_size']
    grid = torch.linspace(0, 1, grid_size)
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model = NeuralDualSolver(**model_config)  # Unpack as keyword args
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Run analysis
    print(f"\nAnalyzing {args.num_samples} samples from {args.data_dir}...")
    summary = comprehensive_drift_analysis(
        model=model,
        data_dir=args.data_dir,
        grid=grid,
        epsilon=args.epsilon,
        device=device,
        num_samples=args.num_samples
    )
    
    if summary:
        print_drift_report(summary, args.epsilon)
        
        # Save to JSON if requested
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✅ Results saved to: {args.save_json}")
    else:
        print("❌ Analysis failed - check data directory")


if __name__ == '__main__':
    main()
