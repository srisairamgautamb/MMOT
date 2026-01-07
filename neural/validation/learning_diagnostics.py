"""
Learning Diagnostics Suite

Comprehensive analysis to determine model fitting status:
- Underfitting (high train + high val error)
- Overfitting (low train + high val error)  
- Well-fit (low train + low val error)

Author: Neural MMOT Team
Date: January 2, 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural.models.architecture import NeuralDualSolver
from neural.data.loader import MMOTDataset


def compute_detailed_errors(model, dataloader, grid, epsilon, device):
    """Compute all error metrics on a dataset."""
    model.eval()
    
    total_losses = []
    u_errors = []
    h_errors = []
    drifts = []
    relative_errors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            marginals = batch['marginals'].to(device)
            u_true = batch['u_star'].to(device)
            h_true = batch['h_star'].to(device)
            
            # Forward pass
            u_pred, h_pred = model(marginals)
            
            # U and H errors
            u_err = torch.abs(u_pred - u_true).mean(dim=(1,2))  # Per-batch item
            h_err = torch.abs(h_pred - h_true).mean(dim=(1,2))
            
            # Distillation loss (same as training)
            distill = F.mse_loss(u_pred, u_true, reduction='none').mean(dim=(1,2)) + \
                      F.mse_loss(h_pred, h_true, reduction='none').mean(dim=(1,2))
            
            # Compute drift per instance
            for i in range(marginals.shape[0]):
                drift = compute_instance_drift(
                    u_pred[i], h_pred[i], marginals[i], grid, epsilon
                )
                drifts.append(drift)
            
            # Relative error as percentage of u_true scale
            u_scale = u_true.abs().mean(dim=(1,2)) + 1e-8
            rel_err = u_err / u_scale * 100  # As percentage
            
            u_errors.extend(u_err.cpu().tolist())
            h_errors.extend(h_err.cpu().tolist())
            total_losses.extend(distill.cpu().tolist())
            relative_errors.extend(rel_err.cpu().tolist())
    
    return {
        'total_loss': np.mean(total_losses),
        'total_loss_std': np.std(total_losses),
        'u_error': np.mean(u_errors),
        'u_error_std': np.std(u_errors),
        'h_error': np.mean(h_errors),
        'h_error_std': np.std(h_errors),
        'drift': np.mean(drifts),
        'drift_std': np.std(drifts),
        'relative_error': np.mean(relative_errors),
        'relative_error_std': np.std(relative_errors),
        'n_samples': len(total_losses)
    }


def compute_instance_drift(u, h, marginals, grid, epsilon):
    """Compute drift for a single instance."""
    N = h.shape[0]
    M = len(grid)
    device = u.device
    
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


def run_diagnostics(config_path='neural/configs/default.yaml',
                   checkpoint_path='checkpoints/best_model.pt'):
    """Main diagnostic routine."""
    
    print("=" * 80)
    print("LEARNING DIAGNOSTICS")
    print("Overfitting vs Underfitting Analysis")
    print("=" * 80)
    
    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_config = {
        'grid_size': config['model']['grid_size'],
        'hidden_dim': config['model']['hidden_dim'],
        'num_layers': config['model']['num_layers'],
        'num_heads': config['model']['num_heads'],
        'dropout': config['model']['dropout']
    }
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = NeuralDualSolver(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Create grid
    grid = torch.linspace(0, 1, model_config['grid_size']).to(device)
    epsilon = 1.0
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MMOTDataset('neural/data/train')
    val_dataset = MMOTDataset('neural/data/val')
    
    # Use subset for faster evaluation
    train_subset = torch.utils.data.Subset(
        train_dataset, range(min(500, len(train_dataset)))
    )
    val_subset = torch.utils.data.Subset(
        val_dataset, range(min(200, len(val_dataset)))
    )
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32)
    
    print(f"  Training samples: {len(train_subset)}")
    print(f"  Validation samples: {len(val_subset)}")
    
    # Compute metrics
    print("\n" + "-" * 80)
    print("Computing training set metrics...")
    train_metrics = compute_detailed_errors(model, train_loader, grid, epsilon, device)
    
    print("Computing validation set metrics...")
    val_metrics = compute_detailed_errors(model, val_loader, grid, epsilon, device)
    
    # Display comparison
    print("\n" + "=" * 80)
    print("METRIC COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'Train':<18} {'Validation':<18} {'Gap'}")
    print("-" * 80)
    
    train_loss = train_metrics['total_loss']
    val_loss = val_metrics['total_loss']
    gap = val_loss - train_loss
    
    print(f"{'Total Loss':<25} {train_loss:<18.2f} {val_loss:<18.2f} {gap:+.2f}")
    print(f"{'U Potential Error':<25} {train_metrics['u_error']:<18.4f} {val_metrics['u_error']:<18.4f}")
    print(f"{'H Potential Error':<25} {train_metrics['h_error']:<18.4f} {val_metrics['h_error']:<18.4f}")
    print(f"{'Relative Error (%)':<25} {train_metrics['relative_error']:<18.2f} {val_metrics['relative_error']:<18.2f}")
    print(f"{'Drift':<25} {train_metrics['drift']:<18.4f} {val_metrics['drift']:<18.4f}")
    
    print("-" * 80)
    
    # DIAGNOSIS
    print("\n" + "=" * 80)
    print("🔍 DIAGNOSIS")
    print("=" * 80)
    
    diagnosis = None
    
    # Loss thresholds for diagnosis
    HIGH_LOSS_THRESHOLD = 50  # Loss > 50 indicates poor fit
    OVERFITTING_GAP = 20       # gap > 20 indicates overfitting
    
    if train_loss > HIGH_LOSS_THRESHOLD and val_loss > HIGH_LOSS_THRESHOLD:
        diagnosis = 'underfitting'
        print("\n❌ UNDERFITTING DETECTED")
        print("\nSymptoms:")
        print(f"  • High training loss: {train_loss:.1f} (threshold: {HIGH_LOSS_THRESHOLD})")
        print(f"  • High validation loss: {val_loss:.1f}")
        print(f"  • Small train-val gap: {gap:.1f} (< {OVERFITTING_GAP} = no overfitting)")
        print("\nRoot Causes:")
        print("  1. Model capacity too small (hidden_dim=256 may be insufficient)")
        print("  2. Training duration too short (50 epochs not enough)")
        print("  3. Learning rate too low (1e-4 too conservative)")
        print(f"  4. Current model: {model_config['hidden_dim']} hidden, {model_config['num_layers']} layers")
        print("\n✅ RECOMMENDED FIXES:")
        print("  → Increase hidden_dim: 256 → 512")
        print("  → Add layers: 3 → 5")
        print("  → Train longer: 50 → 100 epochs") 
        print("  → Use LR schedule: 1e-3 warmup → 1e-5 final")
        print("  → Add attention heads: 4 → 8")
        
    elif gap > OVERFITTING_GAP:
        diagnosis = 'overfitting'
        print("\n❌ OVERFITTING DETECTED")
        print("\nSymptoms:")
        print(f"  • Low training loss: {train_loss:.1f}")
        print(f"  • High validation loss: {val_loss:.1f}")
        print(f"  • Large gap: {gap:.1f} (> {OVERFITTING_GAP})")
        print("\n✅ RECOMMENDED FIXES:")
        print("  → Increase dropout: 0.1 → 0.2")
        print("  → Add weight decay: 1e-4")
        print("  → More training data")
        print("  → Early stopping")
        print("  → Data augmentation")
        
    else:
        diagnosis = 'well_fit'
        print("\n✅ WELL-FIT MODEL")
        print("\nSymptoms:")
        print(f"  • Training loss: {train_loss:.1f}")
        print(f"  • Validation loss: {val_loss:.1f}")
        print(f"  • Small gap: {gap:.1f} (no overfitting)")
        print("\nInterpretation:")
        print("  → Model is learning optimally given current capacity")
        print("  → Current architecture may be near optimal for this data")
        
        if val_metrics['relative_error'] > 1.5:
            print("\n⚠️ BUT: Relative error still above target (1.2%)")
            print("  → Consider increasing model capacity anyway")
            print("  → Or accept this as near-optimal for current data quality")
    
    # Drift check
    print("\n" + "-" * 80)
    print("MARTINGALE CONSTRAINT CHECK:")
    print("-" * 80)
    print(f"  Train Drift: {train_metrics['drift']:.4f}")
    print(f"  Val Drift:   {val_metrics['drift']:.4f}")
    
    if val_metrics['drift'] < 0.1:
        print("  ✅ DRIFT < 0.1: Martingale constraint SATISFIED")
    else:
        print("  ❌ DRIFT >= 0.1: Martingale constraint VIOLATED")
    
    print("=" * 80)
    
    # Return results
    results = {
        'diagnosis': diagnosis,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'gap': gap,
        'model_config': model_config,
        'recommendations': get_recommendations(diagnosis)
    }
    
    # Save results
    output_path = Path('neural/results/validation/diagnostics.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'diagnosis': diagnosis,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'gap': gap,
            'train_drift': train_metrics['drift'],
            'val_drift': val_metrics['drift'],
            'relative_error': val_metrics['relative_error']
        }, f, indent=2)
    print(f"\n✅ Results saved: {output_path}")
    
    return results


def get_recommendations(diagnosis):
    """Get specific recommendations based on diagnosis."""
    if diagnosis == 'underfitting':
        return {
            'hidden_dim': 512,
            'num_layers': 5,
            'num_heads': 8,
            'epochs': 100,
            'learning_rate': 1e-3,
            'scheduler': 'cosine_warmup',
            'warmup_epochs': 10
        }
    elif diagnosis == 'overfitting':
        return {
            'dropout': 0.2,
            'weight_decay': 1e-4,
            'epochs': 50,
            'early_stopping': True,
            'patience': 10
        }
    else:  # well_fit
        return {
            'status': 'current_config_good',
            'optional': 'try_512_hidden_for_marginal_gains'
        }


if __name__ == '__main__':
    results = run_diagnostics()
    print(f"\nDiagnosis: {results['diagnosis'].upper()}")
