"""
Robust training script with Huber loss + data augmentation.

Same architecture, better training for real market generalization.
"""

import argparse
import sys
import yaml
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import NeuralDualSolver
from training.robust_loss import RobustMMOTLoss
from training.augmentation import augment_marginals_simple
from data.loader import MMOTDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def train_robust(config_path='neural/configs/robust_real_data.yaml'):
    """Train with robust configuration."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['hardware']['device']
    print("="*80)
    print("ROBUST TRAINING FOR REAL MARKET GENERALIZATION")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Config: {config_path}")
    
    # Data
    train_dataset = MMOTDataset(config['data']['train_dir'])
    val_dataset = MMOTDataset(config['data']['val_dir'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"\nData:")
    print(f"  Train: {len(train_dataset)} instances")
    print(f"  Val: {len(val_dataset)} instances")
    
    # Model (SAME architecture, just robust training)
    model = NeuralDualSolver(
        grid_size=config['model']['grid_size'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout']  # 0.2 instead of 0.1
    ).to(device)
    
    print(f"\nModel:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Dropout: {config['model']['dropout']}")
    
    # Loss (Huber instead of MSE)
    grid = torch.linspace(0, 1, config['model']['grid_size']).to(device)
    
    loss_fn = RobustMMOTLoss(
        grid=grid,
        epsilon=config['loss']['epsilon'],
        lambda_distill=config['loss']['lambda_distill'],
        lambda_martingale=config['loss']['lambda_martingale'],
        lambda_drift=config['loss']['lambda_drift'],
        lambda_marginal=config['loss']['lambda_marginal'],
        huber_delta=config['loss']['huber_delta']
    )
    
    print(f"\nLoss:")
    print(f"  Type: Huber (robust to outliers)")
    print(f"  λ_martingale: {config['loss']['lambda_martingale']}")
    print(f"  λ_drift: {config['loss']['lambda_drift']}")
    print(f"  Huber δ: {config['loss']['huber_delta']}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['optimizer']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config['scheduler']['patience'],
        factor=config['scheduler']['factor'],
        min_lr=config['scheduler']['min_lr']
    )
    
    print(f"\nOptimizer:")
    print(f"  LR: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['optimizer']['weight_decay']}")
    print(f"  Scheduler: ReduceLROnPlateau")
    
    # Training loop
    best_val_loss = float('inf')
    epochs = config['training']['epochs']
    
    # Data augmentation config
    aug_config = config.get('data_augmentation', {})
    use_aug = aug_config.get('marginal_noise_std', 0) > 0
    
    print(f"\nData Augmentation:")
    print(f"  Enabled: {use_aug}")
    if use_aug:
        print(f"  Marginal noise: {aug_config['marginal_noise_std']}")
    
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING - {epochs} EPOCHS")
    print(f"{'='*80}\n")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            marginals = batch['marginals'].to(device)
            u_true = batch['u_star'].to(device)
            h_true = batch['h_star'].to(device)
            
            # Data augmentation
            if use_aug:
                marginals = augment_marginals_simple(
                    marginals,
                    noise_std=aug_config['marginal_noise_std']
                )
            
            # Forward
            u_pred, h_pred = model(marginals)
            
            loss, loss_dict = loss_fn(u_pred, h_pred, u_true, h_true, marginals)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{np.mean(train_losses):.4f}"})
        
        # Validate
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                marginals = batch['marginals'].to(device)
                u_true = batch['u_star'].to(device)
                h_true = batch['h_star'].to(device)
                
                u_pred, h_pred = model(marginals)
                loss, _ = loss_fn(u_pred, h_pred, u_true, h_true, marginals)
                
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, 'checkpoints/robust/best_model.pt')
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"{'='*80}")
    
    return model


if __name__ == '__main__':
    import os
    os.makedirs('checkpoints/robust', exist_ok=True)
    model = train_robust()
