#!/usr/bin/env python3
"""
train_stage2_resume.py
======================
Continuation training with 3-Stage Protocol for MMOT Neural Solver.

Stage 1 (Epochs 1-100): Already complete - learned general u, h patterns.
Stage 2 (Epochs 101-180): CONSTRAINT LEARNING - Heavy martingale weight.
Stage 3 (Epochs 181-250): FINE-TUNING - Maximize martingale accuracy.

Target: Drift < 0.01 WITHOUT Newton projection.

Author: MMOT Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm

# Add path for imports
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
from architecture_fixed import ImprovedTransformerMMOT


class MoneynessDataset(Dataset):
    """Dataset for moneyness-space MMOT teacher data."""
    
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.marginals = data['marginals']  # List of (N+1, M) arrays
        self.u = data['u']  # List of (N+1, M) arrays
        self.h = data['h']  # List of (N, M) arrays
        self.grid = data['grid']  # (M,) array
        
        # Find max N for padding
        self.max_N = max(m.shape[0] for m in self.marginals) - 1
        self.M = len(self.grid)
        
    def __len__(self):
        return len(self.marginals)
    
    def __getitem__(self, idx):
        m = self.marginals[idx]  # (N+1, M)
        u = self.u[idx]  # (N+1, M)
        h = self.h[idx]  # (N, M)
        
        N = m.shape[0] - 1
        
        # Pad to max_N+1 for marginals/u, max_N for h
        m_padded = np.zeros((self.max_N + 1, self.M), dtype=np.float32)
        u_padded = np.zeros((self.max_N + 1, self.M), dtype=np.float32)
        h_padded = np.zeros((self.max_N, self.M), dtype=np.float32)
        
        m_padded[:N+1] = m
        u_padded[:N+1] = u
        h_padded[:N] = h
        
        return {
            'marginals': torch.from_numpy(m_padded),
            'u': torch.from_numpy(u_padded),
            'h': torch.from_numpy(h_padded),
            'N': N,
            'grid': torch.from_numpy(self.grid.astype(np.float32))
        }


def compute_martingale_loss(u_pred, h_pred, marginals, grid, N_actual, epsilon=0.2):
    """
    Compute the EXACT martingale constraint violation.
    
    The martingale condition is: E[X_{t+1} | X_t = x] = x for all t, x.
    
    This is computed as:
    1. Build the transition kernel from potentials u, h
    2. Compute conditional expectation E[Y|X=x]
    3. Measure deviation from identity
    
    Args:
        u_pred: (batch, N+1, M) predicted u potentials
        h_pred: (batch, N, M) predicted h potentials
        marginals: (batch, N+1, M) marginal distributions
        grid: (M,) spatial grid
        N_actual: (batch,) actual number of time steps per sample
        epsilon: regularization parameter (matches solver)
    
    Returns:
        martingale_loss: scalar, mean squared drift violation
        max_drift: scalar, maximum absolute drift (for monitoring)
    """
    batch_size, _, M = u_pred.shape
    device = u_pred.device
    
    # Precompute grid differences
    x = grid  # (M,)
    x_i = x.unsqueeze(1)  # (M, 1)
    x_j = x.unsqueeze(0)  # (1, M)
    Delta = x_i - x_j  # (M, M) - matches solver definition (x - y)
    
    # Cost matrix
    C = Delta ** 2
    C_max = C.max().item()
    C_scaled = C / C_max
    
    total_drift_loss = 0.0
    max_drift_value = 0.0
    count = 0
    
    for b in range(batch_size):
        N = N_actual[b].item()
        
        for t in range(N):
            u_t = u_pred[b, t]      # (M,)
            u_next = u_pred[b, t+1]  # (M,)
            h_t = h_pred[b, t]       # (M,)
            
            # Build log kernel: LogK[i,j] = (u_t[i] + u_next[j] + h_t[i] * Delta[i,j] - C[i,j]) / epsilon
            term_u = u_t.unsqueeze(1) + u_next.unsqueeze(0)  # (M, M)
            term_h = h_t.unsqueeze(1) * Delta  # (M, M)
            
            LogK = (term_u + term_h - C_scaled) / epsilon
            
            # Apply softmax to get transition probabilities P(y|x)
            # P[i, j] = P(X_{t+1} = x_j | X_t = x_i)
            probs = F.softmax(LogK, dim=1)  # (M, M)
            
            # Compute conditional expectation E[X_{t+1} | X_t = x]
            expected_y = torch.sum(probs * x.unsqueeze(0), dim=1)  # (M,)
            
            # Drift = E[Y|X=x] - x
            drift = expected_y - x  # (M,)
            
            # Squared drift loss (weighted by marginal)
            mu_t = marginals[b, t]  # (M,)
            weighted_drift_sq = (drift ** 2) * (mu_t + 1e-10)  # Weight by marginal mass
            
            drift_loss = weighted_drift_sq.sum()
            total_drift_loss += drift_loss
            
            # Track max drift for monitoring
            max_drift_this = drift.abs().max().item()
            if max_drift_this > max_drift_value:
                max_drift_value = max_drift_this
            
            count += 1
    
    mean_drift_loss = total_drift_loss / (count + 1e-10)
    
    return mean_drift_loss, max_drift_value


def get_loss_weights(epoch):
    """
    Dynamic loss weights across 3-stage training.
    
    Stage 2 (101-180): Shift focus to martingale constraint
    Stage 3 (181-250): Maximum martingale focus, minimal distillation
    """
    if epoch < 101:
        # Stage 1: Warm-up (already done)
        return {'distill': 1.0, 'martingale': 1.0, 'reg': 0.5}
    elif epoch < 181:
        # Stage 2: Constraint Learning
        # Linearly decrease distill, increase martingale
        progress = (epoch - 101) / 80  # 0 to 1
        distill_weight = 1.0 - 0.9 * progress  # 1.0 -> 0.1
        mart_weight = 5.0 + 15.0 * progress  # 5.0 -> 20.0
        return {'distill': distill_weight, 'martingale': mart_weight, 'reg': 0.5}
    else:
        # Stage 3: Fine-tuning
        progress = (epoch - 181) / 69  # 0 to 1
        distill_weight = 0.1 - 0.09 * progress  # 0.1 -> 0.01
        mart_weight = 20.0 + 30.0 * progress  # 20.0 -> 50.0
        return {'distill': distill_weight, 'martingale': mart_weight, 'reg': 1.0}


class ConstraintTrainer:
    """
    Trainer with martingale constraint focus.
    """
    
    def __init__(self, model, device='mps', lr=1e-4, weight_decay=0.01):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=150,  # 150 more epochs
            eta_min=1e-6
        )
        
        # Early stopping
        self.best_val_drift = float('inf')
        self.patience_counter = 0
        self.patience = 25
        
    def train_epoch(self, dataloader, epoch, grid):
        self.model.train()
        total_loss = 0.0
        total_distill = 0.0
        total_mart = 0.0
        max_drift = 0.0
        
        weights = get_loss_weights(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            marginals = batch['marginals'].to(self.device)
            u_teacher = batch['u'].to(self.device)
            h_teacher = batch['h'].to(self.device)
            N_actual = batch['N']
            grid_batch = grid.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            u_pred, h_pred = self.model(marginals, grid_batch)
            
            # Distillation loss (MSE to teacher)
            L_distill = F.mse_loss(u_pred, u_teacher) + F.mse_loss(h_pred, h_teacher)
            
            # Martingale constraint loss (CRITICAL!)
            L_mart, batch_max_drift = compute_martingale_loss(
                u_pred, h_pred, marginals, grid_batch, N_actual
            )
            
            # Regularization (prevent h from exploding)
            L_reg = h_pred.pow(2).mean()
            
            # Total loss
            loss = (weights['distill'] * L_distill + 
                   weights['martingale'] * L_mart + 
                   weights['reg'] * L_reg)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_distill += L_distill.item()
            total_mart += L_mart.item()
            if batch_max_drift > max_drift:
                max_drift = batch_max_drift
            
            pbar.set_postfix({
                'loss': f"{loss.item():.2f}",
                'drift': f"{batch_max_drift:.4f}",
                'wd': f"{weights['distill']:.2f}",
                'wm': f"{weights['martingale']:.1f}"
            })
        
        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'distill': total_distill / n_batches,
            'martingale': total_mart / n_batches,
            'max_drift': max_drift
        }
    
    @torch.no_grad()
    def validate(self, dataloader, grid):
        self.model.eval()
        total_loss = 0.0
        max_drift = 0.0
        all_drifts = []
        
        for batch in dataloader:
            marginals = batch['marginals'].to(self.device)
            u_teacher = batch['u'].to(self.device)
            h_teacher = batch['h'].to(self.device)
            N_actual = batch['N']
            grid_batch = grid.to(self.device)
            
            u_pred, h_pred = self.model(marginals, grid_batch)
            
            L_distill = F.mse_loss(u_pred, u_teacher) + F.mse_loss(h_pred, h_teacher)
            L_mart, batch_max_drift = compute_martingale_loss(
                u_pred, h_pred, marginals, grid_batch, N_actual
            )
            
            total_loss += L_distill.item()
            all_drifts.append(batch_max_drift)
            if batch_max_drift > max_drift:
                max_drift = batch_max_drift
        
        n_batches = len(dataloader)
        mean_drift = np.mean(all_drifts)
        
        return {
            'loss': total_loss / n_batches,
            'max_drift': max_drift,
            'mean_drift': mean_drift
        }
    
    def check_early_stopping(self, val_drift, epoch, save_path):
        """Check if training should stop early."""
        if val_drift < self.best_val_drift:
            self.best_val_drift = val_drift
            self.patience_counter = 0
            torch.save(self.model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            return False, "improved"
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True, "patience_exceeded"
            return False, "no_improvement"


def main():
    parser = argparse.ArgumentParser(description='Stage 2-3 Training for MMOT Neural Solver')
    parser.add_argument('--train_data', type=str, default='data/mmot_teacher_12000_moneyness.npz')
    parser.add_argument('--val_data', type=str, default='data/validation_solved.npz')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--start_epoch', type=int, default=101)
    parser.add_argument('--end_epoch', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--output', type=str, default='checkpoints/')
    parser.add_argument('--target_drift', type=float, default=0.01)
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print(f"Loading training data from {args.train_data}...")
    train_ds = MoneynessDataset(args.train_data)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    
    print(f"Loading validation data from {args.val_data}...")
    val_ds = MoneynessDataset(args.val_data)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    grid = torch.from_numpy(train_ds.grid.astype(np.float32))
    M = len(grid)
    
    # Load model from checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = ImprovedTransformerMMOT(M=M, d_model=128, n_heads=4, n_layers=4)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    
    # Create trainer
    trainer = ConstraintTrainer(model, device=args.device, lr=5e-5)  # Lower LR for fine-tuning
    
    # Training loop
    print("="*60)
    print(f"STAGE 2-3 TRAINING: Epochs {args.start_epoch} to {args.end_epoch}")
    print(f"Target Drift: {args.target_drift}")
    print("="*60)
    
    log_file = open(os.path.join(args.output, 'stage2_training.log'), 'w')
    log_file.write("epoch,train_loss,train_drift,val_loss,val_drift,lr,status\n")
    
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch, grid)
        
        # Validate
        val_metrics = trainer.validate(val_loader, grid)
        
        # Step scheduler
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Check early stopping
        stop, status = trainer.check_early_stopping(val_metrics['max_drift'], epoch, args.output)
        
        # Log
        log_line = f"{epoch},{train_metrics['loss']:.4f},{train_metrics['max_drift']:.6f},"
        log_line += f"{val_metrics['loss']:.4f},{val_metrics['max_drift']:.6f},{current_lr:.2e},{status}\n"
        log_file.write(log_line)
        log_file.flush()
        
        # Print status
        stage = "Stage 2" if epoch < 181 else "Stage 3"
        weights = get_loss_weights(epoch)
        
        print(f"[{stage}] Epoch {epoch}: "
              f"Train Loss={train_metrics['loss']:.2f}, "
              f"Train Drift={train_metrics['max_drift']:.4f}, "
              f"Val Drift={val_metrics['max_drift']:.4f}, "
              f"Best={trainer.best_val_drift:.4f}, "
              f"LR={current_lr:.2e}, "
              f"W[d={weights['distill']:.2f},m={weights['martingale']:.1f}]")
        
        # Check if target reached
        if val_metrics['max_drift'] < args.target_drift:
            print(f"\n🎯 TARGET REACHED! Val Drift = {val_metrics['max_drift']:.6f} < {args.target_drift}")
            torch.save(model.state_dict(), os.path.join(args.output, 'target_met_model.pth'))
            break
        
        # Check early stopping
        if stop:
            print(f"\n⏹️ Early Stopping at Epoch {epoch}. Best Val Drift: {trainer.best_val_drift:.6f}")
            break
        
        # Save periodic checkpoint
        if epoch % 25 == 0:
            torch.save(model.state_dict(), os.path.join(args.output, f'epoch_{epoch}.pth'))
    
    log_file.close()
    
    # Final save
    torch.save(model.state_dict(), os.path.join(args.output, 'final_model.pth'))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Best Validation Drift: {trainer.best_val_drift:.6f}")
    print(f"Target: {args.target_drift}")
    print(f"Status: {'✅ TARGET MET' if trainer.best_val_drift < args.target_drift else '❌ TARGET NOT MET'}")
    print("="*60)


if __name__ == '__main__':
    main()
