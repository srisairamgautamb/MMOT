#!/usr/bin/env python3
"""
NEURAL MMOT TRAINER WITH TEACHER DISTILLATION
==============================================
Trains the neural network using CLASSICAL SOLVER solutions as teacher signal.

This version uses the proper distillation approach:
- Load pre-computed u_classical, h_classical from teacher_data_full.npz
- MSE loss against teacher + martingale regularization
- Target: drift < 0.05 on validation set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from architecture_fixed import ImprovedTransformerMMOT


class TeacherDataset(Dataset):
    """Dataset that includes teacher solutions (u_classical, h_classical)."""
    
    def __init__(self, data_path):
        print(f"Loading teacher data from {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        
        self.marginals = data['marginals'].astype(np.float32)
        self.u_classical = data['u_classical'].astype(np.float32)
        self.h_classical = data['h_classical'].astype(np.float32)
        self.x_grid = torch.from_numpy(data['x_grid'].astype(np.float32))
        self.N_values = data['N_values']
        
        print(f"  Loaded {len(self.marginals)} instances")
        print(f"  Shapes: marginals {self.marginals.shape}, u {self.u_classical.shape}, h {self.h_classical.shape}")

    def __len__(self):
        return len(self.marginals)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.marginals[idx]),
            torch.from_numpy(self.u_classical[idx]),
            torch.from_numpy(self.h_classical[idx]),
            self.x_grid,
            self.N_values[idx]
        )


class DistillationTrainer:
    """Trainer with proper teacher distillation."""
    
    def __init__(self, model, device='mps'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

    def compute_loss(self, marginals, u_teacher, h_teacher, x_grid, N_values):
        """
        Compute loss with teacher distillation.
        
        Loss = MSE(u_pred, u_teacher) + MSE(h_pred, h_teacher) + lambda * L_martingale
        """
        u_pred, h_pred = self.model(marginals, x_grid)
        
        # 1. Distillation loss (MSE against teacher)
        # Mask out padded regions based on actual N for each instance
        batch_size = marginals.shape[0]
        loss_distill = 0.0
        
        for i in range(batch_size):
            N = int(N_values[i]) if hasattr(N_values, '__iter__') else int(N_values)
            # u has N+1 time steps, h has N steps
            loss_distill += F.mse_loss(u_pred[i, :N+1], u_teacher[i, :N+1])
            loss_distill += F.mse_loss(h_pred[i, :N], h_teacher[i, :N])
        
        loss_distill = loss_distill / batch_size
        
        # 2. Martingale constraint loss (soft regularization)
        loss_mart = 0.0
        N_max = h_pred.shape[1]
        
        for t in range(N_max):
            mu_t = marginals[:, t, :]
            h_t = h_pred[:, t, :]
            # Weighted drift
            drift_t = (mu_t * h_t).sum(dim=1)
            loss_mart += drift_t.pow(2).mean()
        
        loss_mart = loss_mart / N_max
        
        # 3. Regularization on h magnitude
        loss_reg = h_pred.pow(2).mean()
        
        # Total loss: distillation + 5.0 * martingale + 0.1 * regularization
        loss_total = loss_distill + 5.0 * loss_mart + 0.1 * loss_reg
        
        return loss_total, {
            'distill': loss_distill.item(),
            'martingale': loss_mart.item(),
            'reg': loss_reg.item()
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        metrics_sum = {'distill': 0, 'martingale': 0, 'reg': 0}
        
        for batch in tqdm(train_loader, desc='Training'):
            marginals, u_teacher, h_teacher, x_grid, N_values = batch
            
            marginals = marginals.to(self.device)
            u_teacher = u_teacher.to(self.device)
            h_teacher = h_teacher.to(self.device)
            x_grid = x_grid[0].to(self.device)  # Same for all
            
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(marginals, u_teacher, h_teacher, x_grid, N_values)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            for k in metrics:
                metrics_sum[k] += metrics[k]
        
        self.scheduler.step()
        n_batches = len(train_loader)
        
        return total_loss / n_batches, {k: v/n_batches for k, v in metrics_sum.items()}

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate model and compute actual drift metric."""
        self.model.eval()
        total_drift = 0.0
        total_error = 0.0
        n_samples = 0
        
        for batch in val_loader:
            marginals, u_teacher, h_teacher, x_grid, N_values = batch
            
            marginals = marginals.to(self.device)
            u_teacher = u_teacher.to(self.device)
            h_teacher = h_teacher.to(self.device)
            x_grid = x_grid[0].to(self.device)
            
            u_pred, h_pred = self.model(marginals, x_grid)
            
            batch_size = marginals.shape[0]
            
            for i in range(batch_size):
                N = int(N_values[i]) if hasattr(N_values, '__iter__') else int(N_values)
                
                # Compute drift for each time step
                for t in range(N):
                    mu_t = marginals[i, t]
                    h_t = h_pred[i, t]
                    drift_t = (mu_t * h_t).sum().abs().item()
                    total_drift += drift_t
                
                # Compute error vs teacher
                error_u = F.mse_loss(u_pred[i, :N+1], u_teacher[i, :N+1]).item()
                error_h = F.mse_loss(h_pred[i, :N], h_teacher[i, :N]).item()
                total_error += (error_u + error_h)
                
                n_samples += N
        
        mean_drift = total_drift / max(n_samples, 1)
        mean_error = total_error / max(len(val_loader.dataset), 1)
        
        return mean_drift, mean_error


def main():
    print("="*70)
    print("NEURAL MMOT TRAINING WITH TEACHER DISTILLATION")
    print("="*70)
    
    # Paths
    train_path = 'teacher_data_full.npz'
    
    if not os.path.exists(train_path):
        print(f"ERROR: Teacher data not found at {train_path}")
        print("Run generate_teacher_data.py first!")
        return
    
    # Load full dataset and split
    full_data = np.load(train_path, allow_pickle=True)
    n_total = full_data['n_instances']
    
    # 90/10 train/val split
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    
    print(f"\nDataset split: {n_train} train, {n_val} validation")
    
    # Create train/val files (just use indices)
    train_indices = np.arange(n_train)
    val_indices = np.arange(n_train, n_total)
    
    # For simplicity, create subset files
    def create_subset(indices, output_path):
        data = np.load(train_path, allow_pickle=True)
        subset = {
            'marginals': data['marginals'][indices],
            'u_classical': data['u_classical'][indices],
            'h_classical': data['h_classical'][indices],
            'x_grid': data['x_grid'],
            'N_values': data['N_values'][indices],
        }
        np.savez_compressed(output_path, **subset)
        print(f"Created {output_path} ({len(indices)} instances)")
    
    if not os.path.exists('train_set.npz'):
        create_subset(train_indices, 'train_set.npz')
    if not os.path.exists('val_set.npz'):
        create_subset(val_indices, 'val_set.npz')
    
    # Load datasets
    train_dataset = TeacherDataset('train_set.npz')
    val_dataset = TeacherDataset('val_set.npz')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Model
    model = ImprovedTransformerMMOT(M=150, N_max=50)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    trainer = DistillationTrainer(model, device=device)
    
    # Training loop
    epochs = 100
    best_drift = float('inf')
    
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING: {epochs} EPOCHS")
    print(f"Target: drift < 0.05")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        train_loss, train_metrics = trainer.train_epoch(train_loader)
        val_drift, val_error = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | "
              f"Distill: {train_metrics['distill']:.4f} | "
              f"Mart: {train_metrics['martingale']:.6f} | "
              f"Val Drift: {val_drift:.4f} | Val Error: {val_error:.4f}")
        
        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'drift': val_drift,
                'loss': train_loss
            }
            torch.save(ckpt, f'checkpoint_epoch{epoch+1}.pth')
            print(f"  >> Saved checkpoint at epoch {epoch+1}")
        
        # Save best model
        if val_drift < best_drift:
            best_drift = val_drift
            torch.save(model.state_dict(), f'best_model_drift{val_drift:.4f}.pth')
            print(f"  >> NEW BEST: drift={val_drift:.4f}")
        
        # Early stopping if target reached
        if val_drift < 0.05:
            print(f"\n{'='*70}")
            print(f"*** TARGET REACHED: Drift {val_drift:.4f} < 0.05 ***")
            print(f"{'='*70}")
            torch.save(model.state_dict(), f'FINAL_MODEL_drift{val_drift:.4f}.pth')
            break
    
    print(f"\nTraining complete. Best drift: {best_drift:.4f}")


if __name__ == "__main__":
    main()
