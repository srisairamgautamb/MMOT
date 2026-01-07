
#!/usr/bin/env python3
"""
TRAIN TRANSFORMER ON MONEYNESS DATA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
from architecture_fixed import ImprovedTransformerMMOT

class MoneynessDataset(Dataset):
    def __init__(self, data_path, max_len=None):
        print(f"Loading {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        
        self.marginals = data['marginals']
        self.u = data['u']
        self.h = data['h']
        self.grid = torch.from_numpy(data['grid'].astype(np.float32))
        
        self.n = len(self.marginals)
        
        # Determine Max N
        self.max_N_steps = 0
        for m in self.marginals:
            # m shape is (N+1, M)
            N = m.shape[0] - 1
            if N > self.max_N_steps:
                self.max_N_steps = N
        
        if max_len:
            self.max_N_steps = max(self.max_N_steps, max_len)
            
        print(f"Dataset size: {self.n}, Max N steps: {self.max_N_steps}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Data
        m = self.marginals[idx].astype(np.float32) # (N+1, M)
        u = self.u[idx].astype(np.float32)         # (N+1, M)
        h = self.h[idx].astype(np.float32)         # (N, M)
        
        N = m.shape[0] - 1
        M = m.shape[1]
        
        # Pad to max_N_steps (+1 for marginals/u)
        pad_size = self.max_N_steps - N
        
        # Pad m and u: (N+1, M) -> (MaxN+1, M)
        if pad_size > 0:
            m_padded = np.pad(m, ((0, pad_size), (0, 0)), mode='constant')
            u_padded = np.pad(u, ((0, pad_size), (0, 0)), mode='constant')
            # Pad h: (N, M) -> (MaxN, M)
            h_padded = np.pad(h, ((0, pad_size), (0, 0)), mode='constant')
        else:
            m_padded = m
            u_padded = u
            h_padded = h
            
        return (
            torch.from_numpy(m_padded),
            torch.from_numpy(u_padded),
            torch.from_numpy(h_padded),
            N
        )

class DistillationTrainer:
    def __init__(self, model, device='mps'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)

    def compute_loss(self, m_batch, u_true_batch, h_true_batch, N_batch, x_grid):
        # Forward
        # x_grid needs to be repeated? Model takes x_grid (M,) usually
        # architecture_fixed.py: forward(marginals, x_grid)
        
        u_pred, h_pred = self.model(m_batch, x_grid)
        
        batch_size = m_batch.shape[0]
        loss_u = 0.0
        loss_h = 0.0
        
        # Masked MSE
        for i in range(batch_size):
            N = N_batch[i]
            loss_u += F.mse_loss(u_pred[i, :N+1], u_true_batch[i, :N+1])
            loss_h += F.mse_loss(h_pred[i, :N], h_true_batch[i, :N])
            
        loss_u /= batch_size
        loss_h /= batch_size
        
        # Total
        loss = loss_u + loss_h
        return loss, loss_u.item(), loss_h.item()

    def train_epoch(self, loader, x_grid):
        self.model.train()
        total_loss = 0
        x_grid = x_grid.to(self.device)
        
        pbar = tqdm(loader, desc="Training")
        for m, u_true, h_true, N in pbar:
            m, u_true, h_true = m.to(self.device), u_true.to(self.device), h_true.to(self.device)
            # N is CPU? or Tensor? From dataloader it is tensor
            
            self.optimizer.zero_grad()
            loss, lu, lh = self.compute_loss(m, u_true, h_true, N, x_grid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Clip grads
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lu': lu, 'lh': lh})
            
        return total_loss / len(loader)

    def validate(self, loader, x_grid):
        self.model.eval()
        total_loss = 0
        x_grid = x_grid.to(self.device)
        
        with torch.no_grad():
            for m, u_true, h_true, N in loader:
                m, u_true, h_true = m.to(self.device), u_true.to(self.device), h_true.to(self.device)
                loss, _, _ = self.compute_loss(m, u_true, h_true, N, x_grid)
                total_loss += loss.item()
                
        return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='checkpoints/')
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    if args.device == 'mps' and not torch.backends.mps.is_available():
         print("MPS not available, using CPU")
         args.device = 'cpu'
         
    os.makedirs(args.output, exist_ok=True)
    
    # Load Datasets
    train_ds = MoneynessDataset(args.train_data)
    val_ds = MoneynessDataset(args.val_data, max_len=train_ds.max_N_steps)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    # Model
    M_points = train_ds.marginals[0].shape[1]
    # Assuming M is constant across dataset (it is)
    model = ImprovedTransformerMMOT(M=M_points, d_model=128, n_heads=4, n_layers=4)
    
    trainer = DistillationTrainer(model, device=args.device)
    
    best_val_loss = float('inf')
    
    x_grid = train_ds.grid
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader, x_grid)
        val_loss = trainer.validate(val_loader, x_grid)
        
        trainer.scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
            print("  New best model saved!")
            
    print(f"Training complete. Best Val Loss: {best_val_loss}")

if __name__ == "__main__":
    main()
