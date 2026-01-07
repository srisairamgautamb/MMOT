"""
Train on Heston data only (2K instances)
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT')

from neural.models.architecture import NeuralDualSolver
from neural.training.loss import MMOTLoss
from neural.data.loader import MMOTDataset
from tqdm import tqdm
import numpy as np

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

print("="*80)
print("TRAINING: HESTON MODEL")
print("="*80)
print(f"Device: {device}")
print(f"Dataset: neural/data/augmented/heston/")
print(f"Instances: 2000")
print("="*80)

dataset = MMOTDataset('neural/data/augmented/heston')
valid_indices = [i for i in range(len(dataset)) if not dataset.files[i].name.startswith('._')]
dataset.files = [dataset.files[i] for i in valid_indices]

print(f"\nLoaded {len(dataset)} instances")

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = NeuralDualSolver(grid_size=150, hidden_dim=256, num_layers=3).to(device)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

grid = torch.linspace(0, 1, 150).to(device)
loss_fn = MMOTLoss(grid=grid, epsilon=1.0, lambda_distill=1.0,
                   lambda_martingale=5.0, lambda_marginal=0.1).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

epochs = 100
best_loss = float('inf')
Path('checkpoints/heston').mkdir(parents=True, exist_ok=True)

print(f"\nStarting training: {epochs} epochs\n")

for epoch in range(epochs):
    model.train()
    losses = []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        marginals = batch['marginals'].to(device)
        u_true = batch['u_star'].to(device)
        h_true = batch['h_star'].to(device)
        
        u_pred, h_pred = model(marginals)
        loss, loss_dict = loss_fn(u_pred, h_pred, u_true, h_true, marginals)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
    
    train_loss = np.mean(losses)
    scheduler.step(train_loss)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
        }, 'checkpoints/heston/best_model.pt')
        print(f"  ✅ Saved (loss: {train_loss:.4f})")

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print(f"Best loss: {best_loss:.4f}")
print(f"Model saved: checkpoints/heston/best_model.pt")
print("="*80)
