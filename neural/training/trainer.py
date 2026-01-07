"""
Main training loop for Neural MMOT Solver.

This module implements the complete training pipeline with:
- Model training and validation
- Checkpoint saving
- Metric logging
- Early stopping support
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import yaml


class MMOTTrainer:
    """
    Main trainer class for Neural MMOT solver.
    
    Handles:
    - Training loop with validation
    - Checkpoint saving/loading
    - Metric logging to TensorBoard
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(self, model, loss_fn, optimizer, scheduler=None,
                 device='cpu', config=None):
        """
        Initialize trainer.
        
        Args:
            model: NeuralDualSolver instance
            loss_fn: MMOTLoss instance
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
            device: 'cpu', 'cuda', or 'mps'
            config: Configuration dict
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup logging
        log_dir = Path(self.config.get('log_dir', 'runs'))
        log_dir.mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Setup checkpointing
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Model parameters: {model.count_parameters():,}")
        print(f"  Log directory: {log_dir}")
        print(f"  Checkpoint directory: {self.checkpoint_dir}")
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: PyTorch DataLoader for training data
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_losses = []
        epoch_loss_dict = {'distill': [], 'martingale': [], 'marginal': []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            marginals = batch['marginals'].to(self.device)
            u_true = batch['u_star'].to(self.device)
            h_true = batch['h_star'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            u_pred, h_pred = self.model(marginals, mask)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(u_pred, h_pred, u_true, h_true, marginals, mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Track losses
            epoch_losses.append(loss.item())
            for key in epoch_loss_dict:
                if key in loss_dict:
                    epoch_loss_dict[key].append(loss_dict[key])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'distill': f"{loss_dict.get('distill', 0):.4f}"
            })
            
            # Log to TensorBoard
            global_step = self.current_epoch * len(train_loader) + batch_idx
            if batch_idx % self.config.get('log_freq', 10) == 0:
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, global_step)
        
        # Compute epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_loss_dict = {k: np.mean(v) for k, v in epoch_loss_dict.items()}
        
        # Log epoch summary
        self.writer.add_scalar('train/epoch_loss', avg_loss, self.current_epoch)
        
        return avg_loss, avg_loss_dict
    
    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate on validation set.
        
        Args:
            val_loader: PyTorch DataLoader for validation data
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        epoch_losses = []
        epoch_loss_dict = {'distill': [], 'martingale': [], 'marginal': []}
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        for batch in pbar:
            # Move data to device
            marginals = batch['marginals'].to(self.device)
            u_true = batch['u_star'].to(self.device)
            h_true = batch['h_star'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            u_pred, h_pred = self.model(marginals, mask)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(u_pred, h_pred, u_true, h_true, marginals, mask)
            
            # Track losses
            epoch_losses.append(loss.item())
            for key in epoch_loss_dict:
                if key in loss_dict:
                    epoch_loss_dict[key].append(loss_dict[key])
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_loss_dict = {k: np.mean(v) for k, v in epoch_loss_dict.items()}
        
        # Log to TensorBoard
        self.writer.add_scalar('val/epoch_loss', avg_loss, self.current_epoch)
        for key, value in avg_loss_dict.items():
            self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
        
        return avg_loss, avg_loss_dict
    
    def save_checkpoint(self, filename, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save as best if applicable
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}")
        print(f"  Best validation loss: {self.best_val_loss:.6f}")
    
    def train(self, train_loader, val_loader, num_epochs, early_stopping_patience=None):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs (optional)
        
        Returns:
            Training history
        """
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print("=" * 70)
        
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_loss, train_loss_dict = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_loss_dict = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train loss: {train_loss:.6f}")
            print(f"  Val loss: {val_loss:.6f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
                print(f"  ✅ New best validation loss: {val_loss:.6f}")
            else:
                epochs_without_improvement += 1
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.get('save_freq', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', is_best=is_best)
            
            # Early stopping check
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                print(f"   No improvement for {early_stopping_patience} epochs")
                break
        
        # Final checkpoint
        self.save_checkpoint('final_model.pt', is_best=False)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Final training loss: {self.train_losses[-1]:.6f}")
        print(f"Final validation loss: {self.val_losses[-1]:.6f}")
        print("=" * 70)
        
        self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("Testing MMOTTrainer...")
    
    # Create dummy components
    from models.architecture import create_model
    from training.loss import MMOTLoss
    from data.loader import MMOTDataset
    from torch.utils.data import DataLoader
    import tempfile
    
    # Small model for testing
    model = create_model({
        'grid_size': 100,
        'hidden_dim': 64,
        'num_heads': 2,
        'num_layers': 1
    })
    
    # Loss function
    grid = torch.linspace(50, 200, 100)
    loss_fn = MMOTLoss(grid)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dummy data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Generate dummy files
        for i in range(10):
            N, M = 5, 100
            data = {
                'marginals': np.random.rand(N+1, M).astype(np.float32),
                'u_star': np.random.rand(N+1, M).astype(np.float32),
                'h_star': np.random.rand(N, M).astype(np.float32),
            }
            data['marginals'] = data['marginals'] / data['marginals'].sum(axis=1, keepdims=True)
            np.savez(tmpdir / f'train_{i}.npz', **data)
        
        for i in range(3):
            N, M = 5, 100
            data = {
                'marginals': np.random.rand(N+1, M).astype(np.float32),
                'u_star': np.random.rand(N+1, M).astype(np.float32),
                'h_star': np.random.rand(N, M).astype(np.float32),
            }
            data['marginals'] = data['marginals'] / data['marginals'].sum(axis=1, keepdims=True)
            np.savez(tmpdir / f'val_{i}.npz', **data)
        
        # Create datasets
        train_dataset = MMOTDataset(tmpdir, max_N=10, grid_size=100)
        val_dataset = MMOTDataset(tmpdir, max_N=10, grid_size=100)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        # Create trainer
        trainer = MMOTTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device='cpu',
            config={'log_freq': 1, 'save_freq': 2}
        )
        
        # Train for 2 epochs
        print("\nRunning 2-epoch test...")
        history = trainer.train(train_loader, val_loader, num_epochs=2)
        
        print("\n✅ Trainer test passed!")
        print(f"   Train losses: {history['train_losses']}")
        print(f"   Val losses: {history['val_losses']}")
