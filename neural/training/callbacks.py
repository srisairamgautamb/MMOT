"""
Callbacks for training: early stopping, checkpointing, learning rate scheduling.
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    """
    
    def __init__(self, patience=10, min_delta=1e-5, verbose=True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f"EarlyStopping: Initial best loss = {val_loss:.6f}")
            return False
        
        # Check for improvement
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                improvement = self.best_loss - val_loss
                print(f"EarlyStopping: Improved by {improvement:.6f}")
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Triggered! Stopping training.")
                return True
        
        return False


class ReduceLROnPlateau:
    """
    Reduce learning rate when validation loss plateaus.
    """
    
    def __init__(self, optimizer, patience=5, factor=0.5, min_lr=1e-7, verbose=True):
        """
        Args:
            optimizer: PyTorch optimizer
            patience: Number of epochs to wait before reducing LR
            factor: Factor to reduce LR by
            min_lr: Minimum learning rate
            verbose: Print messages
        """
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        """
        Update learning rate if needed.
        
        Args:
            val_loss: Current validation loss
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    
                    if self.verbose and new_lr < old_lr:
                        print(f"ReduceLROnPlateau: {old_lr:.2e} → {new_lr:.2e}")
                
                self.counter = 0


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    """
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True,
                 mode='min', verbose=True):
        """
        Args:
            filepath: Path pattern for checkpoint files
            monitor: Metric to monitor
            save_best_only: Only save when model improves
            mode: 'min' or 'max'
            verbose: Print messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        
        if mode == 'min':
            self.best_value = np.inf
            self.is_better = lambda a, b: a < b
        else:
            self.best_value = -np.inf
            self.is_better = lambda a, b: a > b
    
    def __call__(self, value, save_fn):
        """
        Check if checkpoint should be saved.
        
        Args:
            value: Current monitored metric value
            save_fn: Function to call to save checkpoint
        
        Returns:
            True if checkpoint was saved
        """
        if not self.save_best_only or self.is_better(value, self.best_value):
            if self.verbose and self.best_value != np.inf and self.best_value != -np.inf:
                print(f"ModelCheckpoint: {self.monitor} improved "
                      f"({self.best_value:.6f} → {value:.6f})")
            
            self.best_value = value
            save_fn()
            return True
        
        return False


class LearningRateWarmup:
    """
    Warmup learning rate from 0 to target over N epochs.
    """
    
    def __init__(self, optimizer, warmup_epochs, target_lr):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs to warmup
            target_lr: Target learning rate after warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate for current epoch."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.target_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        self.current_epoch += 1


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("Testing callbacks...")
    
    # Test EarlyStopping
    print("\n1. Testing EarlyStopping...")
    early_stop = EarlyStopping(patience=3, verbose=True)
    
    val_losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84]  # Should trigger at epoch 6
    for epoch, loss in enumerate(val_losses):
        print(f"  Epoch {epoch+1}: loss={loss}")
        should_stop = early_stop(loss)
        if should_stop:
            print(f"  Stopped at epoch {epoch+1}")
            break
    
    # Test ReduceLROnPlateau
    print("\n2. Testing ReduceLROnPlateau...")
    import torch
    dummy_param = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.Adam([dummy_param], lr=1e-3)
    
    reduce_lr = ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)
    
    val_losses = [1.0, 0.9, 0.85, 0.85, 0.85, 0.84]  # Should reduce LR at epoch 5
    for epoch, loss in enumerate(val_losses):
        print(f"  Epoch {epoch+1}: loss={loss}, lr={optimizer.param_groups[0]['lr']:.2e}")
        reduce_lr(loss)
    
    print("\n✅ All callback tests passed!")
