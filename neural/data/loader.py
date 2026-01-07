"""
Efficient PyTorch DataLoader for MMOT training data.
Handles variable-length sequences (N varies from 2 to 50).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class MMOTDataset(Dataset):
    """
    Dataset class for neural MMOT training.
    
    Each instance is a tuple:
        - marginals: [N+1, M] probability distributions
        - u_star: [N+1, M] marginal dual potentials
        - h_star: [N, M] martingale dual potentials
    """
    
    def __init__(self, data_dir, max_N=50, grid_size=150, transform=None):
        self.data_dir = Path(data_dir)
        # Filter out macOS AppleDouble files (._*)
        self.files = sorted([
            f for f in self.data_dir.glob('*.npz')
            if not f.name.startswith('._')
        ])
        self.max_N = max_N
        self.grid_size = grid_size
        self.transform = transform
        
        if len(self.files) == 0:
            print(f"WARNING: No data files found in {data_dir}")
        else:
            print(f"Loaded {len(self.files)} instances from {data_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load from disk
        data = np.load(self.files[idx], allow_pickle=True)
        
        marginals = data['marginals']  # [N+1, M]
        u_star = data['u_star']        # [N+1, M]
        h_star = data['h_star']        # [N, M]
        
        N_actual = marginals.shape[0] - 1
        
        # Pad to max_N for batching
        marginals_padded = np.zeros((self.max_N + 1, self.grid_size))
        u_padded = np.zeros((self.max_N + 1, self.grid_size))
        h_padded = np.zeros((self.max_N, self.grid_size))
        
        marginals_padded[:N_actual+1] = marginals
        u_padded[:N_actual+1] = u_star
        h_padded[:N_actual] = h_star
        
        # Create mask for valid timesteps
        mask = np.zeros(self.max_N + 1, dtype=bool)
        mask[:N_actual+1] = True
        
        # Convert to tensors
        batch = {
            'marginals': torch.FloatTensor(marginals_padded),
            'u_star': torch.FloatTensor(u_padded),
            'h_star': torch.FloatTensor(h_padded),
            'mask': torch.BoolTensor(mask),
            'N': N_actual
        }
        
        # Apply transform if provided
        if self.transform:
            batch = self.transform(batch)
            
        return batch


def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4):
    """
    Create train and validation DataLoaders.
    
    Args:
        train_dir: Directory with training .npz files
        val_dir: Directory with validation .npz files
        batch_size: Recommended 32 for M4 (16GB RAM limit)
        num_workers: 4 for M4 (10-core CPU, leave room for OS)
    
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    train_dataset = MMOTDataset(train_dir)
    val_dataset = MMOTDataset(val_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster CPU->GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    # Test dataset loading
    print("Testing MMOTDataset...")
    
    # Create dummy data for testing
    test_dir = Path('data/test_dummy')
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a few dummy instances
    for i in range(5):
        N = np.random.choice([5, 10, 20])
        M = 200
        
        dummy_data = {
            'marginals': np.random.rand(N+1, M).astype(np.float32),
            'u_star': np.random.rand(N+1, M).astype(np.float32),
            'h_star': np.random.rand(N, M).astype(np.float32),
            'dual_value': np.random.rand(),
            'params': {'N': N, 'M': M}
        }
        
        # Normalize marginals to sum to 1
        dummy_data['marginals'] = dummy_data['marginals'] / dummy_data['marginals'].sum(axis=1, keepdims=True)
        
        np.savez(test_dir / f'test_{i:03d}.npz', **dummy_data)
    
    # Test dataset
    dataset = MMOTDataset(test_dir)
    print(f"Dataset size: {len(dataset)}")
    
    # Test getitem
    sample = dataset[0]
    print(f"\nSample structure:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch structure:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}")
        else:
            print(f"  {key}: {val}")
    
    print("\n✅ DataLoader tests passed!")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
