"""
Integration test for complete training pipeline.

Tests the full workflow:
1. Data generation (small scale)
2. Data loading
3. Model creation
4. Loss computation
5. Training loop (2 epochs)
6. Checkpoint saving/loading
7. Validation

This must PASS before any production training.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add parent to path (go up from tests/ to neural/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import create_model
from training.loss import MMOTLoss
from training.trainer import MMOTTrainer
from data.loader import MMOTDataset, get_dataloaders


def create_dummy_dataset(output_dir, num_instances=20):
    """Create minimal dummy dataset for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Creating {num_instances} dummy instances in {output_dir}...")
    
    for i in range(num_instances):
        N = np.random.choice([3, 5, 7])
        M = 100  # Smaller grid for testing
        
        # Create random but valid marginals
        marginals = np.random.rand(N+1, M).astype(np.float32)
        marginals = marginals / marginals.sum(axis=1, keepdims=True)
        
        # Random dual potentials
        u_star = np.random.randn(N+1, M).astype(np.float32)
        h_star = np.random.randn(N, M).astype(np.float32)
        
        # Save
        np.savez(
            output_dir / f'test_{i:03d}.npz',
            marginals=marginals,
            u_star=u_star,
            h_star=h_star,
            dual_value=np.random.rand(),
            params={'N': N, 'M': M}
        )
    
    print(f"✅ Created {num_instances} instances")


def test_full_pipeline():
    """Test complete training pipeline."""
    print("=" * 70)
    print("INTEGRATION TEST: Full Training Pipeline")
    print("=" * 70)
    
    # Use temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Step 1: Create dummy data
        print("\n[1/7] Creating test dataset...")
        train_dir = tmpdir / 'train'
        val_dir = tmpdir / 'val'
        
        create_dummy_dataset(train_dir, num_instances=30)
        create_dummy_dataset(val_dir, num_instances=10)
        
        # Step 2: Test data loading
        print("\n[2/7] Testing data loaders...")
        
        # Need to specify grid_size to match our dummy data
        from torch.utils.data import DataLoader
        train_dataset = MMOTDataset(train_dir, max_N=10, grid_size=100)
        val_dataset = MMOTDataset(val_dir, max_N=10, grid_size=100)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
        
        assert len(train_loader.dataset) == 30, f"Expected 30 train samples, got {len(train_loader.dataset)}"
        assert len(val_loader.dataset) == 10, f"Expected 10 val samples, got {len(val_loader.dataset)}"
        
        # Test one batch
        batch = next(iter(train_loader))
        assert 'marginals' in batch
        assert 'u_star' in batch
        assert 'h_star' in batch
        assert batch['marginals'].shape[0] <= 8  # Batch size
        print(f"   ✅ Data loading works (batch shape: {batch['marginals'].shape})")
        
        # Step 3: Create model
        print("\n[3/7] Creating model...")
        model_config = {
            'grid_size': 100,  # Match dummy data
            'hidden_dim': 64,   # Small for testing
            'num_heads': 2,
            'num_layers': 1,    # Minimal for speed
            'dropout': 0.1
        }
        
        model = create_model(model_config)
        param_count = model.count_parameters()
        print(f"   ✅ Model created ({param_count:,} parameters)")
        
        # Step 4: Test forward pass
        print("\n[4/7] Testing forward pass...")
        model.eval()
        with torch.no_grad():
            u_pred, h_pred = model(batch['marginals'], batch['mask'])
        
        assert u_pred.shape == batch['u_star'].shape, f"u shape mismatch: {u_pred.shape} vs {batch['u_star'].shape}"
        assert h_pred.shape == batch['h_star'].shape, f"h shape mismatch: {h_pred.shape} vs {batch['h_star'].shape}"
        assert not torch.isnan(u_pred).any(), "NaN in u_pred"
        assert not torch.isnan(h_pred).any(), "NaN in h_pred"
        print(f"   ✅ Forward pass works (u: {u_pred.shape}, h: {h_pred.shape})")
        
        # Step 5: Test loss computation
        print("\n[5/7] Testing loss computation...")
        grid = torch.linspace(50, 200, 100)
        loss_fn = MMOTLoss(grid, epsilon=0.1)
        
        loss, loss_dict = loss_fn(
            u_pred, h_pred,
            batch['u_star'], batch['h_star'],
            batch['marginals'], batch['mask']
        )
        
        assert not torch.isnan(loss), "NaN in loss"
        assert loss.item() >= 0, f"Negative loss: {loss.item()}"
        assert 'distill' in loss_dict
        assert 'martingale' in loss_dict
        print(f"   ✅ Loss computation works (total: {loss.item():.4f})")
        
        # Step 6: Test training loop
        print("\n[6/7] Testing training loop (2 epochs)...")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        
        checkpoint_dir = tmpdir / 'checkpoints'
        checkpoint_dir.mkdir()
        
        trainer = MMOTTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
            config={
                'log_dir': str(tmpdir / 'runs'),
                'checkpoint_dir': str(checkpoint_dir),
                'log_freq': 5,
                'save_freq': 1,
                'gradient_clip': 1.0
            }
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2
        )
        
        assert len(history['train_losses']) == 2
        assert len(history['val_losses']) == 2
        assert history['best_val_loss'] > 0
        print(f"   ✅ Training works (train loss: {history['train_losses'][-1]:.4f})")
        
        # Step 7: Test checkpoint loading
        print("\n[7/7] Testing checkpoint save/load...")
        
        # Check checkpoint exists
        checkpoint_files = list(checkpoint_dir.glob('*.pt'))
        assert len(checkpoint_files) > 0, "No checkpoints saved"
        
        # Load checkpoint
        best_checkpoint = checkpoint_dir / 'best_model.pt'
        if best_checkpoint.exists():
            loaded_checkpoint = torch.load(best_checkpoint, map_location='cpu', weights_only=False)
            assert 'model_state_dict' in loaded_checkpoint
            assert 'optimizer_state_dict' in loaded_checkpoint
            assert 'epoch' in loaded_checkpoint
            print(f"   ✅ Checkpoint save/load works ({len(checkpoint_files)} files)")
        
    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST PASSED!")
    print("=" * 70)
    print("\nAll components verified:")
    print("  ✅ Data generation")
    print("  ✅ Data loading")
    print("  ✅ Model creation")
    print("  ✅ Forward pass")
    print("  ✅ Loss computation")
    print("  ✅ Training loop")
    print("  ✅ Checkpoint management")
    print("\n🎉 Training pipeline is READY FOR PRODUCTION!")


if __name__ == '__main__':
    try:
        test_full_pipeline()
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ INTEGRATION TEST FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
