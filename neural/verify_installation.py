#!/usr/bin/env python3
"""
Quick verification script for Neural MMOT Solver.
Tests all core components without requiring training data.
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("=" * 70)
print("NEURAL MMOT SOLVER - COMPONENT VERIFICATION")
print("=" * 70)

# Track results
results = []

def test_component(name, test_func):
    """Run a test and track results."""
    print(f"\n[{len(results)+1}] Testing {name}...")
    try:
        test_func()
        print(f"    ✅ PASSED")
        results.append((name, True, None))
        return True
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        results.append((name, False, str(e)))
        return False

# ============================================================================
# TESTS
# ============================================================================

def test_imports():
    """Test that all modules can be imported."""
    from models.architecture import NeuralDualSolver, create_model
    from models.layers import SinusoidalTimeEmbedding, MarginalEncoder
    from training.loss import MMOTLoss, DistillationLoss, MartingaleLoss
    from data.loader import MMOTDataset, get_dataloaders
    from inference.pricer import NeuralMMOTPricer

def test_architecture():
    """Test neural architecture."""
    from models.architecture import create_model
    
    model = create_model({
        'grid_size': 200,
        'hidden_dim': 256,
        'num_heads': 4,
        'num_layers': 3
    })
    
    # Test forward pass
    marginals = torch.randn(8, 11, 200)
    mask = torch.ones(8, 11, dtype=torch.bool)
    u, h = model(marginals, mask)
    
    assert u.shape == (8, 11, 200)
    assert h.shape == (8, 10, 200)
    assert not torch.isnan(u).any()
    assert not torch.isnan(h).any()

def test_loss_functions():
    """Test loss computation."""
    from training.loss import MMOTLoss
    
    grid = torch.linspace(50, 200, 200)
    loss_fn = MMOTLoss(grid)
    
    B, N, M = 8, 10, 200
    u_pred = torch.randn(B, N+1, M, requires_grad=True)
    h_pred = torch.randn(B, N, M, requires_grad=True)
    u_true = torch.randn(B, N+1, M)
    h_true = torch.randn(B, N, M)
    marginals = torch.softmax(torch.randn(B, N+1, M), dim=-1)
    
    loss, loss_dict = loss_fn(u_pred, h_pred, u_true, h_true, marginals)
    
    assert not torch.isnan(loss)
    assert 'total' in loss_dict
    assert 'distill' in loss_dict

def test_data_loader():
    """Test data loader with dummy data."""
    from data.loader import MMOTDataset
    from torch.utils.data import DataLoader
    
    # Create temporary dummy data
    test_dir = Path('data/test_dummy')
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a few files
    for i in range(3):
        N, M = 5, 200
        data = {
            'marginals': np.random.rand(N+1, M).astype(np.float32),
            'u_star': np.random.rand(N+1, M).astype(np.float32),
            'h_star': np.random.rand(N, M).astype(np.float32),
            'dual_value': np.random.rand(),
            'params': {'N': N}
        }
        data['marginals'] = data['marginals'] / data['marginals'].sum(axis=1, keepdims=True)
        np.savez(test_dir / f'test_{i}.npz', **data)
    
    # Clean up any macOS resource fork files
    for file in test_dir.glob('._*'):
        file.unlink()
    
    # Test loading
    dataset = MMOTDataset(test_dir)
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    
    assert 'marginals' in batch
    assert 'u_star' in batch
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)

def test_inference_engine():
    """Test inference engine."""
    from inference.pricer import NeuralMMOTPricer
    from models.architecture import create_model
    
    # Create and save dummy model
    model = create_model({
        'grid_size': 200,
        'hidden_dim': 128,
        'num_heads': 2,
        'num_layers': 2
    })
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'test_model.pt'
    
    torch.save({'model': model, 'model_state_dict': model.state_dict()}, checkpoint_path)
    
    # Test pricer
    pricer = NeuralMMOTPricer(checkpoint_path, device='cpu')
    
    marginals = np.random.rand(11, 200).astype(np.float32)
    marginals = marginals / marginals.sum(axis=1, keepdims=True)
    
    u, h = pricer.predict_potentials(marginals)
    
    assert u.shape == (11, 200)
    assert h.shape == (10, 200)
    
    # Cleanup
    checkpoint_path.unlink()

def test_configs():
    """Test configuration files."""
    import yaml
    
    default_config = Path('configs/default.yaml')
    m4_config = Path('configs/m4_optimized.yaml')
    
    assert default_config.exists()
    assert m4_config.exists()
    
    with open(default_config) as f:
        config = yaml.safe_load(f)
    
    assert 'model' in config
    assert 'training' in config
    assert config['model']['grid_size'] == 200

# ============================================================================
# RUN ALL TESTS
# ============================================================================

print("\nRunning component tests...\n")

test_component("Module Imports", test_imports)
test_component("Configuration Files", test_configs)
test_component("Neural Architecture", test_architecture)
test_component("Loss Functions", test_loss_functions)
test_component("Data Loader", test_data_loader)
test_component("Inference Engine", test_inference_engine)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

passed = sum(1 for _, success, _ in results if success)
failed = sum(1 for _, success, _ in results if not success)

for name, success, error in results:
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status:10} - {name}")
    if error:
        print(f"           Error: {error}")

print("\n" + "=" * 70)
print(f"Results: {passed}/{len(results)} tests passed")

if failed == 0:
    print("🎉 All components verified successfully!")
    print("✅ Neural MMOT Solver is ready for data generation and training.")
else:
    print(f"⚠️  {failed} test(s) failed. Please review errors above.")

print("=" * 70)

sys.exit(0 if failed == 0 else 1)
