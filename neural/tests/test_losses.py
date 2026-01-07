"""
Unit tests for loss functions.
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.loss import (
    DistillationLoss,
    MartingaleLoss,
    MarginalLoss,
    MMOTLoss,
)


class TestDistillationLoss:
    """Test distillation loss component."""
    
    def test_basic_loss(self):
        """Test basic loss computation."""
        B, N, M = 8, 5, 100
        
        loss_fn = DistillationLoss()
        
        u_pred = torch.randn(B, N+1, M)
        u_true = torch.randn(B, N+1, M)
        h_pred = torch.randn(B, N, M)
        h_true = torch.randn(B, N, M)
        
        loss = loss_fn(u_pred, u_true, h_pred, h_true)
        
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_with_mask(self):
        """Test loss computation with masking."""
        B, N, M = 8, 10, 100
        
        loss_fn = DistillationLoss()
        
        u_pred = torch.randn(B, N+1, M)
        u_true = torch.randn(B, N+1, M)
        h_pred = torch.randn(B, N, M)
        h_true = torch.randn(B, N, M)
        
        mask = torch.ones(B, N+1, dtype=torch.bool)
        mask[0, 5:] = False  # Mask out some timesteps
        
        loss = loss_fn(u_pred, u_true, h_pred, h_true, mask)
        
        assert not torch.isnan(loss)
        assert loss.item() >= 0


class TestMartingaleLoss:
    """Test martingale constraint loss."""
    
    def test_basic_loss(self):
        """Test basic martingale loss."""
        B, N, M = 4, 5, 50
        grid = torch.linspace(50, 200, M)
        
        loss_fn = MartingaleLoss(grid, epsilon=0.1)
        
        u_pred = torch.randn(B, N+1, M)
        h_pred = torch.randn(B, N, M)
        marginals = F.softmax(torch.randn(B, N+1, M), dim=-1)
        
        loss = loss_fn(u_pred, h_pred, marginals)
        
        assert isinstance(loss.item(), float)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestCompositeLoss:
    """Test complete composite loss."""
    
    def test_mmot_loss(self):
        """Test complete MMOT loss function."""
        B, N, M = 8, 10, 200
        grid = torch.linspace(50, 200, M)
        
        loss_fn = MMOTLoss(
            grid=grid,
            epsilon=0.1,
            lambda_distill=1.0,
            lambda_mart=0.5,
            lambda_marg=0.1
        )
        
        u_pred = torch.randn(B, N+1, M, requires_grad=True)
        h_pred = torch.randn(B, N, M, requires_grad=True)
        u_true = torch.randn(B, N+1, M)
        h_true = torch.randn(B, N, M)
        marginals = F.softmax(torch.randn(B, N+1, M), dim=-1)
        mask = torch.ones(B, N+1, dtype=torch.bool)
        
        loss, loss_dict = loss_fn(u_pred, h_pred, u_true, h_true, marginals, mask)
        
        # Check loss value
        assert isinstance(loss.item(), float)
        assert not torch.isnan(loss)
        
        # Check loss dict
        assert 'total' in loss_dict
        assert 'distill' in loss_dict
        assert 'martingale' in loss_dict
        assert 'marginal' in loss_dict
        
        # Check all components are non-negative
        for key, value in loss_dict.items():
            assert value >= 0
    
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        B, N, M = 4, 5, 100
        grid = torch.linspace(50, 200, M)
        
        loss_fn = MMOTLoss(grid)
        
        u_pred = torch.randn(B, N+1, M, requires_grad=True)
        h_pred = torch.randn(B, N, M, requires_grad=True)
        u_true = torch.randn(B, N+1, M)
        h_true = torch.randn(B, N, M)
        marginals = F.softmax(torch.randn(B, N+1, M), dim=-1)
        
        loss, _ = loss_fn(u_pred, h_pred, u_true, h_true, marginals)
        loss.backward()
        
        assert u_pred.grad is not None
        assert h_pred.grad is not None
        assert not torch.isnan(u_pred.grad).any()
        assert not torch.isnan(h_pred.grad).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
