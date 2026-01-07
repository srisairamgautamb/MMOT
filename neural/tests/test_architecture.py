"""
Unit tests for Neural MMOT architecture.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import NeuralDualSolver, create_model
from models.layers import SinusoidalTimeEmbedding, MarginalEncoder


class TestLayers:
    """Test custom neural network layers."""
    
    def test_time_embedding(self):
        """Test sinusoidal time embedding."""
        d_model = 256
        max_len = 100
        batch_size = 16
        seq_len = 11
        
        time_emb = SinusoidalTimeEmbedding(d_model, max_len)
        t_idx = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, -1)
        
        embeddings = time_emb(t_idx)
        
        assert embeddings.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
    
    def test_marginal_encoder(self):
        """Test marginal distribution encoder."""
        grid_size = 200
        hidden_dim = 256
        batch_size = 16
        seq_len = 11
        
        encoder = MarginalEncoder(grid_size, hidden_dim)
        marginals = torch.randn(batch_size, seq_len, grid_size)
        
        latent = encoder(marginals)
        
        assert latent.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(latent).any()
        assert not torch.isinf(latent).any()


class TestArchitecture:
    """Test complete neural architecture."""
    
    def test_model_creation(self):
        """Test model creation with config."""
        config = {
            'grid_size': 200,
            'hidden_dim': 256,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.1
        }
        
        model = create_model(config)
        
        assert isinstance(model, NeuralDualSolver)
        assert model.grid_size == 200
        assert model.hidden_dim == 256
        assert model.count_parameters() > 0
    
    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        batch_size = 16
        N = 10
        M = 200
        
        model = create_model({
            'grid_size': M,
            'hidden_dim': 256,
            'num_heads': 4,
            'num_layers': 3
        })
        
        marginals = torch.randn(batch_size, N+1, M)
        mask = torch.ones(batch_size, N+1, dtype=torch.bool)
        
        u_pred, h_pred = model(marginals, mask)
        
        assert u_pred.shape == (batch_size, N+1, M)
        assert h_pred.shape == (batch_size, N, M)
        assert not torch.isnan(u_pred).any()
        assert not torch.isnan(h_pred).any()
    
    def test_variable_length_sequences(self):
        """Test handling of variable-length sequences."""
        batch_size = 8
        N = 10
        M = 200
        
        model = create_model({
            'grid_size': M,
            'hidden_dim': 256,
            'num_heads': 4,
            'num_layers': 2
        })
        
        marginals = torch.randn(batch_size, N+1, M)
        
        # Create variable length mask
        mask = torch.ones(batch_size, N+1, dtype=torch.bool)
        mask[0, 5:] = False
        mask[1, 8:] = False
        
        u_pred, h_pred = model(marginals, mask)
        
        assert u_pred.shape == (batch_size, N+1, M)
        assert h_pred.shape == (batch_size, N, M)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = create_model({
            'grid_size': 200,
            'hidden_dim': 128,
            'num_heads': 2,
            'num_layers': 2
        })
        
        marginals = torch.randn(4, 6, 200, requires_grad=True)
        u_pred, h_pred = model(marginals)
        
        loss = u_pred.sum() + h_pred.sum()
        loss.backward()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = create_model({
            'grid_size': 200,
            'hidden_dim': 256,
            'num_heads': 4,
            'num_layers': 3
        })
        
        param_count = model.count_parameters()
        
        # Should be around 2.1M parameters
        assert 1_000_000 < param_count < 5_000_000
        
        # Should match manual count
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
