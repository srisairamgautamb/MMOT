"""
Custom neural network layers for MMOT solver.

Components:
- SinusoidalTimeEmbedding: Positional encoding for time steps
- MarginalEncoder: CNN-based encoder for probability distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Positional encoding for time steps (similar to Transformer original paper).
    Helps network distinguish t=0 from t=T.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, t_idx):
        """
        Args:
            t_idx: [Batch, Seq_Len] integer time indices
        Returns:
            [Batch, Seq_Len, d_model] embeddings
        """
        return self.pe[t_idx]


class MarginalEncoder(nn.Module):
    """
    Encode probability distribution μ(x) into latent vector z.
    Uses 1D convolutions to respect spatial structure of densities.
    """
    def __init__(self, grid_size=200, hidden_dim=256):
        super().__init__()
        
        # 1D Convolutions (treat density as 1D signal)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)  # Downsample: 200 → 100 → 50
        
        # MLP to latent space
        self.fc = nn.Sequential(
            nn.Linear(128 * (grid_size // 4), hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, marginals):
        """
        Args:
            marginals: [Batch, Seq_Len, Grid_Size]
        Returns:
            [Batch, Seq_Len, Hidden_Dim] latent encodings
        """
        B, N, M = marginals.shape
        
        # Reshape for conv1d: [B*N, 1, M]
        x = marginals.reshape(B * N, 1, M)
        
        # Convolutions
        x = F.gelu(self.conv1(x))
        x = self.pool(x)
        x = F.gelu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten and MLP
        x = x.view(B * N, -1)
        z = self.fc(x)
        
        # Reshape back: [B, N, Hidden]
        z = z.view(B, N, -1)
        return z


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("Testing custom layers...")
    
    # Test SinusoidalTimeEmbedding
    print("\n1. Testing SinusoidalTimeEmbedding...")
    time_emb = SinusoidalTimeEmbedding(d_model=256, max_len=100)
    t_idx = torch.arange(0, 11).unsqueeze(0)  # [1, 11]
    embeddings = time_emb(t_idx)
    print(f"   Input shape: {t_idx.shape}")
    print(f"   Output shape: {embeddings.shape}")
    assert embeddings.shape == (1, 11, 256)
    print("   ✅ Passed")
    
    # Test MarginalEncoder
    print("\n2. Testing MarginalEncoder...")
    encoder = MarginalEncoder(grid_size=200, hidden_dim=256)
    marginals = torch.randn(16, 11, 200)  # [Batch=16, Seq=11, Grid=200]
    latent = encoder(marginals)
    print(f"   Input shape: {marginals.shape}")
    print(f"   Output shape: {latent.shape}")
    assert latent.shape == (16, 11, 256)
    print("   ✅ Passed")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n   MarginalEncoder parameters: {total_params:,}")
    
    print("\n✅ All layer tests passed!")
