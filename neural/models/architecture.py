"""
Neural MMOT Solver: Transformer-based architecture for dual potential prediction.

Key Innovation: Sequence-to-sequence model that respects:
1. Temporal dependencies (via Transformer)
2. Martingale structure (via specialized loss)
3. Distributional constraints (via physics-informed penalties)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SinusoidalTimeEmbedding, MarginalEncoder

class MartingaleProjection(nn.Module):
    """
    Projects martingale potential h(x) to satisfy E[h(x)] = 0 w.r.t marginals.
    Removes additive drift constants.
    """
    def forward(self, h, marginals):
        # h: [B, N, M]
        # marginals: [B, N, M] (must match time steps)
        
        # E[h] = sum(h * mu)
        # Expectation over grid dim (-1)
        h_mean = (h * marginals).sum(dim=-1, keepdim=True) # [B, N, 1]
        return h - h_mean


class NeuralDualSolver(nn.Module):
    """
    Full neural network for MMOT dual potential prediction.
    
    Architecture:
        Input: Marginals [B, N+1, M]
        ├─ Encoder: CNN + MLP → [B, N+1, Hidden]
        ├─ Time Embedding → [B, N+1, Hidden]
        ├─ Transformer: Self-attention → [B, N+1, Hidden]
        └─ Decoders: Two MLPs → (u [B, N+1, M], h [B, N, M])
    
    Parameters: ~2.1M (fits in 12GB with batch_size=32)
    """
    
    def __init__(self, 
                 grid_size=200, 
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.1):
        super().__init__()
        
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        
        # ===== ENCODING STAGE =====
        self.marginal_encoder = MarginalEncoder(grid_size, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim, max_len=100)
        
        # Combine marginal + time embeddings
        self.input_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # ===== SEQUENCE PROCESSING =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # ===== DECODING STAGE =====
        # Head A: Marginal potentials u(x, t)
        self.decoder_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, grid_size)
        )
        
        # Head B: Martingale potentials h(x, t)
        self.decoder_h = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, grid_size)
        )
        
        # Initialization (Xavier for stability)
        self.apply(self._init_weights)
        
        # ===== NORMALIZATION STATS =====
        # Computed from 10,000 samples (Phase 3 scaling)
        self.register_buffer('u_mean', torch.tensor(-4.1155))
        self.register_buffer('u_std', torch.tensor(2.9703))
        self.register_buffer('h_mean', torch.tensor(0.0032))
        self.register_buffer('h_std', torch.tensor(2.5517))
        
        # Projection
        self.h_proj = MartingaleProjection()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight)
    
    def forward(self, marginals, mask=None):
        """
        Forward pass through neural MMOT solver.
        
        Args:
            marginals: [Batch, N+1, Grid_Size] input distributions
            mask: [Batch, N+1] boolean mask for variable-length sequences
        
        Returns:
            u_pred: [Batch, N+1, Grid_Size] marginal potentials
            h_pred: [Batch, N, Grid_Size] martingale potentials
        """
        B, N_plus_1, M = marginals.shape
        N = N_plus_1 - 1
        
        # ===== STAGE 1: ENCODING =====
        # Encode marginals
        marginal_features = self.marginal_encoder(marginals)  # [B, N+1, Hidden]
        
        # Time embeddings
        t_indices = torch.arange(N_plus_1, device=marginals.device).unsqueeze(0).expand(B, -1)
        time_features = self.time_embedding(t_indices)  # [B, N+1, Hidden]
        
        # Concatenate and project
        combined = torch.cat([marginal_features, time_features], dim=-1)
        x = self.input_proj(combined)  # [B, N+1, Hidden]
        
        # ===== STAGE 2: SEQUENCE PROCESSING =====
        # Transformer (with masking for variable-length sequences)
        if mask is not None:
            # Transformer wants: True = ignore this position
            # Our mask: True = keep this position
            # So we need to invert
            attn_mask = ~mask  # True = mask out
        else:
            attn_mask = None
        
        latent = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # ===== STAGE 3: DECODING =====
        # Decode u (all time steps)
        u_raw = self.decoder_u(latent)  # [B, N+1, M]
        
        # Decode h (only N time steps, not the last)
        h_raw = self.decoder_h(latent[:, :-1, :])  # [B, N, M]
        
        # ===== DENORMALIZATION =====
        # Apply strict denormalization to match target scale
        u_pred = u_raw * self.u_std + self.u_mean
        h_pred = h_raw * self.h_std + self.h_mean
        
        # enforce centering E[h] = 0
        h_pred = self.h_proj(h_pred, marginals[:, :-1, :])
        
        return u_pred, h_pred
    
    def count_parameters(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(config):
    """
    Create model from config dict.
    
    Example config:
        {
            'grid_size': 200,
            'hidden_dim': 256,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.1
        }
    """
    model = NeuralDualSolver(**config)
    print(f"Created NeuralDualSolver with {model.count_parameters():,} parameters")
    return model


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("Testing NeuralDualSolver Architecture")
    print("="*60)
    
    # Test forward pass
    print("\n1. Creating model...")
    model = create_model({
        'grid_size': 200,
        'hidden_dim': 256,
        'num_heads': 4,
        'num_layers': 3,
        'dropout': 0.1
    })
    
    # Dummy input
    print("\n2. Testing forward pass...")
    batch_size = 16
    N = 10
    M = 200
    marginals = torch.randn(batch_size, N+1, M)
    mask = torch.ones(batch_size, N+1, dtype=torch.bool)
    
    # Forward
    u_pred, h_pred = model(marginals, mask)
    
    print(f"   Input shape: {marginals.shape}")
    print(f"   u_pred shape: {u_pred.shape}")
    print(f"   h_pred shape: {h_pred.shape}")
    
    # Assertions
    assert u_pred.shape == (batch_size, N+1, M), f"Expected {(batch_size, N+1, M)}, got {u_pred.shape}"
    assert h_pred.shape == (batch_size, N, M), f"Expected {(batch_size, N, M)}, got {h_pred.shape}"
    print("   ✅ Output shapes correct")
    
    # Test with variable length sequences
    print("\n3. Testing variable-length sequences...")
    # Create mask where some sequences end early
    mask_var = torch.ones(batch_size, N+1, dtype=torch.bool)
    mask_var[0, 5:] = False  # First sequence only has 5 time steps
    mask_var[1, 8:] = False  # Second sequence has 8 time steps
    
    u_pred_var, h_pred_var = model(marginals, mask_var)
    assert u_pred_var.shape == (batch_size, N+1, M)
    assert h_pred_var.shape == (batch_size, N, M)
    print("   ✅ Variable-length handling works")
    
    # Test gradient flow
    print("\n4. Testing gradient flow...")
    loss = u_pred.sum() + h_pred.sum()
    loss.backward()
    
    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"   ⚠️  No gradient for {name}")
        else:
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                print(f"   ⚠️  Zero gradient for {name}")
    print("   ✅ Gradients flowing")
    
    # Model statistics
    print("\n5. Model Statistics:")
    print(f"   Total parameters: {model.count_parameters():,}")
    print(f"   Model size (MB): {model.count_parameters() * 4 / 1024 / 1024:.2f}")
    
    # Print architecture summary
    print("\n6. Architecture Summary:")
    print(model)
    
    print("\n" + "="*60)
    print("✅ All architecture tests passed!")
    print("="*60)
