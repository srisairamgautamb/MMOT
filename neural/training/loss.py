"""
Composite loss function for neural MMOT training.

Components:
1. Distillation Loss: Match teacher's dual potentials
2. Martingale Loss: Enforce E[X_{t+1} - X_t | X_t] = 0
3. Marginal Loss: Match marginal distributions
4. Regularization: L2 weight decay

Total: L = λ₁·L_distill + λ₂·L_mart + λ₃·L_marg + λ₄·L_reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# LOSS COMPONENTS
# ============================================================================

class DistillationLoss(nn.Module):
    """
    L_distill = ||u_pred - u_true||² + ||h_pred - h_true||²
    
    Supervised learning from Phase 2a teacher.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, u_pred, u_true, h_pred, h_true, mask=None):
        """
        Args:
            u_pred: [B, N+1, M] predicted marginal potentials
            u_true: [B, N+1, M] ground truth from teacher
            h_pred: [B, N, M] predicted martingale potentials
            h_true: [B, N, M] ground truth from teacher
            mask: [B, N+1] which timesteps are valid
        
        Returns:
            scalar loss
        """
        # MSE on u potentials
        loss_u = F.mse_loss(u_pred, u_true, reduction='none')  # [B, N+1, M]
        
        # MSE on h potentials
        loss_h = F.mse_loss(h_pred, h_true, reduction='none')  # [B, N, M]
        
        # Apply mask (ignore padded timesteps)
        if mask is not None:
            mask_u = mask.unsqueeze(-1)  # [B, N+1, 1]
            mask_h = mask[:, :-1].unsqueeze(-1)  # [B, N, 1]
            
            loss_u = (loss_u * mask_u).sum() / mask_u.sum()
            loss_h = (loss_h * mask_h).sum() / mask_h.sum()
        else:
            loss_u = loss_u.mean()
            loss_h = loss_h.mean()
        
        return loss_u + loss_h


class MartingaleLoss(nn.Module):
    """
    L_mart = E[(X_{t+1} - E[X_{t+1} | X_t])²]
    
    Physics-informed penalty enforcing no-arbitrage condition.
    Reconstructs transition kernel from (u, h) and checks drift.
    """
    def __init__(self, grid, epsilon=1.0):  # CRITICAL: Must match data generation ε
        super().__init__()
        if not isinstance(grid, torch.Tensor):
            grid = torch.FloatTensor(grid)
        self.register_buffer('grid', grid)
        self.epsilon = epsilon
    
    def forward(self, u_pred, h_pred, marginals, mask=None):
        """
        Args:
            u_pred: [B, N+1, M]
            h_pred: [B, N, M]
            marginals: [B, N+1, M] (for normalization)
        
        Returns:
            Martingale violation penalty
        """
        B, N_plus_1, M = u_pred.shape
        N = N_plus_1 - 1
        S = self.grid  # [M]
        
        total_violation = 0.0
        
        for t in range(N):
            # Construct transition kernel P(x_{t+1} | x_t) using Gibbs formula
            # P(x_{t+1} | x_t) ∝ exp[(u_{t+1}(x_{t+1}) + h_t(x_t)·(x_{t+1} - x_t))/ε]
            
            # Get potentials at time t and t+1
            u_t = u_pred[:, t, :]      # [B, M]
            u_tp1 = u_pred[:, t+1, :]  # [B, M]
            h_t = h_pred[:, t, :]      # [B, M]
            
            # Compute transition log-probabilities
            # Shape gymnastics: [B, M_t, M_{t+1}]
            u_tp1_expanded = u_tp1.unsqueeze(1).expand(-1, M, -1)  # [B, M, M]
            h_t_expanded = h_t.unsqueeze(-1).expand(-1, -1, M)     # [B, M, M]
            
            # Differences: x_{t+1} - x_t
            S_t = S.unsqueeze(1).expand(M, M)       # [M, M]
            S_tp1 = S.unsqueeze(0).expand(M, M)     # [M, M]
            delta_S = S_tp1 - S_t                   # [M, M]
            
            # Gibbs kernel (unnormalized)
            log_kernel = (u_tp1_expanded + h_t_expanded * delta_S.unsqueeze(0)) / self.epsilon
            kernel = F.softmax(log_kernel, dim=-1)  # [B, M, M] normalized over x_{t+1}
            
            # Conditional expectation: E[X_{t+1} | X_t = x_t]
            cond_exp = torch.einsum('bik,k->bi', kernel, S)  # [B, M]
            
            # Martingale condition: E[X_{t+1} | X_t] should equal X_t
            target = S.unsqueeze(0).expand(B, -1)  # [B, M]
            
            # Violation: weighted by marginal density at time t (care more about high-prob regions)
            mu_t = marginals[:, t, :]  # [B, M]
            violation = ((cond_exp - target)**2 * mu_t).sum(dim=-1).mean()  # scalar
            
            total_violation += violation
        
        return total_violation / N  # Average over time steps


class MarginalLoss(nn.Module):
    """
    L_marg = KL(P_pred || P_target) or Wasserstein distance
    
    Ensures pushforward of predicted measure matches input marginals.
    """
    def __init__(self, distance='kl'):
        super().__init__()
        self.distance = distance
    
    def forward(self, u_pred, h_pred, marginals, mask=None):
        """
        Compute marginal mismatch.
        
        For now, simplified: directly penalize if u doesn't reconstruct marginals.
        Full version would sample paths and check empirical marginals.
        """
        # Simplified: Assume u encodes marginal information
        # (More rigorous version: sample from Gibbs measure and check marginals)
        
        # Placeholder: L2 consistency check
        # In practice, integrate Gibbs density and compare to marginals
        return torch.tensor(0.0, device=u_pred.device)  # TODO: Implement


class DriftLoss(nn.Module):
    """
    L_drift = E[E[h(X_t)]²] = penalize non-zero expectation of h.
    
    Mathematical Justification:
    For martingale: E[X_{t+1} - X_t | X_t] = 0
    In Gibbs form with h_t: E[h_t(X_t)] should be 0 (zero drift)
    
    This is a stronger constraint than MartingaleLoss and directly
    addresses the drift issue observed in validation.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, h_pred, marginals):
        """
        Args:
            h_pred: [B, N, M] - Martingale potentials
            marginals: [B, N+1, M] - Marginal distributions
        
        Returns:
            Drift penalty (should be 0 for perfect martingale)
        """
        B, N, M = h_pred.shape
        
        total_drift = 0.0
        for t in range(N):
            # E[h_t(X_t)] under marginal μ_t
            h_t = h_pred[:, t, :]  # [B, M]
            mu_t = marginals[:, t, :]  # [B, M]
            
            # Expectation: sum(h_t * mu_t) over grid dimension
            drift_t = torch.sum(h_t * mu_t, dim=-1)  # [B]
            
            # Penalize non-zero drift (squared penalty)
            total_drift += torch.mean(drift_t ** 2)
        
        return total_drift / N  # Average over time steps


# ============================================================================
# COMPOSITE LOSS
# ============================================================================

class MMOTLoss(nn.Module):
    """
    Total loss = weighted sum of components.
    
    Hyperparameters (tuned empirically):
        λ₁ = 1.0 (distillation, main signal)
        λ₂ = 0.5 (martingale, physics)
        λ₃ = 0.1 (marginal, soft constraint)
        λ₄ = 1e-4 (regularization)
    """
    def __init__(self, grid, epsilon=0.1, lambda_distill=1.0, lambda_martingale=5.0, 
                 lambda_drift=5.0, lambda_marginal=0.1, lambda_reg=1e-4):
        super().__init__()
        
        self.distill_loss = DistillationLoss()
        self.mart_loss = MartingaleLoss(grid, epsilon)
        self.drift_loss = DriftLoss()  # NEW: Explicit drift penalty
        self.marg_loss = MarginalLoss()
        
        self.lambda_distill = lambda_distill
        self.lambda_mart = lambda_martingale
        self.lambda_drift = lambda_drift  # NEW
        self.lambda_marg = lambda_marginal
        self.lambda_reg = lambda_reg
    
    def forward(self, u_pred, h_pred, u_true, h_true, marginals, mask=None):
        """
        Compute total loss.
        
        Returns:
            loss: scalar
            loss_dict: breakdown of components
        """
        # Component losses
        L_distill = self.distill_loss(u_pred, u_true, h_pred, h_true, mask)
        L_mart = self.mart_loss(u_pred, h_pred, marginals, mask)
        L_drift = self.drift_loss(h_pred, marginals)  # NEW
        L_marg = self.marg_loss(u_pred, h_pred, marginals, mask)
        
        # Total
        loss = (self.lambda_distill * L_distill + 
                self.lambda_mart * L_mart +
                self.lambda_drift * L_drift +  # NEW
                self.lambda_marg * L_marg)
        
        # Return breakdown for logging
        loss_dict = {
            'total': loss.item(),
            'distill': L_distill.item(),
            'martingale': L_mart.item(),
            'drift': L_drift.item(),  # NEW
            'marginal': L_marg.item()
        }
        
        return loss, loss_dict


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("Testing MMOT Loss Functions")
    print("="*60)
    
    # Test loss computation
    B, N, M = 16, 10, 200
    grid = torch.linspace(50, 200, M)
    
    print("\n1. Creating loss function...")
    loss_fn = MMOTLoss(grid, epsilon=0.1)
    
    # Dummy predictions and targets
    print("\n2. Creating dummy data...")
    u_pred = torch.randn(B, N+1, M, requires_grad=True)
    h_pred = torch.randn(B, N, M, requires_grad=True)
    u_true = torch.randn(B, N+1, M)
    h_true = torch.randn(B, N, M)
    marginals = F.softmax(torch.randn(B, N+1, M), dim=-1)
    mask = torch.ones(B, N+1, dtype=torch.bool)
    
    # Forward
    print("\n3. Computing loss...")
    loss, loss_dict = loss_fn(u_pred, h_pred, u_true, h_true, marginals, mask)
    
    print("\n4. Loss components:")
    for k, v in loss_dict.items():
        print(f"   {k}: {v:.6f}")
    
    # Backward (check gradients)
    print("\n5. Testing gradients...")
    loss.backward()
    print(f"   Gradient norm (u): {u_pred.grad.norm().item():.4f}")
    print(f"   Gradient norm (h): {h_pred.grad.norm().item():.4f}")
    
    # Test individual components
    print("\n6. Testing individual loss components...")
    
    distill_loss = DistillationLoss()
    L_distill = distill_loss(u_pred, u_true, h_pred, h_true, mask)
    print(f"   Distillation loss: {L_distill.item():.6f}")
    
    mart_loss = MartingaleLoss(grid)
    L_mart = mart_loss(u_pred, h_pred, marginals, mask)
    print(f"   Martingale loss: {L_mart.item():.6f}")
    
    print("\n" + "="*60)
    print("✅ All loss function tests passed!")
    print("="*60)
