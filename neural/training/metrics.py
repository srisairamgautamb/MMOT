"""
Metrics for tracking training progress.

Includes:
- Wasserstein distance (W₁)
- KL divergence
- Martingale violation
- Pricing error
"""

import torch
import numpy as np


def wasserstein_1d(p, q, grid):
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance).
    
    Args:
        p: Probability distribution [batch, M]
        q: Probability distribution [batch, M]
        grid: Grid points [M]
    
    Returns:
        W₁ distance for each sample in batch
    """
    # Ensure distributions sum to 1
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    q = q / (q.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Compute CDFs
    p_cdf = torch.cumsum(p, dim=-1)
    q_cdf = torch.cumsum(q, dim=-1)
    
    # Compute L1 distance between CDFs
    # W₁ = ∫ |F_P(x) - F_Q(x)| dx
    # Approximated as sum over grid
    delta_grid = grid[1] - grid[0]  # Assuming uniform grid
    w1 = torch.abs(p_cdf - q_cdf).sum(dim=-1) * delta_grid
    
    return w1


def kl_divergence(p, q, eps=1e-8):
    """
    Compute KL divergence KL(P || Q).
    
    Args:
        p: Probability distribution [batch, M]
        q: Probability distribution [batch, M]
        eps: Small constant for numerical stability
    
    Returns:
        KL divergence for each sample in batch
    """
    # Normalize
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)
    
    # KL(P||Q) = Σ p(x) log(p(x)/q(x))
    kl = (p * torch.log((p + eps) / (q + eps))).sum(dim=-1)
    
    return kl


def compute_martingale_violation(u_pred, h_pred, marginals, grid, epsilon=0.1):
    """
    Compute empirical martingale violation.
    
    E[X_{t+1} | X_t] should equal X_t
    
    Args:
        u_pred: Marginal potentials [batch, N+1, M]
        h_pred: Martingale potentials [batch, N, M]
        marginals: Input marginals [batch, N+1, M]
        grid: Grid points [M]
        epsilon: Entropic regularization
    
    Returns:
        Average martingale violation
    """
    B, N_plus_1, M = u_pred.shape
    N = N_plus_1 - 1
    
    violations = []
    
    for t in range(N):
        # Get potentials
        u_tp1 = u_pred[:, t+1, :]  # [B, M]
        h_t = h_pred[:, t, :]      # [B, M]
        
        # Compute transition kernel using Gibbs formula
        u_tp1_expanded = u_tp1.unsqueeze(1).expand(-1, M, -1)  # [B, M, M]
        h_t_expanded = h_t.unsqueeze(-1).expand(-1, -1, M)     # [B, M, M]
        
        # Differences
        S_t = grid.unsqueeze(1).expand(M, M)
        S_tp1 = grid.unsqueeze(0).expand(M, M)
        delta_S = S_tp1 - S_t
        
        # Gibbs kernel
        log_kernel = (u_tp1_expanded + h_t_expanded * delta_S.unsqueeze(0)) / epsilon
        kernel = torch.softmax(log_kernel, dim=-1)  # [B, M, M]
        
        # Conditional expectation E[X_{t+1} | X_t]
        cond_exp = torch.einsum('bik,k->bi', kernel, grid)  # [B, M]
        
        # Target: X_t
        target = grid.unsqueeze(0).expand(B, -1)
        
        # Violation weighted by marginal
        mu_t = marginals[:, t, :]
        violation = ((cond_exp - target)**2 * mu_t).sum(dim=-1).mean()
        
        violations.append(violation.item())
    
    return np.mean(violations)


def compute_pricing_error(pred_price, true_price):
    """
    Compute relative pricing error.
    
    Args:
        pred_price: Predicted price
        true_price: True price (from teacher)
    
    Returns:
        Relative error (percentage)
    """
    return abs(pred_price - true_price) / (abs(true_price) + 1e-8) * 100


class MetricsTracker:
    """
    Track multiple metrics during training.
    """
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
            self.metrics[key] = value
    
    def get_current(self, key):
        """Get current value of a metric."""
        return self.metrics.get(key, None)
    
    def get_history(self, key):
        """Get history of a metric."""
        return self.history.get(key, [])
    
    def get_average(self, key, last_n=None):
        """Get average of a metric over last N updates."""
        history = self.history.get(key, [])
        if not history:
            return None
        if last_n is None:
            return np.mean(history)
        return np.mean(history[-last_n:])
    
    def summary(self):
        """Print summary of all metrics."""
        print("Metrics Summary:")
        for key, value in self.metrics.items():
            avg = self.get_average(key)
            print(f"  {key}: current={value:.6f}, avg={avg:.6f}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("Testing metrics...")
    
    # Test Wasserstein distance
    print("\n1. Testing Wasserstein distance...")
    grid = torch.linspace(0, 10, 100)
    p = torch.softmax(torch.randn(4, 100), dim=-1)
    q = torch.softmax(torch.randn(4, 100), dim=-1)
    
    w1 = wasserstein_1d(p, q, grid)
    print(f"   W₁ distances: {w1}")
    assert w1.shape == (4,)
    assert (w1 >= 0).all()
    print("   ✅ Passed")
    
    # Test KL divergence
    print("\n2. Testing KL divergence...")
    kl = kl_divergence(p, q)
    print(f"   KL divergences: {kl}")
    assert kl.shape == (4,)
    assert (kl >= 0).all()  # KL is non-negative
    print("   ✅ Passed")
    
    # Test martingale violation
    print("\n3. Testing martingale violation...")
    B, N, M = 4, 5, 50
    grid = torch.linspace(50, 200, M)
    u_pred = torch.randn(B, N+1, M)
    h_pred = torch.randn(B, N, M)
    marginals = torch.softmax(torch.randn(B, N+1, M), dim=-1)
    
    violation = compute_martingale_violation(u_pred, h_pred, marginals, grid)
    print(f"   Martingale violation: {violation:.6f}")
    assert isinstance(violation, float)
    print("   ✅ Passed")
    
    # Test MetricsTracker
    print("\n4. Testing MetricsTracker...")
    tracker = MetricsTracker()
    
    for i in range(10):
        tracker.update(
            loss=1.0 / (i + 1),
            accuracy=0.5 + 0.05 * i
        )
    
    print(f"   Current loss: {tracker.get_current('loss'):.4f}")
    print(f"   Average loss: {tracker.get_average('loss'):.4f}")
    print(f"   Last 3 accuracy avg: {tracker.get_average('accuracy', last_n=3):.4f}")
    print("   ✅ Passed")
    
    print("\n5. Metrics summary:")
    tracker.summary()
    
    print("\n✅ All metrics tests passed!")
