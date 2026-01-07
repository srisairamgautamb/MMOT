"""
Robust Loss Functions for Real Market Generalization

Adds Huber loss variant that is less sensitive to outliers,
helping model generalize better to real market data which has
fat tails and extreme events not in training distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustDistillationLoss(nn.Module):
    """
    Huber loss for distillation - less sensitive to outliers.
    
    Huber loss = {
        0.5 * x^2           if |x| ≤ δ
        δ * (|x| - 0.5*δ)   if |x| > δ
    }
    
    This helps when teacher solutions have occasional large errors
    or when real data has features outside training distribution.
    """
    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta
    
    def forward(self, u_pred, u_true, h_pred, h_true, mask=None):
        # Huber loss on u potentials
        loss_u = F.huber_loss(u_pred, u_true, delta=self.delta, reduction='none')
        
        # Huber loss on h potentials
        loss_h = F.huber_loss(h_pred, h_true, delta=self.delta, reduction='none')
        
        # Apply mask
        if mask is not None:
            mask_u = mask.unsqueeze(-1)
            mask_h = mask[:, :-1].unsqueeze(-1)
            
            loss_u = (loss_u * mask_u).sum() / mask_u.sum()
            loss_h = (loss_h * mask_h).sum() / mask_h.sum()
        else:
            loss_u = loss_u.mean()
            loss_h = loss_h.mean()
        
        return loss_u + loss_h


# Add this to existing loss.py at the end
class RobustMMOTLoss(nn.Module):
    """
    MMOT loss with robust (Huber) distillation for real market generalization.
    """
    def __init__(self, grid, epsilon=1.0, lambda_distill=1.0, lambda_martingale=10.0,
                 lambda_drift=10.0, lambda_marginal=0.1, huber_delta=0.1):
        super().__init__()
        
        # Import existing components from loss.py
        from neural.training.loss import MartingaleLoss, DriftLoss, MarginalLoss
        
        self.distill_loss = RobustDistillationLoss(delta=huber_delta)
        self.mart_loss = MartingaleLoss(grid, epsilon)
        self.drift_loss = DriftLoss()
        self.marg_loss = MarginalLoss()
        
        self.lambda_distill = lambda_distill
        self.lambda_mart = lambda_martingale
        self.lambda_drift = lambda_drift
        self.lambda_marg = lambda_marginal
    
    def forward(self, u_pred, h_pred, u_true, h_true, marginals, mask=None):
        L_distill = self.distill_loss(u_pred, u_true, h_pred, h_true, mask)
        L_mart = self.mart_loss(u_pred, h_pred, marginals, mask)
        L_drift = self.drift_loss(h_pred, marginals)
        L_marg = self.marg_loss(u_pred, h_pred, marginals, mask)
        
        loss = (self.lambda_distill * L_distill +
                self.lambda_mart * L_mart +
                self.lambda_drift * L_drift +
                self.lambda_marg * L_marg)
        
        loss_dict = {
            'total': loss.item(),
            'distill': L_distill.item(),
            'martingale': L_mart.item(),
            'drift': L_drift.item(),
            'marginal': L_marg.item()
        }
        
        return loss, loss_dict
