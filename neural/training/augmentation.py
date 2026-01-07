"""
Simple data augmentation for marginals during training.

Adds small perturbations to marginals and teacher potentials
to help model generalize beyond the exact training distribution.
"""

import torch
import numpy as np


def augment_training_batch(marginals, u_star, h_star, config):
    """
    Apply data augmentation to a training batch.
    
    Args:
        marginals: [B, N+1, M]
        u_star: [B, N+1, M]
        h_star: [B, N, M]
        config: dict with augmentation parameters
    
    Returns:
        Augmented (marginals, u_star, h_star)
    """
    if not config.get('enabled', False):
        return marginals, u_star, h_star
    
    # 1. Add Gaussian noise to marginals
    if config.get('marginal_noise_std', 0) > 0:
        noise = torch.randn_like(marginals) * config['marginal_noise_std']
        marginals_aug = marginals + noise
        
        # Ensure positive and renormalize
        marginals_aug = torch.clamp(marginals_aug, min=0)
        marginals_aug = marginals_aug / (marginals_aug.sum(dim=-1, keepdim=True) + 1e-10)
    else:
        marginals_aug = marginals
    
    # 2. Add noise to teacher potentials
    if config.get('potential_noise_std', 0) > 0:
        u_noise = torch.randn_like(u_star) * config['potential_noise_std']
        h_noise = torch.randn_like(h_star) * config['potential_noise_std']
        
        u_star_aug = u_star + u_noise
        h_star_aug = h_star + h_noise
    else:
        u_star_aug = u_star
        h_star_aug = h_star
    
    return marginals_aug, u_star_aug, h_star_aug


def augment_marginals_simple(marginals, noise_std=0.02):
    """
    Lightweight augmentation: just add noise and renormalize.
    
    Args:
        marginals: [B, N+1, M] or [N+1, M]
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Augmented marginals (same shape)
    """
    noise = torch.randn_like(marginals) * noise_std
    marginals_aug = marginals + noise
    
    # Ensure positive
    marginals_aug = torch.clamp(marginals_aug, min=0)
    
    # Renormalize
    if marginals.dim() == 3:  # [B, N+1, M]
        marginals_aug = marginals_aug / (marginals_aug.sum(dim=-1, keepdim=True) + 1e-10)
    else:  # [N+1, M]
        marginals_aug = marginals_aug / (marginals_aug.sum(dim=-1, keepdim=True) + 1e-10)
    
    return marginals_aug
