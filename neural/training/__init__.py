"""
Training module for Neural MMOT Solver.

This module contains:
- Composite loss function (distillation + physics-informed)
- Training loop and validation
- Callbacks (early stopping, checkpointing)
- Metrics tracking
"""

from .loss import (
    MMOTLoss,
    DistillationLoss,
    MartingaleLoss,
    MarginalLoss,
)

__all__ = [
    "MMOTLoss",
    "DistillationLoss",
    "MartingaleLoss",
    "MarginalLoss",
]
