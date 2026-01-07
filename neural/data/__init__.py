"""
Data module for Neural MMOT Solver.

This module handles:
- Dataset generation using Phase 2a teacher
- PyTorch DataLoader for training
- Data augmentation for robustness
"""

from .generator import (
    generate_dataset,
    sample_mmot_params,
    generate_marginals,
    solve_instance,
)

from .loader import (
    MMOTDataset,
    get_dataloaders,
)

__all__ = [
    "generate_dataset",
    "sample_mmot_params",
    "generate_marginals",
    "solve_instance",
    "MMOTDataset",
    "get_dataloaders",
]
