"""
Models module for Neural MMOT Solver.

This module contains:
- Neural architecture (Transformer-based dual solver)
- Custom layers (embeddings, encoders)
- Model initialization utilities
"""

from .architecture import (
    NeuralDualSolver,
    create_model,
)

from .layers import (
    SinusoidalTimeEmbedding,
    MarginalEncoder,
)

__all__ = [
    "NeuralDualSolver",
    "create_model",
    "SinusoidalTimeEmbedding",
    "MarginalEncoder",
]
