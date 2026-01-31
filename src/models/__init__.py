"""Models package for NeuralFactors implementation.

Exports main classes and utility functions for VAE-based factor learning.
"""

from .stock_embedder import StockEmbedder
from .prior import StudentTPrior
from .neuralfactors import NeuralFactors

from . import encoder
from . import decoder

__all__ = [
    "NeuralFactors",
    "StockEmbedder", 
    "StudentTPrior",
    "encoder",
    "decoder"
]
