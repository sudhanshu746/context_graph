"""
Context Graph: Learning Structural Embeddings from Agent Trajectories

A Python library for building context graphs that learn node embeddings
from co-occurrence statistics in agent trajectories.
"""

from .graph import ContextGraph
from .embeddings import TrajectoryEmbeddings
from .walker import AdaptiveGraphWalker
from .trajectory import TrajectoryCorpus

__version__ = "0.1.0"
__all__ = [
    "ContextGraph",
    "TrajectoryEmbeddings", 
    "AdaptiveGraphWalker",
    "TrajectoryCorpus",
]
