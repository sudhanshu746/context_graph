"""
Trajectory Corpus - Manage and process agent trajectories

A trajectory is a sequence of nodes visited by an agent.
The corpus maintains multiple trajectories and computes co-occurrence statistics.
"""

from typing import List, Dict, Optional, Iterator, Tuple
from collections import defaultdict
import numpy as np
from .graph import ContextGraph


class TrajectoryCorpus:
    """
    A corpus of agent trajectories for learning graph structure.
    
    Trajectories are sequences of node visits that encode the implicit
    structure of a graph through co-occurrence patterns.
    
    Example:
        >>> corpus = TrajectoryCorpus(window_size=2)
        >>> corpus.add_trajectory(["A", "B", "C", "D"])
        >>> corpus.add_trajectory(["A", "C", "E"])
        >>> matrix = corpus.compute_cooccurrence_matrix()
        >>> print(matrix["A"]["C"])  # Co-occurrence count
    """
    
    def __init__(self, window_size: int = 2):
        """
        Initialize trajectory corpus.
        
        Args:
            window_size: Size of the sliding window for co-occurrence
        """
        self.window_size = window_size
        self.trajectories: List[List[str]] = []
        self._node_set: set = set()
        self._cooccurrence_matrix: Optional[Dict[str, Dict[str, int]]] = None
        self._dirty = True
    
    def add_trajectory(self, trajectory: List[str]) -> None:
        """
        Add a trajectory to the corpus.
        
        Args:
            trajectory: List of node IDs representing the path
        """
        if len(trajectory) >= 2:
            self.trajectories.append(trajectory)
            self._node_set.update(trajectory)
            self._dirty = True
    
    def add_trajectories(self, trajectories: List[List[str]]) -> None:
        """Add multiple trajectories at once."""
        for traj in trajectories:
            self.add_trajectory(traj)
    
    @property
    def nodes(self) -> List[str]:
        """Get all unique nodes in the corpus."""
        return sorted(self._node_set)
    
    @property
    def num_trajectories(self) -> int:
        """Get the number of trajectories."""
        return len(self.trajectories)
    
    def compute_cooccurrence_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Compute the co-occurrence matrix from trajectories.
        
        Uses a sliding window approach where nodes that appear within
        the window are considered to co-occur.
        
        Returns:
            Nested dict mapping (node_i, node_j) -> count
        """
        if not self._dirty and self._cooccurrence_matrix is not None:
            return self._cooccurrence_matrix
        
        cooccur = defaultdict(lambda: defaultdict(int))
        
        for trajectory in self.trajectories:
            for i, node_i in enumerate(trajectory):
                # Look at nodes within the window
                start = max(0, i - self.window_size)
                end = min(len(trajectory), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        node_j = trajectory[j]
                        cooccur[node_i][node_j] += 1
        
        self._cooccurrence_matrix = {k: dict(v) for k, v in cooccur.items()}
        self._dirty = False
        return self._cooccurrence_matrix
    
    def compute_cooccurrence_numpy(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute co-occurrence matrix as numpy array.
        
        Returns:
            Tuple of (matrix, node_to_index mapping)
        """
        nodes = self.nodes
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        
        matrix = np.zeros((n, n), dtype=np.float64)
        cooccur_dict = self.compute_cooccurrence_matrix()
        
        for node_i, neighbors in cooccur_dict.items():
            i = node_to_idx[node_i]
            for node_j, count in neighbors.items():
                j = node_to_idx[node_j]
                matrix[i, j] = count
        
        return matrix, node_to_idx
    
    def compute_pmi_matrix(self, positive: bool = True) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute Pointwise Mutual Information matrix.
        
        PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
        
        Args:
            positive: If True, use Positive PMI (negative values become 0)
        
        Returns:
            Tuple of (PMI matrix, node_to_index mapping)
        """
        cooccur_matrix, node_to_idx = self.compute_cooccurrence_numpy()
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        
        # Total count
        total = cooccur_matrix.sum() + eps
        
        # Marginal probabilities
        p_i = cooccur_matrix.sum(axis=1, keepdims=True) / total + eps
        p_j = cooccur_matrix.sum(axis=0, keepdims=True) / total + eps
        
        # Joint probability
        p_ij = cooccur_matrix / total + eps
        
        # PMI = log(P(i,j) / (P(i) * P(j)))
        pmi = np.log(p_ij / (p_i * p_j))
        
        if positive:
            pmi = np.maximum(pmi, 0)
        
        return pmi, node_to_idx
    
    def to_graph(self, min_weight: float = 0.0) -> ContextGraph:
        """
        Convert co-occurrence statistics to a weighted graph.
        
        Args:
            min_weight: Minimum co-occurrence count to create an edge
        
        Returns:
            ContextGraph with co-occurrence weights
        """
        graph = ContextGraph()
        cooccur = self.compute_cooccurrence_matrix()
        
        for node in self.nodes:
            graph.add_node(node)
        
        for node_i, neighbors in cooccur.items():
            for node_j, count in neighbors.items():
                if count > min_weight:
                    graph.add_edge(node_i, node_j, weight=count)
        
        return graph
    
    def generate_skipgrams(self) -> Iterator[Tuple[str, str]]:
        """
        Generate skip-gram pairs from trajectories.
        
        Yields:
            (center_node, context_node) pairs
        """
        for trajectory in self.trajectories:
            for i, center in enumerate(trajectory):
                start = max(0, i - self.window_size)
                end = min(len(trajectory), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        yield (center, trajectory[j])
    
    def __repr__(self) -> str:
        return f"TrajectoryCorpus(trajectories={self.num_trajectories}, nodes={len(self._node_set)}, window={self.window_size})"
