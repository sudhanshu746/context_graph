"""
Adaptive Graph Walker - Two-phase exploration and exploitation

Implements biased random walks for graph exploration with adaptive
parameters that shift from global exploration to local exploitation.
"""

from typing import List, Optional, Dict, Tuple, Callable
import random
from .graph import ContextGraph


class AdaptiveGraphWalker:
    """
    Adaptive random walk on a context graph.
    
    Implements a two-phase approach:
    - Phase 1 (Global Exploration): Low p (return), high q (explore)
      Discovers structural equivalence across the graph
    - Phase 2 (Local Exploitation): High p (return), low q (explore)
      Focuses on homophily in local neighborhoods
    
    Walk bias parameters:
    - p: Return parameter - likelihood of returning to previous node
    - q: In-out parameter - likelihood of exploring outward vs staying local
    
    Example:
        >>> graph = ContextGraph()
        >>> # ... add nodes and edges ...
        >>> walker = AdaptiveGraphWalker(graph, p=1.0, q=2.0)
        >>> walk = walker.walk(start_node="API", length=10)
        >>> 
        >>> # Adaptive investigation
        >>> investigation = walker.investigate(
        ...     start_node="API",
        ...     evidence_fn=lambda n: n.startswith("Err"),
        ...     max_steps=50
        ... )
    """
    
    def __init__(self, 
                 graph: ContextGraph,
                 p: float = 1.0,
                 q: float = 1.0,
                 seed: Optional[int] = None):
        """
        Initialize walker.
        
        Args:
            graph: ContextGraph to walk on
            p: Return parameter (higher = more likely to return)
            q: In-out parameter (higher = more likely to explore)
            seed: Random seed for reproducibility
        """
        self.graph = graph
        self.p = p
        self.q = q
        self._rng = random.Random(seed)
        self._alias_tables: Dict[Tuple[str, str], Tuple[List[float], List[int]]] = {}
    
    def _compute_transition_probs(self, prev: Optional[str], current: str) -> List[Tuple[str, float]]:
        """
        Compute transition probabilities from current node.
        
        Uses node2vec-style biased sampling based on the relationship
        between previous, current, and candidate next nodes.
        """
        neighbors = self.graph.neighbors(current)
        if not neighbors:
            return []
        
        if prev is None:
            # First step: uniform distribution weighted by edge weight
            weights = []
            for neighbor in neighbors:
                weight = self.graph.get_edge_weight(current, neighbor)
                weights.append((neighbor, weight))
            return weights
        
        # Biased walk
        prev_neighbors = set(self.graph.all_neighbors(prev))
        weights = []
        
        for neighbor in neighbors:
            edge_weight = self.graph.get_edge_weight(current, neighbor)
            
            if neighbor == prev:
                # Return to previous node
                alpha = 1.0 / self.p
            elif neighbor in prev_neighbors:
                # Neighbor of previous (local)
                alpha = 1.0
            else:
                # New exploration
                alpha = 1.0 / self.q
            
            weights.append((neighbor, alpha * edge_weight))
        
        return weights
    
    def _sample_next(self, prev: Optional[str], current: str) -> Optional[str]:
        """Sample next node based on transition probabilities."""
        weights = self._compute_transition_probs(prev, current)
        if not weights:
            return None
        
        nodes, probs = zip(*weights)
        total = sum(probs)
        
        if total == 0:
            return self._rng.choice(nodes)
        
        # Normalize and sample
        r = self._rng.random() * total
        cumsum = 0
        for node, prob in zip(nodes, probs):
            cumsum += prob
            if r <= cumsum:
                return node
        
        return nodes[-1]
    
    def walk(self, 
             start_node: str, 
             length: int,
             allow_revisit: bool = True) -> List[str]:
        """
        Perform a biased random walk.
        
        Args:
            start_node: Starting node for the walk
            length: Maximum walk length
            allow_revisit: Whether to allow revisiting nodes
        
        Returns:
            List of visited node IDs
        """
        if not self.graph.has_node(start_node):
            raise ValueError(f"Start node {start_node} not in graph")
        
        walk = [start_node]
        visited = {start_node} if not allow_revisit else set()
        
        prev = None
        current = start_node
        
        for _ in range(length - 1):
            neighbors = self.graph.neighbors(current)
            
            if not allow_revisit:
                neighbors = [n for n in neighbors if n not in visited]
            
            if not neighbors:
                break
            
            next_node = self._sample_next(prev, current)
            
            if next_node is None:
                break
            
            if not allow_revisit and next_node in visited:
                break
            
            walk.append(next_node)
            visited.add(next_node)
            prev = current
            current = next_node
        
        return walk
    
    def generate_walks(self,
                       num_walks: int,
                       walk_length: int,
                       start_nodes: Optional[List[str]] = None) -> List[List[str]]:
        """
        Generate multiple random walks.
        
        Args:
            num_walks: Number of walks per start node
            walk_length: Length of each walk
            start_nodes: Specific start nodes (default: all nodes)
        
        Returns:
            List of walks
        """
        if start_nodes is None:
            start_nodes = self.graph.nodes
        
        walks = []
        for _ in range(num_walks):
            for start in start_nodes:
                walk = self.walk(start, walk_length)
                walks.append(walk)
        
        return walks
    
    def investigate(self,
                    start_node: str,
                    evidence_fn: Callable[[str], float],
                    max_steps: int = 100,
                    exploration_steps: int = 20,
                    exploitation_threshold: float = 0.5) -> Dict[str, any]:
        """
        Adaptive investigation following evidence.
        
        Phase 1: Global exploration with low p, high q
        Phase 2: Local exploitation when evidence accumulates
        
        Args:
            start_node: Starting node
            evidence_fn: Function returning evidence score (0-1) for a node
            max_steps: Maximum investigation steps
            exploration_steps: Initial exploration phase length
            exploitation_threshold: Evidence threshold to switch to exploitation
        
        Returns:
            Investigation results including path and evidence
        """
        # Store original parameters
        original_p, original_q = self.p, self.q
        
        # Phase 1: Exploration (low p, high q)
        self.p = 0.5  # Less likely to return
        self.q = 2.0  # More likely to explore outward
        
        visited = set()
        path = []
        evidence_scores = {}
        total_evidence = 0.0
        phase = "exploration"
        focus_nodes = []
        
        current = start_node
        prev = None
        
        for step in range(max_steps):
            path.append(current)
            visited.add(current)
            
            # Collect evidence
            evidence = evidence_fn(current)
            evidence_scores[current] = evidence
            total_evidence += evidence
            
            if evidence > exploitation_threshold:
                focus_nodes.append(current)
            
            # Check for phase transition
            if step >= exploration_steps and focus_nodes:
                if phase == "exploration":
                    phase = "exploitation"
                    # High p, low q for local focus
                    self.p = 2.0
                    self.q = 0.5
            
            # Get next node
            neighbors = self.graph.neighbors(current)
            if not neighbors:
                break
            
            # In exploitation phase, prefer high-evidence neighbors
            if phase == "exploitation":
                neighbor_scores = []
                for n in neighbors:
                    n_evidence = evidence_fn(n)
                    neighbor_scores.append((n, n_evidence))
                neighbor_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Weighted selection favoring high evidence
                if neighbor_scores and neighbor_scores[0][1] > 0:
                    # Pick from top candidates
                    top_k = min(3, len(neighbor_scores))
                    candidates = neighbor_scores[:top_k]
                    weights = [s[1] + 0.1 for s in candidates]
                    total_w = sum(weights)
                    r = self._rng.random() * total_w
                    cumsum = 0
                    for node, _ in candidates:
                        cumsum += weights[candidates.index((node, _))]
                        if r <= cumsum:
                            next_node = node
                            break
                    else:
                        next_node = candidates[-1][0]
                else:
                    next_node = self._sample_next(prev, current)
            else:
                next_node = self._sample_next(prev, current)
            
            if next_node is None:
                break
            
            prev = current
            current = next_node
        
        # Restore original parameters
        self.p, self.q = original_p, original_q
        
        return {
            "path": path,
            "evidence_scores": evidence_scores,
            "total_evidence": total_evidence,
            "focus_nodes": focus_nodes,
            "final_phase": phase,
            "nodes_visited": len(visited)
        }
    
    def set_parameters(self, p: float, q: float) -> None:
        """Update walk bias parameters."""
        self.p = p
        self.q = q
        self._alias_tables.clear()
    
    def __repr__(self) -> str:
        return f"AdaptiveGraphWalker(p={self.p}, q={self.q}, nodes={self.graph.num_nodes()})"
