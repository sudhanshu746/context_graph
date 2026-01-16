"""
Context Graph - Core Graph Structure

Represents the underlying graph structure with nodes and edges,
supporting weighted edges and node attributes.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import json


class ContextGraph:
    """
    A context graph that represents relationships between entities.
    
    The graph supports:
    - Weighted edges
    - Node attributes/metadata
    - Edge attributes
    - Serialization/deserialization
    
    Example:
        >>> graph = ContextGraph()
        >>> graph.add_node("API", node_type="service")
        >>> graph.add_node("DB", node_type="database")
        >>> graph.add_edge("API", "DB", weight=1.0, relation="queries")
        >>> print(graph.neighbors("API"))
        ['DB']
    """
    
    def __init__(self):
        """Initialize an empty context graph."""
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self._reverse_edges: Dict[str, Set[str]] = defaultdict(set)
    
    def add_node(self, node_id: str, **attributes) -> None:
        """
        Add a node to the graph with optional attributes.
        
        Args:
            node_id: Unique identifier for the node
            **attributes: Optional key-value attributes for the node
        """
        if node_id not in self._nodes:
            self._nodes[node_id] = {}
        self._nodes[node_id].update(attributes)
    
    def add_edge(self, source: str, target: str, **attributes) -> None:
        """
        Add a directed edge between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            **attributes: Optional edge attributes (e.g., weight, relation)
        """
        # Auto-create nodes if they don't exist
        if source not in self._nodes:
            self.add_node(source)
        if target not in self._nodes:
            self.add_node(target)
        
        # Set default weight if not provided
        if 'weight' not in attributes:
            attributes['weight'] = 1.0
            
        self._edges[source][target] = attributes
        self._reverse_edges[target].add(source)
    
    def add_undirected_edge(self, node1: str, node2: str, **attributes) -> None:
        """Add an undirected edge (two directed edges)."""
        self.add_edge(node1, node2, **attributes)
        self.add_edge(node2, node1, **attributes)
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node attributes."""
        return self._nodes.get(node_id)
    
    def get_edge(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """Get edge attributes."""
        return self._edges.get(source, {}).get(target)
    
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._nodes
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists between two nodes."""
        return target in self._edges.get(source, {})
    
    def neighbors(self, node_id: str) -> List[str]:
        """Get all outgoing neighbors of a node."""
        return list(self._edges.get(node_id, {}).keys())
    
    def predecessors(self, node_id: str) -> List[str]:
        """Get all incoming neighbors of a node."""
        return list(self._reverse_edges.get(node_id, set()))
    
    def all_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors (both incoming and outgoing)."""
        neighbors = set(self.neighbors(node_id))
        neighbors.update(self.predecessors(node_id))
        return list(neighbors)
    
    @property
    def nodes(self) -> List[str]:
        """Get all node IDs."""
        return list(self._nodes.keys())
    
    @property
    def edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all edges as (source, target, attributes) tuples."""
        result = []
        for source, targets in self._edges.items():
            for target, attrs in targets.items():
                result.append((source, target, attrs))
        return result
    
    def num_nodes(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)
    
    def num_edges(self) -> int:
        """Return the number of edges."""
        return sum(len(targets) for targets in self._edges.values())
    
    def degree(self, node_id: str) -> int:
        """Return the out-degree of a node."""
        return len(self._edges.get(node_id, {}))
    
    def in_degree(self, node_id: str) -> int:
        """Return the in-degree of a node."""
        return len(self._reverse_edges.get(node_id, set()))
    
    def get_edge_weight(self, source: str, target: str) -> float:
        """Get the weight of an edge."""
        edge = self.get_edge(source, target)
        return edge.get('weight', 0.0) if edge else 0.0
    
    def set_edge_weight(self, source: str, target: str, weight: float) -> None:
        """Set the weight of an edge."""
        if self.has_edge(source, target):
            self._edges[source][target]['weight'] = weight
    
    def increment_edge_weight(self, source: str, target: str, delta: float = 1.0) -> None:
        """Increment the weight of an edge, creating it if necessary."""
        if not self.has_edge(source, target):
            self.add_edge(source, target, weight=delta)
        else:
            current = self.get_edge_weight(source, target)
            self.set_edge_weight(source, target, current + delta)
    
    def subgraph(self, node_ids: List[str]) -> 'ContextGraph':
        """Create a subgraph containing only the specified nodes."""
        sub = ContextGraph()
        node_set = set(node_ids)
        
        for node_id in node_ids:
            if node_id in self._nodes:
                sub.add_node(node_id, **self._nodes[node_id])
        
        for source, targets in self._edges.items():
            if source in node_set:
                for target, attrs in targets.items():
                    if target in node_set:
                        sub.add_edge(source, target, **attrs)
        
        return sub
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            'nodes': self._nodes,
            'edges': {src: dict(targets) for src, targets in self._edges.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextGraph':
        """Deserialize graph from dictionary."""
        graph = cls()
        for node_id, attrs in data.get('nodes', {}).items():
            graph.add_node(node_id, **attrs)
        for source, targets in data.get('edges', {}).items():
            for target, attrs in targets.items():
                graph.add_edge(source, target, **attrs)
        return graph
    
    def save(self, filepath: str) -> None:
        """Save graph to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ContextGraph':
        """Load graph from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def __repr__(self) -> str:
        return f"ContextGraph(nodes={self.num_nodes()}, edges={self.num_edges()})"
