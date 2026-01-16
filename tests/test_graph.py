"""
Tests for ContextGraph class.
"""

import tempfile
import os
from context_graph import ContextGraph


class TestContextGraph:
    """Tests for the ContextGraph class."""
    
    def test_empty_graph(self):
        """Test creating an empty graph."""
        graph = ContextGraph()
        assert graph.num_nodes() == 0
        assert graph.num_edges() == 0
    
    def test_add_node(self):
        """Test adding nodes."""
        graph = ContextGraph()
        graph.add_node("A", label="Node A", weight=1.0)
        
        assert graph.has_node("A")
        assert graph.num_nodes() == 1
        
        node = graph.get_node("A")
        assert node["label"] == "Node A"
        assert node["weight"] == 1.0
    
    def test_add_edge(self):
        """Test adding edges."""
        graph = ContextGraph()
        graph.add_edge("A", "B", weight=2.0)
        
        assert graph.has_node("A")
        assert graph.has_node("B")
        assert graph.has_edge("A", "B")
        assert not graph.has_edge("B", "A")  # Directed
        assert graph.get_edge_weight("A", "B") == 2.0
    
    def test_add_undirected_edge(self):
        """Test adding undirected edges."""
        graph = ContextGraph()
        graph.add_undirected_edge("A", "B", weight=1.5)
        
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "A")
    
    def test_neighbors(self):
        """Test getting neighbors."""
        graph = ContextGraph()
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("D", "A")
        
        assert set(graph.neighbors("A")) == {"B", "C"}
        assert set(graph.predecessors("A")) == {"D"}
        assert set(graph.all_neighbors("A")) == {"B", "C", "D"}
    
    def test_degree(self):
        """Test degree calculations."""
        graph = ContextGraph()
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("D", "A")
        
        assert graph.degree("A") == 2
        assert graph.in_degree("A") == 1
    
    def test_increment_edge_weight(self):
        """Test incrementing edge weights."""
        graph = ContextGraph()
        graph.add_edge("A", "B", weight=1.0)
        graph.increment_edge_weight("A", "B", 0.5)
        
        assert graph.get_edge_weight("A", "B") == 1.5
        
        # Create new edge
        graph.increment_edge_weight("B", "C", 2.0)
        assert graph.get_edge_weight("B", "C") == 2.0
    
    def test_subgraph(self):
        """Test creating subgraph."""
        graph = ContextGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")
        
        sub = graph.subgraph(["A", "B", "C"])
        
        assert sub.num_nodes() == 3
        assert sub.has_edge("A", "B")
        assert sub.has_edge("B", "C")
        assert not sub.has_node("D")
    
    def test_serialization(self):
        """Test graph serialization."""
        graph = ContextGraph()
        graph.add_node("A", label="Test")
        graph.add_edge("A", "B", weight=2.0)
        
        data = graph.to_dict()
        restored = ContextGraph.from_dict(data)
        
        assert restored.num_nodes() == 2
        assert restored.has_edge("A", "B")
        assert restored.get_node("A")["label"] == "Test"
    
    def test_save_load(self):
        """Test saving and loading graph."""
        graph = ContextGraph()
        graph.add_edge("A", "B", weight=1.5)
        graph.add_node("A", node_type="test")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            graph.save(filepath)
            loaded = ContextGraph.load(filepath)
            
            assert loaded.num_nodes() == 2
            assert loaded.has_edge("A", "B")
            assert loaded.get_edge_weight("A", "B") == 1.5
        finally:
            os.unlink(filepath)
