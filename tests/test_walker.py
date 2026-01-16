"""
Tests for AdaptiveGraphWalker class.
"""

from context_graph import ContextGraph, AdaptiveGraphWalker


class TestAdaptiveGraphWalker:
    """Tests for the AdaptiveGraphWalker class."""
    
    def setup_method(self):
        """Set up test graph."""
        self.graph = ContextGraph()
        # Create a simple graph: A - B - C - D
        #                             |
        #                             E
        self.graph.add_undirected_edge("A", "B")
        self.graph.add_undirected_edge("B", "C")
        self.graph.add_undirected_edge("C", "D")
        self.graph.add_undirected_edge("B", "E")
    
    def test_basic_walk(self):
        """Test basic random walk."""
        walker = AdaptiveGraphWalker(self.graph, seed=42)
        walk = walker.walk("A", length=5)
        
        assert walk[0] == "A"
        assert len(walk) <= 5
        # All nodes should be neighbors
        for i in range(len(walk) - 1):
            assert walk[i + 1] in self.graph.all_neighbors(walk[i])
    
    def test_walk_determinism(self):
        """Test that seeded walks are deterministic."""
        walker1 = AdaptiveGraphWalker(self.graph, seed=42)
        walker2 = AdaptiveGraphWalker(self.graph, seed=42)
        
        walk1 = walker1.walk("A", length=10)
        walk2 = walker2.walk("A", length=10)
        
        assert walk1 == walk2
    
    def test_generate_walks(self):
        """Test generating multiple walks."""
        walker = AdaptiveGraphWalker(self.graph, seed=42)
        walks = walker.generate_walks(num_walks=2, walk_length=4, start_nodes=["A", "B"])
        
        # 2 walks per node * 2 nodes = 4 walks
        assert len(walks) == 4
    
    def test_walk_bias_return(self):
        """Test return bias (high p = more returns)."""
        # This is a probabilistic test - just verify it runs without error
        walker_high_p = AdaptiveGraphWalker(self.graph, p=10.0, q=1.0, seed=42)
        walker_low_p = AdaptiveGraphWalker(self.graph, p=0.1, q=1.0, seed=42)
        
        # Generate walks and verify they're valid
        for _ in range(10):
            walk_high = walker_high_p.walk("B", length=4)
            walk_low = walker_low_p.walk("B", length=4)
            
            assert len(walk_high) >= 1
            assert len(walk_low) >= 1
            assert walk_high[0] == "B"
            assert walk_low[0] == "B"
    
    def test_investigate(self):
        """Test adaptive investigation."""
        walker = AdaptiveGraphWalker(self.graph, seed=42)
        
        # Evidence function: D has high evidence
        def evidence_fn(node):
            return 1.0 if node == "D" else 0.0
        
        result = walker.investigate(
            start_node="A",
            evidence_fn=evidence_fn,
            max_steps=20,
            exploration_steps=5
        )
        
        assert "path" in result
        assert "evidence_scores" in result
        assert "focus_nodes" in result
        assert result["path"][0] == "A"
    
    def test_investigate_finds_evidence(self):
        """Test that investigation collects evidence scores."""
        walker = AdaptiveGraphWalker(self.graph, seed=42)
        
        # E has high evidence
        def evidence_fn(node):
            return 0.9 if node == "E" else 0.1
        
        result = walker.investigate(
            start_node="A",
            evidence_fn=evidence_fn,
            max_steps=30,
            exploration_steps=5
        )
        
        # Should have visited at least some nodes and collected evidence
        assert len(result["evidence_scores"]) > 0
        assert "path" in result
        assert result["path"][0] == "A"
    
    def test_parameter_change(self):
        """Test changing walk parameters."""
        walker = AdaptiveGraphWalker(self.graph, p=1.0, q=1.0)
        
        assert walker.p == 1.0
        assert walker.q == 1.0
        
        walker.set_parameters(p=2.0, q=0.5)
        
        assert walker.p == 2.0
        assert walker.q == 0.5
    
    def test_walk_no_revisit(self):
        """Test walk without revisiting nodes."""
        walker = AdaptiveGraphWalker(self.graph, seed=42)
        walk = walker.walk("A", length=10, allow_revisit=False)
        
        # All nodes should be unique
        assert len(walk) == len(set(walk))
