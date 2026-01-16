"""
Tests for TrajectoryCorpus class.
"""

from context_graph import TrajectoryCorpus


class TestTrajectoryCorpus:
    """Tests for the TrajectoryCorpus class."""
    
    def test_empty_corpus(self):
        """Test creating empty corpus."""
        corpus = TrajectoryCorpus(window_size=2)
        assert corpus.num_trajectories == 0
        assert len(corpus.nodes) == 0
    
    def test_add_trajectory(self):
        """Test adding trajectories."""
        corpus = TrajectoryCorpus(window_size=2)
        corpus.add_trajectory(["A", "B", "C", "D"])
        
        assert corpus.num_trajectories == 1
        assert set(corpus.nodes) == {"A", "B", "C", "D"}
    
    def test_short_trajectory_ignored(self):
        """Test that single-node trajectories are ignored."""
        corpus = TrajectoryCorpus(window_size=2)
        corpus.add_trajectory(["A"])  # Too short
        
        assert corpus.num_trajectories == 0
    
    def test_cooccurrence_matrix(self):
        """Test co-occurrence matrix computation."""
        corpus = TrajectoryCorpus(window_size=2)
        # A → B → C → D
        corpus.add_trajectory(["A", "B", "C", "D"])
        
        cooccur = corpus.compute_cooccurrence_matrix()
        
        # A should co-occur with B (distance 1) and C (distance 2)
        assert cooccur["A"]["B"] > 0
        assert cooccur["A"]["C"] > 0
        # A should not co-occur with D (distance 3 > window 2)
        assert cooccur.get("A", {}).get("D", 0) == 0
    
    def test_window_size_effect(self):
        """Test that window size affects co-occurrences."""
        corpus_small = TrajectoryCorpus(window_size=1)
        corpus_large = TrajectoryCorpus(window_size=3)
        
        trajectory = ["A", "B", "C", "D", "E"]
        corpus_small.add_trajectory(trajectory)
        corpus_large.add_trajectory(trajectory)
        
        cooccur_small = corpus_small.compute_cooccurrence_matrix()
        cooccur_large = corpus_large.compute_cooccurrence_matrix()
        
        # With window=1, A only sees B
        assert cooccur_small.get("A", {}).get("C", 0) == 0
        # With window=3, A sees B, C, D
        assert cooccur_large["A"]["C"] > 0
    
    def test_multiple_trajectories(self):
        """Test co-occurrences from multiple trajectories."""
        corpus = TrajectoryCorpus(window_size=2)
        corpus.add_trajectory(["A", "B", "C"])
        corpus.add_trajectory(["A", "B", "C"])
        
        cooccur = corpus.compute_cooccurrence_matrix()
        
        # A-B should co-occur in each trajectory (bidirectional)
        assert cooccur["A"]["B"] >= 2  # At least 2 co-occurrences
    
    def test_to_graph(self):
        """Test converting corpus to graph."""
        corpus = TrajectoryCorpus(window_size=2)
        corpus.add_trajectory(["A", "B", "C", "D"])
        
        graph = corpus.to_graph(min_weight=0.0)  # Accept any weight
        
        assert graph.num_nodes() == 4
        # Check that some edges were created from co-occurrences
        assert graph.num_edges() > 0
    
    def test_skipgram_generation(self):
        """Test skip-gram pair generation."""
        corpus = TrajectoryCorpus(window_size=1)
        corpus.add_trajectory(["A", "B", "C"])
        
        pairs = list(corpus.generate_skipgrams())
        
        # Should have pairs: (A,B), (B,A), (B,C), (C,B)
        assert ("A", "B") in pairs
        assert ("B", "A") in pairs
        assert ("B", "C") in pairs
        assert ("C", "B") in pairs
