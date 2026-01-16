"""
Tests for TrajectoryEmbeddings class.
"""

from context_graph import TrajectoryCorpus, TrajectoryEmbeddings


class TestTrajectoryEmbeddings:
    """Tests for the TrajectoryEmbeddings class."""
    
    def setup_method(self):
        """Set up test corpus."""
        self.corpus = TrajectoryCorpus(window_size=2)
        self.corpus.add_trajectories([
            ["A", "B", "C", "D"],
            ["A", "C", "E"],
            ["B", "C", "D"],
            ["F", "G", "H"],
        ])
    
    def test_fit_svd(self):
        """Test SVD-based embedding learning."""
        embeddings = TrajectoryEmbeddings(embedding_dim=4, method="svd")
        embeddings.fit(self.corpus)
        
        vec = embeddings.get_embedding("A")
        assert vec.shape == (4,)
    
    def test_fit_skipgram(self):
        """Test skip-gram embedding learning."""
        embeddings = TrajectoryEmbeddings(embedding_dim=4, method="skipgram")
        embeddings.fit(self.corpus)
        
        vec = embeddings.get_embedding("A")
        assert vec.shape == (4,)
    
    def test_similarity(self):
        """Test similarity computation."""
        embeddings = TrajectoryEmbeddings(embedding_dim=8, method="svd")
        embeddings.fit(self.corpus)
        
        sim = embeddings.similarity("A", "C")
        assert -1.0 <= sim <= 1.0
        
        # Self-similarity should be ~1
        self_sim = embeddings.similarity("A", "A")
        assert abs(self_sim - 1.0) < 0.01
    
    def test_most_similar(self):
        """Test finding most similar nodes."""
        embeddings = TrajectoryEmbeddings(embedding_dim=8, method="svd")
        embeddings.fit(self.corpus)
        
        similar = embeddings.most_similar("C", top_k=3)
        
        assert len(similar) == 3
        assert all(isinstance(s, tuple) and len(s) == 2 for s in similar)
        assert "C" not in [n for n, _ in similar]  # Exclude self
    
    def test_cluster_separation(self):
        """Test that separate clusters have different embeddings."""
        embeddings = TrajectoryEmbeddings(embedding_dim=8, method="svd")
        embeddings.fit(self.corpus)
        
        # F,G,H are in a separate cluster from A,B,C,D,E
        # Similarity within cluster should be higher than across
        sim_within = embeddings.similarity("F", "G")
        sim_across = embeddings.similarity("F", "A")
        
        # F should be more similar to G than to A
        assert sim_within > sim_across
    
    def test_get_all_embeddings(self):
        """Test getting all embeddings."""
        embeddings = TrajectoryEmbeddings(embedding_dim=4, method="svd")
        embeddings.fit(self.corpus)
        
        all_emb = embeddings.get_all_embeddings()
        
        assert len(all_emb) == len(self.corpus.nodes)
        assert all(v.shape == (4,) for v in all_emb.values())
    
    def test_unknown_method(self):
        """Test error on unknown method."""
        embeddings = TrajectoryEmbeddings(embedding_dim=4, method="unknown")
        
        try:
            embeddings.fit(self.corpus)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "unknown" in str(e).lower()
