"""
Trajectory Embeddings - Learn node embeddings from co-occurrence statistics

Implements methods similar to Word2Vec Skip-gram for learning
distributed representations of nodes from trajectory data.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .trajectory import TrajectoryCorpus


class TrajectoryEmbeddings:
    """
    Learn node embeddings from trajectory co-occurrence statistics.
    
    Uses SVD on the PMI matrix (similar to implicit matrix factorization
    in Word2Vec) or optionally trains a neural skip-gram model.
    
    The key insight: e_i · e_j ≈ log P(co-occur in window | trajectories)
    
    Example:
        >>> corpus = TrajectoryCorpus(window_size=2)
        >>> corpus.add_trajectory(["A", "B", "C", "D"])
        >>> corpus.add_trajectory(["A", "C", "E"])
        >>> embeddings = TrajectoryEmbeddings(embedding_dim=32)
        >>> embeddings.fit(corpus)
        >>> vec_a = embeddings.get_embedding("A")
        >>> similar = embeddings.most_similar("A", top_k=3)
    """
    
    def __init__(self, embedding_dim: int = 64, method: str = "svd"):
        """
        Initialize embedding model.
        
        Args:
            embedding_dim: Dimensionality of embeddings
            method: "svd" for SVD-based or "skipgram" for neural
        """
        self.embedding_dim = embedding_dim
        self.method = method
        self._embeddings: Optional[np.ndarray] = None
        self._node_to_idx: Optional[Dict[str, int]] = None
        self._idx_to_node: Optional[Dict[int, str]] = None
    
    def fit(self, corpus: TrajectoryCorpus) -> 'TrajectoryEmbeddings':
        """
        Learn embeddings from trajectory corpus.
        
        Args:
            corpus: TrajectoryCorpus with agent trajectories
        
        Returns:
            self for method chaining
        """
        if self.method == "svd":
            self._fit_svd(corpus)
        elif self.method == "skipgram":
            self._fit_skipgram(corpus)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def _fit_svd(self, corpus: TrajectoryCorpus) -> None:
        """Fit using SVD on PMI matrix."""
        # Compute PMI matrix
        pmi_matrix, node_to_idx = corpus.compute_pmi_matrix(positive=True)
        
        self._node_to_idx = node_to_idx
        self._idx_to_node = {i: n for n, i in node_to_idx.items()}
        
        # SVD decomposition
        # PMI ≈ U @ S @ V^T
        # We use U @ sqrt(S) as embeddings
        n_components = min(self.embedding_dim, pmi_matrix.shape[0] - 1)
        
        U, S, Vt = np.linalg.svd(pmi_matrix, full_matrices=False)
        
        # Take top k components
        U_k = U[:, :n_components]
        S_k = S[:n_components]
        
        # Embeddings = U @ sqrt(S)
        self._embeddings = U_k @ np.diag(np.sqrt(S_k))
        
        # Pad if needed
        if self._embeddings.shape[1] < self.embedding_dim:
            padding = np.zeros((self._embeddings.shape[0], 
                              self.embedding_dim - self._embeddings.shape[1]))
            self._embeddings = np.hstack([self._embeddings, padding])
    
    def _fit_skipgram(self, corpus: TrajectoryCorpus, 
                      epochs: int = 10, 
                      learning_rate: float = 0.01,
                      negative_samples: int = 5) -> None:
        """
        Fit using neural skip-gram with negative sampling.
        
        This is a simplified implementation for demonstration.
        """
        nodes = corpus.nodes
        n_nodes = len(nodes)
        
        self._node_to_idx = {n: i for i, n in enumerate(nodes)}
        self._idx_to_node = {i: n for n, i in self._node_to_idx.items()}
        
        # Initialize embeddings randomly
        center_emb = np.random.randn(n_nodes, self.embedding_dim) * 0.1
        context_emb = np.random.randn(n_nodes, self.embedding_dim) * 0.1
        
        # Generate training pairs
        pairs = list(corpus.generate_skipgrams())
        
        for epoch in range(epochs):
            np.random.shuffle(pairs)
            total_loss = 0.0
            
            for center, context in pairs:
                c_idx = self._node_to_idx[center]
                ctx_idx = self._node_to_idx[context]
                
                # Positive sample
                c_vec = center_emb[c_idx]
                ctx_vec = context_emb[ctx_idx]
                
                score = np.dot(c_vec, ctx_vec)
                prob = self._sigmoid(score)
                
                # Gradient for positive sample
                grad_c = (prob - 1) * ctx_vec
                grad_ctx = (prob - 1) * c_vec
                
                # Negative sampling
                for _ in range(negative_samples):
                    neg_idx = np.random.randint(0, n_nodes)
                    if neg_idx == ctx_idx:
                        continue
                    
                    neg_vec = context_emb[neg_idx]
                    neg_score = np.dot(c_vec, neg_vec)
                    neg_prob = self._sigmoid(neg_score)
                    
                    grad_c += neg_prob * neg_vec
                    context_emb[neg_idx] -= learning_rate * neg_prob * c_vec
                
                # Update
                center_emb[c_idx] -= learning_rate * grad_c
                context_emb[ctx_idx] -= learning_rate * grad_ctx
                
                total_loss += -np.log(prob + 1e-10)
        
        # Final embeddings: average of center and context
        self._embeddings = (center_emb + context_emb) / 2
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1 + exp_x)
    
    def get_embedding(self, node: str) -> np.ndarray:
        """Get embedding vector for a node."""
        if self._embeddings is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if node not in self._node_to_idx:
            raise KeyError(f"Unknown node: {node}")
        
        return self._embeddings[self._node_to_idx[node]].copy()
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all embeddings as a dictionary."""
        if self._embeddings is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return {node: self._embeddings[idx].copy() 
                for node, idx in self._node_to_idx.items()}
    
    def similarity(self, node1: str, node2: str) -> float:
        """Compute cosine similarity between two nodes."""
        vec1 = self.get_embedding(node1)
        vec2 = self.get_embedding(node2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def most_similar(self, node: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar nodes.
        
        Args:
            node: Query node
            top_k: Number of results to return
        
        Returns:
            List of (node, similarity) tuples
        """
        if self._embeddings is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        query_vec = self.get_embedding(node)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        # Compute all similarities
        norms = np.linalg.norm(self._embeddings, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        
        similarities = self._embeddings @ query_vec / (norms * query_norm)
        
        # Get top-k (excluding self)
        indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in indices:
            if self._idx_to_node[idx] != node:
                results.append((self._idx_to_node[idx], float(similarities[idx])))
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, filepath: str) -> None:
        """Save embeddings to numpy file."""
        if self._embeddings is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        np.savez(filepath, 
                 embeddings=self._embeddings,
                 nodes=list(self._node_to_idx.keys()))
    
    @classmethod
    def load(cls, filepath: str) -> 'TrajectoryEmbeddings':
        """Load embeddings from numpy file."""
        data = np.load(filepath, allow_pickle=True)
        
        model = cls(embedding_dim=data['embeddings'].shape[1])
        model._embeddings = data['embeddings']
        model._node_to_idx = {n: i for i, n in enumerate(data['nodes'])}
        model._idx_to_node = {i: n for n, i in model._node_to_idx.items()}
        
        return model
    
    def __repr__(self) -> str:
        n_nodes = len(self._node_to_idx) if self._node_to_idx else 0
        return f"TrajectoryEmbeddings(dim={self.embedding_dim}, nodes={n_nodes}, method={self.method})"
