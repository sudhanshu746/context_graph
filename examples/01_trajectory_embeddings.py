"""
Example: Learning Structural Embeddings from Trajectories

This example demonstrates how to:
1. Create trajectory data from agent visits
2. Build a co-occurrence matrix
3. Learn embeddings that capture graph structure
4. Find similar nodes based on learned representations
"""

from context_graph import TrajectoryCorpus, TrajectoryEmbeddings
from context_graph.visualization import print_cooccurrence_matrix, visualize_walk


def main():
    print("=" * 60)
    print("Learning Structural Embeddings from Trajectories")
    print("=" * 60)
    
    # Example trajectories (as shown in the paper)
    # τ1: A → B → C → D
    # τ2: A → C → E  
    # τ3: B → C → D
    # τ4: F → G → H
    
    trajectories = [
        ["A", "B", "C", "D"],  # τ1
        ["A", "C", "E"],       # τ2
        ["B", "C", "D"],       # τ3
        ["F", "G", "H"],       # τ4
    ]
    
    print("\n1. Creating Trajectory Corpus")
    print("-" * 40)
    corpus = TrajectoryCorpus(window_size=2)
    corpus.add_trajectories(trajectories)
    print(corpus)
    
    print("\nTrajectories:")
    for i, traj in enumerate(trajectories, 1):
        print(f"  τ{i}: {visualize_walk(traj)}")
    
    # Compute and display co-occurrence matrix
    print("\n2. Co-occurrence Matrix")
    print("-" * 40)
    cooccur = corpus.compute_cooccurrence_matrix()
    print_cooccurrence_matrix(cooccur)
    
    # Learn embeddings
    print("\n3. Learning Embeddings (SVD method)")
    print("-" * 40)
    embeddings = TrajectoryEmbeddings(embedding_dim=8, method="svd")
    embeddings.fit(corpus)
    print(embeddings)
    
    # Show embedding vectors (first 4 dimensions)
    print("\nEmbedding vectors (first 4 dims):")
    for node in sorted(corpus.nodes):
        vec = embeddings.get_embedding(node)
        vec_str = ", ".join(f"{v:.3f}" for v in vec[:4])
        print(f"  {node}: [{vec_str}, ...]")
    
    # Find similar nodes
    print("\n4. Node Similarities")
    print("-" * 40)
    
    for query_node in ["A", "C", "F"]:
        print(f"\nMost similar to '{query_node}':")
        similar = embeddings.most_similar(query_node, top_k=3)
        for node, score in similar:
            print(f"  {node}: {score:.4f}")
    
    # Demonstrate structural clustering
    print("\n5. Structural Insights")
    print("-" * 40)
    print("Notice how:")
    print("  - C is a hub (central to A,B,C,D,E cluster)")
    print("  - F,G,H form a separate cluster")
    print("  - Nodes frequently co-occurring have similar embeddings")
    
    # Convert to graph
    print("\n6. Converting to Weighted Graph")
    print("-" * 40)
    graph = corpus.to_graph(min_weight=1.0)
    print(graph)
    print("\nEdges with weights:")
    for src, tgt, attrs in sorted(graph.edges, key=lambda x: x[2]['weight'], reverse=True)[:10]:
        print(f"  {src} → {tgt}: weight={attrs['weight']}")
    
    print("\n" + "=" * 60)
    print("Done! Embeddings learned from trajectory co-occurrence.")
    print("=" * 60)


if __name__ == "__main__":
    main()
