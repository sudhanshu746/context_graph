"""
Example: Complete Context Graph Pipeline

This example shows a complete end-to-end workflow:
1. Collect agent trajectories
2. Build context graph from co-occurrences  
3. Learn embeddings
4. Use adaptive walking for exploration
5. Visualize results
"""

from context_graph import (
    TrajectoryCorpus,
    TrajectoryEmbeddings,
    AdaptiveGraphWalker
)
from context_graph.visualization import (
    visualize_walk,
    save_html_visualization
)


def simulate_agent_trajectories():
    """
    Simulate trajectories from multiple agents exploring a codebase.
    
    In a real scenario, these would come from:
    - Developer navigation patterns in IDE
    - Bug tracking system traversals
    - Log analysis pipelines
    - API call graphs
    """
    return [
        # Agent 1: Exploring authentication flow
        ["main.py", "auth/login.py", "auth/session.py", "db/users.py", "auth/logout.py"],
        ["auth/login.py", "auth/password.py", "auth/session.py"],
        ["auth/session.py", "cache/redis.py", "db/users.py"],
        
        # Agent 2: Exploring API endpoints
        ["main.py", "api/routes.py", "api/handlers.py", "db/queries.py"],
        ["api/routes.py", "api/middleware.py", "auth/session.py"],
        ["api/handlers.py", "services/user_service.py", "db/users.py"],
        
        # Agent 3: Database operations
        ["db/connection.py", "db/queries.py", "db/users.py", "db/models.py"],
        ["db/users.py", "db/models.py", "cache/redis.py"],
        ["db/queries.py", "services/user_service.py", "api/handlers.py"],
        
        # Agent 4: Error investigation
        ["logs/errors.log", "auth/login.py", "auth/password.py", "db/users.py"],
        ["logs/errors.log", "api/handlers.py", "services/user_service.py"],
        
        # Agent 5: Config changes
        ["config/settings.py", "main.py", "db/connection.py"],
        ["config/settings.py", "cache/redis.py", "auth/session.py"],
    ]


def main():
    print("=" * 70)
    print("Complete Context Graph Pipeline")
    print("=" * 70)
    
    # Step 1: Collect trajectories
    print("\n📊 Step 1: Collecting Agent Trajectories")
    print("-" * 50)
    trajectories = simulate_agent_trajectories()
    print(f"Collected {len(trajectories)} trajectories")
    
    # Step 2: Build trajectory corpus
    print("\n📚 Step 2: Building Trajectory Corpus")
    print("-" * 50)
    corpus = TrajectoryCorpus(window_size=2)
    corpus.add_trajectories(trajectories)
    print(corpus)
    
    # Step 3: Analyze co-occurrences
    print("\n🔗 Step 3: Analyzing Co-occurrences")
    print("-" * 50)
    cooccur = corpus.compute_cooccurrence_matrix()
    
    # Show most frequent co-occurrences
    pairs = []
    for node_i, neighbors in cooccur.items():
        for node_j, count in neighbors.items():
            if node_i < node_j:  # Avoid duplicates
                pairs.append((node_i, node_j, count))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("Top co-occurring file pairs:")
    for n1, n2, count in pairs[:10]:
        print(f"  {n1} <-> {n2}: {count}")
    
    # Step 4: Learn embeddings
    print("\n🧠 Step 4: Learning Embeddings")
    print("-" * 50)
    embeddings = TrajectoryEmbeddings(embedding_dim=16, method="svd")
    embeddings.fit(corpus)
    print(embeddings)
    
    # Step 5: Build context graph
    print("\n🕸️  Step 5: Building Context Graph")
    print("-" * 50)
    graph = corpus.to_graph(min_weight=1.0)
    
    # Add node types based on path
    for node in graph.nodes:
        if "auth/" in node:
            node_type = "auth"
        elif "api/" in node:
            node_type = "api"
        elif "db/" in node:
            node_type = "database"
        elif "cache/" in node:
            node_type = "cache"
        elif "config/" in node:
            node_type = "config"
        elif "services/" in node:
            node_type = "service"
        elif "logs/" in node:
            node_type = "logs"
        else:
            node_type = "core"
        graph.add_node(node, node_type=node_type)
    
    print(graph)
    
    # Step 6: Find similar files
    print("\n🔍 Step 6: Finding Similar Files")
    print("-" * 50)
    
    query_files = ["auth/login.py", "db/users.py", "api/handlers.py"]
    for query in query_files:
        print(f"\nFiles similar to '{query}':")
        similar = embeddings.most_similar(query, top_k=3)
        for node, score in similar:
            print(f"  {node}: {score:.3f}")
    
    # Step 7: Adaptive exploration
    print("\n🚶 Step 7: Adaptive Graph Walking")
    print("-" * 50)
    
    walker = AdaptiveGraphWalker(graph, seed=42)
    
    # Exploration phase
    print("\nExploration walks (discovering structure):")
    walker.set_parameters(p=0.5, q=2.0)
    for start in ["main.py", "auth/login.py"]:
        walk = walker.walk(start, length=6)
        print(f"  From {start}: {visualize_walk(walk)}")
    
    # Exploitation phase
    print("\nExploitation walks (local focus):")
    walker.set_parameters(p=2.0, q=0.5)
    for start in ["db/users.py", "auth/session.py"]:
        walk = walker.walk(start, length=6)
        print(f"  From {start}: {visualize_walk(walk)}")
    
    # Step 8: Error investigation scenario
    print("\n🔴 Step 8: Error Investigation Scenario")
    print("-" * 50)
    
    def error_evidence(node: str) -> float:
        """Simulate error signals from monitoring."""
        if "db/" in node:
            return 0.6  # Database issues
        elif "auth/" in node:
            return 0.3  # Some auth problems
        elif "logs/" in node:
            return 0.8  # Error logs
        return 0.1
    
    result = walker.investigate(
        start_node="main.py",
        evidence_fn=error_evidence,
        max_steps=20,
        exploration_steps=8
    )
    
    print("Investigation from main.py:")
    print(f"  Path: {visualize_walk(result['path'][:10])}")
    print(f"  Focus areas: {result['focus_nodes']}")
    print(f"  Total evidence: {result['total_evidence']:.2f}")
    
    # Step 9: Save visualization
    print("\n💾 Step 9: Saving Results")
    print("-" * 50)
    
    save_html_visualization(
        graph,
        "examples/codebase_context_graph.html",
        title="Codebase Context Graph",
        width=1000,
        height=700
    )
    print("  Graph visualization: examples/codebase_context_graph.html")
    
    embeddings.save("examples/codebase_embeddings.npz")
    print("  Embeddings: examples/codebase_embeddings.npz")
    
    graph.save("examples/codebase_graph.json")
    print("  Graph data: examples/codebase_graph.json")
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 Summary")
    print("=" * 70)
    print(f"  Total files discovered: {graph.num_nodes()}")
    print(f"  Relationships found: {graph.num_edges() // 2}")  # Divide by 2 for undirected
    print(f"  Embedding dimensions: {embeddings.embedding_dim}")
    print("\nKey insights:")
    print("  - auth/session.py is highly connected (session management)")
    print("  - db/users.py is central to many flows")
    print("  - Clear clusters: auth, api, db, cache")
    print("=" * 70)


if __name__ == "__main__":
    main()
