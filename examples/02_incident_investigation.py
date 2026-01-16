"""
Example: Adaptive Graph Walking for Production Incident Investigation

This example demonstrates:
1. Building a system dependency graph
2. Using adaptive walks to investigate incidents
3. Shifting from global exploration to local exploitation
4. Following evidence to find root causes
"""

from context_graph import ContextGraph, AdaptiveGraphWalker
from context_graph.visualization import visualize_walk, save_html_visualization


def create_production_system_graph() -> ContextGraph:
    """
    Create a sample production system graph.
    
    This mimics the example from the paper showing:
    - Services: API, Auth, DB, Cache
    - Configs: Cfg_A, Cfg_B, Cfg_C
    - Dependencies: D_-1, D_-2, D_-3
    - Errors: Err_1, Err_2, Err_3
    - Other: Bill, Notif
    """
    graph = ContextGraph()
    
    # Add nodes with types
    services = ["API", "Auth", "DB", "Cache"]
    configs = ["Cfg_A", "Cfg_B", "Cfg_C"]
    dependencies = ["D_-1", "D_-2", "D_-3"]
    errors = ["Err_1", "Err_2", "Err_3"]
    others = ["Bill", "Notif"]
    
    for node in services:
        graph.add_node(node, node_type="service")
    for node in configs:
        graph.add_node(node, node_type="config")
    for node in dependencies:
        graph.add_node(node, node_type="dependency")
    for node in errors:
        graph.add_node(node, node_type="error")
    for node in others:
        graph.add_node(node, node_type="auxiliary")
    
    # Add edges (connections between components)
    # Service connections
    graph.add_undirected_edge("API", "Auth", weight=2.0)
    graph.add_undirected_edge("API", "DB", weight=2.0)
    graph.add_undirected_edge("Auth", "Cache", weight=1.5)
    graph.add_undirected_edge("Auth", "DB", weight=1.5)
    graph.add_undirected_edge("DB", "Cache", weight=1.0)
    
    # Auxiliary connections
    graph.add_undirected_edge("API", "Bill", weight=0.5)
    graph.add_undirected_edge("Cache", "Notif", weight=0.5)
    
    # Dependencies
    graph.add_undirected_edge("Auth", "D_-1", weight=1.0)
    graph.add_undirected_edge("DB", "D_-2", weight=1.0)
    graph.add_undirected_edge("DB", "D_-3", weight=1.0)
    
    # Configs connected to central config (Cfg_B)
    graph.add_undirected_edge("D_-1", "Cfg_A", weight=1.0)
    graph.add_undirected_edge("D_-2", "Cfg_B", weight=2.0)
    graph.add_undirected_edge("D_-3", "Cfg_B", weight=2.0)
    graph.add_undirected_edge("Cfg_A", "Cfg_B", weight=1.0)
    graph.add_undirected_edge("Cfg_B", "Cfg_C", weight=1.0)
    
    # Errors emanating from Cfg_B (the problematic config!)
    graph.add_undirected_edge("Cfg_A", "Err_1", weight=0.5)
    graph.add_undirected_edge("Cfg_B", "Err_2", weight=3.0)  # Strong connection!
    graph.add_undirected_edge("Cfg_B", "Err_3", weight=2.5)  # Strong connection!
    
    return graph


def error_evidence_function(node: str) -> float:
    """
    Evidence function for the investigation.
    
    Returns high scores for error-related nodes.
    In real systems, this could check logs, metrics, alerts, etc.
    """
    evidence_scores = {
        # Errors have high evidence
        "Err_1": 0.3,
        "Err_2": 0.9,  # Main error
        "Err_3": 0.85,
        # Cfg_B is the root cause
        "Cfg_B": 0.7,
        # Some evidence in related configs
        "Cfg_A": 0.2,
        "Cfg_C": 0.1,
        # Dependencies show some issues
        "D_-2": 0.4,
        "D_-3": 0.35,
        "D_-1": 0.1,
    }
    return evidence_scores.get(node, 0.0)


def main():
    print("=" * 60)
    print("Adaptive Graph Walking - Production Incident Investigation")
    print("=" * 60)
    
    # Create the system graph
    print("\n1. Creating Production System Graph")
    print("-" * 40)
    graph = create_production_system_graph()
    print(graph)
    print(f"Node types: {set(graph.get_node(n).get('node_type') for n in graph.nodes)}")
    
    # Initialize walker for exploration (Phase 1)
    print("\n2. Phase 1: Global Exploration")
    print("-" * 40)
    print("Parameters: low p (0.5), high q (2.0)")
    print("This discovers structural equivalence across the graph")
    
    walker = AdaptiveGraphWalker(graph, p=0.5, q=2.0, seed=42)
    
    # Generate exploration walks from API (entry point)
    exploration_walks = walker.generate_walks(
        num_walks=3, 
        walk_length=8,
        start_nodes=["API"]
    )
    
    print("\nExploration walks from API:")
    for i, walk in enumerate(exploration_walks, 1):
        print(f"  Walk {i}: {visualize_walk(walk)}")
    
    # Phase 2: Local Exploitation
    print("\n3. Phase 2: Local Exploitation")
    print("-" * 40)
    print("Parameters: high p (2.0), low q (0.5)")
    print("This focuses on homophily in local neighborhoods")
    
    walker.set_parameters(p=2.0, q=0.5)
    
    # Start from a suspicious area
    exploitation_walks = walker.generate_walks(
        num_walks=3,
        walk_length=8,
        start_nodes=["Cfg_B"]
    )
    
    print("\nExploitation walks from Cfg_B (suspicious config):")
    for i, walk in enumerate(exploitation_walks, 1):
        print(f"  Walk {i}: {visualize_walk(walk)}")
    
    # Adaptive Investigation
    print("\n4. Adaptive Investigation")
    print("-" * 40)
    print("Starting investigation from API entry point...")
    print("Evidence function detects errors and misconfigurations")
    
    walker.set_parameters(p=1.0, q=1.0)  # Reset to neutral
    
    result = walker.investigate(
        start_node="API",
        evidence_fn=error_evidence_function,
        max_steps=30,
        exploration_steps=10,
        exploitation_threshold=0.5
    )
    
    print("\nInvestigation Results:")
    print(f"  Path length: {len(result['path'])}")
    print(f"  Nodes visited: {result['nodes_visited']}")
    print(f"  Total evidence: {result['total_evidence']:.2f}")
    print(f"  Final phase: {result['final_phase']}")
    print(f"  Focus nodes (high evidence): {result['focus_nodes']}")
    
    print("\nInvestigation path:")
    print(f"  {visualize_walk(result['path'])}")
    
    print("\n  Evidence at each node:")
    for node in result['path'][:15]:  # Show first 15
        evidence = result['evidence_scores'].get(node, 0)
        bar = "█" * int(evidence * 20)
        print(f"    {node:8s} [{bar:20s}] {evidence:.2f}")
    
    # Analysis
    print("\n5. Root Cause Analysis")
    print("-" * 40)
    
    # Find nodes with highest evidence
    sorted_evidence = sorted(
        result['evidence_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("Top suspects by evidence score:")
    for node, score in sorted_evidence[:5]:
        node_type = graph.get_node(node).get('node_type', 'unknown')
        print(f"  {node:8s} ({node_type:10s}): {score:.2f}")
    
    print("\n🎯 CONCLUSION: Cfg_B appears to be the root cause!")
    print("   It has high evidence and strong connections to Err_2 and Err_3")
    
    # Save visualization
    print("\n6. Saving Visualization")
    print("-" * 40)
    save_html_visualization(
        graph,
        "examples/production_system_graph.html",
        title="Production System - Incident Investigation"
    )
    print("  Saved to: examples/production_system_graph.html")
    
    print("\n" + "=" * 60)
    print("Investigation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
