# Context Graph

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Learn structural embeddings from agent trajectories for context-aware graph exploration.**

Context Graph is a Python library that learns node embeddings from co-occurrence statistics in agent trajectories. Inspired by the principles of Word2Vec applied to graph structures, it enables intelligent navigation and exploration of complex systems.

## 🎯 Key Concepts

### Learning Structural Embeddings from Trajectories

Embeddings are learned from **co-occurrence statistics** in agent trajectories. Nodes visited together frequently get similar vectors—encoding graph structure without explicit schema.

```
Trajectory Corpus:
  τ1: A → B → C → D
  τ2: A → C → E  
  τ3: B → C → D
  τ4: F → G → H

         ↓ (window-based co-occurrence)

Co-occurrence Matrix → SVD → Learned Embeddings

Key insight: e_i · e_j ≈ log P(co-occur in window | trajectories)
```

### Adaptive Graph Walking

The walker adapts based on evidence—starting broad (structural exploration), narrowing as signal accumulates (local exploitation).

- **Phase 1: Global Exploration** (low p, high q)
  - Discovers structural equivalence across the graph
  - Broad sweep to understand the landscape
  
- **Phase 2: Local Exploitation** (high p, low q)
  - Focuses on homophily in local neighborhoods
  - Follows accumulated evidence to find root causes

Walk bias parameters:
- `p = P(return)`: Likelihood of returning to previous node
- `q = P(explore)`: Likelihood of exploring outward vs staying local

## 🚀 Installation

```bash
# Install from source
git clone https://github.com/sudhanshu746/context-graph.git
cd context-graph
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With visualization support
pip install -e ".[viz]"
```

## 📖 Quick Start

### Learning Embeddings from Trajectories

```python
from context_graph import TrajectoryCorpus, TrajectoryEmbeddings

# Create trajectory corpus
corpus = TrajectoryCorpus(window_size=2)
corpus.add_trajectories([
    ["A", "B", "C", "D"],
    ["A", "C", "E"],
    ["B", "C", "D"],
    ["F", "G", "H"],
])

# Learn embeddings
embeddings = TrajectoryEmbeddings(embedding_dim=32, method="svd")
embeddings.fit(corpus)

# Find similar nodes
similar = embeddings.most_similar("C", top_k=3)
print(similar)  # [('B', 0.89), ('D', 0.87), ('A', 0.82)]

# Get embedding vector
vec_c = embeddings.get_embedding("C")
```

### Adaptive Graph Walking

```python
from context_graph import ContextGraph, AdaptiveGraphWalker

# Create graph
graph = ContextGraph()
graph.add_undirected_edge("API", "Auth", weight=2.0)
graph.add_undirected_edge("API", "DB", weight=2.0)
graph.add_undirected_edge("DB", "Cache", weight=1.5)
# ... add more edges

# Create walker
walker = AdaptiveGraphWalker(graph, p=1.0, q=1.0, seed=42)

# Phase 1: Exploration (low p, high q)
walker.set_parameters(p=0.5, q=2.0)
exploration_walk = walker.walk("API", length=10)

# Phase 2: Exploitation (high p, low q)
walker.set_parameters(p=2.0, q=0.5)
exploitation_walk = walker.walk("Cfg_B", length=10)
```

### Adaptive Investigation

```python
# Define evidence function (e.g., from logs, metrics)
def error_evidence(node):
    if "Err" in node:
        return 0.9
    elif "Cfg" in node:
        return 0.5
    return 0.1

# Run adaptive investigation
result = walker.investigate(
    start_node="API",
    evidence_fn=error_evidence,
    max_steps=50,
    exploration_steps=15,
    exploitation_threshold=0.5
)

print(f"Focus nodes: {result['focus_nodes']}")
print(f"Investigation path: {result['path']}")
```

### Visualization

```python
from context_graph.visualization import save_html_visualization

# Generate interactive D3.js visualization
save_html_visualization(
    graph,
    "my_graph.html",
    title="System Dependencies"
)
```

## 🏗️ Architecture

```
context_graph/
├── __init__.py          # Package exports
├── graph.py             # ContextGraph - core graph structure
├── trajectory.py        # TrajectoryCorpus - trajectory management
├── embeddings.py        # TrajectoryEmbeddings - embedding learning
├── walker.py            # AdaptiveGraphWalker - biased random walks
└── visualization.py     # Visualization utilities
```

### Core Classes

| Class | Description |
|-------|-------------|
| `ContextGraph` | Weighted directed graph with node/edge attributes |
| `TrajectoryCorpus` | Manages trajectories, computes co-occurrence matrices |
| `TrajectoryEmbeddings` | Learns embeddings via SVD or skip-gram |
| `AdaptiveGraphWalker` | Biased random walks with adaptive parameters |

## 📊 Examples

### 1. Trajectory Embeddings
```bash
python examples/01_trajectory_embeddings.py
```
Demonstrates learning embeddings from co-occurrence statistics.

### 2. Incident Investigation
```bash
python examples/02_incident_investigation.py
```
Shows adaptive walking for production incident root cause analysis.

### 3. Complete Pipeline
```bash
python examples/03_complete_pipeline.py
```
End-to-end workflow from trajectories to embeddings to investigation.

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=context_graph --cov-report=html

# Run specific test file
pytest tests/test_embeddings.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

- Inspired by research on learning context graphs for AI agents
- Node2Vec: Scalable Feature Learning for Networks
- Word2Vec: Efficient Estimation of Word Representations in Vector Space
- GloVe: Global Vectors for Word Representation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on concepts from "How do you build a Context Graph?" by Jaya Gupta
- Graph embedding techniques from the network science community