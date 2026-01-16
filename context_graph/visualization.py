"""
Visualization utilities for Context Graphs.
"""

from typing import Dict, List, Any
import json
from .graph import ContextGraph


def graph_to_d3_json(graph: ContextGraph) -> Dict[str, Any]:
    """
    Convert graph to D3.js compatible JSON format.
    
    Returns:
        Dict with 'nodes' and 'links' arrays
    """
    nodes = []
    for node_id in graph.nodes:
        node_data = {"id": node_id}
        attrs = graph.get_node(node_id)
        if attrs:
            node_data.update(attrs)
        nodes.append(node_data)
    
    links = []
    for source, target, attrs in graph.edges:
        link = {"source": source, "target": target}
        link.update(attrs)
        links.append(link)
    
    return {"nodes": nodes, "links": links}


def print_cooccurrence_matrix(matrix: Dict[str, Dict[str, int]], 
                               max_nodes: int = 10) -> None:
    """
    Pretty print a co-occurrence matrix.
    """
    nodes = sorted(matrix.keys())[:max_nodes]
    
    # Header
    header = "     " + "  ".join(f"{n[:3]:>3}" for n in nodes)
    print(header)
    print("-" * len(header))
    
    for node_i in nodes:
        row = f"{node_i[:3]:>3} |"
        for node_j in nodes:
            if node_i == node_j:
                row += "  - "
            else:
                count = matrix.get(node_i, {}).get(node_j, 0)
                row += f" {count:>2} "
        print(row)


def visualize_walk(walk: List[str], max_display: int = 20) -> str:
    """
    Create a text visualization of a walk.
    
    Returns:
        String representation like: A → B → C → D
    """
    if len(walk) > max_display:
        displayed = walk[:max_display//2] + ["..."] + walk[-max_display//2:]
    else:
        displayed = walk
    
    return " → ".join(displayed)


def generate_html_visualization(graph: ContextGraph,
                                 title: str = "Context Graph",
                                 width: int = 800,
                                 height: int = 600) -> str:
    """
    Generate an interactive HTML visualization using D3.js.
    
    Args:
        graph: ContextGraph to visualize
        title: Page title
        width: Canvas width
        height: Canvas height
    
    Returns:
        HTML string that can be saved to a file
    """
    graph_data = json.dumps(graph_to_d3_json(graph))
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        #graph {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .node {{
            cursor: pointer;
        }}
        .node circle {{
            stroke: #fff;
            stroke-width: 2px;
        }}
        .node text {{
            font-size: 12px;
            font-weight: 500;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .tooltip {{
            position: absolute;
            padding: 8px 12px;
            background: #333;
            color: white;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <svg id="graph" width="{width}" height="{height}"></svg>
    <script>
        const data = {graph_data};
        
        const svg = d3.select("#graph");
        const width = {width};
        const height = {height};
        
        // Color scale for node types
        const color = d3.scaleOrdinal(d3.schemeCategory10);
        
        // Create simulation
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(80))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));
        
        // Create arrow markers
        svg.append("defs").selectAll("marker")
            .data(["arrow"])
            .join("marker")
            .attr("id", d => d)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("fill", "#999")
            .attr("d", "M0,-5L10,0L0,5");
        
        // Create links
        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.weight || 1))
            .attr("marker-end", "url(#arrow)");
        
        // Create nodes
        const node = svg.append("g")
            .selectAll("g")
            .data(data.nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        node.append("circle")
            .attr("r", 15)
            .attr("fill", d => color(d.node_type || "default"));
        
        node.append("text")
            .attr("dx", 18)
            .attr("dy", 4)
            .text(d => d.id);
        
        // Tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);
        
        node.on("mouseover", function(event, d) {{
            tooltip.transition().duration(200).style("opacity", .9);
            tooltip.html(`<strong>${{d.id}}</strong><br/>Type: ${{d.node_type || 'unknown'}}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        }})
        .on("mouseout", function() {{
            tooltip.transition().duration(500).style("opacity", 0);
        }});
        
        // Update positions
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});
        
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}
        
        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}
        
        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>'''
    
    return html


def save_html_visualization(graph: ContextGraph, 
                            filepath: str,
                            **kwargs) -> None:
    """Save HTML visualization to file."""
    html = generate_html_visualization(graph, **kwargs)
    with open(filepath, 'w') as f:
        f.write(html)
