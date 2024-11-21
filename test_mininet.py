import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout


def visualize_network(topology, path):
    # Draw the network topology
    plt.figure(figsize=(12, 8))
    pos = graphviz_layout(topology, prog="dot")
    
    # Draw nodes
    nx.draw_networkx_nodes(topology, pos, node_size=700, node_color='lightblue')
    # Draw edges
    nx.draw_networkx_edges(topology, pos, edgelist=topology.edges, width=1)
    # Draw labels
    nx.draw_networkx_labels(topology, pos, font_size=16, font_weight='bold')
    
    # Highlight the optimal path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(topology, pos, edgelist=path_edges, edge_color='red', width=2)
    
    plt.title("Network Topology with Optimal Path")
    plt.show()

# Define the network topology
topology = nx.Graph()
edges = [
    (0, 1, {'weight': 2}),
    (0, 2, {'weight': 4}),
    (1, 2, {'weight': 1}),
    (1, 3, {'weight': 7}),
    (2, 3, {'weight': 3}),
    (3, 5, {'weight': 5}),
    (2, 4, {'weight': 2}),
    (3, 4, {'weight': 1}),
    (4, 5, {'weight': 6})
]
topology.add_edges_from(edges)

# Example path from your model
optimal_path = [0, 2, 4, 5]  # Replace with your model's output

visualize_network(topology, optimal_path)
