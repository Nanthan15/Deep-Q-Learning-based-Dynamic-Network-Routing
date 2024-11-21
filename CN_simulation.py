# Filename: mininet_simulation.py
from mininet.net import Mininet
from mininet.node import OVSController
from mininet.cli import CLI
from mininet.link import TCLink
import networkx as nx
import matplotlib.pyplot as plt

# Define network topology in Mininet
def create_network(topology):
    net = Mininet(controller=OVSController, link=TCLink)
    switches = {}
    hosts = {}
    
    # Create switches and hosts
    for node in topology.nodes:
        switches[node] = net.addSwitch(f's{node}')
        hosts[node] = net.addHost(f'h{node}')
    
    # Add links based on topology edges
    for u, v, data in topology.edges(data=True):
        weight = data.get('weight', 1)
        net.addLink(switches[u], switches[v], bw=weight)
    
    # Start network
    net.start()
    return net, hosts

# Define a function to visualize the network and optimal path
def visualize_network(topology, path):
    pos = nx.spring_layout(topology)
    nx.draw(topology, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    nx.draw_networkx_edges(topology, pos, edgelist=[(path[i], path[i + 1]) for i in range(len(path) - 1)], edge_color='red', width=2)
    plt.show()

# Example topology (replace with the new topology)
topology = nx.Graph()
new_edges = [(0, 1, {'weight': 2}),
             (0, 2, {'weight': 4}),
             (1, 2, {'weight': 1}),
             (1, 3, {'weight': 7}),
             (2, 3, {'weight': 3}),
             (3, 5, {'weight': 5}),
             (2, 4, {'weight': 2}),
             (3, 4, {'weight': 1}),
             (4, 5, {'weight': 6})]
topology.add_edges_from(new_edges)

# Optimal path obtained from the ML model
optimal_path = [0, 2, 4, 5]  # Replace with the path returned by your ML model

# Create and start the Mininet network
net, hosts = create_network(topology)

# Visualize the network and optimal path
visualize_network(topology, optimal_path)

# Run CLI for user interaction
CLI(net)

# Stop the network
net.stop()
