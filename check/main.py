from network_module import NetworkEnvironment, DQNAgent, train_dqn_agent, test_dqn_agent
import networkx as nx

# Define the network topology
topology = nx.Graph()
edges = [(0, 1, {'weight': 1}), (1, 2, {'weight': 2}), (2, 3, {'weight': 1}), (3, 4, {'weight': 2}), (0, 2, {'weight': 4}), (1, 3, {'weight': 5})]
topology.add_edges_from(edges)

# Initialize environment and agent
env = NetworkEnvironment(topology)
n_nodes = env.n_nodes
agent = DQNAgent(n_nodes=n_nodes, n_actions=n_nodes)

# Train the agent
train_dqn_agent(env, agent)

# Load the trained model before testing
agent.load("dqn_model_weights.weights.h5")

# Test the agent
start_node = 0
destination_node = n_nodes - 1
optimal_path = test_dqn_agent(env, agent, start_node, destination_node)
print("Optimal Path:", optimal_path)
