import networkx as nx
from env import NetworkEnvironment

def preprocess_state(state, n_nodes):
    state_vec = np.zeros((1, n_nodes))
    state_vec[0][state] = 1
    return state_vec

def train_dqn_agent(env, agent, n_episodes=100):
    for episode in range(n_episodes):
        state = env.reset()
        state = preprocess_state(state, env.n_nodes)
        total_reward = 0
        for time in range(500):
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            next_state, reward, done = env.step(action)
            next_state = preprocess_state(next_state, env.n_nodes)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode: {episode+1}/{n_episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2}")
                break
            agent.replay()

def test_dqn_agent(env, agent, start_node, destination_node):
    state = start_node
    state_vec = preprocess_state(state, env.n_nodes)
    path = [state]
    while state != destination_node:
        valid_actions = env.get_valid_actions()
        action = agent.act(state_vec, valid_actions)
        state, _, done = env.step(action)
        state_vec = preprocess_state(state, env.n_nodes)
        path.append(state)
        if done:
            break
    return path

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

# Test the agent
start_node = 0
destination_node = n_nodes - 1
optimal_path = test_dqn_agent(env, agent, start_node, destination_node)
print("Optimal Path:",optimal_path)
