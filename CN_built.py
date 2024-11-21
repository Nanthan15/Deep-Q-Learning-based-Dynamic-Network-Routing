import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
import pickle
import os

class NetworkEnvironment:
    def __init__(self, topology):
        self.topology = topology
        self.n_nodes = len(topology.nodes)
        self.reset()

    def reset(self):
        self.current_node = np.random.choice(self.n_nodes)
        self.done = False
        return self.current_node

    def step(self, action):
        next_node = action
        if next_node in self.topology[self.current_node]:
            reward = -self.topology[self.current_node][next_node].get('weight', 1)
            self.current_node = next_node
        else:
            reward = -10  # Penalty for invalid action
        if self.current_node == self.n_nodes - 1:  # Assuming the last node is the destination
            self.done = True
            reward = 100
        return self.current_node, reward, self.done

    def get_valid_actions(self):
        return list(self.topology[self.current_node].keys())
    

class DQNAgent:
    def __init__(self, n_nodes, n_actions, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=2000):
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = []
        self.memory_size = memory_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.n_nodes,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        act_values = self.model.predict(state)
        return valid_actions[np.argmax(act_values[0][valid_actions])]

    def replay(self):
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model.get_weights(), f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.model.set_weights(pickle.load(f))


def preprocess_state(state, n_nodes):
    state_vec = np.zeros((1, n_nodes))
    state_vec[0][state] = 1
    return state_vec

def train_dqn_agent(env, agent, n_episodes=10):
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
    # Save the model after training
    agent.save_model("dqn_model_weights.pkl")

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

# Define the initial network topology
topology = nx.Graph()
edges = [(0, 1, {'weight': 1}), (1, 2, {'weight': 2}), (2, 3, {'weight': 1}), (3, 4, {'weight': 2}), (0, 2, {'weight': 4}), (1, 3, {'weight': 5})]
topology.add_edges_from(edges)

# Initialize environment and agent
env = NetworkEnvironment(topology)
n_nodes = env.n_nodes
agent = DQNAgent(n_nodes=n_nodes, n_actions=n_nodes)

# Check if the model weights file exists
if not os.path.exists("dqn_model_weights.pkl"):
    # Train the agent if no pre-trained model is found
    train_dqn_agent(env, agent)
else:
    # Load the pre-trained model
    agent.load_model("dqn_model_weights.pkl")

# Define and set up a different network topology
new_topology = nx.Graph()
# new_edges = [
#     (0, 1, {'weight': 2}),
#     (0, 2, {'weight': 3}),
#     (1, 3, {'weight': 4}),
#     (2, 3, {'weight': 1}),
#     (1, 4, {'weight': 5}),
#     (3, 4, {'weight': 2}),
#     (2, 5, {'weight': 2}),
#     (5, 4, {'weight': 3})
# ]
new_edges = [(0, 1, {'weight': 2}),
(0, 2, {'weight': 4}),
(1, 2, {'weight': 1}),
(1, 3, {'weight': 7}),
(2, 3, {'weight': 3}),
(3, 5, {'weight': 5}),
(2, 4, {'weight': 2}),
(3, 4, {'weight': 1}),
(4, 5, {'weight': 6})]
new_topology.add_edges_from(new_edges)

# Initialize a new environment with the new topology
new_env = NetworkEnvironment(new_topology)
new_n_nodes = new_env.n_nodes

# Test the agent with the new environment
start_node = 0
destination_node = 3
#destination_node = n_nodes - 1
optimal_path = test_dqn_agent(new_env, agent, start_node, destination_node)
print("Optimal Path:", optimal_path)
