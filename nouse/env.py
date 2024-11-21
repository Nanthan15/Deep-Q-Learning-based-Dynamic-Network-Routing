import networkx as nx
import numpy as np

class NetworkEnvironment:
    def _init_(self, topology):
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