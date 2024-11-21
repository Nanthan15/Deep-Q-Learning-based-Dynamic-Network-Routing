import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

# Define the Network Environment
class NetworkEnvironment:
    def __init__(self, topology, source, destination):
        self.topology = topology
        self.source = source
        self.destination = destination
        self.reset()

    def reset(self):
        # Reset the environment to the initial state
        self.current_state = np.zeros(self.topology.shape)
        self.current_position = self.source  # Starting point (source)
        return self.update_state()

    def step(self, action):
        # Define the logic to move in the network based on the action taken
        next_position = tuple(map(sum, zip(self.current_position, action)))
        
        # Ensure next_position is within network bounds and a valid move
        if next_position in self.get_valid_actions():
            self.current_position = next_position
            reward = -1  # Default penalty for each move
            done = False
            
            # Reward for reaching the destination
            if self.current_position == self.destination:
                reward = 100
                done = True
            
            self.current_state = self.update_state()
            return self.current_state, reward, done, {}
        else:
            return self.current_state, -10, False, {}  # Penalty for invalid move

    def update_state(self):
        # Update the state based on the current position
        state = np.zeros(self.topology.shape)
        state[self.current_position] = 1
        return state

    def get_valid_actions(self):
        # Return a list of valid actions from the current position
        actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        valid_actions = []
        for action in actions:
            new_position = tuple(map(sum, zip(self.current_position, action)))
            if (0 <= new_position[0] < self.topology.shape[0] and
                0 <= new_position[1] < self.topology.shape[1] and
                self.topology[new_position] == 0):  # Valid if within bounds and not an obstacle
                valid_actions.append(action)
        return valid_actions

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        act_values = self.model.predict(state)
        return valid_actions[np.argmax(act_values[0])]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][self.get_action_index(action)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action_index(self, action):
        return [(0, 1), (1, 0), (0, -1), (-1, 0)].index(action)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def preprocess_state(self, state):
        return np.reshape(state, [1, self.state_size])

    def train_dqn_agent(self, env, episodes=10, batch_size=32):
        for e in range(episodes):
            state = env.reset()
            state = self.preprocess_state(state)
            for time in range(200):
                valid_actions = env.get_valid_actions()
                action = self.act(state, valid_actions)
                next_state, reward, done, _ = env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"Episode {e}/{episodes} - Score: {time}, Epsilon: {self.epsilon}")
                    break
                if len(self.memory) > batch_size:
                    self.replay(batch_size)

    def test_dqn_agent(self, env):
        state = env.reset()
        state = self.preprocess_state(state)
        done = False
        while not done:
            valid_actions = env.get_valid_actions()
            action = self.act(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            state = self.preprocess_state(next_state)
            print(f"Action taken: {action}, Reward: {reward}, Done: {done}")

# Main Program
if __name__ == "__main__":
    # Define the topology: 0 for open path, 1 for obstacles
    topology = np.array([
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 0]
    ])
    
    source = (0, 0)  # Source node
    destination = (3, 3)  # Destination node
    
    env = NetworkEnvironment(topology, source, destination)
    state_size = np.prod(topology.shape)
    action_size = 4  # There are 4 possible actions: right, down, left, up
    
    agent = DQNAgent(state_size, action_size)
    
    # Train the agent
    agent.train_dqn_agent(env)
    
    # Save the trained model
    agent.save("dqn_model.h5")
    
    # Load the
