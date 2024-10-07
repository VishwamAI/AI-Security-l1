import tensorflow as tf
import numpy as np
from tensorflow import keras

class SecurityEnvironment:
    def __init__(self):
        self.state_size = 10
        self.action_size = 4
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return np.random.rand(self.state_size)

    def step(self, action):
        self.current_step += 1
        next_state = np.random.rand(self.state_size)
        reward = np.random.rand()  # Simplified reward
        done = self.current_step >= self.max_steps
        return next_state, reward, done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training the model
env = SecurityEnvironment()
agent = DQNAgent(env.state_size, env.action_size)
batch_size = 32
episodes = 100

for e in range(episodes):
    state = env.reset()
    for time in range(env.max_steps):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e}/{episodes}, Score: {time}")
            break
    agent.replay(batch_size)

# Save the trained model
agent.model.save('reinforcement_learning_adaptive_security_model.h5')
print("Reinforcement learning adaptive security model created and saved.")

# Function to demonstrate model usage
def use_adaptive_security_model(model, state):
    action = np.argmax(model.predict(state.reshape(1, -1))[0])
    print(f"For the given state, the model recommends action: {action}")

# Example usage
example_state = np.random.rand(env.state_size)
use_adaptive_security_model(agent.model, example_state)
