import gym
import numpy as np

# Initialize the environment
env = gym.make('CartPole-v1')

# Set hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration probability
num_episodes = 1000  # Number of episodes

# Initialize Q-table
state_space = [20, 20, 50, 50]  # Discretization of state space
action_space = env.action_space.n
q_table = np.zeros(state_space + [action_space])

# Discretize the continuous state space
def discretize_state(state):
    state_adj = (state - env.observation_space.low) * np.array([10, 10, 1, 1])
    state_discrete = np.round(state_adj, 0).astype(int)
    state_discrete = np.clip(state_discrete, 0, np.array(state_space) - 1)
    return tuple(state_discrete)

# Q-learning algorithm
for episode in range(num_episodes):
    state, _ = env.reset()  # Adjusted to handle the tuple returned by reset
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.choice(action_space)  # Exploration
        else:
            action = np.argmax(q_table[state])  # Exploitation

        next_state, reward, done, _, _ = env.step(action)  # Adjusted to handle the tuple returned by step
        next_state = discretize_state(next_state)

        # Update Q-table
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state + (best_next_action,)]
        td_error = td_target - q_table[state + (action,)]
        q_table[state + (action,)] += alpha * td_error

        state = next_state
        total_reward += reward

        if done:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print progress
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
