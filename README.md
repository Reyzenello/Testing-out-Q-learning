# Testing-out-Q-learning
Testing out the Gym "Cart-Pole" using Q-learning


Requirements:
pip install gym numpy

python testing-out-q-learning.py


Here are the result:


![image](https://github.com/Reyzenello/Testing-out-Q-learning/assets/43668563/6f31f377-990a-47b2-ac64-294b04ec9442)



This code implements a Q-learning algorithm to solve the CartPole environment from OpenAI Gym. Let's break down the code step by step:

**1. Importing Libraries:**

* `gym`: Provides access to various reinforcement learning environments, including CartPole.
* `numpy`: Used for numerical operations, especially array manipulation.

**2. Initializing the Environment:**

```python
env = gym.make('CartPole-v1')
```
This creates an instance of the CartPole environment.  The `'CartPole-v1'` string specifies the version of the environment.

**3. Setting Hyperparameters:**

* `alpha = 0.1`:  The learning rate, controlling how much the Q-table is updated in each step.
* `gamma = 0.99`: The discount factor, determining the importance of future rewards.  A higher gamma values prioritizes future rewards more.
* `epsilon = 1.0`: The exploration rate (starting value). It determines the probability of choosing a random action (exploration) versus the best action based on the Q-table (exploitation).
* `epsilon_min = 0.01`:  The minimum exploration rate.  Epsilon decays over time, but it won't go below this value.
* `epsilon_decay = 0.995`:  The decay rate for epsilon. After each episode, epsilon is multiplied by this value to encourage more exploitation over time.
* `num_episodes = 1000`:  The total number of episodes (training iterations) to run.

**4. Initializing the Q-table:**

```python
state_space = [20, 20, 50, 50]  # Discretization of state space
action_space = env.action_space.n
q_table = np.zeros(state_space + [action_space])
```

* `state_space`: A list defining the number of bins to use for discretizing each of the four continuous state variables of the CartPole environment (cart position, cart velocity, pole angle, pole velocity at tip). These values are important for discretizing a continous space.
* `action_space`: The number of possible actions (2 in CartPole: move left or move right).
* `q_table`: The Q-table, a multi-dimensional array storing Q-values for each state-action pair. It's initialized to all zeros. The shape of your Q-table in this case will be (20,20,50,50,2).

**5. Discretizing the State Space (`discretize_state`):**

```python
def discretize_state(state):
    state_adj = (state - env.observation_space.low) * np.array([10, 10, 1, 1])
    state_discrete = np.round(state_adj, 0).astype(int)
    state_discrete = np.clip(state_discrete, 0, np.array(state_space) - 1)
    return tuple(state_discrete)
```

The CartPole environment has a continuous state space.  This function discretizes it into discrete bins to make it suitable for Q-learning with a Q-table.

* `state`: The continuous state vector from the environment.
* `env.observation_space.low`: The minimum values for each state variable, used to adjust the state.
* It then scales, rounds, and clips the values to fit within the predefined `state_space`.

**6. Q-learning Algorithm:**

```python
for episode in range(num_episodes):
    # ...
```
This loop runs for the specified number of episodes, performing the Q-learning update in each step.

* `state, _ = env.reset()`:  Resets the environment to a starting state at the beginning of each episode. The "_" is used to discard the "info" returned by the `reset` function which we don't need here.
* `done = False`:  A flag indicating whether the episode has finished (e.g., the pole fell over).
* `total_reward = 0`: Accumulates the reward for the current episode.

```python
    while not done:
        # ...
```

This inner loop runs for each step within an episode until the episode ends.

* `if np.random.rand() < epsilon`:  Chooses a random action with probability `epsilon`. This is the exploration part.
* `else: action = np.argmax(q_table[state])`:  Chooses the action with the highest Q-value in the current state according to the Q-table. This is the exploitation part.

```python
        next_state, reward, done, _, _ = env.step(action)
```
Takes the chosen action in the environment. It returns:

* `next_state`:  The new state after taking the action.
* `reward`: The reward received for taking the action.
* `done`:  Whether the episode is finished.

* `next_state = discretize_state(next_state)`:  Discretizes the next state.

```python
        # Update Q-table
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state + (best_next_action,)]
        td_error = td_target - q_table[state + (action,)]
        q_table[state + (action,)] += alpha * td_error

```
This is the core Q-learning update.

* `td_target`: Calculates the target Q-value using the Bellman equation.
* `td_error`: The difference between the target and the current Q-value (Temporal Difference error).
* The Q-table is updated using the learning rate and the TD error.

```python
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

```
* Epsilon is decayed after each episode to reduce exploration over time.
* Progress is printed every 100 episodes.
* `env.close()`: Releases the environment resources.



