import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import gymnasium as gym
import numpy as np

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode=None)  # Set render_mode='human' if you want to see it run

# Q-learning parameters
num_episodes = 10
num_bins = (1, 1, 6, 12)
num_actions = env.action_space.n
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Initialize Q-table
Q = np.zeros(num_bins + (num_actions,))

# Discretize observation function
def discretize(observation):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], np.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -np.radians(50)]
    ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(observation))]
    new_obs = [int(round((num_bins[i] - 1) * ratios[i])) for i in range(len(observation))]
    new_obs = [min(num_bins[i] - 1, max(0, new_obs[i])) for i in range(len(observation))]
    return tuple(new_obs)

# Epsilon-greedy policy
def epsilon_greedy_policy(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# Training loop
for episode in range(num_episodes):
    observation, _ = env.reset()
    state = discretize(observation)
    done = False
    total_reward = 0

    while not done:
        action = epsilon_greedy_policy(state)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_observation)

        # Q-learning update
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Evaluation
total_rewards = []
for _ in range(100):
    observation, _ = env.reset()
    state = discretize(observation)
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(Q[state])
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = discretize(observation)
        total_reward += reward

    total_rewards.append(total_reward)

print(f"\nAverage Reward over 100 Episodes: {np.mean(total_rewards)}")
env.close()
