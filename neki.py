import gym_examples
import gym
env = gym.make('gym_examples/GridWorld-v0', size=5, render_mode='human')
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    action_space = env.action_space
    action = action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()