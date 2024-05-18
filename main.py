import environments
import gym
from agents.basic_agent import BasicAgent

env = gym.make('environments/Maze-v0', size=20, render_mode='human')
# env.action_space.seed(42)
agent = BasicAgent(env.size)
agent.train(env, num_episodes=10000)

for _ in range(15):
    agent.play(env)

env.close()