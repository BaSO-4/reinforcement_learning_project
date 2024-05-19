import environments
import gym
from agents.basic_agent import BasicAgent
from agents.advanced_agent import AdvancedAgent

env = gym.make('environments/AdvancedMaze', render_mode='human')
# env.action_space.seed(42)
agent = AdvancedAgent()
agent.train(env, num_iterations=500000)

for _ in range(15):
    agent.play(env)

env.close()