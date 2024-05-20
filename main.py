import environments
import gym
from agents.basic_agent import BasicAgent
from agents.advanced_agent import AdvancedAgent

if __name__ == "__main__":
    # basic agent
    basic_env = gym.make('environments/Maze-v0', render_mode='human')
    basic_agent = BasicAgent(basic_env.size)
    basic_agent.train(basic_env, num_iterations=5000)
    for _ in range(15):
        basic_agent.play(basic_env)
    basic_env.close()


    # advanced agent
    # advanced_env = gym.make('environments/AdvancedMaze-v0', render_mode='human')
    # advanced_agent = AdvancedAgent()
    # advanced_agent.train(advanced_env, num_iterations=500000)
    # for _ in range(15):
    #     advanced_agent.play(advanced_env)
    # basic_env.close()