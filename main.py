import environments
import gym
from agents.basic_agent import BasicAgent
from agents.advanced_agent import AdvancedAgent
from agents.zero_knowledge_agent import ZeroKnowledgeAgent
from utility_functions import play


if __name__ == "__main__":
    '''
    #basic agent on an example maze
    basic_env = gym.make('environments/ExampleMaze-v0', render_mode='human')
    basic_agent = BasicAgent(basic_env.size)
    basic_agent.train(basic_env, num_iterations=10000)
    for _ in range(10):
        basic_agent.play(basic_env)
    basic_env.close()
    '''
    
    # basic agent on random maze
    # basic_env = gym.make('environments/Maze-v0', render_mode='human')
    # basic_agent = BasicAgent(basic_env.size)
    # basic_agent.train(basic_env, num_iterations=5000)
    # for _ in range(15):
    #     basic_agent.play(basic_env)
    # basic_env.close()
    
    #basic agent on an example maze, enemy chasing him
    # basic_env = gym.make('environments/ExampleMaze-v2', render_mode='human')
    # basic_agent = BasicAgent(basic_env.size)
    # basic_agent.train(basic_env, num_iterations=10000)
    # for _ in range(20):
    #     basic_agent.play(basic_env)
    # basic_env.close()

    # advanced agent
    # advanced_env = gym.make('environments/AdvancedMaze-v0', render_mode='human')
    # advanced_agent = AdvancedAgent()
    # # advanced_agent.train(advanced_env, num_iterations=100000000)
    # # ! tole spodaj predvaja kako se že natreniran agent obnaša (ne rabiš še enkrat trenirat)
    # play(advanced_agent, 'Q_matrices/advanced.npy', advanced_env, 20)

    # zero knowledge, only memory is 2 last moves and last choice of move in this cell
    advanced_env = gym.make('environments/ZeroKnowledgeMaze', render_mode='human')
    advanced_agent = ZeroKnowledgeAgent()
    advanced_agent.train(advanced_env, num_iterations=100000000)
    # # ! tole spodaj predvaja kako se že natreniran agent obnaša (ne rabiš še enkrat trenirat)
    # play(advanced_agent, 'Q_matrices/advanced.npy', advanced_env, 20)