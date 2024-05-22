import numpy as np
from utility_functions import plot_q_matrix

class BasicAgent:
    def __init__(self, env_size):
        self.Q = np.zeros((env_size, env_size, 4))
    
    def choose_action(self, state, epsilon, action_space):
        if np.random.rand() < epsilon:
            return np.random.randint(4)
        evaluations = self.Q[state[0], state[1]]
        best = np.argmax(evaluations)
        return best

    def train(self, env, num_iterations=100, alpha=0.2, gamma=0.7, start_epsilon=0.5):
        print("Learning...")
        env.set_training_status(True)
        limit = env.size**2 / 2
        for k in range(num_iterations):
            epsilon = start_epsilon * (1 - k / num_iterations)
            # alpha = start_alpha * (1 - k / (2*num_iterations))
            # print("Iteration", k, "epsilon=", epsilon, "alpha=", alpha)
            state = env.reset()["position"]
            done = False
            i = 0
            while not done:
                action_space = env.action_space
                action = self.choose_action(state, epsilon, action_space)
                next_state, reward, done_env, _, _ = env.step(action)
                next_state = next_state["position"]
                done = done_env
                self.Q[state[0], state[1], action] += alpha * (reward + gamma * np.max(self.Q[next_state[0], next_state[1]]) - self.Q[state[0], state[1], action])
                state = next_state
                if i > limit:
                    break
                i += 1
        print("Learning done!")
        # plot_q_matrix(self.Q) # to še zmer ne dela hahaha

    def play(self, env):
        limit = 3 * env.size
        env.set_training_status(False)
        state = env.reset()["position"]
        done = False
        i = 0
        while not done:
            action_space = env.action_space
            action = self.choose_action(state, 0, action_space)
            next_state, _, done, _, _ = env.step(action)
            state = next_state["position"]
            if i > limit:
                break
            i += 1
