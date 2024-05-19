import numpy as np

class AdvancedAgent:
    def __init__(self):
        # 2s are for binary encodings. 4s are for: which direction is least explored, which is second least explored ... last 4 is for moves
        self.Q = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4))
    
    def decode_state(self, state):
        walls = state["next_to_a_wall"] 

        borders = state["next_to_a_border"]
        
        target = state["target_in_sight"]
        
        explored_down = state["explored"]["down"]
        explored_right = state["explored"]["right"]
        explored_up = state["explored"]["up"]
        explored_left = state["explored"]["left"]
        exploration_list = [explored_down, explored_right, explored_up, explored_left]
        least_explored = np.argmin(exploration_list)
        exploration_list.remove(least_explored)
        second_least_explored = np.argmin(exploration_list)
        exploration_list.remove(second_least_explored)
        third_least_explored = np.argmin(exploration_list)

        return walls, borders, target, least_explored, second_least_explored, third_least_explored

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(4)

        walls, borders, target, least_explored, second_least_explored, third_least_explored = self.decode_state(state)

        evaluations = self.Q[walls[0], walls[1], walls[2], walls[3], borders[0], borders[1], borders[2], borders[3], target[0], target[1], target[2], target[3], least_explored, second_least_explored, third_least_explored]
        best = np.argmax(evaluations)
        return best

    def train(self, env, num_iterations=100, alpha=0.2, gamma=0.7, start_epsilon=0.5):
        print("Learning...")
        env.set_training_status(True)
        limit = env.size**4
        for k in range(num_iterations):
            epsilon = start_epsilon * (1 - k / num_iterations)
            state = env.reset()
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, epsilon)
                next_state, reward, done_env = env.step(action)
                done = done_env
                walls, borders, target, least_explored, second_least_explored, third_least_explored = self.decode_state(state)
                next_walls, next_borders, next_target, next_least_explored, next_second_least_explored, next_third_least_explored = self.decode_state(next_state)
                self.Q[walls[0], walls[1], walls[2], walls[3], borders[0], borders[1], borders[2], borders[3], target[0], target[1], target[2], target[3], least_explored, second_least_explored, third_least_explored, action] += alpha * (reward + gamma * np.max(self.Q[next_walls[0], next_walls[1], next_walls[2], next_walls[3], next_borders[0], next_borders[1], next_borders[2], next_borders[3], next_target[0], next_target[1], next_target[2], next_target[3], next_least_explored, next_second_least_explored, next_third_least_explored]) - self.Q[walls[0], walls[1], walls[2], walls[3], borders[0], borders[1], borders[2], borders[3], target[0], target[1], target[2], target[3], least_explored, second_least_explored, third_least_explored, action])
                state = next_state
                if i > limit:
                    break
                i += 1
            if k % 100 == 0:
                print("Iteration", k)
        print("Learning done!")
        # env.print_solved(self.Q)

    def play(self, env):
        limit = 3 * env.size
        env.set_training_status(False)
        state = env.reset()
        done = False
        i = 0
        while not done:
            action = self.choose_action(state, 0)
            next_state, _, done = env.step(action)
            state = next_state
            if i > limit:
                break
            i += 1
