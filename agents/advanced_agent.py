import numpy as np

class AdvancedAgent:
    def __init__(self):
        # 2s are for binary encodings. 4s are for: which direction is least explored, which is second least explored... 5s are for last moves (d, r, u, l, None) last 4 is for moves
        self.Q = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 5, 5, 5, 5, 4))
        # self.Q = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 5, 5, 5, 5, 4))
    
    def decode_state(self, state):
        walls = state["next to a wall"] 

        borders = state["next to a border"]
        
        # dead_ends = state["dead end in sight"]

        target = state["target in sight"]
        
        explored_down = state["explored"]["down"]
        explored_right = state["explored"]["right"]
        explored_up = state["explored"]["up"]
        explored_left = state["explored"]["left"]
        exploration_list = [explored_down, explored_right, explored_up, explored_left]
        least_explored = np.argmin(exploration_list)
        exploration_list.remove(exploration_list[least_explored])
        second_least_explored = np.argmin(exploration_list)
        exploration_list.remove(exploration_list[second_least_explored])
        third_least_explored = np.argmin(exploration_list)
        last_moves = state["last moves"]
        last_move_0 = last_moves[3] if last_moves[3] is not None else 4
        last_move_1 = last_moves[2] if last_moves[2] is not None else 4
        last_move_2 = last_moves[1] if last_moves[1] is not None else 4
        last_move_3 = last_moves[0] if last_moves[0] is not None else 4


        return walls, borders, target, least_explored, second_least_explored, third_least_explored, last_move_0, last_move_1, last_move_2, last_move_3
        # return walls, borders, dead_ends, target, least_explored, second_least_explored, third_least_explored

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(4)

        walls, borders, target, least_explored, second_least_explored, third_least_explored, last_move_0, last_move_1, last_move_2, last_move_3 = self.decode_state(state)
        # walls, borders, dead_ends, target, least_explored, second_least_explored, third_least_explored = self.decode_state(state)

        # evaluations = self.Q[walls[0], walls[1], walls[2], walls[3], borders[0], borders[1], borders[2], borders[3], dead_ends[0], dead_ends[1], dead_ends[2], dead_ends[3], target[0], target[1], target[2], target[3], least_explored, second_least_explored, third_least_explored]
        evaluations = self.Q[walls[0], walls[1], walls[2], walls[3], borders[0], borders[1], borders[2], borders[3], target[0], target[1], target[2], target[3], least_explored, second_least_explored, third_least_explored, last_move_0, last_move_1, last_move_2, last_move_3]
        best = np.argmax(evaluations)
        return best

    def train(self, env, num_iterations, alpha=0.2, gamma=0.7, start_epsilon=0.1):
        print("Learning...")
        env.set_training_status(True)
        for k in range(num_iterations):
            epsilon = start_epsilon
            state = env.reset()
            limit = env.size**2
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, epsilon)
                next_state, reward, done_env, _, _ = env.step(action)
                done = done_env
                walls, borders, target, least_explored, second_least_explored, third_least_explored, last_move_0, last_move_1, last_move_2, last_move_3 = self.decode_state(state)
                next_walls, next_borders, next_target, next_least_explored, next_second_least_explored, next_third_least_explored, next_last_move_0, next_last_move_1, next_last_move_2, next_last_move_3 = self.decode_state(next_state)
                self.Q[walls[0], walls[1], walls[2], walls[3], borders[0], borders[1], borders[2], borders[3], target[0], target[1], target[2], target[3], least_explored, second_least_explored, third_least_explored, last_move_0, last_move_1, last_move_2, last_move_3, action] += alpha * (reward + gamma * np.max(self.Q[next_walls[0], next_walls[1], next_walls[2], next_walls[3], next_borders[0], next_borders[1], next_borders[2], next_borders[3], next_target[0], next_target[1], next_target[2], next_target[3], next_least_explored, next_second_least_explored, next_third_least_explored, next_last_move_0, next_last_move_1, next_last_move_2, next_last_move_3]) - self.Q[walls[0], walls[1], walls[2], walls[3], borders[0], borders[1], borders[2], borders[3], target[0], target[1], target[2], target[3], least_explored, second_least_explored, third_least_explored, last_move_0, last_move_1, last_move_2, last_move_3, action])
                state = next_state
                if i > limit:
                    break
                i += 1
            if k % 10000 == 0:
                print("Iteration", k)
                np.save('Q_matrices/advanced.npy', self.Q)
        print("Learning done!")

    def play(self, env):
        env.set_training_status(False)
        state = env.reset()
        done = False
        i = 0
        while not done:
            limit = 3 * env.size
            action = self.choose_action(state, 0)
            next_state, _, done, _, _ = env.step(action)
            state = next_state
            if i > limit:
                break
            i += 1
