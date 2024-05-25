import numpy as np

# zbriš dead end, namest % dej število že znanih za vsako smer

class ZeroKnowledgeAgent:
    def __init__(self):
        self.Q = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 4))
        self.prev_step = None
    
    def decode_state(self, state):
        walls_or_borders = state["next to a wall or border"] 

        target = state["target in sight"]
        target_direction = state["target in direction"]

        last_moves = state["last moves"]
        last_move_0 = last_moves[1] if last_moves[1] is not None else 4
        last_move_1 = last_moves[0] if last_moves[0] is not None else 4

        return walls_or_borders, target, target_direction, last_move_0, last_move_1

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(4)

        walls_or_borders, target, target_direction, last_move_0, last_move_1 = self.decode_state(state)

        if target:
            evaluations = self.Q[0, 0, 0, 0, target_direction[0], target_direction[1], target_direction[2], target_direction[3], 0, 0]
        else:
            evaluations = self.Q[walls_or_borders[0], walls_or_borders[1], walls_or_borders[2], walls_or_borders[3], 0, 0, 0, 0, last_move_0, last_move_1]
        max_value = np.max(evaluations)
        max_indices = np.flatnonzero(evaluations == max_value)
        chosen = np.random.choice(max_indices)

        if chosen == self.prev_step:
            return np.random.choice(max_indices)

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
                next_state, reward, done_env, _, info = env.step(action)
                done = done_env
                self.prev_step = info["prev"]

                walls_or_borders, target, target_direction, last_move_0, last_move_1 = self.decode_state(state)
                next_walls_or_borders, next_target, next_target_direction, next_last_move_0, next_last_move_1 = self.decode_state(next_state)

                if target:                    
                    access_vector = [0, 0, 0, 0, target_direction[0], target_direction[1], target_direction[2], target_direction[3], 0, 0, action]
                else:
                    access_vector = [walls_or_borders[0], walls_or_borders[1], walls_or_borders[2], walls_or_borders[3], 0, 0, 0, 0, last_move_0, last_move_1, action]
                if next_target:
                    choosing_vector = [0, 0, 0, 0, next_target_direction[0], next_target_direction[1], next_target_direction[2], next_target_direction[3], 0, 0, 0, 0, 0]
                else:
                    choosing_vector = [next_walls_or_borders[0], next_walls_or_borders[1], next_walls_or_borders[2], next_walls_or_borders[3], 0, 0, 0, 0, next_last_move_0, next_last_move_1]
                
                self.Q[tuple(access_vector)] += alpha * (reward + gamma * np.max(self.Q[tuple(choosing_vector)]) - self.Q[tuple(access_vector)])
                state = next_state
                
                if i > limit:
                    break
                i += 1
            if k % 10000 == 0:
                np.save('Q_matrices/advanced.npy', self.Q)
            if k % 5000 == 0:
                print("Iteration", k)
        print("Learning done!")

    def play(self, env):
        env.set_training_status(False)
        state = env.reset()
        done = False
        i = 0
        while not done:
            limit = 6 * env.size
            action = self.choose_action(state, 0)
            next_state, _, done, _, info = env.step(action)
            self.prev_step = info["prev"]
            state = next_state
            if i > limit:
                break
            i += 1
