import numpy as np
import matplotlib.pyplot as plt
# the mazes are without a target, so that we can set the target randomly and have a lot of different maazes this way (need this for the advanced maze)

maze_5_1 = np.array([   
    [0, 0, 0, 0, 0],
    [0, 2, 2, 2, 0],    
    [0, 0, 0, 0, 0],    
    [0, 2, 2, 2, 0],    
    [0, 0, 0, 0, 0],
], dtype=np.int8)

maze_5_2 = np.array([
    [0, 0, 0, 2, 0],
    [0, 2, 0, 2, 0],    
    [0, 2, 0, 2, 0],    
    [0, 2, 0, 2, 0],    
    [0, 0, 0, 0, 0],
], dtype=np.int8)

maze_5_3 = np.array([
    [0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0],    
    [0, 0, 0, 0, 0],    
    [0, 2, 2, 2, 2],    
    [0, 0, 0, 0, 0],
], dtype=np.int8)

maze_5_4 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0],
], dtype=np.int8)

maze_5_5 = np.array([
    [0, 2, 0, 0, 0],
    [0, 0, 2, 2, 0],    
    [0, 0, 0, 2, 0],    
    [0, 2, 2, 2, 0],    
    [0, 0, 0, 0, 0],
], dtype=np.int8)

maze_7_1 = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 0, 0, 0, 2, 0],
            [0, 2, 0, 0, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.int8)

maze_7_2 = np.array([
            [0, 2, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 2, 0],
            [2, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 2, 0],
            [2, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.int8)

maze_10_1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 2, 0],    
    [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],    
    [0, 2, 0, 0, 0, 0, 0, 0, 2, 0],    
    [0, 2, 0, 0, 0, 0, 2, 0, 2, 0],    
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],    
    [0, 2, 0, 2, 2, 2, 2, 0, 2, 0],    
    [0, 2, 0, 2, 0, 0, 0, 0, 2, 0],    
    [0, 2, 2, 2, 2, 2, 2, 0, 2, 0],    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.int8)

maze_10_2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],    
    [0, 0, 2, 2, 0, 0, 2, 2, 2, 0],    
    [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],    
    [0, 2, 0, 0, 2, 2, 2, 0, 0, 0],    
    [0, 2, 0, 0, 2, 0, 0, 0, 0, 0],    
    [0, 2, 0, 0, 2, 0, 0, 0, 0, 0],    
    [0, 2, 2, 2, 2, 2, 2, 2, 0, 0],    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],    
], dtype=np.int8)

maze_10_3 = np.array([
    [0, 2, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],    
    [0, 0, 2, 2, 0, 0, 2, 0, 2, 0],    
    [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],    
    [0, 0, 0, 0, 2, 2, 0, 0, 2, 0],    
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],    
    [0, 2, 0, 0, 0, 0, 0, 2, 2, 0],    
    [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],    
    [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 2],    
], dtype=np.int8)

maze_10_4 = np.array([
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],    
    [0, 2, 0, 0, 2, 0, 2, 0, 0, 0],    
    [0, 0, 2, 0, 2, 0, 0, 2, 2, 0],    
    [2, 0, 2, 0, 2, 2, 0, 0, 2, 0],    
    [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],    
    [0, 2, 2, 2, 2, 0, 2, 2, 2, 2],    
    [0, 0, 0, 0, 2, 0, 2, 0, 0, 0],    
    [2, 2, 2, 0, 2, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 2, 0],    
], dtype=np.int8)

maze_10_5 = np.array([
    [0, 0, 0, 0, 2, 0, 0, 0, 2, 0],
    [0, 2, 2, 0, 2, 0, 2, 0, 2, 0],    
    [0, 0, 2, 0, 2, 0, 2, 0, 2, 0],    
    [2, 0, 2, 0, 0, 0, 2, 0, 0, 0],    
    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2],    
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],    
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],    
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],    
    [2, 2, 2, 2, 2, 0, 2, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],    
], dtype=np.int8)

maze_10_6 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    
], dtype=np.int8)

def my_mazes():
    # return np.array([maze_5_1, maze_5_2, maze_5_3, maze_5_4, maze_5_5])
    return np.array([maze_5_1, maze_5_2, maze_5_3, maze_5_4, maze_5_5, maze_7_1, maze_7_2, maze_10_1, maze_10_2, maze_10_3, maze_10_4, maze_10_5, maze_10_6])


# play with pretrained Q matrix
def play(agent, Q_path, env, times):
    Q = np.load(Q_path)
    agent.Q = Q
    for _ in range(times):
        agent.play(env)
    env.close()


def get_best_action_and_value(q_values):
    best_action = np.argmax(q_values)
    best_value = q_values[best_action]
    return best_action, best_value

# Normalize values to be between 0 and 1
def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)

# Plot the Q-matrix with optimal moves and colors for certainty
def plot_q_matrix(q_matrix):
    moves = range(4)
    directions = {
        0: (1, 0),
        1: (0, 1),
        2: (-1, 0),
        3: (0, -1)
    }
    nrows, ncols, _ = q_matrix.shape
    _, ax = plt.subplots()
    ax.set_xticks(np.arange(0, ncols, 1))
    ax.set_yticks(np.arange(0, nrows, 1))
    ax.grid(True)

    opt_directions = np.max(q_matrix, axis=2)
    normalized_opt_directions = normalize(opt_directions)

    for i in range(nrows):
        for j in range(ncols):
            best_action = np.argmax(opt_directions[i, j])
            action = moves[best_action]
            action_vector = directions[action]
            color = plt.cm.viridis(normalized_opt_directions[i, j])
            ax.arrow(j, i, action_vector[1] * 0.3, action_vector[0] * 0.3,
                     head_width=0.1, head_length=0.1, fc=color, ec=color)

        plt.gca().invert_yaxis()
        plt.show()