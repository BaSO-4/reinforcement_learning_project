import gym
from gym import spaces
import pygame
import numpy as np
import matplotlib.pyplot as plt

#! all coordinates are in the form (y, x)!!!! this is due to matrix indexing
class Maze(gym.Env):
    # maze:
    # 0 - empty cell
    # 1 - target cell
    # -1 - visited cell
    # 2 - wall

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}
    frodo_concerned = pygame.image.load('images\\frodo_concerned.jpg')
    frodo_happy = pygame.image.load('images\\frodo_happy.jpg')
    mordor = pygame.image.load('images\\mordor.jpg')


    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the maze
        self.window_size = 512  # The size of the PyGame window

        #! if youre changing this: observation space must be a dictionary
        # each location is a 2D vector
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )


        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]), 
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }


        self.maze = np.zeros((size, size), dtype=int)
            
        # add walls
        # horizontally
        for i in range(size):
            is_here = np.random.rand() < (size / 5) # expected number of walls in each direction is size/3
            if not is_here:
                continue
            length = np.random.randint(0, size * 5 / 12)
            start = np.random.randint(0, size - length)
            self.maze[start:start+length, i] = 2
        # vertically
        for i in range(size):
            is_here = np.random.rand() < (size / 5) # expected number of walls in each direction is size/3
            if not is_here:
                continue
            length = np.random.randint(0, size * 5 / 12)
            start = np.random.randint(0, size - length)
            self.maze[i, start:start+length] = 2

        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while self.maze[self._target_location[0], self._target_location[1]] == 2:
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self.maze[self._target_location[0], self._target_location[1]] = 1


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def set_training_status(self, is_training):
        self.is_training = is_training

    def _get_obs(self):
        return {"position": self._agent_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # We sample the agent's location randomly until it does not coincide with the target's location or a wall
        self._agent_location = self._target_location
        while np.array_equal(self._target_location, self._agent_location) or self.maze[self._agent_location[0], self._agent_location[1]] == 2:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        # info = self._get_info()

        if not self.is_training and self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):        
        direction = self._action_to_direction[action]
        # we check if there was an attempt to leave the grid or walk into a wall
        attempted_location = self._agent_location + direction
        if np.any(attempted_location < 0) or np.any(attempted_location >= self.size):
            reward = -0.8
        elif self.maze[attempted_location[0], attempted_location[1]] == 2:
            reward = -0.8
        else:
            reward = -0.04 # small negative reward for each step
            self._agent_location = attempted_location
            # mark visited cell
            self.maze[self._agent_location[0], self._agent_location[1]] = -1
        
        # negative reward for revisiting cells
        if self.maze[self._agent_location[0], self._agent_location[1]] == -1:
            reward += -0.25

        # An iteration is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        if terminated:
            reward = 1.0

        observation = self._get_obs()
        info = self._get_info()

        if not self.is_training and self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return self._render_frame()

    def _render_frame(self, Q=None):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        mordor = pygame.transform.scale(
            self.mordor, 
            (pix_square_size, pix_square_size)
        )
        canvas.blit(mordor, self._target_location * pix_square_size)
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )

        # Now we draw the agent
        frodo = pygame.transform.scale(
            (self.frodo_concerned if self.maze[self._agent_location[0], self._agent_location[1]] == -1 else self.frodo_happy),
            (pix_square_size, pix_square_size)
        )
        canvas.blit(frodo, self._agent_location * pix_square_size)
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (self._agent_location + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )

        # add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # add walls
        for i in range(self.size):
            for j in range(self.size):
                if self.maze[i, j] == 2:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.array([i, j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def print_solved(self, Q):
        #todo: not working!
        plt.clf()
        maze = np.zeros(self.maze.shape) # white cells
        maze[self.maze == 1] = 1 # black target
        plt.imshow(maze.T, cmap='binary')
        plt.grid(which='both', color='black', linestyle='-', linewidth=2)
        plt.xticks(np.arange(0.5, self.maze.shape[1], 1), [])
        plt.yticks(np.arange(0.5, self.maze.shape[0], 1), [])
        plt.tick_params(axis='both', which='both', length=0)
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] == 1:
                    continue
                best_direction = np.argmax(Q[i, j])
                if best_direction == 0:
                    plt.arrow(j, i, 0.25, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif best_direction == 1:
                    plt.arrow(j, i, 0, 0.25, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif best_direction == 2:
                    plt.arrow(j, i, -0.25, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif best_direction == 3:
                    plt.arrow(j, i, 0, -0.25, head_width=0.1, head_length=0.1, fc='k', ec='k')
        plt.show()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
