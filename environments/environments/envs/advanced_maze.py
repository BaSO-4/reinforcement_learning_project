import gym
from gym import spaces
import pygame
import numpy as np
import matplotlib.pyplot as plt

#! all coordinates are in the form (y, x)!!!! this is due to matrix indexing
class AdvancedMaze(gym.Env):
    # maze:
    # 0 - empty cell
    # 1 - target cell
    # -1 - visited cell
    # -2 - "seen" cell
    # 2 - wall

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, render_mode=None):
        self.size = 7  # The size of the maze
        self.window_size = 512  # The size of the PyGame window

        #! if youre changing this: observation space must be a dictionary
        self.observation_space = spaces.Dict(
            {
                # some binary values (donw, right, up, left)
                "next to a wall": spaces.MultiDiscrete([2, 2, 2, 2]),
                "next to a border": spaces.MultiDiscrete([2, 2, 2, 2]),
                "target in sight": spaces.MultiDiscrete([2, 2, 2, 2]),
                # info about past exploration
                "explored": spaces.Dict(
                    {
                        "down": spaces.Box(0, 1, shape=(1,), dtype=int),
                        "right": spaces.Box(0, 1, shape=(1,), dtype=int),
                        "up": spaces.Box(0, 1, shape=(1,), dtype=int),
                        "left": spaces.Box(0, 1, shape=(1,), dtype=int),
                    }
                )
            }
        )


        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]), # down
            1: np.array([0, 1]), # right
            2: np.array([-1, 0]), # up
            3: np.array([0, -1]), # left
        }


        # this is a handmade maze, to try things out. if it works, upgrade it to be random
        self.maze = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 1, 0, 0, 2, 0],
            [0, 2, 0, 0, 0, 2, 0],
            [0, 2, 0, 0, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])

        self._target_location = np.array([2, 2])

        self.locs_that_see_target_bellow = np.array([])
        self.locs_that_see_target_on_the_right = np.array([])
        self.locs_that_see_target_above = np.array([np.array([3, 2]), np.array([3, 3]), np.array([3, 4]), np.array([4, 2]), np.array([4, 3]), np.array([4, 4]), np.array([4, 5])])
        self.locs_that_see_target_on_the_left = np.array([np.array([2, 3]), np.array([2, 4]), np.array([3, 3]), np.array([3, 4]), np.array([4, 3]), np.array([4, 4]), np.array([4, 5])])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def set_training_status(self, is_training):
        self.is_training = is_training

    def _get_obs(self):
        obs = {}
        #  next to a border
        obs["next to a border"] = np.array([0, 0, 0, 0])
        if self._agent_location[0] == self.size - 1:
            obs["next to a border"][0] = 1
        if self._agent_location[1] == self.size - 1:
            obs["next to a border"][1] = 1
        if self._agent_location[0] == 0:
            obs["next to a border"][2] = 1
        if self._agent_location[1] == 0:
            obs["next to a border"][3] = 1
        # next to a wall
        obs["next to a wall"] = np.array([0, 0, 0, 0])
        if obs["next to a border"][0] != 1 and self.maze[self._agent_location[0] + 1, self._agent_location[1]] == 2:
            obs["next to a wall"][0] = 1
        if obs["next to a border"][1] != 1 and self.maze[self._agent_location[0], self._agent_location[1] + 1] == 2:
            obs["next to a wall"][1] = 1
        if obs["next to a border"][2] != 1 and self.maze[self._agent_location[0] - 1, self._agent_location[1]] == 2:
            obs["next to a wall"][2] = 1
        if obs["next to a border"][2] != 1 and self.maze[self._agent_location[0], self._agent_location[1] - 1] == 2:
            obs["next to a wall"][3] = 1
        # target in sight
        obs["target in sight"] = np.array([0, 0, 0, 0])
        if self._agent_location in self.locs_that_see_target_bellow:
            obs["target in sight"][0] = 1
        if self._agent_location in self.locs_that_see_target_on_the_right:
            obs["target in sight"][1] = 1
        if self._agent_location in self.locs_that_see_target_above:
            obs["target in sight"][2] = 1
        if self._agent_location in self.locs_that_see_target_on_the_left:
            obs["target in sight"][3] = 1
        # info about past exploration
        obs["explored"] = {
            "down": (np.sum(self.maze[self._agent_location[0]+1:, :] < 0)) / (self.maze[self._agent_location[0]+1:, :].size),
            "right": (np.sum(self.maze[:, self._agent_location[1]+1:] < 0)) / (self.maze[:, self._agent_location[1]+1:].size),
            "up": (np.sum(self.maze[:self._agent_location[0]-1, :] < 0)) / (self.maze[:self._agent_location[0]-1, :].size),
            "left": (np.sum(self.maze[:, :self._agent_location[1]-1] < 0)) / (self.maze[:, :self._agent_location[1]-1:].size)
        }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # just for testing
        self._agent_location = np.array([2, 6])

        observation = self._get_obs()

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
            reward = -0.01 # small negative reward for each step
            self._agent_location = attempted_location
            # mark visited cell
            self.maze[self._agent_location[0], self._agent_location[1]] = -2
            self.mark_seen_cells()
        
        # negative reward for revisiting cells
        if self.maze[self._agent_location[0], self._agent_location[1]] == -1:
            reward += -0.25

        # An iteration is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        if terminated:
            reward = 1.0

        observation = self._get_obs()

        if not self.is_training and self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, None, {}
    
    def mark_seen_cells(self):
        i = 0
        # go down
        while self._agent_location[0] + i < self.size:
            if self.maze[self._agent_location[0] + i, self._agent_location[1]] == 2:
                break
            self.maze[self._agent_location[0] + i, self._agent_location[1]] = -2
            i += 1
        # go right
        i = 0
        while self._agent_location[1] + i < self.size:
            if self.maze[self._agent_location[0], self._agent_location[1] + i] == 2:
                break
            self.maze[self._agent_location[0], self._agent_location[1] + i] = -2
            i += 1
        # go up
        i = 0
        while self._agent_location[0] - i >= 0:
            if self.maze[self._agent_location[0] - i, self._agent_location[1]] == 2:
                break
            self.maze[self._agent_location[0] - i, self._agent_location[1]] = -2
            i += 1
        # go down
        i = 0
        while self._agent_location[1] - i >= 0:
            if self.maze[self._agent_location[0], self._agent_location[1] - 1] == 2:
                break
            self.maze[self._agent_location[0], self._agent_location[1] - i] = -2
            i += 1


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
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

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


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
