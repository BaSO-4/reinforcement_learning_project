import gym
from gym import spaces
import pygame
import numpy as np

#! all coordinates are in the form (y, x)!!!! this is due to matrix indexing
class ExampleMaze2(gym.Env):
    # maze:
    # 0 - empty cell
    # 1 - target cell
    # -1 - visited cell
    # 2 - wall

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    frodo_concerned = pygame.image.load('images\\frodo_concerned.jpg')
    frodo_happy = pygame.image.load('images\\frodo_happy.jpg')
    mordor = pygame.image.load('images\\mordor.jpg')
    wall = pygame.image.load('images\\wall.jpg')
    enemy = pygame.image.load('images\\nazgul.jpg')



    def __init__(self, render_mode=None, size=8):
        self.size = size  # The size of the maze
        self.window_size = 512  # The size of the PyGame window
        self.step_count = 0


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
            3: np.array([0, -1])
        }


        self.maze = np.array([
                [0, 2, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 2, 0, 2, 0, 0],
                [0, 0, 0, 2, 2, 0, 2, 0],
                [2, 2, 0, 2, 0, 0, 0, 0],
                [2, 0, 0, 2, 0, 2, 0, 0],
                [0, 0, 0, 2, 0, 2, 2, 2],
                [0, 2, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0, 0]
            ])
    
    
        #self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = np.array([3,7])
        self._enemy_location = np.array([6,0])
        #while self.maze[self._enemy_location[0], self._enemy_location[1]] == 2 or np.array_equal(self._target_location, self._enemy_location):
        #    self._enemy_location = self.np_random.integers(0, self.size, size=2, dtype=int)
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

        # We sample the agent's location randomly until it does not coincide with the target's location, enemys location or a wall
        self._agent_location = self._target_location
        self._enemy_location = np.array([6,0])
        while np.array_equal(self._target_location, self._agent_location) or np.array_equal(self._enemy_location, self._agent_location) or self.maze[self._agent_location[0], self._agent_location[1]] == 2:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        # info = self._get_info()

        if not self.is_training and self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        direction = self._action_to_direction[action]
        attempted_location = self._agent_location + direction
        
        if np.any(attempted_location < 0) or np.any(attempted_location >= self.size): #out of maze
            reward = -0.8
        elif self.maze[attempted_location[0], attempted_location[1]] == 2: # wall
            reward = -0.8
        else:
            reward = -0.3 #step
            self._agent_location = attempted_location
            self.maze[self._agent_location[0], self._agent_location[1]] = -1

            if np.array_equal(self._agent_location, self._enemy_location):
                reward = -5.0

        if np.array_equal(self._agent_location, self._target_location):#reaches target
            reward = 10.0
            terminated = True
        else:
            terminated = False

        observation = self._get_obs()
        info = self._get_info()

        if not self.is_training and self.render_mode == "human":
            self._render_frame()

        def move_enemy_towards_agent(enemy_location, agent_location):
            direction = agent_location - enemy_location
            possible_directions = [np.array([np.sign(direction[0]), 0]), np.array([0, np.sign(direction[1])])]
            
            # Check if the primary direction is valid
            for direction in possible_directions:
                attempted_enemy_location = enemy_location + direction
                if (np.all(attempted_enemy_location >= 0) and 
                    np.all(attempted_enemy_location < self.size) and 
                    self.maze[attempted_enemy_location[0], attempted_enemy_location[1]] != 2):
                    return attempted_enemy_location

            # If the primary direction is not valid, choose a random valid direction
            possible_directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
            np.random.shuffle(possible_directions)
            for direction in possible_directions:
                attempted_enemy_location = enemy_location + direction
                if (np.all(attempted_enemy_location >= 0) and 
                    np.all(attempted_enemy_location < self.size) and 
                    self.maze[attempted_enemy_location[0], attempted_enemy_location[1]] != 2):
                    return attempted_enemy_location

            return enemy_location

        self._enemy_location = move_enemy_towards_agent(self._enemy_location, self._agent_location)
        '''
        # Penalty for being adjacent to the enemy
        adjacent_positions = [
            self._agent_location + np.array([1, 0]),
            self._agent_location + np.array([-1, 0]),
            self._agent_location + np.array([0, 1]),
            self._agent_location + np.array([0, -1])
        ]

        for pos in adjacent_positions:
            if np.array_equal(pos, self._enemy_location):
                reward -= -1.0
        '''
        # Increment step count
        self.step_count += 1

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

        # Now we draw the enemy
        enemy = pygame.transform.scale(
            self.enemy, 
            (pix_square_size, pix_square_size)
        )

        canvas.blit(enemy, self._enemy_location * pix_square_size)


        # Now we draw the agent
        frodo = pygame.transform.scale(
            (self.frodo_concerned if self.maze[self._agent_location[0], self._agent_location[1]] == -1 else self.frodo_happy),
            (pix_square_size, pix_square_size)
        )
        canvas.blit(frodo, self._agent_location * pix_square_size)
        '''
        # add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )
        '''
        ### add walls
        wall = pygame.transform.scale(
            self.wall, 
            (pix_square_size, pix_square_size))
        
        for i in range(self.size):
            for j in range(self.size):
                if self.maze[j, i] == 2:
                    canvas.blit(wall, (j * pix_square_size, i * pix_square_size))


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
