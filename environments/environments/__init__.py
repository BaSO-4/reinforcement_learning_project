from gym.envs.registration import register

register(
    id="environments/Maze-v0",
    entry_point="environments.envs:Maze",
)
register(
    id="environments/AdvancedMaze-v0",
    entry_point="environments.envs:AdvancedMaze",
)
