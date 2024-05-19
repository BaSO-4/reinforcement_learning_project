todo ideas:
allow the maze to be a rectangle, not only square -> tried, is nontrivial due to pygame's different indexing to np
find best parameters and updating of parameters (alpha, epsilon, gamma, number of walls)
fix the visualizaion of learned Q matrix, to be displayed when learning completes. now its not working
add visualizations, statistics, analysis to be included in the report
 - for example: after training, put the agent on random locations and let it play. for each run, compare the distance from start to target and steps it took to find target
set the treshold of reward - when are we convinced that we know what to do in certain state -> during training, visualize which cells have been "learned" - we will see how the learned region expands
- position walls smarter, so that there are no cells with no escape, or be careful that we dont place an agent in such cells or add portals

optional:
- create new agents
- create an agent (and envirnment) that can learn to solve ANY maze (difficult) - distance to walls in each direction, does it see a target, is there a visible dead-end in sight in each direction ...
- instead of blue circle and red square, use frodo and mordor or something funny
- add portals, monsters, tasks ... add multiple targets?