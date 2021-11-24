from maze import Maze
from agent import Agent
from policy import Policy

maze = Maze(4, 4, -1, {(3, 0): 40, (2, 1): -10, (3, 1): -10, (0, 3): 10, (1, 3): -1}, [(3, 0), (0, 3)])
policy = Policy()
agent = Agent(maze, policy, (2, 3), 1)

agent.value_iteration()
agent.simulate()
agent.visualize()
