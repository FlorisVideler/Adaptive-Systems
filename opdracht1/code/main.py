from agent import Agent
from maze import Maze
from policy import Policy


maze = Maze(
    lenght=4,
    height=4,
    all_rewards=-1,
    special_rewards={
        (3, 0): 40,
        (2, 1): -10,
        (3, 1): -10,
        (0, 3): 10,
        (1, 3): -1},
    end_positions=[(3, 0), (0, 3)]
)
policy = Policy()
agent = Agent(maze, policy, (2, 3), 1)

agent.value_iteration()
agent.simulate()
agent.visualize()
