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
        (1, 3): -2},
    end_positions=[(3, 0), (0, 3)]
)
policy = Policy(lenght=4, height=4)
agent = Agent(maze, policy, (2, 3), 1)


# agent.generate_episode()
# agent.visualize()

# agent.value_iteration()
# agent.update_policy()
# agent.simulate()
# agent.visualize()
# Random policy
agent.first_visit_mc_prediction()

agent.discount = 0.9
agent.first_visit_mc_prediction()

# Optimal policy
agent.value_iteration()
agent.update_policy()

agent.discount = 1
agent.first_visit_mc_prediction()

agent.discount = 0.9
agent.first_visit_mc_prediction()
# agent.visualize()