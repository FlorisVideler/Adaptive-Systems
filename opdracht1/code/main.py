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
random_policy = Policy(lenght=4, height=4)
optimal_policy = Policy(lenght=4, height=4)

agent = Agent(maze, optimal_policy, (2, 3), 1)
agent.value_iteration()
agent.update_policy()


# agent.generate_episode()
# agent.visualize()

# agent.value_iteration()
# agent.update_policy()
# agent.simulate()
# agent.visualize()


# print('Monte Carlo first visit; policy = random; discount = 1; 10000 episodes')
# agent.policy = random_policy
# agent.first_visit_mc_prediction()
# print('\n')
# print('Monte Carlo first visit; policy = random; discount = 0.9; 10000 episodes')
# agent.discount = 0.9
# agent.first_visit_mc_prediction()
# print('\n')

# print('Monte Carlo first visit = optimal; discount = 1; 10000 episodes')
# agent.policy = optimal_policy
# agent.discount = 1
# agent.first_visit_mc_prediction()
# print('\n')
# print('Monte Carlo first visit; policy = optimal; discount = 0.9; 10000 episodes')
# agent.discount = 0.9
# agent.first_visit_mc_prediction()
# print('\n')




# print('Tubular TD; policy = random; discount = 1; 10000 episodes')
# agent.policy = random_policy
# agent.tabular_td()
# print('\n')
# print('Tubular TD; policy = random; discount = 0.9; 10000 episodes')
# agent.discount = 0.9
# agent.tabular_td()
# print('\n')

# print('Tubular TD; policy = optimal; discount = 1; 10000 episodes')
# agent.policy = optimal_policy
# agent.discount = 1
# agent.tabular_td()
# print('\n')
# print('Tubular TD; policy = optimal; discount = 0.9; 10000 episodes')
# agent.discount = 0.9
# agent.tabular_td()
# print('\n')

agent.on_policy_first_vist_mc()