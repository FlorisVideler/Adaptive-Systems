import imp
import re
import gym

from classes.transition import Transition
from classes.agent import Agent
from classes.epsilongreedypolicy import EpsilonGreedyPolicy

env = gym.make('LunarLander-v2')
env.reset()

amount_of_episodes = 10
max_steps = 1_000

memory_size = 100_000

policy = EpsilonGreedyPolicy(env.action_space.n, 0.5)

agent = Agent(memory_size, 0, 0, 0)

agent.copy_model()

# for i_episode in range(amount_of_episodes):
#     state = env.reset()
#     for t in range(max_steps):
#         env.render()
#         action = policy.select_action(state, agent.policy_network.model)
#         next_state, reward, done, info = env.step(action)
#         transition = Transition(state, action, reward, next_state, done)
#         agent.memory.append_memory(transition)
#         state = next_state
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
#     agent.learn(64)    
env.close()

agent.policy_network.save()