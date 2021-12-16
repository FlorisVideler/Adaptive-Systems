import gym
import matplotlib.pyplot as plt

from classes.transition import Transition
from classes.agent import Agent
from classes.epsilongreedypolicy import EpsilonGreedyPolicy

env = gym.make('LunarLander-v2')
env.reset()

tau = 0.8

amount_of_episodes = 20
max_steps = 1_000

memory_size = 100_000

policy = EpsilonGreedyPolicy(env.action_space.n, 0.2)

agent = Agent(memory_size, 0, 0, 0)

rewards = []

for i_episode in range(amount_of_episodes):
    state = env.reset()
    print(f'Episode {i_episode}')
    total_reward = 0
    for t in range(max_steps):
        env.render()
        action = policy.select_action(state, agent.policy_network.model)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        transition = Transition(state, action, reward, next_state, done)
        agent.memory.append_memory(transition)
        state = next_state
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    agent.learn(64)
    if i_episode % 10 == 0 and i_episode >  0:
        print(f'Copying policy to target {tau}')
        agent.copy_model(tau)
    rewards.append(total_reward)
env.close()

agent.policy_network.save()


plt.plot([i for i in range(amount_of_episodes)], rewards)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()
