import time
start_time = time.time()


import gym
import matplotlib.pyplot as plt
import numpy as np

from classes.transition import Transition
from classes.agent import Agent


env = gym.make('LunarLander-v2')
env.reset()

tau = 1
epsilon = 1
min_epsilon = 0.08
epsilon_decay = 0.996

amount_of_episodes = 5_000
max_steps = 1_000

memory_size = 100_000

agent = Agent(n_actions=env.action_space.n, n_states=env.observation_space.shape[0], memory_size=memory_size, gamma=0.9, alpha=0.001, epsilon=epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay)

# agent.policy_network.load_model('/home/floris/Desktop/School/AS/opdracht/double-deep-q-learning-networks-in-the-gym/models/policy-model-20211216-211218')

rewards = []
average_rewards = []

for i_episode in range(amount_of_episodes):
    state = env.reset()
    print(f'Episode {i_episode}')
    total_reward = 0
    done = False

    while not done:
        action = agent.policy.select_action(state, agent.policy_network.model)
        next_state, reward, done, info = env.step(action)

        # env.render()

        total_reward += reward
        # transition = Transition(state, action, reward, next_state, done)
        agent.memory.append_memory((state, action, reward, next_state, done))

        state = next_state
        agent.learn(96)

    agent.copy_model(tau)

    agent.policy.decay_epsilon()

    rewards.append(total_reward)
    avg_reward = np.mean(rewards)
    average_rewards.append(avg_reward)
    print(f'Current reward: {total_reward}, avg reward: {avg_reward}')
    if np.mean(rewards[-100:]) >= 200:
        break
env.close()

# agent.save_models()

print("--- %s seconds ---" % (time.time() - start_time))
# plt.plot([i for i in range(amount_of_episodes)], rewards)
plt.plot(average_rewards)
plt.plot(rewards)
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend(['Reward', 'Average reward'], loc='upper right')
plt.show()
