import gym
import matplotlib.pyplot as plt

from classes.transition import Transition
from classes.agent import Agent


env = gym.make('LunarLander-v2')
env.reset()

tau = 0.8
epsilon = 1
min_epsilon = 0.05
epsilon_decay = 0.95

amount_of_episodes = 200
max_steps = 1_000

memory_size = 10_000

agent = Agent(n_actions=env.action_space.n, memory_size=memory_size, gamma=0.9, alpha=0.001, epsilon=epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay)

rewards = []

for i_episode in range(amount_of_episodes):
    state = env.reset()
    print(f'Episode {i_episode}')
    total_reward = 0
    for t in range(max_steps):
        env.render()
        action = agent.policy.select_action(state, agent.policy_network.model)
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
