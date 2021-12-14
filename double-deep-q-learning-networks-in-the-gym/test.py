import gym
env = gym.make('LunarLander-v2')
env.reset()
for i_episode in range(1):
    observation = env.reset()
    for t in range(10):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        
        print(env.action_space)
        observation, reward, done, info = env.step(action)
        # print(reward, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()