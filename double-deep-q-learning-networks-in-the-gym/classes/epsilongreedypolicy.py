import numpy as np



class EpsilonGreedyPolicy:
    def __init__(self, actions, epsilon):
        self.actions = actions
        self.epsilon = epsilon

    def select_action(self, state, model):
        if np.random.uniform(0.0, 1.0) < self.epsilon: # exploration
            action = np.random.choice(self.actions)
        else:
            q_values = model.predict(np.array([state])) # output Q(s,a) for all a of current state
            # print(q_values[0])
            action = np.argmax(q_values[0]) # because the output is m * n, so we need to consider the dimension [0]
            # print(action)
        return action

    def __str__(self):
        return "...."