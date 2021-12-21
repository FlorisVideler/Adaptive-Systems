import numpy as np



class EpsilonGreedyPolicy:
    """Epsilon greddy policy class"""
    def __init__(self, actions, epsilon, min_epsilon, epsilon_decay, n_states):
        self.actions = actions
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon  # The minimal epsilon
        self.epsilon_decay = epsilon_decay
        self.n_states = n_states

    def select_action(self, state, model):
        """Selects action based on the current state"""
        if np.random.uniform(0.0, 1.0) < self.epsilon:  # exploration
            action = np.random.choice(self.actions)
        else:
            state = np.reshape(state, [1, self.n_states])
            q_values = model.predict(state)  # output Q(s,a) for all a of current state
            action = np.argmax(q_values[0])  # because the output is m * n, so we need to consider the dimension [0]
        return action

    def decay_epsilon(self):
        """This function will decay the epsilon over time"""
        self.epsilon = max(self.min_epsilon, self.epsilon_decay * self.epsilon)

    def __str__(self):
        return f'Actions: {self.actions}, Epsilon: {self.epsilon}, Minimal epsilon: {self.min_epsilon}, Epsilon decay: {self.epsilon_decay}, Number of states: {self.n_states}'
