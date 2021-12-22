import numpy as np
from tensorflow.keras import Sequential


class EpsilonGreedyPolicy:
    def __init__(self, actions: list, epsilon: float, min_epsilon: float, epsilon_decay: float, n_states: int) -> None:
        """
        The constructor of the epsilon greedy policy class.

        Args:
            actions (list): The actions that are available.
            epsilon (float): The current epsilon.
            min_epsilon (float): The lowest the epsilon can go.
            epsilon_decay (float): The factor the epsilon decays by.
            n_states (int): The lenght of the states.
        """
        self.actions = actions
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.n_states = n_states

    def select_action(self, state: np.ndarray, model: Sequential) -> int:
        """
        Selects an action based on a state, using a model as Q-table.

        Args:
            state (np.ndarray): The state to base the action of.
            model (Sequential): The model to use.

        Returns:
            int: The action to take.
        """
        if np.random.uniform(0.0, 1.0) < self.epsilon:  # exploration
            action = np.random.choice(self.actions)
        else:
            state = np.reshape(state, [1, self.n_states])
            q_values = model.predict(state)  # output Q(s,a) for all a of current state
            action = np.argmax(q_values[0])  # because the output is m * n, so we need to consider the dimension [0]
        return action

    def decay_epsilon(self) -> None:
        """
        Decays the epsilon
        """
        self.epsilon = max(self.min_epsilon, self.epsilon_decay * self.epsilon)

    def __str__(self):
        return f'Actions: {self.actions}, Epsilon: {self.epsilon}, Minimal epsilon: {self.min_epsilon}, Epsilon decay: {self.epsilon_decay}, Number of states: {self.n_states}'
