import random
from state import State
from util import get_all_max


# TODO: Make policy save a value function that picks randomly.
class Policy:
    """
    A class to represent a policy. Could be a function,
    but kept like this for later use.
    """

    def __init__(self, lenght: int, height: int, greedy=True) -> None:
        # up ,right, down, left
        self.greedy = greedy
        self.legal_actions = [0, 1, 2, 3]
        self.lenght = lenght
        self.height = height
        self.policy_matrix = self.generate_random_matrix(lenght, height)

    def generate_random_matrix(self, lenght: int, height: int) -> list:
        """
        Generates a random policy matrix.

        Args:
            lenght (int): The lenght of the matrix.
            height (int): The height of the matrix

        Returns:
            list: The matrix.
        """
        policy_matrix = []
        for y in range(height):
            y_row = []
            for x in range(lenght):
                actions_row = []
                for legal_action in self.legal_actions:
                    actions_row.append(1 / len(self.legal_actions))
                y_row.append(actions_row)
            policy_matrix.append(y_row)
        return policy_matrix

    def select_action(self, state: State) -> int:
        """
        Selects an action based on a State.

        Args:
            state (State): The state to base the action on.

        Returns:
            int: The best action.
        """
        x, y = state.location
        action_chances = self.policy_matrix[y][x]
        if self.greedy:
            return random.choice(get_all_max(action_chances))
        else:
            return random.choices(self.legal_actions, action_chances)[0]

    def update_policy(self, state: State, q_function: list, epsilon: float) -> None:
        """
        Updates the policy.

        Args:
            state (State): The current state.
            q_function (list): The current qfunction.
            epsilon (float): The epsilon to use.
        """
        x, y = state.location
        max_value = max(q_function[y][x])
        best_action = q_function[y][x].index(max_value)
        all_maxes = get_all_max(q_function[y][x])
        if len(all_maxes) > 1:
            best_action = random.choice(all_maxes)
        for action, value in enumerate(q_function[y][x]):
            if action == best_action:
                chance = 1 - epsilon + epsilon / len(q_function[y][x])
            else:
                chance = epsilon / len(q_function[y][x])
            self.policy_matrix[y][x][action] = chance

    def reset_policy(self) -> None:
        """
        Resets the policy.
        """
        self.policy_matrix = self.generate_random_matrix(
            self.lenght, self.height)
