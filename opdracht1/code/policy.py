import random
from state import State
from util import (get_positions_around, get_possible_states, all_max_bellman, get_all_max)


# TODO: Make policy save a value function that picks randomly.
class Policy:
    """
    A class to represent a policy. Could be a function,
    but kept like this for later use.
    """

    def __init__(self, lenght, height, greedy=True) -> None:
        # up ,right, down, left
        self.greedy = greedy
        self.legal_actions = [0, 1, 2, 3]
        self.lenght = lenght
        self.height = height
        self.policy_matrix = self.generate_random_matrix(lenght, height)

    def generate_random_matrix(self, lenght, height):
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

    def select_action(self, state):
        """
        Selects an action based on a State.

        Args:
            maze (List): The maze to use.
            state (State): The state to base the action on.
            discount (float): the discount value to use in calculations.

        Returns:
            tuple: The best step (value, index).
            The index corresponds with the action.
        """
        x, y = state.location
        action_chances = self.policy_matrix[y][x]
        if self.greedy:
            return random.choice(get_all_max(action_chances))
        else:
            return random.choices(self.legal_actions, action_chances)[0]

    def select_actions(self, maze: list, state: State, discount: float) -> int:
        surrounding_positions = get_positions_around(state.location)
        possible_states = get_possible_states(maze, surrounding_positions)
        best_steps = all_max_bellman(discount, possible_states)
        return best_steps

    def update_policy(self, state, q_function, epsilon):
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

    def reset_policy(self):
        self.policy_matrix = self.generate_random_matrix(
            self.lenght, self.height)
