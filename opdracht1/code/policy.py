import random
from state import State
from util import (get_positions_around, get_possible_states,
                  max_bellman, all_max_bellman, get_all_max)


# TODO: Make policy save a value function that picks randomly.
class Policy:
    """
    A class to represent a policy. Could be a function,
    but kept like this for later use.
    """

    def __init__(self, lenght, height, greedy = True) -> None:
        # up ,right, down, left
        self.greedy = greedy
        self.legal_actions = [0, 1, 2, 3]
        self.policy_matrix = self.generate_random_matrix(lenght, height)

    def generate_random_matrix(self, lenght, height):
        # policy_matrix = []
        # for y in range(height):
        #     y_row = []
        #     for x in range(lenght):
        #         y_row.append(self.legal_actions)
        #     policy_matrix.append(y_row)
        # return policy_matrix
        policy_matrix = []
        for y in range(height):
            y_row = []
            for x in range(lenght):
                actions_row = []
                for legal_action in self.legal_actions:
                    # Start all actions with the same chance
                    actions_row.append(1 / len(self.legal_actions))
                y_row.append(actions_row)
            policy_matrix.append(y_row)
        return policy_matrix

    # def update_to_deterministic_policy_matrix(self, discount, maze, value_function):
    #     new_policy = []
    #     for y, y_row in enumerate(value_function):
    #         new_policy_y_row = []
    #         for x, x_row in enumerate(y_row): 
    #             positions_around = get_positions_around((x, y))
    #             possible_states = get_possible_states(maze, positions_around)
    #             all_max_actions = all_max_bellman(discount, possible_states, value_function)
    #             new_policy_y_row.append(all_max_actions)
    #         new_policy.append(new_policy_y_row)
    #     self.policy_matrix = new_policy 


    # def select_action(self, maze: list,
    #                       state: State, discount: float) -> tuple:
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
        return random.choice(self.policy_matrix[y][x])
        surrounding_positions = get_positions_around(state.location)
        possible_states = get_possible_states(maze, surrounding_positions)
        best_step = max_bellman(discount, possible_states)
        return best_step

    def select_actions(self, maze: list,
                       state: State, discount: float) -> int:
        surrounding_positions = get_positions_around(state.location)
        possible_states = get_possible_states(maze, surrounding_positions)
        best_steps = all_max_bellman(discount, possible_states)
        return best_steps


    def reset_policy(self):
        #TODO: self.lenght en height
        self.policy_matrix = self.generate_random_matrix(4, 4)