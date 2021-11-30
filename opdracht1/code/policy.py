from state import State
from util import get_positions_around, get_possible_states, max_bellman


# TODO: Make policy save a value function that picks randomly.
class Policy:
    """
    A class to represent a policy. Could be a function,
    but kept like this for later use.
    """
    def __init__(self) -> None:
        # up ,right, down, left
        self.legal_actions = [0, 1, 2, 3]

    def select_action(self, maze: list,
                      state: State, discount: float) -> tuple:
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
        surrounding_positions = get_positions_around(state.location)
        possible_states = get_possible_states(maze, surrounding_positions)
        best_step = max_bellman(discount, possible_states)
        return best_step
