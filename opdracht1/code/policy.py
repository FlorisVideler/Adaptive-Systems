from state import State
from util import get_positions_around, max_bellman, get_possible_states
import random

class Policy:
    def select_action(self, maze, state: State, discount):
        current_x, current_y = state.location
        # value_current_position = values[current_y][current_x]
        surrounding_positions = get_positions_around(state.location)
        possible_states = get_possible_states(maze, surrounding_positions)
        best_step = max_bellman(discount, possible_states)
        return best_step
