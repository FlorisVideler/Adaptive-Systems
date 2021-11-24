from state import State
from maze import Maze
from util import get_positions_around, max_bellman, get_possible_states

class Policy:
    def select_action(self, maze: Maze, state: State, discount: float) -> tuple:
        surrounding_positions = get_positions_around(state.location)
        possible_states = get_possible_states(maze, surrounding_positions)
        best_step = max_bellman(discount, possible_states)
        return best_step
