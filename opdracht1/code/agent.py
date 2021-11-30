import copy

import numpy as np

from maze import Maze
from policy import Policy
from state import State
from util import get_positions_around, get_possible_states, max_bellman


class Agent:
    """
    Class to represent an agent in the simulation.
    """
    def __init__(self, maze: Maze, policy: Policy,
                 start_position: tuple = (2, 3), discount: int = 1) -> None:
        """
        Constructor for the Agent class.

        Args:
            maze (Maze): The Maze to navigate in.
            policy (Policy): The policy to use.
            start_position (tuple, optional): The start position of the Agent.
            Defaults to (2, 3).
            discount (int, optional): The discount that is used for
            calculations. Defaults to 1.
        """
        self.maze: Maze = maze
        self.policy: Policy = policy
        self.discount = discount
        self.threshold = 0.1
        start_x, start_y = start_position
        self.state: State = self.maze.maze[start_y][start_x]

    def pick_action(self, state: State) -> int:
        """
        Picks an action based on the state.

        Args:
            state (State): The state to base the action on.

        Returns:
            int: The action as an int.
        """
        return self.policy.select_action(self.maze.maze,
                                         state, self.discount)[1]

    def first_visit_mc_prediction(self):
        values = copy.deepcopy(self.maze.maze)
        values_flat = list(np.array(values).flatten())
        empty_lists = [[] for _ in range(len(values_flat))]
        returns = dict(zip(values_flat, empty_lists))
        print(returns)

    def value_iteration(self) -> None:
        """
        Does the value iteration algorithm.
        """
        c = 0
        print(f'\nSweep {c}: ')
        visual_maze = np.zeros_like(np.array(self.maze.maze))
        print(visual_maze)
        delta = self.threshold+1
        while delta > self.threshold:
            delta = 0
            new_maze = copy.deepcopy(self.maze.maze)
            for y, y_row in enumerate(self.maze.maze):
                for x, x_row in enumerate(y_row):
                    old_value = self.maze.maze[y][x].value
                    if (x, y) not in self.maze.end_positions:
                        positions_around = get_positions_around((x, y))
                        possible_states = get_possible_states(
                            self.maze.maze, positions_around)
                        new_value = max_bellman(self.discount,
                                                possible_states)[0]
                        new_maze[y][x] = State(
                            (x, y), self.maze.maze[y][x].reward,
                            new_value, self.maze.maze[y][x].done)
                        visual_maze[y][x] = new_value
                        delta = max(delta, abs(old_value-new_value))

            self.maze.maze = new_maze
            if delta > self.threshold:
                c += 1
                print(f'Sweep {c}: ')
                print(visual_maze)

        print(f'Done after {c} sweeps!\n')

    def simulate(self) -> None:
        """
        Simulates walking through the maze.
        """
        print(f'\nSimulating agent starting on {self.state.location}')
        while not self.state.done:
            action = self.pick_action(self.state)
            next_state = self.maze.do_step(self.state, action)
            print(
                f'Moving from {self.state.location} to '
                f'{next_state.location} {self.maze.actions[action]}')

            self.state = next_state
        print(f'Finished simulation om {self.state.location}\n')

    def visualize(self) -> None:
        """
        Visualizes the values and the policy.
        """
        output_str = f'\n{"Values:":16}{"Policy:"}\n'
        for y, y_row in enumerate(self.maze.maze):
            output_row_values = ''
            output_row_policy = ''
            for x, x_row in enumerate(y_row):
                state = self.maze.maze[y][x]
                value = state.value
                if state.done:
                    policy = 9
                else:
                    policy = self.pick_action(state)
                output_row_values += f'{value:3}'
                output_row_policy += f'{self.maze.actions[policy]:3}'
            output_str += f'{output_row_values:14}  {output_row_policy} \n'
        print(output_str)
