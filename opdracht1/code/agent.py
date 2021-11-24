from policy import Policy
from maze import Maze
from state import State
import copy
from util import get_positions_around, max_bellman, get_possible_states


import numpy as np

class Agent:
    def __init__(self, maze: Maze, policy: Policy, start_position: tuple = (2, 3), discount: int = 1) -> None:
        self.maze = maze
        self.policy = policy
        self.discount = discount
        self.threshold = 0.1
        start_x, start_y = start_position
        self.state: State = self.maze.maze[start_y][start_x]       

    def pick_action(self, state: State):
        return self.policy.select_action(self.maze.maze, state, self.discount)[1]

    def value_iteration(self):
        c = 0
        print(f'\nSweep {c}: ')
        print(np.matrix(self.maze.maze))
        delta = self.threshold+1
        while delta > self.threshold:
            delta = 0
            new_maze = copy.deepcopy(self.maze.maze)
            for y, y_row in enumerate(self.maze.maze):
                for x, x_row in enumerate(y_row):
                    old_value = self.maze.maze[y][x].value
                    if (x, y) not in self.maze.end_positions:
                        positions_around = get_positions_around((x, y))
                        possible_states = get_possible_states(self.maze.maze, positions_around)
                        new_value = max_bellman(self.discount, possible_states)[0]
                        new_maze[y][x] = State((x, y), self.maze.maze[y][x].reward, new_value, self.maze.maze[y][x].done)
                        delta = max(delta, abs(old_value-new_value))
                    
            self.maze.maze = new_maze
            if delta > self.threshold:
                c+=1
                print(f'Sweep {c}: ')
                print(np.matrix(self.maze.maze))
        
        print(f'Done after {c} sweeps!\n')

    def simulate(self):
        print(f'\nSimulating agent starting on {self.state.location}')
        while not self.state.done:
            # print(self.state.location)
            action = self.pick_action(self.state)
            # print(f'action: {action}')
            next_state = self.maze.do_step(self.state, action)
            print(f'Moving from {self.state.location} to {next_state.location} {self.maze.actions[action]}')

            self.state = next_state
        print(f'Finished simulation om {self.state.location}\n')

    def visualize(self):
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