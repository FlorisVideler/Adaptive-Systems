from state import State
from util import get_positions_around


class Maze:
    def __init__(self, lenght: int, height: int, all_rewards: int,
                 special_rewards: dict, end_positions: list) -> None:
        self.lenght = lenght
        self.height = height
        self.all_rewards = all_rewards
        self.special_rewards = special_rewards
        self.end_positions = end_positions

        self.actions = {
            0: '↑',
            1: '→',
            2: '↓',
            3: '←',
            9: '⦾'
        }

        self.maze = self.generate_maze()

    def generate_maze(self) -> list:
        maze = []
        for y in range(self.height):
            y_row = []
            for x in range(self.height):
                location = x, y
                reward = self.all_rewards
                if location in self.special_rewards:
                    reward = self.special_rewards[location]
                end_position = location in self.end_positions
                state = State((x, y), reward, 0, end_position)
                y_row.append(state)
            maze.append(y_row)
        return maze

    def do_step(self, state: State, action: int) -> State:
        surrounding_positions = get_positions_around(state.location)
        next_x, next_y = surrounding_positions[action]
        next_state = self.maze[next_y][next_x]
        return next_state
