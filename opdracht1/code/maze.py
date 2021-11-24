from state import State
from util import get_positions_around


class Maze:
    """
    Represent a maze.
    """
    def __init__(self, lenght: int, height: int, all_rewards: int,
                 special_rewards: dict, end_positions: list) -> None:
        """
        Constructer for the Maze class.

        Args:
            lenght (int): The lenght of the Maze.
            height (int): The height of the Maze.
            all_rewards (int): The standard reward of the states.
            special_rewards (dict): A dictonairy with locations
            and what there rewards is.
            end_positions (list): A list of tuples with end locations.
        """
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
        """
        Generates the maze of the Maze. We use states with to represent this.

        Returns:
            list: A 2D list of the grid.
        """
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
        """
        Given a state and an action, there will be given the next State.

        Args:
            state (State): The State that the next state will be based of.
            action (int): The action that needs to take place.

        Returns:
            State: The next state.
        """
        surrounding_positions = get_positions_around(state.location)
        next_x, next_y = surrounding_positions[action]
        next_state = self.maze[next_y][next_x]
        return next_state
