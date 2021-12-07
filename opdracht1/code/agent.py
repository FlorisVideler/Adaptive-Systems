import copy
import random

import numpy as np

from maze import Maze
from policy import Policy
from state import State
from util import get_positions_around, get_possible_states, max_bellman, all_max_bellman


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

        self.value_function = self.generate_empty_value_function(
            maze.lenght, maze.height)

    def generate_empty_value_function(self, lenght, height):
        value_function = []
        for y in range(height):
            y_row = []
            for x in range(lenght):
                y_row.append(0)
            value_function.append(y_row)
        return value_function

    def generate_empty_q_function(self, lenght, height):
        q_function = []
        for y in range(height):
            y_row = []
            for x in range(lenght):
                actions_row = []
                for legal_action in self.policy.legal_actions:
                    actions_row.append(0)
                y_row.append(actions_row)
            q_function.append(y_row)
        return q_function

    def pick_action(self, state: State) -> int:
        """
        Picks an action based on the state.

        Args:
            state (State): The state to base the action on.

        Returns:
            int: The action as an int.
        """
        return self.policy.select_action(state)

    def generate_episode(self):
        start_x, start_y = random.randrange(4), random.randrange(4)
        start_state = self.maze.maze[start_y][start_x]
        states_episode = [start_state]
        action_episode = []
        reward_episode = [None]
        current_state: State = start_state
        while not current_state.done:
            current_action = self.pick_action(current_state)
            action_episode.append(current_action)
            next_state = self.maze.do_step(current_state, current_action)
            current_reward = next_state.reward
            reward_episode.append(current_reward)
            if not next_state.done:
                states_episode.append(next_state)
            current_state = next_state
        return states_episode, action_episode, reward_episode

    def first_visit_mc_prediction(self, max_episodes=10_000, threshold=0.01, converged_threshold=10):
        local_value_function = self.generate_empty_value_function(
            4, 4)
        maze_positions_flat = [s.location for s in list(
            np.array(self.maze.maze).flatten())]
        empty_lists = [[] for _ in range(len(maze_positions_flat))]
        returns = dict(zip(maze_positions_flat, empty_lists))
        converged_count = 0
        for i in range(max_episodes):
            states_episode, action_episode, reward_episode = self.generate_episode()
            delta = 0
            if not states_episode[0].done:
                g = 0
                for t in range(len(states_episode)-1, -1, -1):
                    g = self.discount*g+reward_episode[t+1]
                    current_state = states_episode[t]
                    if current_state not in states_episode[:t]:
                        x, y = current_state.location
                        returns[current_state.location].append(g)
                        new_value = sum(
                            returns[current_state.location]) / len(returns[current_state.location])
                        old_value = local_value_function[y][x]
                        delta = max(delta, abs(old_value - new_value))
                        local_value_function[y][x] = new_value

            converged = False

            if delta < threshold:
                converged_count += 1
                if converged_count == converged_threshold:
                    converged = True
            else:
                converged_count = 0

            if converged:
                print(f'Stopped after {i-converged_threshold} episodes.')
                print(np.array(local_value_function))
                break
        if not converged:
            print(f'Did not converge within {max_episodes} episodes.')
        print(np.array(local_value_function))

    def tabular_td(self, max_episodes=10_000, threshold=0.01, converged_threshold=10, alpha=0.1):
        local_value_function = self.generate_empty_value_function(4, 4)
        converged_count = 0
        for i in range(max_episodes):
            delta = 0
            start_x, start_y = random.randrange(4), random.randrange(4)
            current_state = self.maze.maze[start_y][start_x]
            while not current_state.done:
                current_action = self.pick_action(current_state)
                next_state = self.maze.do_step(current_state, current_action)
                reward = next_state.reward
                x_current_state, y_current_state = current_state.location
                x_next_state, y_next_state = next_state.location
                old_value = local_value_function[y_current_state][x_current_state]
                new_value = local_value_function[y_current_state][x_current_state] + alpha * (
                    reward + self.discount * local_value_function[y_next_state][x_next_state] - local_value_function[y_current_state][x_current_state])
                delta = max(delta, abs(old_value - new_value))
                local_value_function[y_current_state][x_current_state] = new_value
                current_state = next_state

            converged = False

            if delta < threshold:
                converged_count += 1
                if converged_count == converged_threshold:
                    converged = True
            else:
                converged_count = 0

            if converged:
                print(f'Stopped after {i-converged_threshold} episodes.')
                break

        if not converged:
            print(f'Delta did not become small enough in {max_episodes} episodes')

        print(np.array(local_value_function))

    def on_policy_first_vist_mc(self, max_episodes=10_000, threshold=0.01, converged_threshold=10, epsilon=0.1):
        maze_positions_flat = [s.location for s in list(
            np.array(self.maze.maze).flatten())]
        returns_list = []
        for position in maze_positions_flat:
            for legal_action in self.policy.legal_actions:
                returns_list.append((position, legal_action))
        empty_lists = [[] for _ in range(len(returns_list))]
        returns = dict(zip(returns_list, empty_lists))
        converged_count = 0
        q_function = self.generate_empty_q_function(4, 4)

        all_routes = []

        for i in range(max_episodes):
            states_episode, action_episode, reward_episode = self.generate_episode()
            all_routes.append(len(states_episode))
            delta = 0
            if not states_episode[0].done:
                g = 0
                for t in range(len(states_episode)-1, -1, -1):
                    g = self.discount*g+reward_episode[t+1]
                    current_state = states_episode[t]
                    current_action = action_episode[t]
                    if current_state not in states_episode[:t] and current_action not in action_episode[:t]:
                        returns[current_state.location,
                                current_action].append(g)
                        x_current_state, y_current_state = current_state.location
                        old_value = q_function[y_current_state][x_current_state][current_action]
                        new_value = sum(returns[current_state.location, current_action]) / len(
                            returns[current_state.location, current_action])
                        delta = max(delta, abs(old_value - new_value))
                        q_function[y_current_state][x_current_state][current_action] = new_value

                        self.policy.update_policy(
                            current_state, q_function, epsilon)
            converged = False

            if delta < threshold:
                converged_count += 1
                if converged_count == converged_threshold:
                    converged = True
            else:
                converged_count = 0

            if converged:
                print(f'Stopped after {i-converged_threshold} episodes.')
                break

        if not converged:
            print(f'Delta did not become small enough in {max_episodes} episodes')
        return q_function

    def sarsa(self, max_episodes=10_000, threshold=0.01, converged_threshold=10, alpha=0.1, epsilon=0.1):
        q_function = self.generate_empty_q_function(4, 4)
        converged_count = 0
        for i in range(max_episodes):
            start_x, start_y = random.randrange(4), random.randrange(4)
            current_state = self.maze.maze[start_y][start_x]
            if not current_state.done:
                self.policy.update_policy(current_state, q_function, epsilon)
            current_action = self.pick_action(current_state)
            delta = 0
            while not current_state.done:
                next_state = self.maze.do_step(current_state, current_action)
                x_current_state, y_current_state = current_state.location
                x_next_state, y_next_state = next_state.location
                reward = next_state.reward
                next_action = self.pick_action(next_state)
                old_value = q_function[y_current_state][x_current_state][current_action]
                new_value = q_function[y_current_state][x_current_state][current_action] + alpha * (
                    reward + self.discount * q_function[y_next_state][x_next_state][next_action] - q_function[y_current_state][x_current_state][current_action])
                delta = max(delta, abs(old_value - new_value))
                q_function[y_current_state][x_current_state][current_action] = new_value
                current_state = next_state
                current_action = next_action
            converged = False

            if delta < threshold:
                converged_count += 1
                if converged_count == converged_threshold:
                    converged = True
            else:
                converged_count = 0

            if converged:
                print(f'Stopped after {i-converged_threshold} episodes.')
                break

        if not converged:
            print(f'Delta did not become small enough in {max_episodes} episodes')
        return q_function

    def q_learning(self, max_episodes=10_000, threshold=0.1, converged_threshold=10, alpha=0.1, epsilon=0.1):
        q_function = self.generate_empty_q_function(4, 4)
        converged_count = 0
        for i in range(max_episodes):
            delta = 0
            start_x, start_y = random.randrange(4), random.randrange(4)
            current_state = self.maze.maze[start_y][start_x]
            while not current_state.done:
                self.policy.update_policy(current_state, q_function, epsilon)
                current_action = self.pick_action(current_state)
                next_state = self.maze.do_step(current_state, current_action)
                x_current_state, y_current_state = current_state.location
                x_next_state, y_next_state = next_state.location
                reward = next_state.reward
                old_value = q_function[y_current_state][x_current_state][current_action]
                new_value = q_function[y_current_state][x_current_state][current_action] + alpha * (reward + self.discount * max(
                    q_function[y_next_state][x_next_state]) - q_function[y_current_state][x_current_state][current_action])
                delta = delta = max(delta, abs(old_value - new_value))
                q_function[y_current_state][x_current_state][current_action] = new_value
                current_state = next_state

            converged = False

            if delta < threshold:
                converged_count += 1
                if converged_count == converged_threshold:
                    converged = True
            else:
                converged_count = 0

            if converged:
                print(f'Stopped after {i-converged_threshold} episodes.')
                break

        if not converged:
            print(f'Delta did not become small enough in {max_episodes} episodes')
        return q_function

    def double_q_learning(self, max_episodes=10_000, threshold=0.1, converged_threshold=10, alpha=0.1, epsilon=0.1):
        q_function_1 = self.generate_empty_q_function(4, 4)
        q_function_2 = self.generate_empty_q_function(4, 4)
        converged_count = 0
        for i in range(max_episodes):
            start_x, start_y = random.randrange(4), random.randrange(4)
            current_state = self.maze.maze[start_y][start_x]
            delta = 0
            while not current_state.done:
                q_funtion_sum = np.array(q_function_1) + np.array(q_function_2)
                q_funtion_sum = q_funtion_sum.tolist()
                self.policy.update_policy(
                    current_state, q_funtion_sum, epsilon)
                current_action = self.pick_action(current_state)
                next_state = self.maze.do_step(current_state, current_action)
                x_current_state, y_current_state = current_state.location
                x_next_state, y_next_state = next_state.location
                reward = next_state.reward
                if random.random() < 0.5:
                    old_value = q_function_1[y_current_state][x_current_state][current_action]
                    new_value = q_function_1[y_current_state][x_current_state][current_action] + alpha * (reward + self.discount * q_function_2[y_next_state][x_next_state][q_function_1[y_next_state][x_next_state].index(
                        max(q_function_1[y_next_state][x_next_state]))] - q_function_1[y_current_state][x_current_state][current_action])
                    q_function_1[y_current_state][x_current_state][current_action] = new_value
                else:
                    old_value = q_function_2[y_current_state][x_current_state][current_action]
                    new_value = q_function_2[y_current_state][x_current_state][current_action] + alpha * (reward + self.discount * q_function_1[y_next_state][x_next_state][q_function_2[y_next_state][x_next_state].index(
                        max(q_function_2[y_next_state][x_next_state]))] - q_function_2[y_current_state][x_current_state][current_action])
                    q_function_2[y_current_state][x_current_state][current_action] = new_value
                delta = max(delta, abs(old_value - new_value))
                current_state = next_state
            converged = False

            if delta < threshold:
                converged_count += 1
                if converged_count == converged_threshold:
                    converged = True
            else:
                converged_count = 0

            if converged:
                print(f'Stopped after {i-converged_threshold} episodes.')
                break

        if not converged:
            print(f'Delta did not become small enough in {max_episodes} episodes')
        return q_function_1, q_function_2

    def value_iteration(self) -> None:
        """
        Does the value iteration algorithm.
        """
        c = 0
        print(f'\nSweep {c}: ')
        visual_maze = np.zeros_like(np.array(self.value_function))
        print(visual_maze)
        delta = self.threshold+1
        while delta > self.threshold:
            delta = 0
            new_value_function = copy.deepcopy(self.value_function)
            for y, y_row in enumerate(self.value_function):
                for x, x_row in enumerate(y_row):
                    old_value = self.value_function[y][x]
                    if (x, y) not in self.maze.end_positions:
                        positions_around = get_positions_around((x, y))
                        possible_states = get_possible_states(
                            self.maze.maze, positions_around)
                        new_value = max_bellman(
                            self.discount, possible_states, self.value_function)[0]
                        new_value_function[y][x] = new_value
                        visual_maze[y][x] = new_value
                        delta = max(delta, abs(old_value-new_value))

            self.value_function = new_value_function
            if delta > self.threshold:
                c += 1
                print(f'Sweep {c}: ')
                print(visual_maze)

        print(f'Done after {c} sweeps!\n')

    def update_policy_to_deterministic(self):
        new_policy = copy.deepcopy(self.policy.policy_matrix)

        for y, y_row in enumerate(new_policy):
            for x, x_row in enumerate(y_row):
                positions_around = get_positions_around((x, y))
                possible_states = get_possible_states(
                    self.maze.maze, positions_around)
                all_max_actions = all_max_bellman(
                    self.discount, possible_states, self.value_function)
                max_chance = 1 / len(all_max_actions)
                for action, chance in enumerate(x_row):
                    if action in all_max_actions:
                        x_row[action] = max_chance
                    else:
                        x_row[action] = 0

        self.policy.policy_matrix = new_policy

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
        for y, y_row in enumerate(self.value_function):
            output_row_values = ''
            output_row_policy = ''
            for x, x_row in enumerate(y_row):
                state = self.maze.maze[y][x]
                value = self.value_function[y][x]
                if state.done:
                    policy = 9
                else:
                    policy = self.pick_action(state)
                output_row_values += f'{value:3}'
                output_row_policy += f'{self.maze.actions[policy]:3}'
            output_str += f'{output_row_values:14}  {output_row_policy} \n'
        print(output_str)
