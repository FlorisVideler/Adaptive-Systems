from state import State

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np


def get_positions_around(position: tuple):
    """
    Gets the positions around a point.
    When the location doesn't exist, the position itself is used.

    Args:
        position (tuple): The position where to get the points around.

    Returns:
        list: A list with tuples of positions.
    """
    x, y = position
    positions = [(x, y)] * 4

    if y - 1 >= 0:
        positions[0] = (x, y-1)

    if x + 1 <= 3:
        positions[1] = (x+1, y)

    if y + 1 <= 3:
        positions[2] = (x, y+1)

    if x - 1 >= 0:
        positions[3] = (x-1, y)

    return positions


def get_possible_states(all_states: list, positions: tuple) -> list:
    """
    Gets the possible states around a postions.

    Args:
        all_states (list): The possible states.
        positions (tuple): The position where to get the states around.

    Returns:
        list: A list of possible states.
    """
    states = []
    for x, y in positions:
        states.append(all_states[y][x])
    return states


def bellman(discount: float, state: State, value) -> float:
    """
    Calculates the value using the bellman function.

    Args:
        discount (float): The discount used for the calculation.
        state (State): The state to use for calculation.

    Returns:
        float: The value of a state.
    """
    return state.reward + discount * value


def max_bellman(discount: float, states: list, value_function) -> tuple:
    """
    Gets the maximun value of multiple states.

    Args:
        discount (float): The discount used for the calculations
        states (list): The states that the value needs to be calculated for.

    Returns:
        tuple: A tuple with the max value and
        the index (action) to get to this State.
    """
    all_results = []
    for state in states:
        x, y = state.location
        all_results.append(bellman(discount, state, value_function[y][x]))
    # Sometimes there are two maxes, we just take te first one.
    return max(all_results), all_results.index(max(all_results))


def all_max_bellman(discount: float, states: list, value_function) -> list:
    """
    Gets the max actions from a list with bellman.

    Args:
        discount (float): The discount to use.
        states (list): A list with states.
        value_function ([type]): The value function.

    Returns:
        list: List with the best actions.
    """
    all_results = []
    for state in states:
        x, y = state.location
        all_results.append(bellman(discount, state, value_function[y][x]))
    actions = []
    max_result = max(all_results)
    for index, result in enumerate(all_results):
        if result == max_result:
            actions.append(index)
    return actions


def get_all_max(full_list: list) -> list:
    """
    Gets all the max values from a list and returns their index.

    Args:
        full_list (list): The input list.

    Returns:
        list: The indexes with the heighest value.
    """
    max_value = max(full_list)
    max_indexes = []
    for index, item in enumerate(full_list):
        if item == max_value:
            max_indexes.append(index)
    return max_indexes


def triangulation_for_triheatmap(M: int, N: int) -> list:
    """
    Generate a heightmap.

    Args:
        M (int): Lenght.
        N (int): Height.

    Returns:
        list: List with the right values and positions.
    """
    # vertices of the little squares
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N)
                         )  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (M + 1) * (N + 1)  # indices of the centers

    trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]


def transform_policy_to_matrix_values(policy_matrix: list) -> list:
    """
    Transforms list to list to plot in heatmap.

    Args:
        policy_matrix (list): The matrix to plot.

    Returns:
        list: The lists in the right order.
    """
    policy_matrix = np.array(policy_matrix)
    up_values = policy_matrix[:, :, 0]
    right_values = policy_matrix[:, :, 1]
    down_values = policy_matrix[:, :, 2]
    left_values = policy_matrix[:, :, 3]
    return [up_values, right_values, down_values, left_values]


def plot_matrix(M: int, N: int, values: list, title: str) -> None:
    """
    Plots matrix as heatmap

    Args:
        M (int): Height.
        N (int): Lenght.
        values (list): Values to plot.
        title (str): Title of the plot.
    """
    values = np.array(values)
    triangul = triangulation_for_triheatmap(M, N)
    fig, ax = plt.subplots()
    imgs = [ax.tripcolor(t, val.ravel(), cmap='RdYlGn', vmin=0, vmax=1, ec='white')
            for t, val in zip(triangul, values)]
    for val, dir in zip(values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
        for i in range(M):
            for j in range(N):
                v = val[j, i]
                ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0],
                        f'{v:.2f}', color='k' if 0.2 < v < 0.8 else 'w', ha='center', va='center')
    fig.colorbar(imgs[0], ax=ax)
    ax.set_xticks(range(M))
    ax.set_yticks(range(N))
    ax.invert_yaxis()
    ax.margins(x=0, y=0)
    ax.set_aspect('equal', 'box')  # square cells
    plt.title(title)
    plt.tight_layout()
    plt.show()
