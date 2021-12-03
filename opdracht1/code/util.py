from state import State


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


def all_max_bellman(discount: float, states: list, value_function) -> tuple:
    all_results = []
    # print(f'States: {states}')
    for state in states:
        x, y = state.location
        all_results.append(bellman(discount, state, value_function[y][x]))
    actions = []
    max_result = max(all_results)
    for index, result in enumerate(all_results):
        if result == max_result:
            actions.append(index)
    return actions
