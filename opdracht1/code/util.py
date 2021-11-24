from state import State

def get_positions_around(position: tuple):
    x, y = position
    positions = [(x,y)] * 4

    if y - 1 >= 0:
        positions[0] = (x, y-1)

    if x + 1 <= 3:
        positions[1] = (x+1, y)

    if y + 1 <= 3:
        positions[2] = (x, y+1)

    if x - 1 >= 0:
        positions[3] = (x-1, y)
    
    return positions


def get_possible_states(all_states: list, positions: tuple):
        states = []
        for x, y in positions:
            states.append(all_states[y][x])
        return states


def bellman(discount: float, state: State):
    return state.reward + discount * state.value


def max_bellman(discount: float, states: list):
    all_results = []
    for state in states:
        all_results.append(bellman(discount, state))
    return max(all_results), all_results.index(max(all_results))