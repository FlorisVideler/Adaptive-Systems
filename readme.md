# Adaptive Systems
Floris Videler<br>
1758374<br>
floris.videler@student.hu.nl

This repository contains all the assignments for the course Adaptive Systems. The repository also contains the theory assignments.

## Assignment 1.3.0 (the code)
The code contains 3 classes, 1 dataclass, 1 utility functions file and 1 file to demo the functionality.

- Maze class (maze.py): Contains all the information about the grid and builds the maze itself.
- Policy class (policy.py): Represents a policy. Could also be a function for now, but is kept as class for later use.
- Agent class (agent.py): Represents an agent, does all the hard work. Calculates the values and moves in the maze.
- State dataclass (state.py): Represents the state as a dataclass.
- Utility functions (util.py): Has a couple of helper functions that are used multiple times.

## Assignment 1.3.0 (result)
Values:
|||||
| -- | -- | -- | -- |
| 38 | 39 | 40 | 0 |
| 37 | 38 | 39 | 40 |
| 36 | 37 | 36 | 35 |
| 0 | 36 | 35 | 34 |

Policy:
|||||
| -- | -- | -- | -- |
| → | → | → | ⦾ |
| ↑ | ↑ | ↑ | ↑ |
| ↑ | ↑ | ← | ← |
| ⦾ | ↑ | ↑ | ↑ |


## Assignment 2 (the code)
The code contains 3 classes, 1 dataclass, 1 utility functions file and 1 file to demo the functionality.

- Maze class (maze.py): Contains all the information about the grid and builds the maze itself.
- Policy class (policy.py): Represents a policy. Could also be a function for now, but is kept as class for later use.
- Agent class (agent.py): Represents an agent, does all the hard work. Calculates the values and moves in the maze.
- State dataclass (state.py): Represents the state as a dataclass.
- Utility functions (util.py): Has a couple of helper functions that are used multiple times.

## Results
Can be found in the [notebook](https://github.com/FlorisVideler/Adaptive-Systems/blob/main/code/demo.ipynb) or you can download the [HTML](https://github.com/FlorisVideler/Adaptive-Systems/blob/main/code/demo.html)

## Limitations
All the algoriths work as expected and give correct outcomes. However the calculation of when a algorithm is done, using delta's is not working the full 100%. The reason for not fixing this is that this is nowhere near the mail assignment and there are other courses that cost time.