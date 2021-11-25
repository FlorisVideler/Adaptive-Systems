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
- State dataclass (State.py): Represents the state as a dataclass.
- Utility functions (util.py): Has a couple of helper functions that are used multiple times.