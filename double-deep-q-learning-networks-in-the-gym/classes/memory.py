from collections import deque
from random import sample
from classes.transition import Transition


class Memory:
    def __init__(self, size: int) -> None:
        """
        The constructor of the memory class.

        Args:
            size (int): The max size of the memory.
        """
        self.size = size
        self.transitions = deque(maxlen=size)

    def sample(self, sample_size: int) -> list:
        """
        Takes a sample of the memory.

        Args:
            sample_size (int): The amount of transitions to take from the memory.

        Returns:
            list: The sample.
        """
        return sample(self.transitions, sample_size)

    def append_memory(self, transition: Transition):
        """
        Appends a transition to the memory.

        Args:
            transition (Transition): The transition to append.
        """
        self.transitions.append(transition)

    def __str__(self):
        return f'Memory size: {self.size}, Transistions: {self.transitions}'
