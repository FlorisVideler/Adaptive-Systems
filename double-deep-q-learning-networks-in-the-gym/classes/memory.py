from collections import deque
from random import sample
from classes.transition import Transition


class Memory:
    """The memory class where the transitions are saved"""

    def __init__(self, size):
        self.size = size
        self.transitions = deque(maxlen=size)

    def sample(self, sample_size):
        """This function gives a sample of states of the current memory"""
        return sample(self.transitions, sample_size)

    def append_memory(self, transition: Transition):
        """This function adds a transition to its memory"""
        self.transitions.append(transition)

    def __str__(self):
        return f'Memory size: {self.size}, Transistions: {self.transitions}'
