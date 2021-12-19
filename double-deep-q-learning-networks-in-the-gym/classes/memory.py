from collections import deque
from random import sample
from classes.transition import Transition

class Memory:
    def __init__(self, size):
        self.size = size
        self.transitions = deque(maxlen=size)

    def sample(self, sample_size):
        return sample(self.transitions, sample_size)

    def append_memory(self, transition: Transition):
        self.transitions.append(transition)

    def __str__(self):
        return "...."