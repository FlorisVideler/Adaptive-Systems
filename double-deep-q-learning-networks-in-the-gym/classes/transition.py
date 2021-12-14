from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: int
    next_state: np.ndarray
    done: bool
