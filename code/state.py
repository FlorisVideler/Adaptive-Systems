from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class State:
    """
    Dataclass that represents a State.

    Args:
        location (tuple): The location where this State takes place.
        reward (int): The rewards of the State.
        value (float): The value of this State.
        done (bool): Wheter this state is an end state.
    """
    location: tuple
    reward: int
    done: bool

    def __str__(self) -> str:
        return str(self.value)
