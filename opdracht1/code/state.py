from dataclasses import dataclass

@dataclass
class State:
    location: tuple
    reward: int
    value: float
    done: bool

    def __repr__(self) -> str:
        return str(self.value)