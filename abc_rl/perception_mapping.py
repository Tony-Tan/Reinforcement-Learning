from abc import ABC, abstractmethod
import numpy as np


class PerceptionMapping(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def map(self, state: np.ndarray) -> np.ndarray:
        ...
