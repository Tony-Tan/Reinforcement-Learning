from abc import ABC, abstractmethod
import numpy as np
from utils.commons import Logger


class PerceptionMapping(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def __call__(self,  *args, **kwargs) -> np.ndarray:
        ...
