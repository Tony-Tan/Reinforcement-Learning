from collections import deque
import random
from abc import ABC, abstractmethod
import numpy as np


class ExperienceReplay(ABC):
    @abstractmethod
    def store(self, experience):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass



