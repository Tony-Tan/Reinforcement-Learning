import numpy as np
import random
from utils.commons import Logger
from abc import ABC, abstractmethod


class Exploration(ABC):
    def __init__(self):
        # logger('Exploration method: \'{}\' is initialized'.format(self.__class__.__name__))
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass


