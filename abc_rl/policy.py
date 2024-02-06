import numpy as np
from utils.commons import *
from abc_rl.exploration import *
from environments.envwrapper import *
from abc_rl.experience_replay import *
from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def select_action(self, state:np.ndarray) -> np.ndarray:
        ...
    #
    # @abstractmethod
    # def save(self):
    #     ...
    #
    # @abstractmethod
    # def load(self):
    #     ...


class ValueFunction(ABC):

    def __init__(self):
        ...

    @abstractmethod
    def value(self, **kwargs) -> np.ndarray:
        ...

    @abstractmethod
    def update(self,  **kwargs):
        ...


# class ModelBased(ABC):
#     def __init__(self):
#         ...
#
#     @abstractmethod
#     def plane(self):
#         ...
#
#     @abstractmethod
#     def update(self):
#         ...
#
#     @abstractmethod
#     def select_action(self):
#         ...