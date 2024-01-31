import numpy as np
from utils.commons import *
from core.exploration import *
from environments.envwrapper import *
from core.experience_replay import *
from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def __init__(self, state_space, action_space, exploration_method):
        self.exploration = exploration_method
        self.state_space = state_space
        self.action_space = action_space
        pass

    # def select_action(self, states: np.ndarray):
    #     raise MethodNotImplement('Policy need have \'select_action\' method')

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def react(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError


class ValueBased(ABC):
    def __init__(self, env: EnvWrapper, exploration_method):
        super().__init__(env, exploration_method)

    def value(self, **kwargs) -> np.ndarray:
        raise NotImplementedError


class ModelBased(Policy):
    def __init__(self, env: EnvWrapper, exploration_method: Exploration):
        super().__init__(env,exploration_method)
