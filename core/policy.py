import numpy as np
from utils.commons import *
from core.exploration import *
from environments.envwrapper import *
from core.replay_memory import *


class Policy:
    def __init__(self, env: EnvWrapper, exploration_method):
        self.exploration = exploration_method
        self.state_space = env.state_space
        self.action_space = env.action_space
        pass

    # def select_action(self, states: np.ndarray):
    #     raise MethodNotImplement('Policy need have \'select_action\' method')

    def update(self, **kwargs):
        raise MethodNotImplement('Policy need have \'update\' method')

    def react(self, **kwargs) -> np.ndarray:
        raise MethodNotImplement('Policy need have \'react\' method')

    def save(self):
        raise MethodNotImplement('Policy need have \'save\' method')

    def load(self):
        raise MethodNotImplement('Policy need have \'load\' method')


class ValueBasedPolicy(Policy):
    def __init__(self, env: EnvWrapper, exploration_method):
        super().__init__(env, exploration_method)

    def value(self, **kwargs) -> np.ndarray:
        raise MethodNotImplement('ValueBasedPolicy need have \'value\' method')


class PolicyBasedPolicy(Policy):
    def __init__(self, env: EnvWrapper, exploration_method: Exploration):
        super().__init__(env, exploration_method)

    def distribution(self, state: np.ndarray, **kwargs) -> np.ndarray:
        raise MethodNotImplement('PolicyBasedPolicy need have \'distribution\' method')



class ModelBasedPolicy(Policy):
    def __init__(self, env: EnvWrapper, exploration_method: Exploration):
        super().__init__(env,exploration_method)
