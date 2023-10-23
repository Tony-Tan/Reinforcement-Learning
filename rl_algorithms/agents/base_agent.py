import numpy as np
from collections import deque
from rl_algorithms.common.core import *
from rl_algorithms.environments.env import *
from rl_algorithms.common.exceptions import *


class Agent:
    def __init__(self, env: Env, replay_buffer_size: int, save_path: str, logger: Logger):
        self.env = env
        self.logger = logger
        self.save_path = save_path
        self.replay_buffer = deque(maxlen=replay_buffer_size)

    def react(self, *args) -> np.ndarray:
        raise MethodNotImplement("Design the `act` method that returns the agent's action based on the current state.")

    def replay_buffer_append(self, on_r_t_t_info: list):
        self.replay_buffer.append(on_r_t_t_info)

    def learn(self, *args):
        raise MethodNotImplement("This method is responsible for training the agent. It takes the total number of "
                                 "time steps as input and updates the agent's policy and value function based on "
                                 "interactions with the environment.")

    def save(self):
        raise MethodNotImplement("store and restore agent parameters")

    def load(self):
        raise MethodNotImplement("store and restore agent parameters")

