from collections import deque
import copy
import numpy as np
from utils.commons import *
from utils.replay_memory import *


class AgentOnline:
    def __init__(self, memory_size: int, *args):
        self.memory_size = memory_size
        self.replay_buffer = ReplayBuffer(memory_size)
        raise MethodNotImplement("Design the `act` method that returns the agent's action based on the current state.")

    def select_action(self, states: np.ndarray, **kwargs) -> np.ndarray:
        raise MethodNotImplement("Design the `act` method that returns the agent's action based on the current state.")

    def replay_buffer_append(self, transition: list):
        self.replay_buffer.append(transition)

    def learn(self, *args):
        raise MethodNotImplement("This method is responsible for training the agent. It takes the total number of "
                                 "time steps as input and updates the agent's policy and value function based on "
                                 "interactions with the environment.")

    def test(self, *args):
        raise MethodNotImplement("test model")

    def save(self):
        raise MethodNotImplement("store and restore agent parameters")

    def load(self):
        raise MethodNotImplement("store and restore agent parameters")
