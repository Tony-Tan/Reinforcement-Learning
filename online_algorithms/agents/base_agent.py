from utils.commons import *
from core.replay_memory import *
from core.policy import *
from gymnasium.spaces import Space
from environments.envwrapper import *


class AgentOnline:
    def __init__(self, env: EnvWrapper, memory_size: int):
        self.state_space = env.state_space
        self.action_space = env.action_space
        self.memory_size = memory_size
        self.replay_buffer = ReplayBuffer(memory_size)

    def react(self, states: np.ndarray, **kwargs) -> np.ndarray:
        raise MethodNotImplement("Design the `react` method that returns the agent's action based on the current state.")

    def observe(self, transition: list):
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
