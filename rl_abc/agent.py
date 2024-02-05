from utils.commons import *
from abc.experience_replay import *
from abc.policy import *
from gymnasium.spaces import Space
from environments.envwrapper import *


class AgentOnline:
    def __init__(self, memory_size: int, save_path: str):
        self.memory_size = memory_size
        self.save_path = save_path
        self.replay_buffer = ReplayBuffer(memory_size)
        self.__save_folder_create()

    def __save_folder_create(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def react(self, **kwargs) -> np.ndarray:
        raise MethodNotImplement("react of the agent should be implemented")

    def observe(self, **kwargs):
        self.replay_buffer.append(kwargs['transition'])

    def learn(self, **kwargs):
        raise MethodNotImplement("This method is responsible for training the agent. It takes the total number of "
                                 "time steps as input and updates the agent's policy and value function based on "
                                 "interactions with the environment.")

    def test(self, **kwargs):
        raise MethodNotImplement("test model")

    def save(self, **kwargs):
        raise MethodNotImplement("store and restore agent parameters")

    def load(self, **kwargs):
        raise MethodNotImplement("store and restore agent parameters")
