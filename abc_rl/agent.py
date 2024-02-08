from utils.commons import *
from abc import ABC, abstractmethod
from gymnasium.spaces import Space
from environments.envwrapper import *


class Agent(ABC):
    def __init__(self):
        pass

    # def __save_folder_create(self):
    #     if not os.path.exists(self.save_path):
    #         os.makedirs(self.save_path)
    @abstractmethod
    def select_action(self,  **kwargs) -> np.ndarray:
        ...

    @abstractmethod
    def store(self, **kwargs):
        ...

    @abstractmethod
    def learn(self, **kwargs):
        ...

    @abstractmethod
    def test(self, **kwargs):
        ...


