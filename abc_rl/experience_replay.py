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


class UniformExperienceReplay(ExperienceReplay):
    def __init__(self, memory_size: int):
        self.replay_buffer = deque(maxlen=memory_size)

    def store(self, transition: list):
        if len(self.replay_buffer) == 0:
            for t_i in transition:
                self.replay_buffer.append([t_i])
        else:
            for i, t_i in enumerate(transition):
                self.replay_buffer[i].append(t_i)

    def sample(self, batch_size: int):
        idx = np.arange(self.__len__())
        selected = np.random.choice(idx, batch_size, replace=True)
        sampled_transitions = []
        for element_i in self.replay_buffer:
            data_i = []
            for selected_idx in selected:
                data_i.append(element_i[selected_idx])
            sampled_transitions.append(np.array(data_i, dtype=np.float32))
        return sampled_transitions

    def __len__(self):
        if len(self.replay_buffer) == 0:
            return 0
        else:
            return len(self.replay_buffer[0])

    def __getitem__(self, item):
        return self.replay_buffer[item]

    # def __setitem__(self, key, transition: list):
    #     return self.replay_buffer[key]

