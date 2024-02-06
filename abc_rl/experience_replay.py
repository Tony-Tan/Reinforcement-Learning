from collections import deque
import random
from abc import ABC, abstractmethod


class ExperienceReplay(ABC):
    @abstractmethod
    def store(self, experience):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass


class UniformExerienceReplay(ExperienceReplay):
    def __init__(self, memory_size:int):
        self.memory = deque(maxlen=memory_size)

    def store(self, transition: list):
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]

    def __setitem__(self, key, value):
        self.memory[key] = value

#
# class ReplayBuffer(ExperienceReplay):
#     def __init__(self, max_memory_size):
#         self.max_memory_size = max_memory_size
#         self.buffer = deque(maxlen=max_memory_size)
#
#     def append(self, new_data):
#         self.buffer.append(new_data)
#
#     def append_trajectory(self, new_data):
#         for step_i in new_data:
#             self.buffer.append(step_i)
#
#     def __getitem__(self, item):
#         return self.buffer[item]
#
#     def __setitem__(self, key, value):
#         self.buffer[key] = value
#
#     def __len__(self):
#         return len(self.buffer)
#
#     def sample(self, size: int):
#         return random.sample(self.buffer, size)

