from collections import deque
import random


class ReplayBuffer:
    def __init__(self, max_memory_size):
        self.max_memory_size = max_memory_size
        self.buffer = deque(maxlen=max_memory_size)

    def append_step(self, new_data):
        self.buffer.append(new_data)

    def append_trajectory(self, new_data):
        for step_i in new_data:
            self.buffer.append(step_i)

    def __getitem__(self, item):
        return self.buffer[item]

    def __len__(self):
        return len(self.buffer)

    def sample(self, size: int):
        return random.sample(self.buffer, size)