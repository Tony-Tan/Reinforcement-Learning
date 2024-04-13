from collections import deque
import random
from abc import ABC, abstractmethod
import numpy as np
import torch


class ExperienceReplay(ABC):
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def store(self, observation, action, reward, next_observation, done, truncated):
        self.buffer.append([observation, action, reward, next_observation, done, truncated])

    def __len__(self):
        return len(self.buffer)

    def get_items(self, idx, dtyp: np.dtype, device: torch.device):
        obs = []
        action = []
        reward = []
        next_obs = []
        done = []
        truncated = []
        for idx_i in idx:
            o, a, r, n, d, t = self.buffer[idx_i]
            obs.append(o)
            action.append(a)
            reward.append(r)
            next_obs.append(n)
            done.append(d)
            truncated.append(t)
        obs = np.array(obs, dtype=dtyp)
        action = np.array(action, dtype=dtyp)
        reward = np.array(reward, dtype=dtyp)
        next_obs = np.array(next_obs, dtype=dtyp)
        done = np.array(done, dtype=dtyp)
        truncated = np.array(truncated, dtype=dtyp)
        return (torch.from_numpy(obs).to(device),
                torch.as_tensor(action).to(device),
                torch.as_tensor(reward).to(device),
                torch.as_tensor(next_obs).to(device),
                torch.as_tensor(done).to(device),
                torch.as_tensor(truncated).to(device))

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass
