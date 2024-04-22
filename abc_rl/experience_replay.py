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
        obs = np.empty([len(idx), *self.buffer[0][0].shape], dtype=dtyp)
        # obs = []
        action = np.empty([len(idx), *self.buffer[0][1].shape], dtype=dtyp)
        reward = np.empty([len(idx), *self.buffer[0][2].shape], dtype=dtyp)
        next_obs = np.empty([len(idx), *self.buffer[0][3].shape], dtype=dtyp)
        done = np.empty([len(idx), *self.buffer[0][4].shape], dtype=dtyp)
        truncated = np.empty([len(idx), *self.buffer[0][5].shape], dtype=dtyp)
        for i, idx_i in enumerate(idx):
            o, a, r, n, d, t = self.buffer[idx_i]
            obs[i] = o
            action[i] = a
            reward[i] = r
            next_obs[i] = n
            done[i] = d
            truncated[i] = t
        # obs = np.array(obs, dtype=dtyp)
        # action = np.array(action, dtype=dtyp)
        # reward = np.array(reward, dtype=dtyp)
        # next_obs = np.array(next_obs, dtype=dtyp)
        # done = np.array(done, dtype=dtyp)
        # truncated = np.array(truncated, dtype=dtyp)
        # make a cuda stream
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            obs = torch.from_numpy(obs).to(device)
            action = torch.from_numpy(action).to(device)
            reward = torch.from_numpy(reward).to(device)
            next_obs = torch.from_numpy(next_obs).to(device)
            done = torch.from_numpy(done).to(device)
            truncated = torch.from_numpy(truncated).to(device)
        stream.synchronize()
        return obs,action,reward,next_obs,done,truncated

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass
