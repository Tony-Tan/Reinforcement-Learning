from abc import ABC, abstractmethod
import numpy as np
import torch
from multiprocessing import Pool


class ExperienceReplay(ABC):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0  # This will keep track of the next position to insert into, for overwriting old records

    def store(self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_observation: np.ndarray, done: np.ndarray, truncated: np.ndarray):
        if len(self.buffer) < self.capacity:
            # If there is still space in the buffer, append the data
            self.buffer.append([observation, action, reward, next_observation, done, truncated])
        else:
            # Overwrite the oldest data if the buffer is at capacity
            self.buffer[self.position] = [observation, action, reward, next_observation, done, truncated]

        # Update the position in a circular manner
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)

    def get_items(self, idx):
        idx_size = len(idx)
        obs = np.empty([idx_size, *self.buffer[0][0].shape], dtype=np.float32)
        action = np.empty([idx_size, *self.buffer[0][1].shape], dtype=np.float32)
        reward = np.empty([idx_size, *self.buffer[0][2].shape], dtype=np.float32)
        next_obs = np.empty([idx_size, *self.buffer[0][0].shape], dtype=np.float32)
        done = np.empty([idx_size, 1], dtype=np.float32)
        truncated = np.empty([idx_size, 1], dtype=np.float32)
        for i, idx_i in enumerate(idx):
            o, a, r, n, d, t = self.buffer[idx_i]
            # the observation is store from 0 to idx_size in obs_next_obs
            obs[i] = o
            # the next observation is store from idx_size to -1 in obs_next_obs
            next_obs[i] = n
            action[i] = a
            reward[i] = r
            # the truncated is store from 0 to idx_size in done_truncated
            done[i] = d
            # the truncated is store from idx_size to -1 in done_truncated
            truncated[i] = t

        # from numpy to tensor
        obs = torch.from_numpy(obs)
        next_obs = torch.from_numpy(next_obs)
        action = torch.from_numpy(action)
        reward = torch.from_numpy(reward)
        done = torch.from_numpy(done)
        truncated = torch.from_numpy(truncated)
        return obs, action, reward, next_obs, done, truncated

    # from multiprocessing import Pool
    #
    # def get_item(self, idx_i):
    #     o, a, r, n, d, t = self.buffer[idx_i]
    #     return o, a, r, n, d, t
    #
    # def get_items(self, idx):
    #     idx_size = len(idx)
    #     with Pool() as p:
    #         results = p.map(self.get_item, idx)
    #
    #     obs, action, reward, next_obs, done, truncated = zip(*results)
    #
    #     # from numpy to tensor
    #     obs = torch.from_numpy(np.array(obs, dtype=np.float32))
    #     next_obs = torch.from_numpy(np.array(next_obs, dtype=np.float32))
    #     action = torch.from_numpy(np.array(action, dtype=np.float32))
    #     reward = torch.from_numpy(np.array(reward, dtype=np.float32))
    #     done = torch.from_numpy(np.array(done, dtype=np.float32))
    #     truncated = torch.from_numpy(np.array(truncated, dtype=np.float32))
    #     return obs, action, reward, next_obs, done, truncated
    @abstractmethod
    def sample(self, *args, **kwargs):
        pass
