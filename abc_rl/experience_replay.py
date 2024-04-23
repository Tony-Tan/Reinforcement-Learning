from collections import deque
from abc import ABC, abstractmethod
import numpy as np
import torch
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor


#
# class ExperienceReplay(ABC):
#     def __init__(self, capacity: int):
#         self.max_size = capacity
#         self.ptr = 0
#         self.size = 0
#         self.state_shape = None
#         self.action_shape = None
#         self.reward_shape = None
#         self.done_shape = None
#         self.truncated_shape = None
#         self.buffer = None
#         self.state_size = 0
#         self.next_state_size = 0
#         self.action_size = 0
#         self.reward_size = 0
#         self.done_size = 0
#         self.truncated_size = 0
#         self.state_segment = []
#         self.action_segment = []
#         self.reward_segment = []
#         self.next_state_segment = []
#         self.done_segment = []
#         self.truncated_segment = []
#
#     def __init_buffer(self, state: np.array, action: np.array, reward: np.array, next_state: np.array,
#                     done: np.array, truncated: np.array):
#         assert self.buffer is None
#         self.state_shape = state.shape
#         self.state_size = int(np.prod(state.shape))
#         self.state_segment = [0, self.state_size]
#
#         self.action_shape = action.shape
#         self.action_size = int(np.prod(action.shape))
#         self.action_segment = [self.state_segment[1], self.state_segment[1] + self.action_size]
#
#         self.reward_shape = reward.shape
#         self.reward_size = int(np.prod(reward.shape))
#         self.reward_segment = [self.action_segment[1],self.action_segment[1]+self.reward_size]
#
#         self.next_state_size = int(np.prod(state.shape))
#         self.next_state_segment = [self.reward_segment[1],self.reward_segment[1]+self.next_state_size]
#
#         self.done_shape = reward.shape
#         self.done_size = int(np.prod(done.shape))
#         self.done_segment = [self.next_state_segment[1],self.next_state_segment[1]+self.done_size]
#
#         self.truncated_shape = reward.shape
#         self.truncated_size = int(np.prod(truncated.shape))
#         self.truncated_segment = [self.done_segment[1],self.done_segment[1]+self.truncated_size]
#
#         sample_size = int(self.state_size + self.action_size + self.reward_size + self.next_state_size
#                        + self.done_size + self.truncated_size)
#         # 初始化存储数组
#         self.buffer = np.empty((self.max_size, sample_size), dtype=np.float32)
#
#     def store(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
#               done: np.ndarray, truncated: np.ndarray):
#         if self.size == 0:
#             self.__init_buffer(state, action, reward, next_state, done, truncated)
#         if self.size < self.max_size:
#             self.size += 1
#
#         experience = np.concatenate([state.flatten(), action.flatten(), reward.flatten(),
#                                     next_state.flatten(), done.flatten(), truncated.flatten()])
#         self.buffer[self.ptr] = experience
#         self.ptr = (self.ptr + 1) % self.max_size
#
#
#     def get_items(self, idx, device:torch.device=None):
#         assert self.size > 0
#         samples = self.buffer[idx]
#         if device is not None:
#             stream = torch.cuda.Stream()
#             with torch.cuda.stream(stream):
#                 samples = torch.from_numpy(samples).to(device, non_blocking=True)
#             stream.synchronize()
#         else:
#             samples = self.buffer[idx]
#         states = samples[:, self.state_segment[0]:self.state_segment[1]].reshape(-1, *self.state_shape)
#         actions = samples[:, self.action_segment[0]: self.action_segment[1]].reshape(-1, *self.action_shape)
#         rewards = samples[:,self.reward_segment[0]:self.reward_segment[1]].reshape(-1,*self.reward_shape)
#         next_states = samples[:, self.next_state_segment[0]: self.next_state_segment[1]].reshape(-1, *self.state_shape)
#         done = samples[:,self.done_segment[0]:self.done_segment[1]].reshape(-1, *self.done_shape)
#         truncated = samples[:,self.truncated_segment[0]:self.truncated_segment[1]].reshape(-1, *self.truncated_shape)
#
#         return states, actions, rewards, next_states, done, truncated
#
#     def __len__(self):
#         return self.size


class ExperienceReplay(ABC):
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def store(self, observation, action, reward, next_observation, done, truncated):
        self.buffer.append([observation, action, reward, next_observation, done, truncated])

    def __len__(self):
        return len(self.buffer)

    def get_items(self, idx):
        idx_size = len(idx)
        obs= np.empty([idx_size, *self.buffer[0][0].shape], dtype=np.float32)
        action = np.empty([idx_size, *self.buffer[0][1].shape], dtype=np.float32)
        reward = np.empty([idx_size, *self.buffer[0][2].shape], dtype=np.float32)
        next_obs = np.empty([idx_size , *self.buffer[0][0].shape], dtype=np.float32)
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

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass



