from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.multiprocessing as mp


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

    def clear(self):
        self.buffer = []
        self.position = 0

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

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass





class ExperienceReplayMP(ABC):
    """
    Shared memory implementation of the Uniform Experience Replay buffer.
    """
    def __init__(self, capacity: int, obs_shape: np.shape, action_shape:  np.shape):
        self.capacity = capacity
        self.obs_shape= obs_shape
        self.obs_dim = int(np.prod(obs_shape))
        self.action_shape = action_shape
        self.action_dim = int(np.prod(action_shape))
        self.state_buffer = mp.Array('f', capacity * self.obs_dim)
        self.next_state_buffer = mp.Array('f', capacity * self.obs_dim)
        self.action_buffer = mp.Array('f', capacity * self.action_dim)
        self.reward_buffer = mp.Array('f', capacity)
        self.done_buffer = mp.Array('f', capacity)
        self.truncated_buffer = mp.Array('f', capacity)
        self.ptr = mp.Value('i', 0)
        self.size = mp.Value('i', 0)
        self.lock = mp.Lock()
    @staticmethod
    def _store_in_shared_array(shared_array, data, index, dim):
        for i in range(dim):
            shared_array[index * dim + i] = data[i]

    def store(self, observation, action, reward, next_observation, done, truncated):
        with self.lock:
            index = self.ptr.value % self.capacity
            self._store_in_shared_array(self.state_buffer, observation.flatten(), index, self.obs_dim)
            self._store_in_shared_array(self.next_state_buffer, next_observation.flatten(), index, self.obs_dim)
            self._store_in_shared_array(self.action_buffer, action, index, self.action_dim)
            self.reward_buffer[index] = reward
            self.done_buffer[index] = done
            self.truncated_buffer[index] = truncated
            self.ptr.value += 1
            self.size.value = min(self.size.value + 1, self.capacity)

    def clear(self):
        with self.lock:
            self.ptr.value = 0
            self.size.value = 0

    def __len__(self):
        return self.size.value

    @staticmethod
    def _load_from_shared_array(shared_array, indices, dim):
        data = np.zeros((len(indices), dim), dtype=np.float32)
        for j, index in enumerate(indices):
            for i in range(dim):
                data[j, i] = shared_array[index * dim + i]
        return data

    def get_items(self, indices):
        with self.lock:
            batch_size = len(indices)
            states = self._load_from_shared_array(self.state_buffer, indices,
                                                  self.obs_dim).reshape(batch_size, *self.obs_shape)
            next_states = self._load_from_shared_array(self.next_state_buffer, indices,
                                                       self.obs_dim).reshape(batch_size, *self.obs_shape)
            actions = self._load_from_shared_array(self.action_buffer, indices, self.action_dim)
            rewards = np.array([self.reward_buffer[i] for i in indices], dtype=np.float32)
            truncated = np.array([self.truncated_buffer[i] for i in indices], dtype=np.float32)
            dones = np.array([self.done_buffer[i] for i in indices], dtype=np.float32)
        return (torch.as_tensor(states), torch.as_tensor(actions), torch.as_tensor(rewards),
                torch.as_tensor(next_states), torch.as_tensor(dones), torch.as_tensor(truncated))

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass