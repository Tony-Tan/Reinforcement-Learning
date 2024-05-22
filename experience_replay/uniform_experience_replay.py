import numpy as np
from abc_rl.experience_replay import *
from abc_rl.experience_replay import ExperienceReplay
import torch.multiprocessing as mp


class UniformExperienceReplay(ExperienceReplay):
    def __init__(self, capacity: int):
        super(UniformExperienceReplay,self).__init__(capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(np.arange(self.__len__()), batch_size,  replace=False)
        return self.get_items(idx)


class UniformExperienceReplayMP(SharedExperience):
    """
    Shared memory implementation of the Uniform Experience Replay buffer.
    """

    def __init__(self, capacity: int, obs_shape: np.shape, action_shape: np.shape):
        super(UniformExperienceReplayMP, self).__init__(capacity, obs_shape, action_shape)


    def store(self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_observation: np.ndarray, done: np.ndarray, truncated: np.ndarray):
        super(UniformExperienceReplayMP, self).store(observation, action, reward, next_observation, done, truncated)

    def sample(self, batch_size: int = 0 ):
        return self.get_items()