from abc_rl.experience_replay import *
import numpy as np


class ProportionalPrioritization(ExperienceReplay):
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        super(ProportionalPrioritization, self).__init__(capacity)
        self.p = np.zeros(capacity)
        self.alpha = alpha
        self.beta = beta

    def store(self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_observation: np.ndarray, done: np.ndarray, truncated: np.ndarray):
        super(ProportionalPrioritization, self).store(observation, action, reward, next_observation, done, truncated)
        p_position = self.position - 1 if self.position > 0 else self.capacity - 1
        self.p[p_position] = np.max(self.p) if self.__len__() > 1 else 1.0

    def sample(self, batch_size: int, beta:float=None):
        n = self.__len__()
        # normalize the p as a probability distribution
        p = self.p[:n] / self.p[:n].sum()
        # select the index of the samples
        idx = np.random.choice(np.arange(n), batch_size, p=p, replace=False)
        if beta is None:
            w = (n * p) ** -self.beta
        else:
            w = (n * p) ** -beta
        w = w[idx] / np.max(w)
        return self.get_items(idx), w, idx


# class RankPrioritization(ExperienceReplay):
#     def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
#         super(RankPrioritization, self).__init__(capacity)
#         self.p = np.zeros(capacity)
#         self.alpha = alpha
#         self.beta = beta
#
#     def store(self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray,
#               next_observation: np.ndarray, done: np.ndarray, truncated: np.ndarray):
#         super(ProportionalPrioritization, self).store(observation, action, reward, next_observation, done, truncated)
#         p_position = self.position - 1 if self.position > 0 else self.capacity - 1
#         self.p[p_position] = np.max(self.p) if self.__len__() > 1 else 1.0
#
#     def sample(self, batch_size: int, beta:float=None):
#         n = self.__len__()
#         # normalize the p as a probability distribution
#         p = self.p[:n] / self.p[:n].sum()
#         # select the index of the samples
#         idx = np.random.choice(np.arange(n), batch_size, p=p, replace=False)
#         if beta is None:
#             w = (n * p) ** -self.beta
#         else:
#             w = (n * p) ** -beta
#         w = w[idx] / np.max(w)
#         return self.get_items(idx), w, idx
