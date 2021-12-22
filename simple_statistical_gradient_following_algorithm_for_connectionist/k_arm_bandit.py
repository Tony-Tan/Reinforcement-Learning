import copy

import numpy as np


class BasicConceptError(Exception):
    def __init__(self, value_):
        self.__value = value_

    def __str__(self):
        print('[Basic Concepts Error!]: ' + self.__value)


class Space:
    def __init__(self, initial_list):
        if not isinstance(initial_list, list):
            raise BasicConceptError('Space should be initialized by a list!')
        if len(initial_list) == 0:
            raise BasicConceptError('Empty space may destroy your whole project!')
        self.__list = initial_list
        self.n = len(initial_list)

    def __getitem__(self, index):
        return self.__list[index]


# class bandit:
#     def __init__(self, value_mean=0.0, value_var=1.0):
#         """
#         bandit, reward is produced by a normal distribution with mean and variance;
#         :param value_mean: mean
#         :param value_var: variance
#         """
#         self.value_mean = value_mean
#         self.value_var = value_var
#
#     def run(self):
#         return np.random.normal(self.value_mean, self.value_var, 1)[0]


class KArmedBandit:
    def __init__(self, value_mean_array_, value_deviation_array_):
        self._k_value_mean = value_mean_array_
        self._k_value_deviation = value_deviation_array_
        self._k = len(value_mean_array_)
        self.action_space = Space([i for i in range(self._k)])
        self.optimal_action = np.flatnonzero(self._k_value_mean == self._k_value_mean.max())

    def reset(self):
        pass

    def step(self, action_):
        current_state = []
        if action_ < self._k:
            for i in self.action_space:
                current_state.append(
                    np.random.normal(self._k_value_mean[i], self._k_value_deviation[i], 1)[0])
            return current_state, current_state[action_], False, {}
        else:
            raise ValueError("action must be a number less than k")


if __name__ == '__main__':
    env = KArmedBandit(np.random.normal(.0, 1.0, 10), np.ones(10))