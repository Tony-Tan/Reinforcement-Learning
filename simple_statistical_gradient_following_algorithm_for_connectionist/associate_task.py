import random

from k_arm_bandit import KArmedBandit
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


SQRT_PI = np.sqrt(np.pi)
K = 5


def squashing_function(input):
    return 1. / (1. + np.exp(-input))


class GaussianUnit:
    def __init__(self, input_size_, action_space_):
        self.weights_mean = np.zeros(input_size_)
        # self.weights_std = np.zeros(input_size_)
        self.mean = 0
        self.std = 0.1
        self.action_space = np.array(action_space_)
        pass

    def __call__(self, input):
        self.mean = squashing_function(self.weights_mean.dot(input)) * (K - 1)
        # self.std = squashing_function(self.weights_std.dot(input))*10 + 0.01
        # action_prob = 1 / (2 * SQRT_PI * self.std) * np.exp(
        #     -0.5 * ((self.action_space - self.mean) / self.std) ** 2) + 0.01
        # action_prob /= np.sum(action_prob)
        # print(action_prob)
        # return np.random.choice(self.action_space, 1, p=action_prob)[0]
        return np.random.normal(self.mean, self.std)

    def characteristic_eligibility(self, x, y):
        partial_mu = (y - self.mean) / (self.std ** 2)
        # partial_std = ((y-self.mean)**2-self.std**2)/(self.std**3)
        s = self.weights_mean.dot(x)
        nabla_weights_mean = partial_mu * x * (np.exp(-s) / (1 + np.exp(-s)) ** 2)
        # nabla_weights_std = partial_std * x
        return nabla_weights_mean  # , nabla_weights_std


def reinforce_algorithm(repeat_times=1000000,random_seed=0):
    # 10 arm bandits
    reward_log_list = np.zeros(int(repeat_times/100))
    np.random.seed(random_seed)
    env_mean = np.random.normal(.0, 1.0, K)

    env = KArmedBandit(env_mean, np.ones(K))
    gu = GaussianUnit(K, range(K))
    state, reward, is_done, _ = env.step(0)
    alpha_mean = 0.001
    # alpha_std = 0.01
    # base_line_std = 10
    base_line_mean = 0

    for repeat_i in range(repeat_times):
        x = np.array(state)
        y = gu(x)
        if y >= K or y < 0:
            state, reward, is_done, _ = env.step(0)
            reward = -10.
        state, reward, is_done, _ = env.step(int(y))
        r = reward
        # nabla_mean, nabla_std = gu.characteristic_eligibility(x, y)
        nabla_mean = gu.characteristic_eligibility(x, y)
        # gu.weights_std += (alpha_std * (r-base_line_std)*nabla_std)
        gu.weights_mean += (alpha_mean * (r - base_line_mean) * nabla_mean)
        if repeat_i % 100 == 0:
            reward_log_list[int(repeat_i/100)] = r
    print('mean:')
    print(gu.mean)
    print('environment mean:')
    print(env_mean)
    # print('weights for mean:')
    # print(gu.weights_mean)
    return reward_log_list


if __name__ == '__main__':
    experiment_time = 1000
    seed_seq = np.random.randint(0, 100000, experiment_time)
    pool = Pool()
    repeat_times = 100000
    thread_num = 4
    reward_matrix = []
    for experiment_i in range(int(experiment_time/thread_num)):
        result_0 = pool.apply_async(reinforce_algorithm, [repeat_times, seed_seq[experiment_i*4]])
        result_1 = pool.apply_async(reinforce_algorithm, [repeat_times, seed_seq[experiment_i*4]+1])
        result_2 = pool.apply_async(reinforce_algorithm, [repeat_times, seed_seq[experiment_i*4]+2])
        result_3 = pool.apply_async(reinforce_algorithm, [repeat_times, seed_seq[experiment_i*4]+3])
        answer0 = result_0.get(timeout=50)
        answer1 = result_1.get(timeout=50)
        answer2 = result_2.get(timeout=50)
        answer3 = result_3.get(timeout=50)
        reward_matrix.append(answer0)
        reward_matrix.append(answer1)
        reward_matrix.append(answer2)
        reward_matrix.append(answer3)
    average_reward_list = np.zeros(len(reward_matrix[0]))
    for i in reward_matrix:
        average_reward_list += i
    plt.plot(average_reward_list/experiment_time)
    plt.show()
