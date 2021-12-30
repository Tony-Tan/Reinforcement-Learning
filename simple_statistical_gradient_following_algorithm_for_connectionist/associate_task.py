import random

from k_arm_bandit_finite_states import KArmedBandit
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

SQRT_PI = np.sqrt(np.pi)
K = 2


def squashing_function(input):
    return 1. / (1. + np.exp(-input))


class GaussianUnit:
    def __init__(self, input_size_, action_space_):
        self.weights_mean = np.zeros(input_size_)
        self.mean = np.random.random(1)*(K-1)
        self.std = 0.1
        self.action_space = np.array(action_space_)
        pass

    def __call__(self, input):
        self.mean = squashing_function(self.weights_mean.dot(input)) * (K - 1)
        return np.random.normal(self.mean, self.std)

    def characteristic_eligibility(self, x, y):
        s_mean = self.weights_mean.dot(x)
        mu = squashing_function(s_mean)
        partial_mu = (y - mu) / (self.std**2)
        nabla_weights_mean = partial_mu * np.exp(-s_mean) / ((1 + np.exp(-s_mean)) ** 2) * x
        return nabla_weights_mean


def reinforce_algorithm(repeat_times=1000000, random_seed=0):
    # 10 arm bandits
    reward_log_list = np.zeros(int(repeat_times / 100))
    optimal_action_hit_list = np.zeros(int(repeat_times / 100))
    np.random.seed(random_seed)
    env_mean = np.random.normal(.0, 1.0, K)

    env = KArmedBandit(env_mean, np.ones(K))
    gu = GaussianUnit(K, range(K))
    state, reward, is_done, _ = env.step(0)
    alpha = 0.0001
    base_line_mean = 0

    for repeat_i in range(repeat_times):
        x = np.array(state)
        y = gu(x)
        action = round(y)
        if action >= K or action < 0:
            state, reward, is_done, _ = env.step(0)
            reward = -10
        else:
            state, reward, is_done, _ = env.step(action)
        r = reward
        nabla_mean = gu.characteristic_eligibility(x, y)
        alpha_mean = alpha
        gu.weights_mean += (alpha_mean * (r - base_line_mean) * nabla_mean)
        if repeat_i % 100 == 0:
            # alpha *= 0.995
            reward_log_list[int(repeat_i / 100)] = r
            optimal_action_hit_list[int(repeat_i / 100)] = 1 if action == env.optimal_action else 0
    print('---------------------------------------------------')
    print('optimal action:')
    print(env.optimal_action)
    print('mean:')
    print(gu.mean)
    print('std:')
    print(gu.std)
    print('mean learning rate:')
    print(alpha_mean)
    return optimal_action_hit_list, reward_log_list


def experiment():
    experiment_time = 100
    seed_seq = np.random.randint(0, 100000, experiment_time)
    pool = Pool()
    repeat_times = 100000
    thread_num = 4
    reward_matrix = []
    optimal_action_hit_matrix = []
    for experiment_i in range(int(experiment_time / thread_num)):
        result_0 = pool.apply_async(reinforce_algorithm, [repeat_times, seed_seq[experiment_i * 4]])
        result_1 = pool.apply_async(reinforce_algorithm, [repeat_times, seed_seq[experiment_i * 4] + 1])
        result_2 = pool.apply_async(reinforce_algorithm, [repeat_times, seed_seq[experiment_i * 4] + 2])
        result_3 = pool.apply_async(reinforce_algorithm, [repeat_times, seed_seq[experiment_i * 4] + 3])
        optimal_action_hit_thread_0, reward_thread_0 = result_0.get(timeout=500)
        optimal_action_hit_thread_1, reward_thread_1 = result_1.get(timeout=500)
        optimal_action_hit_thread_2, reward_thread_2 = result_2.get(timeout=500)
        optimal_action_hit_thread_3, reward_thread_3 = result_3.get(timeout=500)
        reward_matrix.append(reward_thread_0)
        reward_matrix.append(reward_thread_1)
        reward_matrix.append(reward_thread_2)
        reward_matrix.append(reward_thread_3)
        optimal_action_hit_matrix.append(optimal_action_hit_thread_0)
        optimal_action_hit_matrix.append(optimal_action_hit_thread_1)
        optimal_action_hit_matrix.append(optimal_action_hit_thread_2)
        optimal_action_hit_matrix.append(optimal_action_hit_thread_3)
    average_reward_list = np.zeros(len(reward_matrix[0]))
    average_optimal_action_hit_list = np.zeros(len(optimal_action_hit_matrix[0]))
    for i in reward_matrix:
        average_reward_list += i
    for i in optimal_action_hit_matrix:
        average_optimal_action_hit_list += i
    # plt.plot(average_reward_list / experiment_time, label='reward')
    plt.plot(average_optimal_action_hit_list / experiment_time, label='optimal action rate')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # optimal_action_list, reward_list = reinforce_algorithm()
    experiment()
