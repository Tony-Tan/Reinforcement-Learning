import collections
import random

import matplotlib.pyplot as plt
import numpy as np
from environment.random_walk_19_states import RandomWalk


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Agent:
    def __init__(self, env, n):
        self.env = env
        self.n = n
        self.policies = collections.defaultdict(constant_factory(2))
        self.value_of_state = collections.defaultdict(lambda: 0.5)

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def estimating(self, iteration_times, alpha=0.9, gamma=0.9):
        for _ in range(iteration_times):
            current_stat = self.env.reset()
            action = self.select_action(current_stat)
            # the doc of deque can be found: https://docs.python.org/3/library/collections.html#collections.deque
            n_queue = collections.deque()
            new_state, reward, is_done, _ = self.env.step(action)
            while True:
                n_queue.append([new_state, reward, is_done])
                if is_done:
                    while len(n_queue) != 0:
                        state_updated, _, _ = n_queue.popleft()
                        gamma_temp = 1.0
                        g_value = 0.0
                        for iter_n in n_queue:
                            g_value += gamma_temp * iter_n[1]
                            gamma_temp *= gamma
                        self.value_of_state[state_updated] += (alpha * (g_value - self.value_of_state[state_updated]))
                    break
                else:
                    if len(n_queue) == self.n + 1:
                        state_updated, _, _ = n_queue.popleft()
                        gamma_temp = 1.0
                        g_value = 0.0
                        for iter_n in n_queue:
                            g_value += gamma_temp * iter_n[1]
                            gamma_temp *= gamma
                        action_next = self.select_action(new_state)
                        new_state, reward, is_done, _ = self.env.step(action_next)
                        g_value += (reward * gamma_temp + self.value_of_state[new_state])
                        self.value_of_state[state_updated] += (alpha * (g_value - self.value_of_state[state_updated]))
                    else:
                        action_next = self.select_action(new_state)
                        new_state, reward, is_done, _ = self.env.step(action_next)

    def estimating_with_generated_randomwalk(self, random_walk_trace_list, alpha=0.9, gamma=0.9):
        for random_walk in random_walk_trace_list:
            n_queue = collections.deque()
            new_state, reward, is_done, _ = random_walk[0]
            random_walk_step = 0
            while True:
                n_queue.append([new_state, reward, is_done])
                if is_done:
                    while len(n_queue) != 0:
                        state_updated, _, _ = n_queue.popleft()
                        gamma_temp = 1.0
                        g_value = 0.0
                        for iter_n in n_queue:
                            g_value += gamma_temp * iter_n[1]
                            gamma_temp *= gamma
                        self.value_of_state[state_updated] += (alpha * (g_value - self.value_of_state[state_updated]))
                    break
                else:
                    if len(n_queue) == self.n + 1:
                        state_updated, _, _ = n_queue.popleft()
                        gamma_temp = 1.0
                        g_value = 0.0
                        for iter_n in n_queue:
                            g_value += gamma_temp * iter_n[1]
                            gamma_temp *= gamma
                        random_walk_step += 1
                        new_state, reward, is_done, _ = random_walk[random_walk_step]
                        g_value += (reward * gamma_temp + self.value_of_state[new_state])
                        self.value_of_state[state_updated] += (alpha * (g_value - self.value_of_state[state_updated]))


def generate_random_walk_trace_list(env, agent):
    current_stat = env.reset()
    action = agent.select_action(current_stat)
    walk_trace = [env.step(action)]
    new_state, reward, is_done, _ = walk_trace[0]
    while not is_done:
        action = agent.select_action(new_state)
        walk_trace.append(env.step(action))
        new_state, reward, is_done, _ = walk_trace[-1]
    return walk_trace


if __name__ == '__main__':
    env = RandomWalk(19)
    ground_truth = []
    for i in range(0, 19):
        ground_truth.append(-1 + i / 9)
    alpha_array = [i / 100. for i in range(0, 100)]
    plt.figure(0)
    agents = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for agent_n in agents:
        rms_array = np.zeros(100)
        for _ in range(100):
            walk_trace_list = []
            for i in range(10):
                agent = Agent(env, 8)
                walk_trace_list.append(generate_random_walk_trace_list(env, agent))
            alpha_num = 0
            for alpha_i in alpha_array:
                value_list_of_state = np.zeros(19)
                agent = Agent(env, agent_n)
                agent.estimating_with_generated_randomwalk(walk_trace_list, alpha_i, gamma=1)
                for i in range(1, env.state_space.n - 1):
                    value_list_of_state[i] = (agent.value_of_state[i])
                rms_array[alpha_num] += np.sqrt(np.sum((np.array(value_list_of_state[1:-1]) -
                                                        np.array(ground_truth[1:-1])) ** 2) / 17)
                alpha_num += 1
        rms_array = rms_array / 100
        plt.plot(np.array(alpha_array), rms_array, color=(random.random(), random.random(), random.random()),
                 label='n=' + str(agent_n))
    plt.legend()
    plt.show()
