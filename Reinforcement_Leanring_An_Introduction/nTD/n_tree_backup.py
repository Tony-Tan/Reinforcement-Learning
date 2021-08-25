import collections

import matplotlib.pyplot as plt
import numpy as np
from environment.random_walk_19_states import RandomWalk


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Agent:
    def __init__(self, environment_, n):
        self.env = environment_
        self.n = n
        self.policies = collections.defaultdict(constant_factory(self.env.action_space.n))
        self.value_of_state_action = collections.defaultdict(lambda: 0)

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def estimating(self, iteration_times, alpha=0.001, gamma=0.9, epsilon=0.1):
        for iteration_time in range(iteration_times):
            current_stat = self.env.reset()
            action = self.select_action(current_stat)
            new_state, reward, is_done, _ = self.env.step(action)
            # the doc of deque can be found: https://docs.python.org/3/library/collections.html#collections.deque
            n_queue = collections.deque()
            n_queue.append([current_stat, action, reward])
            while True:
                if is_done:
                    while len(n_queue) != 0:
                        state_updated, action_updated, reward = n_queue.popleft()
                        gamma_temp_policy = gamma
                        g_value = reward
                        for iter_n in n_queue:
                            g_value += gamma_temp_policy * iter_n[2]
                            expected_value = 0
                            for action_iter in range(self.env.action_space.n):
                                if action_iter != iter_n[1]:
                                    expected_value += self.policies[iter_n[0]][action_iter] * \
                                                      self.value_of_state_action[(iter_n[0], action_iter)]
                            g_value += gamma_temp_policy * gamma * expected_value
                            gamma_temp_policy *= gamma * self.policies[iter_n[0]][iter_n[1]]
                        self.value_of_state_action[(state_updated, action_updated)] += \
                            (alpha * (g_value - self.value_of_state_action[(state_updated, action_updated)]))
                    break
                else:
                    if len(n_queue) == self.n + 1:
                        state_updated, action_updated, reward = n_queue.popleft()
                        gamma_temp = gamma
                        gamma_temp_policy = gamma
                        g_value = reward
                        for iter_n in n_queue:
                            g_value += gamma_temp_policy * iter_n[2]
                            expected_value = 0
                            for action_iter in range(self.env.action_space.n):
                                if action_iter != iter_n[1]:
                                    expected_value += self.policies[iter_n[0]][action_iter] * \
                                                      self.value_of_state_action[(iter_n[0], action_iter)]
                            g_value += gamma_temp_policy * gamma * expected_value
                            gamma_temp_policy *= gamma * self.policies[iter_n[0]][iter_n[1]]
                            gamma_temp *= gamma
                        # new
                        current_stat = new_state
                        action = self.select_action(current_stat)
                        new_state, reward, is_done, _ = self.env.step(action)
                        n_queue.append([current_stat, action, reward])
                        expected_g_state_value = 0
                        for action_iter in range(self.env.action_space.n):
                            expected_g_state_value += self.value_of_state_action[(current_stat, action_iter)] * \
                                                      self.policies[current_stat][action_iter]
                        g_value += self.value_of_state_action[(current_stat, action)] * gamma_temp_policy
                        self.value_of_state_action[(state_updated, action_updated)] += \
                            (alpha * (g_value - self.value_of_state_action[(state_updated, action_updated)]))
                    else:
                        current_stat = new_state
                        action = self.select_action(current_stat)
                        new_state, reward, is_done, _ = self.env.step(action)
                        n_queue.append([current_stat, action, reward])
        # update policy
        for state_iter in range(self.env.state_space.n):
            value_of_action_list = []
            for action_iter in range(self.env.action_space.n):
                value_of_action_list.append(self.value_of_state_action[(state_iter, action_iter)])
            value_of_action_list = np.array(value_of_action_list)
            optimal_action = np.random.choice(
                np.flatnonzero(value_of_action_list == value_of_action_list.max()))
            for action_iter in range(self.env.action_space.n):
                if action_iter == optimal_action:
                    self.policies[state_iter][
                        action_iter] = 1 - epsilon + epsilon / self.env.action_space.n
                else:
                    self.policies[state_iter][action_iter] = epsilon / self.env.action_space.n


if __name__ == '__main__':
    env = RandomWalk(19)
    ground_truth = []
    for i in range(0, 19):
        ground_truth.append(-1 + i / 9)
    agent = Agent(env, 4)
    agent.estimating(10000)
    estimating_value = np.zeros(19)
    for i in range(env.state_space.n):
        for j in range(env.action_space.n):
            estimating_value[i] += agent.value_of_state_action[(i, j)] * agent.policies[i][j]
    print(estimating_value)
    plt.figure(0)
    plt.plot(estimating_value[1:-1])
    plt.plot(ground_truth[1:-1])
    plt.show()
