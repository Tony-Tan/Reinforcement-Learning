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
        self.behaviors_policies = collections.defaultdict(constant_factory(self.env.action_space.n))
        self.policies = collections.defaultdict(constant_factory(self.env.action_space.n))
        self.value_of_state_action = collections.defaultdict(lambda: 0)

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def estimating(self, iteration_times, alpha=0.1, gamma=0.9, epsilon=0.1):
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
                        gamma_temp = gamma
                        g_value = reward
                        p_value = self.policies[state_updated][action_updated] / self.behaviors_policies[
                            state_updated][action_updated]
                        for iter_n in n_queue:
                            # iter_n[2] is the reward in the queue
                            g_value += gamma_temp * iter_n[2]
                            p_value *= self.policies[iter_n[0]][iter_n[1]] / self.behaviors_policies[iter_n[0]][
                                iter_n[1]]
                            gamma_temp *= gamma
                        self.value_of_state_action[(state_updated, action_updated)] += \
                            (alpha * p_value * (g_value - self.value_of_state_action[(state_updated, action_updated)]))
                        # update policy
                        value_of_action_list = []
                        for action_iter in range(self.env.action_space.n):
                            value_of_action_list.append(self.value_of_state_action[(state_updated, action_iter)])
                        value_of_action_list = np.array(value_of_action_list)
                        optimal_action = np.random.choice(
                            np.flatnonzero(value_of_action_list == value_of_action_list.max()))
                        for action_iter in range(self.env.action_space.n):
                            if action_iter == optimal_action:
                                self.policies[state_updated][
                                    action_iter] = 1
                            else:
                                self.policies[state_updated][action_iter] = 0
                    break
                else:
                    if len(n_queue) == self.n + 1:
                        state_updated, action_updated, reward = n_queue.popleft()
                        gamma_temp = gamma
                        g_value = reward
                        p_value = self.policies[state_updated][action_updated] / self.behaviors_policies[
                            state_updated][action_updated]
                        for iter_n in n_queue:
                            g_value += gamma_temp * iter_n[2]
                            gamma_temp *= gamma
                            p_value *= self.policies[iter_n[0]][iter_n[1]] / self.behaviors_policies[iter_n[0]][
                                iter_n[1]]
                        # new
                        current_stat = new_state
                        action = self.select_action(current_stat)
                        new_state, reward, is_done, _ = self.env.step(action)
                        n_queue.append([current_stat, action, reward])
                        g_value += self.value_of_state_action[(current_stat, action)] * gamma_temp
                        self.value_of_state_action[(state_updated, action_updated)] += \
                            (alpha * p_value * (g_value - self.value_of_state_action[(state_updated, action_updated)]))
                        # update policy
                        value_of_action_list = []
                        for action_iter in range(self.env.action_space.n):
                            value_of_action_list.append(self.value_of_state_action[(state_updated, action_iter)])
                        value_of_action_list = np.array(value_of_action_list)
                        optimal_action = np.random.choice(
                            np.flatnonzero(value_of_action_list == value_of_action_list.max()))
                        for action_iter in range(self.env.action_space.n):
                            if action_iter == optimal_action:
                                self.policies[state_updated][action_iter] = 1
                            else:
                                self.policies[state_updated][action_iter] = 0
                    else:
                        current_stat = new_state
                        action = self.select_action(current_stat)
                        new_state, reward, is_done, _ = self.env.step(action)
                        n_queue.append([current_stat, action, reward])


if __name__ == '__main__':
    env = RandomWalk(19)
    ground_truth = []
    for i in range(0, 19):
        ground_truth.append(-1 + i / 9)
    agent = Agent(env, 1)
    agent.estimating(10000)
    estimating_value = np.zeros(19)
    for i in range(env.state_space.n):
        for j in range(env.action_space.n):
            estimating_value[i] = agent.value_of_state_action[(i, j)]
    print(estimating_value)
    plt.figure(0)
    plt.plot(estimating_value[1:-1])
    plt.plot(ground_truth[1:-1])
    plt.show()
