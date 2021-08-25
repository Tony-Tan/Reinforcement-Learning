import collections

import matplotlib.pyplot as plt
import numpy as np
from environment.random_walk_1000_states import RandomWalk1000


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class StateAggregation:
    def __init__(self, min_state, max_state, aggregation_size):
        self.min_state = min_state
        self.max_state = max_state
        self.aggregation_size = aggregation_size
        self.aggregation_num = int((max_state - min_state) / aggregation_size) + 1
        if (max_state - min_state) % aggregation_size == 0:
            self.aggregation_num -= 1
        self.weight = np.zeros(self.aggregation_num)

    def __call__(self, x):
        current_position = int(x / self.aggregation_size)
        return self.weight[current_position]

    def derivation(self, x):
        derivative = np.zeros(self.aggregation_num)
        current_position = int(x / self.aggregation_size)
        derivative[current_position] = 1.0
        return derivative


class Agent:
    def __init__(self, env, n, min_state, max_state, aggregation_size):
        self.env = env
        self.n = n
        self.value_state = StateAggregation(min_state, max_state, aggregation_size)
        self.policies = collections.defaultdict(constant_factory(env.action_space.n))

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def SGTD_n_app(self, number_of_episodes, learning_rate, state_num=1000, gamma=1.):
        mu = np.zeros(state_num)
        for _ in range(number_of_episodes):
            n_queue = collections.deque()
            state = self.env.reset()
            action = self.select_action(state)
            mu[state] += 1.0
            new_state, reward, is_done, _ = self.env.step(action)
            while True:
                mu[new_state] += 1.0
                n_queue.append([new_state, reward, is_done])
                if is_done:
                    while len(n_queue) != 0:
                        state_updated, _, _ = n_queue.popleft()
                        if state_updated is None:
                            break
                        gamma_temp = 1.0
                        g_value = 0.0
                        for iter_n in n_queue:
                            g_value += gamma_temp * iter_n[1]
                            gamma_temp *= gamma

                        if new_state is not None:
                            self.value_state.weight += learning_rate * (reward +
                                                                        gamma * self.value_state(new_state) -
                                                                        self.value_state(state_updated)) * delta_value
                        else:
                            self.value_state.weight += learning_rate * (
                                        reward - self.value_state(state_updated)) * delta_value
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
                        new_state, reward, is_done, _ = env.step(action_next)
                        if new_state is not None:
                            g_value += (reward * gamma_temp + self.value_state(new_state))
                        else:
                            g_value += reward * gamma_temp
                        delta_value = self.value_state.derivation(state_updated)
                        if new_state is not None:
                            self.value_state.weight += learning_rate * (reward +
                                                                        gamma * self.value_state(new_state) -
                                                                        self.value_state(state_updated)) * delta_value
                        else:
                            self.value_state.weight += learning_rate * (
                                        reward - self.value_state(state_updated)) * delta_value
                    else:
                        action_next = self.select_action(new_state)
                        new_state, reward, is_done, _ = env.step(action_next)

        return mu


if __name__ == '__main__':
    env = RandomWalk1000()
    agent = Agent(env, 0, 0, 1000, 100)
    mu = agent.SGTD_n_app(10000, 1e-2, gamma=0.99)
    mu = mu / np.sum(mu)
    x = np.arange(1, 999, 1.)
    y = np.arange(1, 999, 1.)
    # for i in range(1, x.size, 2):
    #     y[i-1] = agent.value_state(x[i-1] + 50)
    #     y[i] = agent.value_state(x[i] - 50)
    print(agent.value_state.weight)
    for i in range(x.size):
        y[i] = agent.value_state(x[i])
    plt.figure(0)
    plt.plot(x, y)
    plt.figure(1)
    plt.bar(range(len(mu)), mu, color='gray', width=1)
    plt.show()
