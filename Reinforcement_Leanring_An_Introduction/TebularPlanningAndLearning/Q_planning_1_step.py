import collections
import random

import matplotlib.pyplot as plt
import numpy as np
from environment.random_walk import RandomWalk


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Agent:
    def __init__(self, env, epsilon=0.4, initial_value=0):
        self.env = env
        self.epsilon = epsilon
        self.value_state_action = collections.defaultdict(lambda: initial_value)
        self.policies = collections.defaultdict(constant_factory(env.action_space.n))

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def q_planning(self, number_of_episodes, alpha=0.1, gamma=1):
        for _ in range(number_of_episodes):
            state_selected = random.choice(range(self.env.state_space.n))
            action_selected = random.choice(range(self.env.action_space.n))
            new_state, reward, is_done, _ = self.env.step(action_selected, state_selected)
            value_of_action_list = []
            for action_iter in range(self.env.action_space.n):
                value_of_action_list.append(self.value_state_action[(new_state, action_iter)])
            value_of_action_list = np.array(value_of_action_list)
            optimal_action = \
                np.random.choice(np.flatnonzero(value_of_action_list == value_of_action_list.max()))
            self.value_state_action[(state_selected, action_selected)] += \
                alpha * (reward + gamma * self.value_state_action[(new_state, optimal_action)] -
                         self.value_state_action[(state_selected, action_selected)])


if __name__ == '__main__':
    env = RandomWalk()
    agent = Agent(env, initial_value=0.0)
    agent.q_planning(100000, alpha=0.1, gamma=0.9)
    value_list = []
    ground_truth = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    for state_i in range(env.state_space.n):
        value = 0
        for action_i in range(env.action_space.n):
            value += \
                agent.value_state_action[(state_i, action_i)] * \
                agent.policies[state_i][action_i]
        value_list.append(value)
        print('value of state %d is %f' % (env.state_space[state_i], value))
    plt.plot(env.state_space[1:-1], value_list[1:-1], c='b')
    plt.plot(env.state_space[1:-1], ground_truth, c='r')
    plt.show()
