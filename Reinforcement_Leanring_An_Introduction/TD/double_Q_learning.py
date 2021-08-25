import collections

import matplotlib.pyplot as plt
import numpy as np
from random_walk import RandomWalk


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Agent:
    def __init__(self, env, epsilon=0.4, initial_value=0):
        self.env = env
        self.epsilon = epsilon
        self.value_state_action_1 = collections.defaultdict(lambda: initial_value)
        self.value_state_action_2 = collections.defaultdict(lambda: initial_value)
        self.policies = collections.defaultdict(constant_factory(env.action_space.n))

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def q_control(self, number_of_episodes, alpha=0.1, gamma=1, epsilon=0.4, only_evaluation=False):
        for _ in range(number_of_episodes):
            state = self.env.reset()
            while True:
                action = self.select_action(state)
                new_state, reward, is_done, _ = self.env.step(action)
                # next state-action return
                random_switch = np.random.choice([0, 1])
                q_state_next = []
                if random_switch:
                    for action_iter in range(self.env.action_space.n):
                        q_state_next.append(self.value_state_action_1[(new_state, action_iter)])
                    q_state_next = max(q_state_next)
                    q_state_current = self.value_state_action_2[(state, action)]
                    self.value_state_action_2[(state, action)] = \
                        q_state_current + alpha * (reward + gamma * q_state_next - q_state_current)
                else:
                    for action_iter in range(self.env.action_space.n):
                        q_state_next.append(self.value_state_action_2[(new_state, action_iter)])
                    q_state_next = max(q_state_next)
                    q_state_current = self.value_state_action_1[(state, action)]
                    self.value_state_action_1[(state, action)] = \
                        q_state_current + alpha * (reward + gamma * q_state_next - q_state_current)

                if not only_evaluation:
                    # control epsilon-greedy
                    value_of_action_list = []
                    for action_iter in range(self.env.action_space.n):
                        value_of_action_list.append((self.value_state_action_1[(state, action_iter)] +
                                                     self.value_state_action_2[(state, action_iter)]) / 2)
                    value_of_action_list = np.array(value_of_action_list)
                    optimal_action = np.random.choice(
                        np.flatnonzero(value_of_action_list == value_of_action_list.max()))
                    for action_iter in range(self.env.action_space.n):
                        if action_iter == optimal_action:
                            self.policies[state][action_iter] = 1 - epsilon + epsilon / self.env.action_space.n
                        else:
                            self.policies[state][action_iter] = epsilon / self.env.action_space.n
                if is_done:
                    break
                state = new_state


if __name__ == '__main__':
    env = RandomWalk()
    agent = Agent(env, initial_value=0.0)
    agent.q_control(100, alpha=0.1, gamma=0.9, only_evaluation=False)
    value_list = []
    ground_truth = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    for state_i in range(env.state_space.n):
        value = 0
        for action_i in range(env.action_space.n):
            value += \
                agent.value_state_action_1[(state_i, action_i)] * \
                agent.policies[state_i][action_i]
        value_list.append(value)
        print('value of state %d is %f' % (env.state_space[state_i], value))
    plt.plot(env.state_space[1:-1], value_list[1:-1], c='b')
    plt.plot(env.state_space[1:-1], ground_truth, c='r')
    plt.show()
