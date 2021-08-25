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
        self.value_state_action = collections.defaultdict(lambda: initial_value)
        self.policies = collections.defaultdict(constant_factory(env.action_space.n))

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def sarsa_evaluation_control(self, number_of_episodes, alpha=0.1, gamma=1, epsilon=0.1, only_evaluation=False):
        for _ in range(number_of_episodes):
            state = self.env.reset()
            while True:
                action = self.select_action(state)
                new_state, reward, is_done, _ = self.env.step(action)
                # next state-action return
                if not is_done:
                    action_next = self.select_action(new_state)
                    q_state_next = self.value_state_action[(new_state, action_next)]
                    q_state_current = self.value_state_action[(state, action)]
                    self.value_state_action[(state, action)] = \
                        q_state_current + alpha * (reward + gamma * q_state_next - q_state_current)
                else:
                    q_state_current = self.value_state_action[(state, action)]
                    self.value_state_action[(state, action)] = \
                        q_state_current + alpha * (reward - q_state_current)
                if not only_evaluation:
                    # control epsilon-greedy
                    value_of_action_list = []
                    for action_iter in range(self.env.action_space.n):
                        value_of_action_list.append(self.value_state_action[(state, action_iter)])
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
    agent = Agent(env, initial_value=0.5)
    agent.sarsa_evaluation_control(10000, alpha=0.1, only_evaluation=True)
    print(agent.value_state_action)
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
