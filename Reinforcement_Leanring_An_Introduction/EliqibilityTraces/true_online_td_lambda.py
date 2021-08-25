import collections

import matplotlib.pyplot as plt
import numpy as np
from environment.random_walk_19_states import RandomWalk


class LinearFunction:
    def __init__(self):
        self.weight = np.zeros(2)

    def __call__(self, x):
        return self.weight[0] * x + self.weight[1]

    def derivative(self, x):
        return np.array([x, 1.])


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Agent:
    def __init__(self, env):
        self.env = env
        self.policies = collections.defaultdict(constant_factory(2))
        self.value_of_state = LinearFunction()

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def estimating(self, iteration_times, lambda_coe=1., alpha=0.1, gamma=0.9, epsilon=0.3):
        for iter_time in range(iteration_times):
            eligibility_trace = np.zeros(2)
            v_value_old = 0
            current_state = self.env.reset()
            current_action = self.select_action(current_state)
            next_state, reward, is_done, _ = self.env.step(current_action)
            while True:
                x = np.array([current_state / 19., 1])
                if next_state is None:
                    x_next = np.array([0, 0])
                else:
                    x_next = np.array([next_state / 19., 1])

                v_value = self.value_of_state.weight.transpose().dot(x)
                v_value_next = self.value_of_state.weight.transpose().dot(x_next)

                delta = reward + gamma * v_value_next - v_value
                eligibility_trace = \
                    gamma * lambda_coe * eligibility_trace + \
                    (1. - alpha * gamma * lambda_coe * eligibility_trace.transpose().dot(x)) * x
                self.value_of_state.weight += alpha * ((delta + v_value - v_value_old) * eligibility_trace -
                                                       (v_value - v_value_old) * x)
                v_value_old = v_value_next

                # update policy
                # value_of_action_list = []
                # for action_iter in range(self.env.action_space.n):
                #     value_of_action_list.append(self.value_of_state[current_state])
                # value_of_action_list = np.array(value_of_action_list)
                # optimal_action = np.random.choice(
                #     np.flatnonzero(value_of_action_list == value_of_action_list.max()))
                # for action_iter in range(self.env.action_space.n):
                #     if action_iter == optimal_action:
                #         self.policies[current_state][
                #             action_iter] = 1 - epsilon + epsilon / self.env.action_space.n
                #     else:
                #         self.policies[current_state][action_iter] = epsilon / self.env.action_space.n

                if is_done:
                    break
                current_state = next_state
                current_action = self.select_action(current_state)
                next_state, reward, is_done, _ = self.env.step(current_action)


if __name__ == '__main__':
    env = RandomWalk(19)
    agent = Agent(env)
    agent.estimating(2000, 0.8, 0.01, 0.9)
    value_of_state = []
    for i_state in range(1, env.state_space.n - 1):
        value_of_state.append(agent.value_of_state(i_state / 19.))
    plt.plot(value_of_state)
    plt.show()
