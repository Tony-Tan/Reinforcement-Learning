# Baird, Leemon. “Residual Algorithms: Reinforcement Learning with Function Approximation.”
# In Machine Learning Proceedings 1995, 30–37. Elsevier, 1995.
# https://doi.org/10.1016/B978-1-55860-377-6.50013-X.

import numpy as np
import random
import matplotlib.pyplot as plt


class StarProblem:
    def __init__(self, number_of_states_):
        self.number_of_states = number_of_states_
        self.state_space = range(1, self.number_of_states + 1)
        self.last_state = self.state_space[-1]

    def next_state(self, current_state_):
        if current_state_ != self.last_state:
            return self.last_state, 0, False, {}
        else:
            return self.last_state, 0, True, {}


def direct_residual_gradient_experiment(repeat_times, gamma, learning_rate, learning_rate_decay=0.9, method='direct'):
    # env_star_problem = StarProblem(6)
    weight_list = []
    number_of_states = 6
    parameter_num = number_of_states + 1
    weights = np.random.randint(0, 100, parameter_num)
    # weights = np.ones(parameter_num)
    weights[-1] = 100000
    last_state = number_of_states
    reward = 0
    for repeat_i in range(repeat_times):
        delta_weight = 0
        for state_i in range(1, number_of_states+1):
            nebula_w_state_i = np.zeros(parameter_num)
            if state_i != last_state:
                nebula_w_state_i[0] = 1.
                nebula_w_state_i[state_i] = 2
                value_v = weights[0] + weights[state_i]
            else:
                nebula_w_state_i[0] = 2.
                nebula_w_state_i[state_i] = 1.
                value_v = weights[0] * 2 + weights[state_i]
            value_v_next = weights[0] * 2 + weights[last_state]
            if method == 'direct':
                delta_weight -= nebula_w_state_i * (reward + gamma * value_v_next - value_v)
            elif method == 'residual_gradient':
                nebula_w_next_state = np.zeros(parameter_num)
                nebula_w_next_state[0] = 2.
                nebula_w_next_state[last_state] = 1
                delta_weight += (gamma * nebula_w_next_state - nebula_w_state_i) * (
                            reward + gamma * value_v_next - value_v)
        delta_weight *= -learning_rate
        learning_rate *= learning_rate_decay
        weights = weights + delta_weight
        if repeat_i % 20 == 0:
            weight_list.append(weights)
    return weight_list


if __name__ == '__main__':
    method = 'residual_gradient'
    weights_process = direct_residual_gradient_experiment(2000, 0.9, 0.1, 1, method)
    plt.figure(figsize=(16, 9))
    plt.title(method)
    weights_mat = np.array(weights_process)
    for i in range(len(weights_mat[0])):
        plt.plot(weights_mat[:, i], label='$w_'+str(i)+'$')
    plt.legend()
    plt.show()
    print(weights_process)
