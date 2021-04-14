# the first algorithm is a kind of finite-difference algorithm and
# employed to:
# 1. Regenerative process
# 2. Markov Chain
# the value of each state x_i(0<i<N) = int(100*exp(-(theta-N/2))/(N*5))
# non-associative sequence which mean the next state does not depend on current state
# the probability of each state x_i = normalized(exp(-(theta-i - 4)^2/2))

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

STATE_NUM = 8


# probability and value functions
def probability_assign(theta):
    probability = np.ones(STATE_NUM) / STATE_NUM
    for i in range(STATE_NUM):
        probability[i] = np.exp(-(theta - i/STATE_NUM) ** 2 / 0.5)
    probability /= probability.sum()
    return probability


def probability_assign2d(theta):
    probability = np.ones([STATE_NUM, STATE_NUM])
    for i in range(STATE_NUM):
        for j in range(STATE_NUM):
            probability[i][j] = np.exp(-((i-theta-STATE_NUM/2.)**2/(j+1) + (j-theta-STATE_NUM/2.)**2/(i+1)) / 5)
        probability[i] /= np.sum(probability[i])
    return probability


def value_list(theta):
    value = np.ones(STATE_NUM) / STATE_NUM
    for i in range(STATE_NUM):
        value[i] = -np.arctan(theta * i) - theta
    return value


def step_size(n):
    return .01 / (n + 1)


def c_value(n):
    return np.power(n + 1, 0.25)


def calculate_y_t(sequence):
    Y = 0
    # assuming state at state = the first state and end with state = the first state
    # cut out the first loop
    for s, v in sequence:
        Y += v
    t = len(sequence)
    return Y, t


def SequenceGenerator(theta):
    probability_array = probability_assign(theta)
    value_array = value_list(theta)
    start_state = np.random.choice(STATE_NUM, 1, p=probability_array)[0]
    sequence = [[start_state, value_array[start_state]]]
    next_state = np.random.choice(STATE_NUM, 1, p=probability_array)[0]
    while next_state != start_state:
        sequence.append([next_state, value_array[next_state]])
        next_state = np.random.choice(STATE_NUM, 1, p=probability_array)[0]
    return sequence


def MarkovChainGenerator(theta, start_state=0):
    probability_array2d = probability_assign2d(theta)
    value_array = value_list(theta)
    current_state = start_state
    sequence = [[current_state, value_array[current_state]]]
    next_state = np.random.choice(STATE_NUM, 1, p=probability_array2d[current_state])[0]
    while next_state != start_state:
        current_state = next_state
        sequence.append([current_state, value_array[current_state]])
        next_state = np.random.choice(STATE_NUM, 1, p=probability_array2d[current_state])[0]
    return sequence


def AlgorithmA(sequence_generator_type, step_num=10000):
    if sequence_generator_type == 'mc':
        generator = MarkovChainGenerator
    else:
        generator = SequenceGenerator
    theta = np.pi * (np.random.rand() - 0.5)
    y_t = []
    theta_list = []
    for i in range(step_num):
        c = c_value(i)
        sq = generator(theta)
        sq_c = generator(np.arctan(np.tan(theta) + c))
        Y, t = calculate_y_t(sq)
        Y_c, t_c = calculate_y_t(sq_c)
        if t == 0 or t_c == 0:
            i -= 1
            print('not available sequence')
            continue
        eta = (Y_c * t - Y * t_c)*c
        # alpha = step_size(i)
        # theta -= alpha * eta
        theta = np.arctan(np.tan(theta) - (1/(i+1.)) * eta)
        if i % 100 == 0:
            y_t.append(Y / t)
            theta_list.append(theta)
    plt.figure(0)
    plt.subplot(321)
    plt.plot(value_list(theta), c='r', label='value')
    plt.legend()
    plt.subplot(322)
    plt.plot(probability_assign(theta), c='b', label='probability')
    plt.legend()
    plt.subplot(312)
    plt.plot(y_t, c='g', label='Y/t')
    plt.legend()
    plt.subplot(313)
    plt.plot(theta_list, c='y', label='$\\theta$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # sg = SequenceGenerator(1000)
    AlgorithmA('sg')
