# Gambler's problem:
# flip a coin, if it comes up head, the gambler wins as many dollars as he has staked on that flip
# if it is tail, he loses his stack
# terminate: win 100 dollars or lose all money
# hind:
# undiscounted, episodic, finite MDP
# states \in \{1,2,...,99\}
# actions \in \{0,1,..., min(s,100-s)\}
# reward: +1 when gambler reaches his goal; others are 0

import numpy as np
from matplotlib import pyplot as plt
import copy

def value_iteration(p, theta, state_space, gamma):
    # record results for the 1,2,3,32 and final sweep
    temple_result_to_record = [1,2,3,32]
    state_space_list = []
    state_value = np.zeros(100)
    policy = np.zeros(100)
    loop_i = 1
    while True:
        delta = 0.0
        for s in state_space:
            v = state_value[s]
            state_value_of_every_action = np.zeros(s + 1)
            for action_i in range(s):
                if action_i + s >= 100:
                    state_value_of_every_action[action_i] = p * 1. + \
                                                            (1 - p) * (0 + gamma * state_value[s - action_i])
                else:
                    state_value_of_every_action[action_i] = p * (0 + gamma * state_value[s + action_i]) + \
                                                            (1 - p) * (0 + gamma * state_value[s - action_i])

            state_value[s] = np.max(state_value_of_every_action)
            delta = np.max([delta, np.abs(state_value[s] - v)])
        if delta < theta:
            state_space_list.append(copy.deepcopy(state_value))
            break
        if loop_i in temple_result_to_record:
            state_space_list.append(copy.deepcopy(state_value))
        loop_i += 1


    for s in state_space:
        state_value_of_every_action = np.zeros(s + 1)
        for action_i in range(s):
            if action_i + s >= 100:
                state_value_of_every_action[action_i] = p * 1. + \
                                                        (1 - p) * (0 + gamma * state_value[s - action_i])
            else:
                state_value_of_every_action[action_i] = p * (0 + gamma * state_value[s + action_i]) + \
                                                        (1 - p) * (0 + gamma * state_value[s - action_i])

        policy[s] = np.argmax(state_value_of_every_action)

    return state_space_list, policy


if __name__ == '__main__':
    state_space = range(100)
    state_value_list, policy = value_iteration(0.4, 1e-100, state_space, 0.99)
    plt.figure(0)
    for state_value_i in state_value_list:
        plt.plot(state_value_i, linewidth=1)
    plt.figure(1)
    plt.plot(policy)
    plt.show()
