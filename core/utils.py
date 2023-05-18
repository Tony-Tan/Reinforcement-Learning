import numpy as np
import random
from datetime import datetime
import argparse
import time
import os


def epsilon_greedy(values: np.ndarray, epsilon: float, ) -> int:
    optimal_action = np.random.choice(
        np.flatnonzero(values == np.max(values)))
    if random.randint(0, 10000000) < epsilon * 10000000:
        return random.randint(0, len(values) - 1)
    else:
        return optimal_action


def standard_info_print(info: str):
    print("[%s]: %s" % (datetime.now(), info))


def script_args(args_list: list, description: str):
    parser = argparse.ArgumentParser(description=description)
    for arg_i in args_list:
        # '--actor_hidden_layer', default=[400, 300], nargs='+', type=int,
        #                     help='acot hidden perceptron size'
        parser.add_argument(arg_i[0], default=arg_i[1], type=arg_i[2], help=arg_i[3])
    args = parser.parse_args()
    return args


def discount_cumulate(reward_array: np.ndarray, termination_array: np.ndarray, discount_rate=0.99):
    """
    :param reward_array: nx1 float matrix
    :param discount_rate: float number
    :param termination_array: termination array, nx1 int matrix, 1 for True and 0 for False,
           discounting restart from the termination element which was 1
    :return:
    """
    step_size = len(reward_array)
    g = np.zeros(step_size, dtype=np.float32)
    g[-1] = reward_array[-1]
    for step_i in reversed(range(step_size - 1)):
        reward_i = reward_array[step_i]
        if termination_array is not None:
            if termination_array[step_i] == 1.0:
                g[step_i] = reward_i
            else:
                g[step_i] = reward_i + discount_rate * g[step_i + 1]
        else:
            g[step_i] = reward_i + discount_rate * g[step_i + 1]
    return g.reshape((-1, 1))

