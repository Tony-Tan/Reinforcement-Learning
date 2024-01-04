import numpy as np
import random


def epsilon_greedy(values: np.ndarray, epsilon: float, ) -> int:
    optimal_action = np.random.choice(
        np.flatnonzero(values == np.max(values)))
    if random.randint(0, 10000000) < epsilon * 10000000:
        return random.randint(0, len(values) - 1)
    else:
        return optimal_action
