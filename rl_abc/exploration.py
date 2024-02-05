import numpy as np
import random
from utils.commons import Logger


class Exploration:
    def __init__(self, logger: Logger):
        logger('Exploration method: \'{}\' is initialized'.format(self.__class__.__name__))
        pass

    def __call__(self, **kwargs):
        pass


class EpsilonGreedy(Exploration):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def __call__(self, values: np.ndarray, epsilon: float):
        optimal_action = np.random.choice(
            np.flatnonzero(values == np.max(values)))
        if random.randint(0, 10000000) < epsilon * 10000000:
            return random.randint(0, len(values) - 1)
        else:
            return optimal_action


if __name__ == '__main__':
    logger_ = Logger()
    ep_greedy = EpsilonGreedy(logger_)