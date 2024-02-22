from abc_rl.exploration import *


class EpsilonGreedy(Exploration):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        super().__init__()

    def __call__(self, values: np.ndarray):
        optimal_action = np.random.choice(
            np.flatnonzero(values == np.max(values)))
        if random.randint(0, 10000000) < self.epsilon * 10000000:
            return random.randint(0, len(values) - 1)
        else:
            return optimal_action


class DecayingEpsilonGreedy(Exploration):
    def __init__(self, max_epsilon=1, min_epsilon=0.1, total_step=100000):
        super().__init__()
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = (max_epsilon - min_epsilon)/total_step
        self.decaying_counter = 0

    def __call__(self, values: np.ndarray):
        self.decaying_counter += 1
        self.epsilon = max(self.max_epsilon - self.decaying_counter * self.decay_rate, self.min_epsilon)
        optimal_action = np.random.choice(
            np.flatnonzero(values == np.max(values)))
        if random.randint(0, 10000000) < self.epsilon * 10000000:
            return random.randint(0, len(values) - 1)
        else:
            return optimal_action
