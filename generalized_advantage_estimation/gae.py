from core.rl_elements import *
from core.basic import *


class GAE:
    def __init__(self, lambda_value=0.95, gamma=0.99):
        self.lambda_value = lambda_value
        self.gamma = gamma

    def __call__(self, reward_np: np.ndarray, state_value_np: np.ndarray, termination_np: np.ndarray ):
        if reward_np.shape != state_value_np.shape != termination_np.shape:
            raise ShapeNotMatch
        delta_np = reward_np - state_value_np
        delta_np[:-1] = (delta_np[:-1] + self.gamma * state_value_np[1:]) * (1. - termination_np[:-1])
        # generate advantage
        n = len(termination_np)
        advantage = np.zeros((n, 1), dtype=np.float32)
        advantage[n - 1] = 0
        for i in reversed(range(n - 1)):
            if termination_np[i] == 1:
                advantage[i] = delta_np[i]
            else:
                advantage[i] = delta_np[i] + self.gamma * self.lambda_value * advantage[i + 1]
        return advantage
