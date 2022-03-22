# environment to test natural policy gradient
import numpy as np
# state i: 0, j:1
# action a_1: -1, a_2: 1

#       action     -1        1
# state
#   0              1         0
#   1              2         0


class TwoStatesMDP:
    def __init__(self, initial_distribution):
        self.action_space = [-1, 1]
        self.state_space = [0, 1]
        self.initial_distribution = initial_distribution
        self.current_state = -1
        pass

    def reset(self):
        if np.random.rand() > self.initial_distribution[0]:
            self.current_state = self.state_space[1]
        else:
            self.current_state = self.state_space[0]
        return self.current_state

    def step(self, action):
        if action == -1:
            if self.current_state == 0:
                return self.current_state, 1, False, {}
            else:
                return self.current_state, 2, False, {}
        else:
            self.current_state = (self.current_state+1) % 2
            return self.current_state, 0, False, {}
