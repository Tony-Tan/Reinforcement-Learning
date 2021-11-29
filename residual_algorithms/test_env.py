import numpy as np


class TheStarProblem:
    def __init__(self, num_of_nodes_):
        self.num_of_nodes = num_of_nodes_
        self.last_state = num_of_nodes_ - 1
        self.current_state = 0
        pass

    def reset(self):
        self.current_state = np.random.randint(0, self.num_of_nodes - 1)
        pass

    def step(self):
        if self.current_state != self.last_state:
            self.current_state = self.last_state
            return self.current_state, 0, False, {}
        else:
            return self.current_state, 0, True, {}
