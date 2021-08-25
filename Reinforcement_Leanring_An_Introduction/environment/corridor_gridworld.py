import collections
import matplotlib.pyplot as plt
import random

class Space:
    def __init__(self, n_list):
        self.n_list = n_list
        self.n = len(n_list)

    def __getitem__(self, item):
        return self.n_list[item]


class ShortCorridor:
    def __init__(self):
        self.size = 4
        self.state_space = Space([i for i in range(self.size)])
        self.action_space = Space([0, 1])
        self.current_state = 0
        self.start_position = 0

    def reset(self):
        # self.current_state = self.state_space[int(self.grid_size * self.grid_size / 2)]
        self.current_state = self.start_position
        return self.current_state

    def step(self, action):
        # punish_value = 0
        if action == 0:
            if self.current_state == 0:
                self.current_state = 0
            elif self.current_state == 1:
                self.current_state += 1
            else:
                self.current_state -= 1
            return self.current_state, -1, False, {}
        if action == 1:
            if self.current_state == 1:
                self.current_state -= 1
            else:
                self.current_state += 1
            if self.current_state == 3:
                return self.current_state, 0, True, {}
            else:
                return self.current_state, -1, False, {}


if __name__ == '__main__':
    sc = ShortCorridor()
    for i in range(10):
        if random.randint(0, 100) <50:
            action = 0
        else:
            action = 1
        state, r, is_done, _ = sc.step(action)
        print('action:{} state:{} reward:{} is_done:{} '.format(action, state, r, is_done))
