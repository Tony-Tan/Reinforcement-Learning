# this environment comes from example 13.1 on page 323 of
# the second edition on Reinforcement learning: an introduction
# Richard S. Sutton and Andrew G.Barto
#
import numpy as np


class ShortCorridorWithSwitchedActions:
    def __init__(self):
        self.current_position = 0
        self.action_space = [0, 1]

    def reset(self):
        self.current_position = 0
        return self.current_position

    def step(self, action_):
        if self.current_position == 0:
            if action_ == 0:
                return self.current_position, -1, False, {}
            elif action_ == 1:
                self.current_position += 1
                return self.current_position, -1, False, {}
        elif self.current_position == 1:
            if action_ == 0:
                self.current_position += 1
                return self.current_position, -1, False, {}
            elif action_ == 1:
                self.current_position -= 1
                return self.current_position, -1, False, {}
        elif self.current_position == 2:
            if action_ == 0:
                self.current_position -= 1
                return self.current_position, -1, False, {}
            elif action_ == 1:
                self.current_position += 1
                return self.current_position, 0, True, {}


if __name__ == '__main__':
    env = ShortCorridorWithSwitchedActions()