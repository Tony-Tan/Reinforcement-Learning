# Example 4.1: 4x4 gridworld shown below
#       x  x  x  x
#       x  x  x  x
#       x  x  x  x
#       x  x  x  x
# R_t = -1 for all transitions
# reward of all actions from A is +10 and its next state is A'
# reward of all actions from B is +5 and its next state is B'
# action could be 0:left, 1:right, 2:up, and 3:down
from environment.basic_classes import Space
from environment.environment_template import ENV

class GridWorld(ENV):
    def __init__(self):
        super().__init__()
        self.__row = 4
        self.__col = 4
        # 0:left, 1:right, 2:up, and 3:down
        self.action_space = Space([i for i in range(4)])
        self.state_space = Space([i for i in range(self.__row * self.__col)])
        self.current_state = 0

    def reset(self):
        self.current_state = 0

    def set_current_state(self, state_):
        self.current_state = state_

    def step(self, action_):
        if self.current_state == 0 or self.current_state == self.__col * self.__row - 1:
            return self.current_state, 0, True, {}
        if action_ == 0:
            if self.current_state % self.__col != 0:
                self.current_state -= 1
        elif action_ == 1:
            if (self.current_state + 1) % self.__col != 0:
                self.current_state += 1
        elif action_ == 2:
            if self.current_state >= self.__col:
                self.current_state -= self.__col
        elif action_ == 3:
            if self.current_state < (self.__row - 1) * self.__col:
                self.current_state += self.__col
        return self.current_state, -1., False, {}


if __name__ == '__main__':
    gw4x4 = GridWorld()
    gw4x4.reset()

