from environment.basic_classes import Space
from environment.environment_template import ENV


# Example 3.5: Gridworld Figure 3.2 left
#
#       x  A  x  B  x
#       x  x  x  x  x
#       x  x  x  B' x
#       x  A' x  x  x
#       x  x  x  x  x
#
# reward of all actions from A is +10 and its next state is A'
# reward of all actions from B is +5 and its next state is B'
# action could be 0:left, 1:right, 2:up, and 3:down
# actions taking the agent out of grid, they hold its position and get a reward of -1
# and other actions

class GridWorld(ENV):
    def __init__(self):
        super().__init__()
        self.__row = 5
        self.__col = 5
        # 0:left, 1:right, 2:up, and 3:down
        self.action_space = Space([i for i in range(4)])
        self.state_space = Space([i for i in range(self.__row * self.__col)])
        self.current_state = 0

    def reset(self):
        self.current_state = 0

    def set_current_state(self, state_):
        self.current_state = state_

    def step(self, action_):
        if self.current_state == 1:
            self.current_state = 21
            return self.current_state, 10., False, {}
        elif self.current_state == 3:
            self.current_state = 13
            return self.current_state, 5., False, {}
        elif action_ == 0:
            if self.current_state % self.__col == 0:
                return self.current_state, -1., False, {}
            else:
                self.current_state -= 1
                return self.current_state, 0., False, {}
        elif action_ == 1:
            if (self.current_state + 1) % self.__col == 0:
                return self.current_state, -1., False, {}
            else:
                self.current_state += 1
                return self.current_state, 0., False, {}
        elif action_ == 2:
            if self.current_state < self.__col:
                return self.current_state, -1., False, {}
            else:
                self.current_state -= self.__col
                return self.current_state, 0., False, {}
        elif action_ == 3:
            if self.current_state >= (self.__row - 1) * self.__col:
                return self.current_state, -1., False, {}
            else:
                self.current_state += self.__col
                return self.current_state, 0., False, {}
