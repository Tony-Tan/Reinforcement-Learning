import numpy


class Space:
    def __init__(self, initial_list):
        self.list = initial_list
        self.n = len(initial_list)

    def __getitem__(self, index):
        return self.list[index]


class RandomWalk:
    def __init__(self):
        self.state_space = Space([0, 1, 2, 3, 4, 5, 6])
        self.action_space = Space([-1, +1])
        self.current_state = 3

    def reset(self):
        self.current_state = 3
        return 3

    def step(self, action, state=None):
        if state is None:
            state = self.current_state + self.action_space[action]
        else:
            state += self.action_space[action]
        if state == 0:
            return None, 0, True, {}
        elif state == 6:
            return None, 1, True, {}
        else:
            self.current_state = state
            return state, 0, False, {}


if __name__ == '__main__':
    env = RandomWalk()
    print(env.state_space[0])