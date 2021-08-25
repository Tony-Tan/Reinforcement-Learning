class Space:
    def __init__(self, initial_list):
        self.list = initial_list
        self.n = len(initial_list)

    def __getitem__(self, index):
        return self.list[index]


class RandomWalk:
    def __init__(self, n):
        self.start_state = int(n/2)
        self.state_space = Space([i for i in range(n)])
        self.action_space = Space([0,1])
        self.current_state = 0

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        if action == 0:
            next_state = self.current_state - 1
            if next_state == 0:
                return next_state, -1, True, {}
            else:
                self.current_state -= 1
                return self.current_state, 0, False, {}
        if action == 1:
            next_state = self.current_state + 1
            if next_state == self.state_space.n - 1:
                return next_state, 1, True, {}
            else:
                self.current_state += 1
                return self.current_state, 0, False, {}