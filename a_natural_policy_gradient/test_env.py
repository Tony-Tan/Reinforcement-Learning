class TwoStateMDP:
    def __init__(self):
        self.state_space = [0,1]
        self.action_space = [0,1]
        self.current_state = 0
        pass

    def reset(self):
        self.current_state = 0
        pass

    def step(self, action):
        if self.current_state == 0:
            if action == 0:
                return self.current_state, 1, False, {}
            elif action == 1:
                self.current_state = 1
                return self.current_state, 0, False, {}
        elif self.current_state == 1:
            if action == 0:
                return self.current_state, 2, False, {}
            elif action == 1:
                self.current_state = 0
                return self.current_state, 0, False, {}

