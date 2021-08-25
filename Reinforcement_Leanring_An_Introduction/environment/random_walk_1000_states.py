import numpy
import random

class Space:
    def __init__(self, initial_list):
        self.list = initial_list
        self.n = len(initial_list)

    def __getitem__(self, index):
        return self.list[index]


class RandomWalk1000:
    def __init__(self):
        self.state_space = Space([i for i in range(1, 1001)])
        self.action_space = Space([-1, +1])
        self.current_state = int(self.state_space.n/2)

    def reset(self):
        self.current_state = int(self.state_space.n/2)
        return self.current_state

    def step(self, action):
        step_size = random.randint(1, 100)
        new_position = self.current_state + self.action_space[action]*step_size

        if new_position <= 0:
            return None, -1, True, {}
        elif new_position >= 1000:
            return None, 1, True, {}
        else:
            self.current_state = new_position
            return self.current_state, 0, False, {}


if __name__ == '__main__':
    env = RandomWalk1000()
    current_state = env.reset()
    for i in range(500):
        action = random.randint(0, 1)
        new_state, reward, is_done, _ = env.step(action)
        if new_state is not None:
            print('current state: %3d ----> action: %d ----> reward: %d ----> new state: %d'
                  % (current_state, action, reward, new_state))
            current_state = new_state
        else:
            print('current state: %3d ----> action: %d ----> reward: %d ----> new state: termination'
                  % (current_state, action, reward))
            current_state = env.reset()