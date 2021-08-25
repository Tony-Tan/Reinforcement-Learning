import collections
import matplotlib.pyplot as plt
import random
import math
from basic_classes import Space


def bound(x, bound_value):
    if x > bound_value[1]:
        return bound_value[1]
    elif x < bound_value[0]:
        return bound_value[0]
    else:
        return x


class MountainCar:
    def __init__(self):
        self.position_bound = [-1.2, 0.5]
        self.velocity_bound = [-0.07, 0.07]
        self.initial_position_range = [-.6, -.4]
        self.initial_velocity = 0
        self.current_position = 0
        self.current_velocity = 0
        self.action_space = Space([-1, 0, 1])

    def reset(self):
        self.current_position = random.uniform(self.initial_position_range[0], self.initial_position_range[1])
        self.current_velocity = 0
        return [self.current_position, self.current_velocity]

    def step(self, action):
        self.current_velocity = self.current_velocity + self.action_space[action]*0.001 - \
                                0.0025*math.cos(3*self.current_position)
        self.current_velocity = bound(self.current_velocity, self.velocity_bound)

        self.current_position = self.current_position + self.current_velocity
        self.current_position = bound(self.current_position, self.position_bound)
        if self.current_position == self.position_bound[0]:
            self.current_velocity = 0
        if self.current_position == self.position_bound[1]:
            return [self.current_position, self.current_velocity], 0, True, {}
        else:
            return [self.current_position, self.current_velocity], -1, False, {}


if __name__ == '__main__':
    action_taken = 2
    env = MountainCar()
    env.current_velocity = 0
    env.current_position = -1.2
    position_list = []
    velocity_list = []
    for step_num in range(200):
        action_taken = 2
        new_state, reward, is_done, _ = env.step(action_taken)
        if is_done:
            break
        position_list.append(new_state[0])
    plt.plot(position_list, label='position')
    # plt.plot(velocity_list, label='velocity')
    plt.legend()
    plt.show()