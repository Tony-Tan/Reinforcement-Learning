import collections

import matplotlib.pyplot as plt
import numpy as np
from environment.mountain_car import MountainCar


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class LinearFunction:
    def __init__(self, n):
        self.n = n
        self.weight = np.zeros(n)

    def __call__(self, x):
        sum = 0
        for i in x:
            sum += self.weight[i]
        return sum


class Agent:
    def __init__(self, environment_):
        self.env = environment_
        self.tiling_block_num = 8
        self.tiling_num = 8
        self.value_of_state_action = LinearFunction(
            self.tiling_num * self.tiling_block_num * self.tiling_block_num * 10)
        # parameters for feature extraction
        width = self.env.position_bound[1] - self.env.position_bound[0]
        height = self.env.velocity_bound[1] - self.env.velocity_bound[0]
        self.block_width = width / (self.tiling_block_num - 1)
        self.block_height = height / (self.tiling_block_num - 1)
        self.width_step = self.block_width / self.tiling_num
        self.height_step = self.block_height / self.tiling_num
        self.tiling_dict = {}

    def state_feature_extract(self, state, action):
        position, velocity = state
        feature = []
        x = position - self.env.position_bound[0]
        y = velocity - self.env.velocity_bound[0]
        for i in range(self.tiling_num):
            # x_ = x - self.block_start_position[i][0]
            # y_ = y - self.block_start_position[i][1]
            x_ = x - i * self.width_step
            y_ = y - i * self.height_step
            x_position = int(x_ / self.block_width + 1)
            y_position = int(y_ / self.block_height + 1)
            org_feature = (i * self.tiling_block_num * self.tiling_block_num +
                           y_position * self.tiling_block_num + x_position) * 10 + action

            if org_feature in self.tiling_dict.keys():
                feature.append(self.tiling_dict[org_feature])
            else:
                self.tiling_dict[org_feature] = len(self.tiling_dict.keys())
                feature.append(self.tiling_dict[org_feature])
        return feature

    def select_action(self, state, epsilon=0.1):
        value_of_action_list = []
        policies = np.zeros(self.env.action_space.n)
        for action_iter in range(self.env.action_space.n):
            state_action_feature = self.state_feature_extract(state, action_iter)
            value_of_action_list.append(self.value_of_state_action(state_action_feature))
        value_of_action_list = np.array(value_of_action_list)
        optimal_action = np.random.choice(
            np.flatnonzero(value_of_action_list == value_of_action_list.max()))
        for action_iter in range(self.env.action_space.n):
            if action_iter == optimal_action:
                policies[action_iter] = 1 - epsilon + epsilon / self.env.action_space.n
            else:
                policies[action_iter] = epsilon / self.env.action_space.n
        probability_distribution = policies
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def running(self, iteration_times, n, alpha=0.01, gamma=0.9):
        total_step = []
        for iteration_time in range(iteration_times):
            step_num = 0
            state = self.env.reset()
            action = self.select_action(state)
            # get reward R and next state S'
            n_queue = collections.deque()
            while True:
                next_state, reward, is_done, _ = self.env.step(action)
                n_queue.append([state, reward, action])
                step_num += 1
                if is_done:
                    g = 0.
                    gamma_ = 1
                    for iter_q in n_queue:
                        g += gamma_ * iter_q[1]
                        gamma_ *= gamma
                    while len(n_queue) != 0:
                        state_2_update, r, action_2_update = n_queue.popleft()
                        state_action_feature = self.state_feature_extract(state_2_update, action_2_update)
                        delta_value = g - self.value_of_state_action(state_action_feature)
                        for idx_feature in state_action_feature:
                            self.value_of_state_action.weight[idx_feature] += alpha * delta_value * 1.0
                        g -= r
                        g /= gamma
                    break
                else:
                    if len(n_queue) < n + 1:
                        state = next_state
                        action = self.select_action(state)
                        continue
                    else:
                        next_action = self.select_action(next_state)
                        state_2_update, r, action_2_update = n_queue.popleft()
                        # calculate G
                        g = r
                        gamma_ = gamma
                        for iter_q in n_queue:
                            g += gamma_ * iter_q[1]
                            gamma_ *= gamma
                        next_state_action_feature = self.state_feature_extract(next_state, next_action)
                        state_action_feature = self.state_feature_extract(state_2_update, action_2_update)
                        g += self.value_of_state_action(next_state_action_feature) * gamma_
                        delta_value = g - self.value_of_state_action(state_action_feature)
                        for idx_feature in state_action_feature:
                            self.value_of_state_action.weight[idx_feature] += alpha * delta_value * 1.0
                        state = next_state
                        action = self.select_action(state)
            print(iteration_time, step_num)
            total_step.append(step_num)
        return np.array(total_step)


if __name__ == '__main__':
    env = MountainCar()
    repeat_times = 30
    n = 1
    episode_num = 200
    step_num_list = np.zeros(episode_num)
    for _ in range(repeat_times):
        print('1 round ' + str(_))
        agent = Agent(env)
        step_num_list += agent.running(episode_num, n, alpha=0.01 / 8.)
    plt.plot(step_num_list / float(repeat_times), c='g', alpha=0.7, label='n=1 and $\\alpha=0.1/8$ and $\\gamma=0.9$')

    n = 8
    step_num_list = np.zeros(episode_num)
    for _ in range(repeat_times):
        print('2 round ' + str(_))
        agent = Agent(env)
        step_num_list += agent.running(episode_num, n, alpha=0.01 / 8.)
    plt.plot(step_num_list / float(repeat_times), c='r', alpha=0.7, label='n=8 and $\\alpha=0.1/8$ and $\\gamma=0.9$')

    plt.legend()
    plt.show()
