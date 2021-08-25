import matplotlib.pyplot as plt
import numpy as np
from environment.mountain_car import MountainCar


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class LinearFunction:
    def __init__(self, n):
        self.n = n
        self.weight = np.zeros(n + 1)

    def __call__(self, x_1, x_2):
        sum = 0
        for i in x_1:
            sum += self.weight[i]
        sum += self.weight[self.n] * x_2
        return sum

    def update_weight(self, delta_value, alpha, x_1, x_2):
        for i in x_1:
            self.weight[i] += alpha * delta_value * 1
        self.weight[self.n] += alpha * delta_value * x_2


class Agent:
    def __init__(self, environment_):
        self.env = environment_
        self.tiling_block_num = 8
        self.tiling_num = 8
        self.value_of_state_action = LinearFunction(self.tiling_num * self.tiling_block_num * self.tiling_block_num)
        # parameters for feature extraction
        width = self.env.position_bound[1] - self.env.position_bound[0]
        height = self.env.velocity_bound[1] - self.env.velocity_bound[0]
        self.block_width = width / (self.tiling_block_num - 1)
        self.block_height = height / (self.tiling_block_num - 1)
        self.width_step = self.block_width / self.tiling_num
        self.height_step = self.block_height / self.tiling_num

    def state_feature_extract(self, state):
        position, velocity = state
        feature = []
        x = position - self.env.position_bound[0]
        y = velocity - self.env.velocity_bound[0]
        for i in range(self.tiling_num):
            x_ = x - i * self.width_step
            y_ = y - i * self.height_step
            x_position = int(x_ / self.block_width + 1)
            y_position = int(y_ / self.block_height + 1)
            feature.append(
                i * self.tiling_block_num * self.tiling_block_num + y_position * self.tiling_block_num + x_position)
        return feature

    def select_action(self, state_feature, epsilon=0.3):
        value_of_action_list = []
        policies = np.zeros(self.env.action_space.n)
        for action_iter in range(self.env.action_space.n):
            value_of_action_list.append(self.value_of_state_action(state_feature, action_iter))
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

    def running(self, iteration_times, alpha=0.01, gamma=0.9):
        total_step = []
        for iteration_time in range(iteration_times):
            step_num = 0
            state = self.env.reset()
            state_feature = self.state_feature_extract(state)
            action = self.select_action(state_feature)
            # get reward R and next state S'
            while True:
                next_state, reward, is_done, _ = self.env.step(action)
                step_num += 1
                next_state_feature = 0
                if next_state is not None:
                    next_state_feature = self.state_feature_extract(next_state)
                if is_done:
                    self.value_of_state_action.update_weight(
                        reward - self.value_of_state_action(state_feature, action),
                        alpha, state_feature, action)
                    break
                next_action = self.select_action(next_state_feature)
                self.value_of_state_action.update_weight(
                    reward + gamma * self.value_of_state_action(next_state_feature, next_action)
                    - self.value_of_state_action(state_feature, action), alpha, state_feature, action)
                state_feature = next_state_feature
                action = self.select_action(state_feature)
            total_step.append(step_num)
        return np.array(total_step)


if __name__ == '__main__':
    env = MountainCar()
    repeat_times = 30
    step_num_list = np.zeros(100)
    for _ in range(repeat_times):
        print('1 round ' + str(_))
        agent = Agent(env)
        step_num_list += agent.running(100, alpha=0.1 / 8.)
    plt.plot(step_num_list / float(repeat_times), c='g', alpha=0.7, label='$\\alpha$=0.1/8')

    step_num_list = np.zeros(100)
    for _ in range(repeat_times):
        print('2 round ' + str(_))
        agent = Agent(env)
        step_num_list += agent.running(100, alpha=0.2 / 8.)
    plt.plot(step_num_list / float(repeat_times), c='b', alpha=0.7, label='$\\alpha$=0.2/8')

    step_num_list = np.zeros(100)
    for _ in range(repeat_times):
        print('3 round ' + str(_))
        agent = Agent(env)
        step_num_list += agent.running(100, alpha=0.5 / 8.)
    plt.plot(step_num_list / float(repeat_times), c='r', alpha=0.7, label='$\\alpha$=0.5/8')

    plt.legend()
    plt.show()
